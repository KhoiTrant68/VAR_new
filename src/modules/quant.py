from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import distributed
from torch import nn as nn
from torch.nn import functional as F

__all__ = [
    "VectorQuantizer",
]


class Phi(nn.Conv2d):
    """Quantization residual convolutional layer."""

    def __init__(self, embed_dim: int, quant_resi: float):
        """
        Initialize the Phi layer.

        Args:
            embed_dim (int): Embedding dimension
            quant_resi (float): Residual quantization ratio
        """
        kernel_size = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            h_BChw (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed tensor
        """
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(
            self.resi_ratio
        )


class PhiShared(nn.Module):
    """Shared Phi layer for quantization."""

    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """Partially shared Phi layers with intelligent selection."""

    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):
    """Non-shared Phi layers with scale-aware selection."""

    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(
            np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()
        )

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class VectorQuantizer(nn.Module):
    """
    Advanced Vector Quantization module with multi-scale support.

    This implementation supports:
    - Multi-scale vector quantization
    - Normalized and non-normalized embedding
    - Flexible quantization residual handling
    """

    def __init__(
        self,
        vocab_size: int,
        Cvae: int,
        using_znorm: bool,
        beta: float = 0.25,
        default_qresi_counts: int = 0,
        v_patch_nums: Optional[Tuple[int, ...]] = None,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int, ...] = v_patch_nums or []

        self.quant_resi_ratio = quant_resi
        self.quant_resi = self._create_quantization_residual(
            Cvae, quant_resi, default_qresi_counts, share_quant_resi
        )

        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        self.prog_si = -1

    def _create_quantization_residual(
        self,
        Cvae: int,
        quant_resi: float,
        default_qresi_counts: int,
        share_quant_resi: int,
    ):
        """Create quantization residual based on sharing strategy."""
        phi_creator = lambda: (
            Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
        )

        if share_quant_resi == 0:
            return PhiNonShared(
                [
                    phi_creator()
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:
            return PhiShared(phi_creator())
        else:
            return PhiPartiallyShared(
                nn.ModuleList([phi_creator() for _ in range(share_quant_resi)])
            )

    def eini(self, eini: float):
        """
        Initialize embedding weights.

        Args:
            eini (float): Initialization scale
        """
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight, std=eini)
        else:
            nn.init.uniform_(
                self.embedding.weight,
                -abs(eini) / self.vocab_size,
                abs(eini) / self.vocab_size,
            )

    def extra_repr(self) -> str:
        return (
            f"{self.v_patch_nums}, znorm={self.using_znorm}, "
            f"beta={self.beta} | S={len(self.v_patch_nums)}, "
            f"quant_resi={self.quant_resi_ratio}"
        )

    def forward(
        self, f_BChw: torch.Tensor, ret_usages: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[float]], torch.Tensor]:
        """
        Forward pass for vector quantization.

        Args:
            f_BChw (torch.Tensor): Input feature tensor
            ret_usages (bool): Whether to return vocab usage stats

        Returns:
            Tuple of quantized features, usage stats, and VQ loss
        """
        # Ensure float32 precision
        f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape

        with torch.no_grad():
            f_no_grad = f_BChw.clone()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        mean_vq_loss: torch.Tensor = 0.0
        vocab_hit_V = torch.zeros(
            self.vocab_size, dtype=torch.float, device=f_BChw.device
        )
        SN = len(self.v_patch_nums)

        for si, pn in enumerate(self.v_patch_nums):
            # Compute embedding distances
            rest_NC = (
                F.interpolate(f_rest, size=(pn, pn), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )

            if self.using_znorm:
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(
                    rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(
                    rest_NC.square(), dim=1, keepdim=True
                ) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(
                    rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )
                idx_N = torch.argmin(d_no_grad, dim=1)

            hit_V = idx_N.bincount(minlength=self.vocab_size).float()

            # Distributed training sync
            if self.training and dist.initialized():
                distributed.all_reduce(hit_V, async_op=True)

            # Quantization and reconstruction
            idx_Bhw = idx_N.view(B, pn, pn)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)

            # Update vocabulary hit statistics
            vocab_hit_V.add_(hit_V)
            mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(
                f_hat, f_no_grad
            )

        mean_vq_loss *= 1.0 / SN
        f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        # Compute usage statistics
        margin = (
            distributed.get_world_size()
            * (f_BChw.numel() / f_BChw.shape[1])
            / self.vocab_size
            * 0.08
        )

        usages = None
        if ret_usages:
            usages = [
                (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                for si, pn in enumerate(self.v_patch_nums)
            ]

        return f_hat, usages, mean_vq_loss
