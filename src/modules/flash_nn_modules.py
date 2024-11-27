import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import DropPath

__all__ = ["FFN", "AdaLNSelfAttn", "AdaLNBeforeHead"]

# Import fused operators if available
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    fused_mlp_func = None

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from torch.nn.functional import scaled_dot_product_attention as slow_attn
except ImportError:

    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query @ key.transpose(-2, -1) * scale
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p, inplace=True)
        return attn @ value


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                )
            )
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
    ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim) if not attn_l2_norm else 1

        self.mat_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.zero_k_bias = torch.zeros(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

    def forward(self, x, attn_bias=None):
        B, L, C = x.shape

        # Compute QKV with bias
        qkv = F.linear(
            x,
            self.mat_qkv.weight,
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)
        dropout_p = self.attn_drop if self.training else 0.0

        if self.using_flash:
            return self.proj_drop(
                self.proj(
                    flash_attn_func(
                        q, k, v, dropout_p=dropout_p, softmax_scale=self.scale
                    )
                )
            )
        elif self.using_xform:
            return self.proj_drop(
                self.proj(
                    memory_efficient_attention(
                        q, k, v, attn_bias, dropout_p, scale=self.scale
                    )
                )
            )
        else:
            attn_output = slow_attn(
                q, k, v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p
            )
            return self.proj_drop(self.proj(attn_output))


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        cond_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.attn = SelfAttention(
            embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.ffn = FFN(embed_dim, int(embed_dim * mlp_ratio), drop=drop)
        self.norm = nn.LayerNorm(embed_dim)
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cond, attn_bias):
        cond_params = self.ada_lin(cond).view(-1, 1, 6, x.size(-1)).unbind(2)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = cond_params

        norm_x = self.norm(x)
        x = x + self.drop_path(
            self.attn(norm_x * (1 + scale1) + shift1, attn_bias) * gamma1
        )
        norm_x = self.norm(x)
        x = x + self.drop_path(self.ffn(norm_x * (1 + scale2) + shift2) * gamma2)
        return x
