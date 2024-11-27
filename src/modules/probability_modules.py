import torch
from torch import nn
from torch.nn import functional as F


def sample_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples=1,
) -> torch.Tensor:
    B, l, V = logits_BlV.shape

    # Apply top-k filtering if needed
    if top_k > 0:
        top_k_values, _ = logits_BlV.topk(top_k, dim=-1, largest=True, sorted=False)
        logits_BlV = logits_BlV.masked_fill(
            logits_BlV < top_k_values[..., -1, None], -float("inf")
        )

    # Apply top-p (nucleus) sampling if needed
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs > (1 - top_p)
        mask[..., 0] = False  # Ensure the most probable token remains
        sorted_logits.masked_fill_(mask, -float("inf"))
        logits_BlV = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    probs = logits_BlV.softmax(dim=-1)
    return torch.multinomial(
        probs.view(-1, V),
        num_samples=num_samples,
        replacement=num_samples >= 0,
        generator=rng,
    ).view(B, l, num_samples)


def gumbel_softmax_with_rng(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    rng: torch.Generator = None,
) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = -torch.empty_like(logits).exponential_(generator=rng).log()
    logits = (logits + gumbels) / tau
    y_soft = logits.softmax(dim)

    if hard:
        _, index = y_soft.max(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    random_tensor = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(
        keep_prob
    )

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor
