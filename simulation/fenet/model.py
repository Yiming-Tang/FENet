"""Deterministic Gram features and the empirical 256--64 classifier."""

from __future__ import annotations

import torch
from torch import nn


def build_v_matrix(
    window_length: int,
    convention: str = "paper",
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return V with shape (T, T-1).

    ``paper`` uses v_t[i] = I(i < t)/T - t/T**2, with t=1,...,T-1.
    ``legacy-notebook`` reproduces the original notebook's constant tail
    -1/T**2. It is provided for compatibility and is a different embedding.
    """
    if window_length < 2:
        raise ValueError("window_length must be at least 2.")
    if convention not in {"paper", "legacy-notebook"}:
        raise ValueError("convention must be 'paper' or 'legacy-notebook'.")
    t = torch.arange(1, window_length, dtype=dtype, device=device)
    row = torch.arange(window_length, device=device).unsqueeze(1)
    first = row < t.unsqueeze(0)
    if convention == "paper":
        return first.to(dtype) / window_length - t.unsqueeze(0) / window_length**2
    return torch.where(first, 1 / window_length - t / window_length**2, -1 / window_length**2)


def embedding_from_gram(gram: torch.Tensor, convention: str = "paper") -> torch.Tensor:
    """Flatten S V in row-major order for square Gram matrices (..., T, T).

    The paper convention uses prefix sums and costs O(T**2) after forming S.
    No quadrature weights or normalization are applied to S by this function.
    """
    if gram.ndim < 2 or gram.shape[-1] != gram.shape[-2] or gram.shape[-1] < 2:
        raise ValueError("gram must have square final dimensions of size at least 2.")
    if not gram.is_floating_point():
        raise TypeError("gram must be a floating-point tensor.")
    t = gram.shape[-1]
    if convention == "paper":
        prefix = gram.cumsum(dim=-1)[..., :-1]
        total = gram.sum(dim=-1, keepdim=True)
        split = torch.arange(1, t, dtype=gram.dtype, device=gram.device)
        embedded = prefix / t - total * split / t**2
    else:
        embedded = gram @ build_v_matrix(t, convention, dtype=gram.dtype, device=gram.device)
    return embedded.flatten(start_dim=-2)


def gram_embedding(observations: torch.Tensor, convention: str = "paper") -> torch.Tensor:
    """Embed curves shaped (..., T, J), using S = X X^T as in the source code."""
    if observations.ndim < 2:
        raise ValueError("observations must have shape (..., T, J).")
    return embedding_from_gram(observations @ observations.transpose(-1, -2), convention)


class FENet(nn.Module):
    """MLP for precomputed features of dimension T(T-1); output is a logit."""

    def __init__(self, window_length: int, hidden_dims: tuple[int, ...] = (256, 64)):
        super().__init__()
        if window_length < 2 or any(width < 1 for width in hidden_dims):
            raise ValueError("Invalid window length or hidden-layer width.")
        self.window_length = int(window_length)
        self.hidden_dims = tuple(hidden_dims)
        dimensions = [window_length * (window_length - 1), *hidden_dims, 1]
        layers: list[nn.Module] = []
        for index, (incoming, outgoing) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(incoming, outgoing))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
