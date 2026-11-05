"""Sliding-window localization adapted from the simulation notebook."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import gram_embedding


@torch.no_grad()
def detect_changepoints(
    model: nn.Module,
    observations: torch.Tensor,
    *,
    window_size: int | None = None,
    gamma: float = 0.5,
    smooth_window: int = 5,
    convention: str = "paper",
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, list]:
    """Score one (T*, J) sequence and return source-style change locations.

    Scores receive a centered, zero-padded moving average (NumPy ``same``),
    then each maximal segment >= gamma yields its earliest maximum plus
    floor(window_size/2). Locations are zero-based curve indices. This follows
    the notebook's practical localization, not the trailing average in the
    manuscript Algorithm 1. A model must be trained at the same window length.
    """
    window_size = window_size or getattr(model, "window_length", None)
    if observations.ndim != 2 or window_size is None or window_size < 2:
        raise ValueError("Expected observations (T*, J) and window_size >= 2.")
    if len(observations) < window_size or batch_size < 1 or smooth_window < 1:
        raise ValueError("Sequence, batch size, or smoothing width is invalid.")
    if getattr(model, "window_length", window_size) != window_size:
        raise ValueError("Detection and training window lengths must match.")
    if not 0 <= gamma <= 1 or not torch.isfinite(observations).all():
        raise ValueError("Expected finite observations and gamma in [0, 1].")
    model.to(device).eval()
    count = len(observations) - window_size + 1
    if smooth_window > count:
        raise ValueError("smooth_window cannot exceed the number of sliding windows.")
    probabilities = []
    for start in range(0, count, batch_size):
        windows = torch.stack([observations[i:i + window_size]
                               for i in range(start, min(start + batch_size, count))])
        features = gram_embedding(windows.float().to(device), convention)
        probabilities.extend(torch.sigmoid(model(features)).cpu().tolist())
    smoothed = np.convolve(probabilities, np.ones(smooth_window) / smooth_window, mode="same")
    above = smoothed >= gamma
    boundaries = np.diff(np.r_[False, above, False].astype(int))
    starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    locations = [int(start + np.argmax(smoothed[start:stop]) + window_size // 2)
                 for start, stop in zip(starts, stops)]
    return {"changepoints": locations, "probabilities": probabilities,
            "smoothed_probabilities": smoothed.tolist()}
