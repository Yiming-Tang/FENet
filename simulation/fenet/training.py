"""Small reusable training and classification helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without promising bitwise GPU equality."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_classifier(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Fit supplied model with Adam and BCEWithLogitsLoss; return epoch losses.

    Seed the model construction separately with ``set_seed`` if its initial
    weights also need to be repeatable. Features are cached on the CPU.
    """
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Expected features (N, P) and labels (N,).")
    if not len(labels) or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Training data and positive hyperparameters are required.")
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError("Training features and labels must be finite.")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary.")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features.detach().float().cpu(), labels.detach().float().cpu()),
        batch_size=batch_size, shuffle=True, generator=generator,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        history.append({"epoch": epoch, "loss": total_loss / len(labels)})
    return history


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, features: torch.Tensor, *, batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return CPU probabilities in input order."""
    if features.ndim != 2 or batch_size < 1:
        raise ValueError("Expected a feature matrix and a positive batch size.")
    model.to(device).eval()
    chunks = [torch.sigmoid(model(chunk.float().to(device))).cpu()
              for chunk in features.split(batch_size)]
    return torch.cat(chunks) if chunks else torch.empty(0)


def classification_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Compute binary metrics; undefined denominators are reported as NaN."""
    y = np.asarray(labels).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError("Expected nonempty labels and finite probabilities of equal length.")
    if not np.isin(y, [0, 1]).all() or not 0 <= threshold <= 1:
        raise ValueError("Expected binary labels and threshold in [0, 1].")
    prediction = p >= threshold
    tp = int(((y == 1) & prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    ratio = lambda a, b: a / b if b else float("nan")
    return {
        "Accuracy": (tp + tn) / len(y), "TPR": ratio(tp, tp + fn),
        "FAR": ratio(fp, fp + tn), "Precision": ratio(tp, tp + fp),
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }
