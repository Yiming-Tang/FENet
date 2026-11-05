"""Functional embedding neural networks for change-point detection."""

from .model import FENet, build_v_matrix, embedding_from_gram, gram_embedding
from .training import classification_metrics, fit_classifier, predict_probabilities, set_seed
from .detection import detect_changepoints

__all__ = [
    "FENet", "build_v_matrix", "embedding_from_gram", "gram_embedding",
    "classification_metrics", "fit_classifier", "predict_probabilities", "set_seed",
    "detect_changepoints",
]
