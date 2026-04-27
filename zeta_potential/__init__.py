"""Public API for the zeta potential demo."""

from .pipeline import predict_bulk, predict_single

__all__ = [
    "predict_single",
    "predict_bulk",
]
