"""Ensemble learning methods for AMP prediction."""

from .ensemble_classifier import EnsembleClassifier
from .ensemble_regressor import EnsembleRegressor
from .voting import SoftVoting, HardVoting, WeightedVoting

__all__ = [
    'EnsembleClassifier',
    'EnsembleRegressor',
    'SoftVoting',
    'HardVoting',
    'WeightedVoting'
]