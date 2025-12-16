"""Voting strategies for ensemble learning."""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod


class VotingStrategy(ABC):
    """Abstract base class for voting strategies."""

    @abstractmethod
    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Combine predictions from multiple models.

        Args:
            predictions: Dictionary mapping model names to prediction tensors

        Returns:
            Combined predictions
        """
        pass


class HardVoting(VotingStrategy):
    """Hard voting strategy for classification ensembles.

    Combines binary predictions by majority vote.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize hard voting.

        Args:
            threshold: Threshold for converting probabilities to binary predictions
        """
        self.threshold = threshold

    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Combine predictions using majority voting.

        Args:
            predictions: Dictionary mapping model names to prediction tensors

        Returns:
            Binary predictions based on majority vote
        """
        # Convert probabilities to binary predictions if needed
        binary_preds = []
        for model_name, preds in predictions.items():
            if preds.dtype == torch.float:
                # Assume these are probabilities, convert to binary
                binary_pred = (preds >= self.threshold).int()
            else:
                # Assume these are already binary
                binary_pred = preds.int()
            binary_preds.append(binary_pred)

        # Stack predictions and compute majority vote
        stacked_preds = torch.stack(binary_preds, dim=0)  # [num_models, num_samples]
        majority_threshold = len(predictions) // 2 + 1
        ensemble_preds = (stacked_preds.sum(dim=0) >= majority_threshold).int()

        return ensemble_preds


class SoftVoting(VotingStrategy):
    """Soft voting strategy for classification ensembles.

    Combines probability predictions by averaging.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize soft voting.

        Args:
            threshold: Threshold for final binary classification
        """
        self.threshold = threshold

    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Combine predictions using probability averaging.

        Args:
            predictions: Dictionary mapping model names to prediction tensors
            probabilities: Dictionary mapping model names to probability tensors

        Returns:
            Dictionary with ensemble probabilities and predictions
        """
        if probabilities is not None:
            # Use provided probabilities
            prob_tensors = list(probabilities.values())
        else:
            # Assume predictions are probabilities
            prob_tensors = list(predictions.values())

        # Average probabilities across models
        stacked_probs = torch.stack(prob_tensors, dim=0)  # [num_models, num_samples]
        ensemble_probs = stacked_probs.mean(dim=0)  # [num_samples]

        # Convert to binary predictions
        ensemble_preds = (ensemble_probs >= self.threshold).int()

        return {
            'probabilities': ensemble_probs,
            'predictions': ensemble_preds
        }


class WeightedVoting(VotingStrategy):
    """Weighted voting strategy for ensembles.

    Combines predictions using model-specific weights.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        """Initialize weighted voting.

        Args:
            weights: Dictionary mapping model names to weights
            threshold: Threshold for final binary classification
        """
        self.weights = weights
        self.threshold = threshold

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set or update model weights.

        Args:
            weights: Dictionary mapping model names to weights
        """
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}

    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Combine predictions using weighted averaging.

        Args:
            predictions: Dictionary mapping model names to prediction tensors
            probabilities: Dictionary mapping model names to probability tensors

        Returns:
            Dictionary with ensemble probabilities and predictions
        """
        if self.weights is None:
            raise ValueError("Weights must be set before combining predictions")

        if probabilities is not None:
            prob_dict = probabilities
        else:
            prob_dict = predictions

        # Weighted average of probabilities
        ensemble_probs = None
        for model_name, probs in prob_dict.items():
            weight = self.weights.get(model_name, 0.0)
            if ensemble_probs is None:
                ensemble_probs = probs * weight
            else:
                ensemble_probs += probs * weight

        # Convert to binary predictions
        ensemble_preds = (ensemble_probs >= self.threshold).int()

        return {
            'probabilities': ensemble_probs,
            'predictions': ensemble_preds
        }


class WeightedRegressionVoting:
    """Weighted voting strategy for regression ensembles."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize weighted regression voting.

        Args:
            weights: Dictionary mapping model names to weights
        """
        self.weights = weights

    def set_weights_from_performance(
        self,
        performance_metrics: Dict[str, float],
        metric_type: str = "mse",
        epsilon: float = 1e-8
    ) -> None:
        """Set weights based on model performance.

        Args:
            performance_metrics: Dictionary mapping model names to performance scores
            metric_type: Type of metric ("mse", "mae", "r2")
            epsilon: Small value to avoid division by zero
        """
        if metric_type.lower() in ["mse", "mae"]:
            # For error metrics, use inverse weights (lower error = higher weight)
            inv_metrics = {k: 1 / (v + epsilon) for k, v in performance_metrics.items()}
            total_inv = sum(inv_metrics.values())
            self.weights = {k: v / total_inv for k, v in inv_metrics.items()}
        elif metric_type.lower() == "r2":
            # For R² metric, use direct weights (higher R² = higher weight)
            total_r2 = sum(performance_metrics.values())
            self.weights = {k: v / total_r2 for k, v in performance_metrics.items()}
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combine regression predictions using weighted averaging.

        Args:
            predictions: Dictionary mapping model names to prediction tensors

        Returns:
            Weighted ensemble predictions
        """
        if self.weights is None:
            # Equal weights if not specified
            self.weights = {k: 1.0 / len(predictions) for k in predictions.keys()}

        ensemble_preds = None
        for model_name, preds in predictions.items():
            weight = self.weights.get(model_name, 0.0)
            if ensemble_preds is None:
                ensemble_preds = preds * weight
            else:
                ensemble_preds += preds * weight

        return ensemble_preds


class AdaptiveVoting:
    """Adaptive voting strategy that adjusts weights based on confidence."""

    def __init__(self, base_strategy: VotingStrategy):
        """Initialize adaptive voting.

        Args:
            base_strategy: Base voting strategy to adapt
        """
        self.base_strategy = base_strategy

    def combine_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        confidences: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Combine predictions with confidence-based adaptation.

        Args:
            predictions: Dictionary mapping model names to prediction tensors
            confidences: Dictionary mapping model names to confidence scores

        Returns:
            Adapted ensemble predictions
        """
        if confidences is not None:
            # Weight predictions by confidence
            weighted_predictions = {}
            for model_name, preds in predictions.items():
                conf = confidences.get(model_name, torch.ones_like(preds))
                weighted_predictions[model_name] = preds * conf

            return self.base_strategy.combine_predictions(weighted_predictions, **kwargs)
        else:
            return self.base_strategy.combine_predictions(predictions, **kwargs)