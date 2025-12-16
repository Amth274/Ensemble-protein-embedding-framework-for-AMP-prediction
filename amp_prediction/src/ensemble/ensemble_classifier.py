"""Ensemble classifier for AMP prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import logging

from ..models.base import BaseAMPModel
from .voting import SoftVoting, HardVoting, WeightedVoting

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """Ensemble classifier combining multiple AMP prediction models."""

    def __init__(
        self,
        models: Dict[str, BaseAMPModel],
        device: str = "auto",
        voting_strategy: str = "soft"
    ):
        """Initialize ensemble classifier.

        Args:
            models: Dictionary mapping model names to model instances
            device: Device to run inference on
            voting_strategy: Voting strategy ("soft", "hard", "weighted")
        """
        self.models = models
        self.device = self._get_device(device)
        self.voting_strategy = voting_strategy
        self.model_weights = None

        # Move models to device
        for model in self.models.values():
            model.to(self.device)

        # Initialize voting strategy
        self._init_voting_strategy()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _init_voting_strategy(self) -> None:
        """Initialize the voting strategy."""
        if self.voting_strategy == "soft":
            self.voter = SoftVoting()
        elif self.voting_strategy == "hard":
            self.voter = HardVoting()
        elif self.voting_strategy == "weighted":
            self.voter = WeightedVoting()
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def set_model_weights(self, weights: Dict[str, float]) -> None:
        """Set model weights for weighted voting.

        Args:
            weights: Dictionary mapping model names to weights
        """
        if isinstance(self.voter, WeightedVoting):
            self.voter.set_weights(weights)
            self.model_weights = weights
        else:
            logger.warning("Model weights can only be set for weighted voting strategy")

    def predict_single_model(
        self,
        model_name: str,
        dataloader: DataLoader,
        return_probabilities: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get predictions from a single model.

        Args:
            model_name: Name of the model
            dataloader: DataLoader for the dataset
            return_probabilities: Whether to return probabilities

        Returns:
            Dictionary with predictions and optionally probabilities
        """
        model = self.models[model_name]
        model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x, y, _ = batch  # For regression datasets with values
                    x, y = x.to(self.device), y.to(self.device)

                # Handle different model input requirements
                if model_name.lower() == 'logistic':
                    # Logistic model needs mean pooled input
                    x_input = x.mean(dim=1)
                else:
                    x_input = x

                logits = model(x_input)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        results = {'logits': all_logits, 'labels': all_labels}

        if return_probabilities:
            probabilities = torch.sigmoid(all_logits)
            results['probabilities'] = probabilities

        return results

    def predict_all_models(
        self,
        dataloader: DataLoader,
        return_probabilities: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get predictions from all models.

        Args:
            dataloader: DataLoader for the dataset
            return_probabilities: Whether to return probabilities

        Returns:
            Dictionary mapping model names to their predictions
        """
        all_predictions = {}

        for model_name in tqdm(self.models.keys(), desc="Getting model predictions"):
            predictions = self.predict_single_model(
                model_name, dataloader, return_probabilities
            )
            all_predictions[model_name] = predictions

        return all_predictions

    def predict_ensemble(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Get ensemble predictions.

        Args:
            dataloader: DataLoader for the dataset
            threshold: Threshold for binary classification

        Returns:
            Dictionary with ensemble predictions and metadata
        """
        # Get predictions from all models
        all_predictions = self.predict_all_models(dataloader)

        # Extract predictions and probabilities
        model_predictions = {}
        model_probabilities = {}
        labels = None

        for model_name, results in all_predictions.items():
            model_predictions[model_name] = (results['probabilities'] >= threshold).int()
            model_probabilities[model_name] = results['probabilities']
            if labels is None:
                labels = results['labels']

        # Combine using voting strategy
        if self.voting_strategy in ["soft", "weighted"]:
            ensemble_results = self.voter.combine_predictions(
                model_predictions,
                probabilities=model_probabilities
            )
            ensemble_probs = ensemble_results['probabilities']
            ensemble_preds = ensemble_results['predictions']
        else:  # hard voting
            ensemble_preds = self.voter.combine_predictions(model_predictions)
            ensemble_probs = None

        results = {
            'predictions': ensemble_preds,
            'labels': labels,
            'model_predictions': model_predictions,
            'model_probabilities': model_probabilities
        }

        if ensemble_probs is not None:
            results['probabilities'] = ensemble_probs

        return results

    def evaluate_single_model(
        self,
        model_name: str,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate a single model.

        Args:
            model_name: Name of the model to evaluate
            dataloader: DataLoader for the dataset
            threshold: Threshold for binary classification

        Returns:
            Dictionary with evaluation metrics
        """
        results = self.predict_single_model(model_name, dataloader)

        y_true = results['labels'].numpy()
        y_probs = results['probabilities'].numpy()
        y_pred = (y_probs >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_probs)
        }

        return metrics

    def evaluate_ensemble(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate ensemble performance.

        Args:
            dataloader: DataLoader for the dataset
            threshold: Threshold for binary classification

        Returns:
            Dictionary with ensemble and individual model metrics
        """
        ensemble_results = self.predict_ensemble(dataloader, threshold)

        y_true = ensemble_results['labels'].numpy()
        y_pred = ensemble_results['predictions'].numpy()

        # Ensemble metrics
        ensemble_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

        if 'probabilities' in ensemble_results:
            y_probs = ensemble_results['probabilities'].numpy()
            ensemble_metrics['roc_auc'] = roc_auc_score(y_true, y_probs)

        # Individual model metrics
        individual_metrics = {}
        for model_name in self.models.keys():
            individual_metrics[model_name] = self.evaluate_single_model(
                model_name, dataloader, threshold
            )

        return {
            'ensemble': ensemble_metrics,
            'individual': individual_metrics
        }

    def save_ensemble(self, save_dir: str) -> None:
        """Save ensemble models.

        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{model_name}.pt")
            model.save_model(save_path)

        # Save ensemble configuration
        config = {
            'voting_strategy': self.voting_strategy,
            'model_weights': self.model_weights,
            'model_names': list(self.models.keys())
        }

        config_path = os.path.join(save_dir, "ensemble_config.pt")
        torch.save(config, config_path)

        logger.info(f"Ensemble saved to {save_dir}")

    def load_ensemble(self, save_dir: str) -> None:
        """Load ensemble models.

        Args:
            save_dir: Directory to load models from
        """
        import os

        # Load ensemble configuration
        config_path = os.path.join(save_dir, "ensemble_config.pt")
        config = torch.load(config_path)

        self.voting_strategy = config['voting_strategy']
        self.model_weights = config['model_weights']

        # Load individual models
        for model_name in config['model_names']:
            if model_name in self.models:
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                self.models[model_name].load_model(model_path, self.device)

        # Reinitialize voting strategy
        self._init_voting_strategy()
        if self.model_weights:
            self.set_model_weights(self.model_weights)

        logger.info(f"Ensemble loaded from {save_dir}")