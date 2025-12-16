"""Ensemble regressor for AMP MIC prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import logging

from ..models.base import BaseAMPModel
from .voting import WeightedRegressionVoting

logger = logging.getLogger(__name__)


class EnsembleRegressor:
    """Ensemble regressor for antimicrobial peptide MIC prediction."""

    def __init__(
        self,
        models: Dict[str, BaseAMPModel],
        device: str = "auto"
    ):
        """Initialize ensemble regressor.

        Args:
            models: Dictionary mapping model names to model instances
            device: Device to run inference on
        """
        self.models = models
        self.device = self._get_device(device)
        self.weighted_voter = WeightedRegressionVoting()
        self.model_weights = None

        # Move models to device
        for model in self.models.values():
            model.to(self.device)

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def set_model_weights(
        self,
        weights: Dict[str, float]
    ) -> None:
        """Set model weights for weighted averaging.

        Args:
            weights: Dictionary mapping model names to weights
        """
        self.weighted_voter.weights = weights
        self.model_weights = weights

    def set_weights_from_performance(
        self,
        performance_metrics: Dict[str, float],
        metric_type: str = "mse"
    ) -> None:
        """Set weights based on model performance.

        Args:
            performance_metrics: Dictionary mapping model names to performance scores
            metric_type: Type of metric ("mse", "mae", "r2")
        """
        self.weighted_voter.set_weights_from_performance(
            performance_metrics, metric_type
        )
        self.model_weights = self.weighted_voter.weights

    def predict_single_model(
        self,
        model_name: str,
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """Get predictions from a single model.

        Args:
            model_name: Name of the model
            dataloader: DataLoader for the dataset

        Returns:
            Dictionary with predictions and targets
        """
        model = self.models[model_name]
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # x, label, value
                    x, _, values = batch
                    x, values = x.to(self.device), values.to(self.device)
                    targets = values
                else:  # x, targets
                    x, targets = batch
                    x, targets = x.to(self.device), targets.to(self.device)

                # Handle different model input requirements
                if model_name.lower() == 'logistic':
                    # Logistic model needs mean pooled input
                    x_input = x.mean(dim=1)
                else:
                    x_input = x

                predictions = model(x_input)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return {
            'predictions': all_predictions,
            'targets': all_targets
        }

    def predict_all_models(
        self,
        dataloader: DataLoader
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get predictions from all models.

        Args:
            dataloader: DataLoader for the dataset

        Returns:
            Dictionary mapping model names to their predictions
        """
        all_predictions = {}

        for model_name in tqdm(self.models.keys(), desc="Getting model predictions"):
            predictions = self.predict_single_model(model_name, dataloader)
            all_predictions[model_name] = predictions

        return all_predictions

    def predict_ensemble(
        self,
        dataloader: DataLoader,
        method: str = "weighted"
    ) -> Dict[str, torch.Tensor]:
        """Get ensemble predictions.

        Args:
            dataloader: DataLoader for the dataset
            method: Ensemble method ("weighted", "average")

        Returns:
            Dictionary with ensemble predictions
        """
        # Get predictions from all models
        all_predictions = self.predict_all_models(dataloader)

        # Extract model predictions
        model_predictions = {}
        targets = None

        for model_name, results in all_predictions.items():
            model_predictions[model_name] = results['predictions']
            if targets is None:
                targets = results['targets']

        # Combine predictions
        if method == "weighted" and self.model_weights is not None:
            ensemble_preds = self.weighted_voter.combine_predictions(model_predictions)
        else:
            # Simple averaging
            pred_tensors = list(model_predictions.values())
            stacked_preds = torch.stack(pred_tensors, dim=0)
            ensemble_preds = stacked_preds.mean(dim=0)

        return {
            'predictions': ensemble_preds,
            'targets': targets,
            'model_predictions': model_predictions
        }

    def evaluate_single_model(
        self,
        model_name: str,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate a single model.

        Args:
            model_name: Name of the model to evaluate
            dataloader: DataLoader for the dataset

        Returns:
            Dictionary with evaluation metrics
        """
        results = self.predict_single_model(model_name, dataloader)

        y_true = results['targets'].numpy()
        y_pred = results['predictions'].numpy()

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        # Calculate Pearson correlation
        if len(y_true) > 1:
            r, _ = pearsonr(y_true, y_pred)
            metrics['pearson_r'] = r
        else:
            metrics['pearson_r'] = 0.0

        return metrics

    def evaluate_ensemble(
        self,
        dataloader: DataLoader,
        method: str = "weighted"
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate ensemble performance.

        Args:
            dataloader: DataLoader for the dataset
            method: Ensemble method ("weighted", "average")

        Returns:
            Dictionary with ensemble and individual model metrics
        """
        ensemble_results = self.predict_ensemble(dataloader, method)

        y_true = ensemble_results['targets'].numpy()
        y_pred = ensemble_results['predictions'].numpy()

        # Ensemble metrics
        ensemble_metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        # Calculate Pearson correlation
        if len(y_true) > 1:
            r, _ = pearsonr(y_true, y_pred)
            ensemble_metrics['pearson_r'] = r
        else:
            ensemble_metrics['pearson_r'] = 0.0

        # Individual model metrics
        individual_metrics = {}
        for model_name in self.models.keys():
            individual_metrics[model_name] = self.evaluate_single_model(
                model_name, dataloader
            )

        return {
            'ensemble': ensemble_metrics,
            'individual': individual_metrics
        }

    def get_top_predictions(
        self,
        dataloader: DataLoader,
        k: int = 50,
        method: str = "weighted",
        criterion: str = "lowest"
    ) -> Dict[str, np.ndarray]:
        """Get top-k predictions based on specified criterion.

        Args:
            dataloader: DataLoader for the dataset
            k: Number of top predictions to return
            method: Ensemble method ("weighted", "average")
            criterion: Selection criterion ("lowest", "highest")

        Returns:
            Dictionary with top-k results
        """
        ensemble_results = self.predict_ensemble(dataloader, method)

        predictions = ensemble_results['predictions'].numpy()
        targets = ensemble_results['targets'].numpy()

        # Get indices for top-k predictions
        if criterion == "lowest":
            top_indices = np.argsort(predictions)[:k]
        elif criterion == "highest":
            top_indices = np.argsort(predictions)[-k:]
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return {
            'indices': top_indices,
            'predictions': predictions[top_indices],
            'targets': targets[top_indices]
        }

    def save_ensemble(self, save_dir: str) -> None:
        """Save ensemble models.

        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{model_name}_regression.pt")
            model.save_model(save_path)

        # Save ensemble configuration
        config = {
            'model_weights': self.model_weights,
            'model_names': list(self.models.keys())
        }

        config_path = os.path.join(save_dir, "ensemble_regressor_config.pt")
        torch.save(config, config_path)

        logger.info(f"Ensemble regressor saved to {save_dir}")

    def load_ensemble(self, save_dir: str) -> None:
        """Load ensemble models.

        Args:
            save_dir: Directory to load models from
        """
        import os

        # Load ensemble configuration
        config_path = os.path.join(save_dir, "ensemble_regressor_config.pt")
        config = torch.load(config_path)

        self.model_weights = config['model_weights']

        # Load individual models
        for model_name in config['model_names']:
            if model_name in self.models:
                model_path = os.path.join(save_dir, f"{model_name}_regression.pt")
                self.models[model_name].load_model(model_path, self.device)

        # Set weights if available
        if self.model_weights:
            self.set_model_weights(self.model_weights)

        logger.info(f"Ensemble regressor loaded from {save_dir}")

    def predict_with_uncertainty(
        self,
        dataloader: DataLoader,
        method: str = "weighted",
        n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Get ensemble predictions with uncertainty estimates.

        Args:
            dataloader: DataLoader for the dataset
            method: Ensemble method
            n_samples: Number of bootstrap samples for uncertainty

        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        # Get base predictions
        base_results = self.predict_ensemble(dataloader, method)

        # Bootstrap sampling for uncertainty estimation
        model_predictions = base_results['model_predictions']
        n_models = len(model_predictions)
        n_samples_per_model = max(1, n_samples // n_models)

        bootstrap_predictions = []

        for _ in range(n_samples_per_model):
            # Randomly sample models with replacement
            sampled_models = np.random.choice(
                list(model_predictions.keys()),
                size=n_models,
                replace=True
            )

            # Get predictions from sampled models
            sampled_preds = []
            for model_name in sampled_models:
                sampled_preds.append(model_predictions[model_name])

            # Average sampled predictions
            stacked_preds = torch.stack(sampled_preds, dim=0)
            bootstrap_pred = stacked_preds.mean(dim=0)
            bootstrap_predictions.append(bootstrap_pred)

        # Calculate uncertainty as standard deviation
        bootstrap_stack = torch.stack(bootstrap_predictions, dim=0)
        uncertainty = bootstrap_stack.std(dim=0)

        return {
            'predictions': base_results['predictions'],
            'targets': base_results['targets'],
            'uncertainty': uncertainty,
            'confidence_interval': {
                'lower': base_results['predictions'] - 1.96 * uncertainty,
                'upper': base_results['predictions'] + 1.96 * uncertainty
            }
        }