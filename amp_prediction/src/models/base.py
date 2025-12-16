"""Base classes and utilities for neural network models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAMPModel(nn.Module, ABC):
    """Base class for all AMP prediction models."""

    def __init__(self, embedding_dim: int = 1280, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameters count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout
        }

    def save_model(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str, device: str = 'cpu') -> None:
        """Load model state dict."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()