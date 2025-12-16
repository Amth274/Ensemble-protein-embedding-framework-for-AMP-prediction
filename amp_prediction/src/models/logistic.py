"""Logistic regression model for AMP prediction."""

import torch
import torch.nn as nn
from .base import BaseAMPModel


class LogisticRegression(BaseAMPModel):
    """Logistic regression classifier for antimicrobial peptide prediction.

    Simple linear model that operates on mean-pooled embeddings.
    Serves as a baseline model for comparison with more complex architectures.
    """

    def __init__(self, input_dim: int = 1280):
        super().__init__(input_dim, dropout=0.0)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the logistic regression model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
               or [batch_size, embedding_dim] for pre-pooled inputs

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        # If input is 3D, apply mean pooling over sequence dimension
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, D]

        return self.linear(x).squeeze(1)  # [B]