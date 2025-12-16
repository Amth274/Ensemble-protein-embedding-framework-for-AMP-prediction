"""CNN-based models for AMP prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAMPModel


class CNN1DAMPClassifier(BaseAMPModel):
    """1D CNN classifier for antimicrobial peptide prediction.

    This model uses three convolutional layers with different kernel sizes
    to capture multi-scale features from protein sequences.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        seq_len: int = 100,
        num_classes: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(embedding_dim, dropout)
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=512,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)

        # Adaptive pooling to handle variable sequence lengths
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        # Reshape for Conv1D: [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)

        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 512, L]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 256, L]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, L]

        # Global max pooling
        x = self.pool(x)  # [B, 128, 1]

        # Classification
        out = self.classifier(x)  # [B, num_classes]
        return out.squeeze(1)  # [B]