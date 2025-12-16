"""Hybrid CNN-LSTM models for AMP prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAMPModel


class CNN_BiLSTM_Classifier(BaseAMPModel):
    """Hybrid CNN-BiLSTM classifier for antimicrobial peptide prediction.

    This model combines convolutional layers for local feature extraction
    with bidirectional LSTM for capturing long-range dependencies.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        cnn_out_channels: int = 256,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.5
    ):
        super().__init__(input_dim, dropout)
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # CNN layer for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out_channels,
            kernel_size=5,
            padding=2
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid CNN-BiLSTM.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        # CNN feature extraction
        x = x.transpose(1, 2)  # [B, D, L] for Conv1D
        x = self.conv1d(x)     # [B, C, L]
        x = self.relu(x)
        x = self.maxpool(x)    # [B, C, L//2]

        # Prepare for LSTM
        x = x.transpose(1, 2)  # [B, L//2, C]

        # BiLSTM processing
        output, _ = self.bilstm(x)  # [B, L//2, 2*H]
        out = output[:, -1, :]      # Take last time step [B, 2*H]

        # Classification
        out = self.dropout_layer(out)
        out = self.classifier(out)  # [B, 1]
        return out.squeeze(1)       # [B]