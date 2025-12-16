"""LSTM-based models for AMP prediction."""

import torch
import torch.nn as nn
from .base import BaseAMPModel


class AMPBilstmClassifier(BaseAMPModel):
    """Bidirectional LSTM classifier for antimicrobial peptide prediction.

    Uses bidirectional LSTM to capture both forward and backward dependencies
    in protein sequences, followed by mean pooling and classification layers.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(embedding_dim, dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BiLSTM.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        lstm_out, _ = self.bilstm(x)  # [B, L, 2*H]
        pooled = torch.mean(lstm_out, dim=1)  # Mean pooling over sequence length
        return self.classifier(pooled).squeeze(1)  # [B]


class AMP_BiRNN(BaseAMPModel):
    """Bidirectional RNN using LSTM cells for AMP prediction.

    This model uses the last hidden state from the bidirectional LSTM
    instead of mean pooling for classification.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__(input_dim, dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BiRNN.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        out, _ = self.rnn(x)  # [B, T, 2*H]
        out = out[:, -1, :]   # Use last hidden state [B, 2*H]
        out = self.classifier(out)  # [B, 1]
        return out.squeeze(1)  # [B]