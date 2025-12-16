"""GRU-based models for AMP prediction."""

import torch
import torch.nn as nn
from .base import BaseAMPModel


class GRUClassifier(BaseAMPModel):
    """Bidirectional GRU classifier for antimicrobial peptide prediction.

    Uses bidirectional GRU with layer normalization and concatenates
    the final hidden states from both directions for classification.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__(embedding_dim, dropout)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_directions * hidden_dim),
            nn.Linear(self.num_directions * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        output, hidden = self.gru(x)
        # output: [B, L, num_directions*H]
        # hidden: [num_layers*num_directions, B, H]

        if self.bidirectional:
            # Concatenate last hidden states from both directions
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2*H]
        else:
            hidden_cat = hidden[-1]  # [B, H]

        return self.classifier(hidden_cat).squeeze(1)  # [B]