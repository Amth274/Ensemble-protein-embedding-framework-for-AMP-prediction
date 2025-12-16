"""Transformer-based models for AMP prediction."""

import torch
import torch.nn as nn
from .base import BaseAMPModel


class AMPTransformerClassifier(BaseAMPModel):
    """Transformer encoder classifier for antimicrobial peptide prediction.

    Uses multi-head attention mechanisms to capture complex relationships
    between amino acids in protein sequences.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        seq_len: int = 100,
        num_heads: int = 1,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(embedding_dim, dropout)
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        # Transformer encoding
        x = self.transformer(x)  # [B, L, D]

        # Mean pooling over sequence length
        x = x.mean(dim=1)  # [B, D]

        # Classification
        return self.classifier(x).squeeze(1)  # [B]


class AMPTransformerWithPositionalEncoding(BaseAMPModel):
    """Transformer with positional encoding for AMP prediction.

    Enhanced version with positional encoding to better capture
    sequence order information.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        seq_len: int = 100,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__(embedding_dim, dropout)
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(seq_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def _create_positional_encoding(self, seq_len: int, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, seq_len, embedding_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer with positional encoding.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor of shape [batch_size] for binary classification
        """
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = x + pos_enc

        # Transformer encoding
        x = self.transformer(x)  # [B, L, D]

        # Mean pooling over sequence length
        x = x.mean(dim=1)  # [B, D]

        # Classification
        return self.classifier(x).squeeze(1)  # [B]