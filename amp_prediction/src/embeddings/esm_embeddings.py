"""ESM-based embedding generation for protein sequences."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Union, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESMEmbedding:
    """Base class for ESM embedding generation."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "auto"
    ):
        """Initialize ESM embedding generator.

        Args:
            model_name: HuggingFace model identifier for ESM model
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading ESM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _tokenize_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Tokenize a protein sequence."""
        return self.tokenizer(
            sequence,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)


class ESMSequenceEmbedding(ESMEmbedding):
    """Generate sequence-level embeddings using ESM models."""

    def __init__(
        self,
        model_name: str = "facebook/esm1b_t33_650M_UR50S",
        device: str = "auto",
        pooling_strategy: str = "mean"
    ):
        """Initialize sequence-level embedding generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            pooling_strategy: Strategy for pooling token embeddings ('mean', 'max', 'cls')
        """
        super().__init__(model_name, device)
        self.pooling_strategy = pooling_strategy

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sequence-level embeddings."""
        if self.pooling_strategy == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return (summed / counts).squeeze(0)

        elif self.pooling_strategy == "max":
            # Max pooling
            return torch.max(token_embeddings, dim=1)[0].squeeze(0)

        elif self.pooling_strategy == "cls":
            # Use CLS token embedding (first token)
            return token_embeddings[0, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Generate embedding for a single sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            Sequence embedding tensor of shape [embedding_dim]
        """
        inputs = self._tokenize_sequence(sequence)

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state

            embedding = self._pool_embeddings(
                token_embeddings,
                inputs['attention_mask']
            )

        return embedding.cpu()

    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[torch.Tensor]:
        """Generate embeddings for multiple sequences.

        Args:
            sequences: List of protein sequence strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of sequence embeddings
        """
        embeddings = []
        iterator = tqdm(sequences, desc="Generating embeddings") if show_progress else sequences

        for sequence in iterator:
            embedding = self.embed_sequence(sequence)
            embeddings.append(embedding)

        return embeddings

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        sequence_column: str = "Sequence",
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """Generate embeddings for sequences in a DataFrame.

        Args:
            df: DataFrame containing protein sequences
            sequence_column: Name of the column containing sequences
            batch_size: Batch size for processing

        Returns:
            List of sequence embeddings
        """
        sequences = df[sequence_column].tolist()
        return self.embed_sequences(sequences, batch_size)


class ESMAminoAcidEmbedding(ESMEmbedding):
    """Generate amino acid-level embeddings using ESM models."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "auto",
        max_length: int = 100
    ):
        """Initialize amino acid-level embedding generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            max_length: Maximum sequence length for padding/truncation
        """
        super().__init__(model_name, device)
        self.max_length = max_length

    def _pad_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad or truncate embedding to fixed length."""
        if tensor.size(0) >= self.max_length:
            return tensor[:self.max_length]
        else:
            # Pad along sequence dimension
            return F.pad(tensor, (0, 0, 0, self.max_length - tensor.size(0)))

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Generate amino acid-level embeddings for a sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            Padded embedding tensor of shape [max_length, embedding_dim]
        """
        inputs = self._tokenize_sequence(sequence)

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state  # [1, seq_len+2, embedding_dim]

            # Remove special tokens (CLS and SEP)
            aa_embeddings = token_embeddings[0, 1:len(sequence)+1, :].cpu()

            # Pad to fixed length
            padded_embeddings = self._pad_embedding(aa_embeddings)

        return padded_embeddings

    def embed_sequences_with_metadata(
        self,
        sequences: List[str],
        labels: Optional[List[Union[int, float]]] = None,
        values: Optional[List[float]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """Generate embeddings with metadata for multiple sequences.

        Args:
            sequences: List of protein sequence strings
            labels: Optional list of labels for classification
            values: Optional list of values for regression
            ids: Optional list of sequence IDs
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries containing embeddings and metadata
        """
        embeddings_data = []
        iterator = enumerate(sequences)

        if show_progress:
            iterator = tqdm(iterator, total=len(sequences), desc="Generating embeddings")

        for i, sequence in iterator:
            embedding = self.embed_sequence(sequence)

            data = {
                'ID': ids[i] if ids else f'seq_{i:05d}',
                'embeddings': embedding,
            }

            if labels is not None:
                data['label'] = torch.tensor(labels[i], dtype=torch.float)

            if values is not None:
                data['value'] = torch.tensor(values[i], dtype=torch.float)

            embeddings_data.append(data)

        return embeddings_data

    def save_embeddings(
        self,
        embeddings_data: List[Dict],
        save_path: str
    ) -> None:
        """Save embeddings to file.

        Args:
            embeddings_data: List of embedding dictionaries
            save_path: Path to save the embeddings
        """
        torch.save(embeddings_data, save_path)
        logger.info(f"Embeddings saved to {save_path}")

    def load_embeddings(self, load_path: str) -> List[Dict]:
        """Load embeddings from file.

        Args:
            load_path: Path to load embeddings from

        Returns:
            List of embedding dictionaries
        """
        embeddings_data = torch.load(load_path)
        logger.info(f"Embeddings loaded from {load_path}")
        return embeddings_data


def create_amino_acid_embeddings():
    """Create amino acid embeddings for all 20 standard amino acids."""
    model_name = "facebook/esm2_t33_650M_UR50D"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    aa_tokens = "ACDEFGHIKLMNPQRSTVWY"
    aa_embeddings = {}

    logger.info("Creating amino acid embeddings...")

    with torch.no_grad():
        for aa in tqdm(aa_tokens, desc="Processing amino acids"):
            tokens = tokenizer(aa, return_tensors="pt").to(device)
            outputs = model(**tokens)
            # Get embedding for the amino acid token (position 1, after CLS)
            embedding = outputs.last_hidden_state[0, 1].cpu()
            aa_embeddings[aa] = embedding

    # Stack into matrix for easier access
    aa_embed_matrix = torch.stack([aa_embeddings[aa] for aa in aa_tokens])

    return aa_embeddings, aa_embed_matrix