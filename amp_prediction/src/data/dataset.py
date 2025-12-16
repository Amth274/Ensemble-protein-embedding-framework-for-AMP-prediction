"""PyTorch Dataset classes for AMP prediction."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Union, Tuple, Optional
import pandas as pd


class AMPDataset(Dataset):
    """Dataset for antimicrobial peptide sequences with embeddings.

    This dataset handles both classification and regression tasks,
    supporting variable-length sequences with padding.
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        max_length: int = 100,
        task_type: str = "classification"
    ):
        """Initialize AMP dataset.

        Args:
            data: List of dictionaries with embeddings and labels, or path to saved data
            max_length: Maximum sequence length for padding
            task_type: Either "classification" or "regression"
        """
        self.max_length = max_length
        self.task_type = task_type

        if isinstance(data, str):
            self.data = torch.load(data)
        else:
            self.data = data

        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that data contains required fields."""
        if not self.data:
            raise ValueError("Data cannot be empty")

        required_fields = ['embeddings']
        if self.task_type == "classification":
            required_fields.append('label')
        elif self.task_type == "regression":
            required_fields.extend(['label', 'value'])

        for field in required_fields:
            if field not in self.data[0]:
                raise ValueError(f"Missing required field: {field}")

    def _pad_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad or truncate embedding to fixed length."""
        if tensor.size(0) >= self.max_length:
            return tensor[:self.max_length]
        else:
            return F.pad(tensor, (0, 0, 0, self.max_length - tensor.size(0)))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get item from dataset.

        Returns:
            For classification: (embeddings, label)
            For regression: (embeddings, label, value)
        """
        entry = self.data[idx]
        embeddings = self._pad_embedding(entry['embeddings'])
        label = entry['label']

        if self.task_type == "classification":
            return embeddings, label
        else:  # regression
            value = entry['value']
            return embeddings, label, value


class EnsembleDataset(Dataset):
    """Dataset for ensemble learning with pre-computed embeddings.

    This dataset is optimized for ensemble training and inference,
    with consistent padding and batching strategies.
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        max_length: int = 100,
        task_type: str = "classification"
    ):
        """Initialize ensemble dataset.

        Args:
            data: List of dictionaries with embeddings and labels, or path to saved data
            max_length: Maximum sequence length for padding
            task_type: Either "classification" or "regression"
        """
        self.max_length = max_length
        self.task_type = task_type

        if isinstance(data, str):
            self.data = torch.load(data)
        else:
            self.data = data

    def _pad_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad or truncate embedding to fixed length."""
        if tensor.size(0) >= self.max_length:
            return tensor[:self.max_length]
        else:
            return F.pad(tensor, (0, 0, 0, self.max_length - tensor.size(0)))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get item from dataset.

        Returns:
            For classification: (embeddings, label)
            For regression: (embeddings, label, value)
        """
        entry = self.data[idx]
        embeddings = self._pad_embedding(entry['embeddings'])
        label = entry['label']

        if self.task_type == "classification":
            return embeddings, label
        else:  # regression
            value = entry['value']
            return embeddings, label, value


class SequenceDataset(Dataset):
    """Dataset for raw protein sequences (without pre-computed embeddings).

    This dataset is useful for on-the-fly embedding generation or
    when working with different embedding strategies.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[Union[int, float]],
        values: Optional[List[float]] = None,
        ids: Optional[List[str]] = None,
        task_type: str = "classification"
    ):
        """Initialize sequence dataset.

        Args:
            sequences: List of protein sequence strings
            labels: List of labels (int for classification, float for regression)
            values: Optional list of values for regression tasks
            ids: Optional list of sequence identifiers
            task_type: Either "classification" or "regression"
        """
        self.sequences = sequences
        self.labels = labels
        self.values = values
        self.ids = ids or [f"seq_{i:05d}" for i in range(len(sequences))]
        self.task_type = task_type

        # Validation
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have the same length")

        if task_type == "regression" and values is not None:
            if len(sequences) != len(values):
                raise ValueError("Sequences and values must have the same length")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, float]]:
        """Get item from dataset.

        Returns:
            Dictionary containing sequence, label, and optional value/id
        """
        item = {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'id': self.ids[idx]
        }

        if self.task_type == "regression" and self.values is not None:
            item['value'] = self.values[idx]

        return item

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        sequence_column: str = "Sequence",
        label_column: str = "label",
        value_column: Optional[str] = None,
        id_column: Optional[str] = None,
        task_type: str = "classification"
    ) -> "SequenceDataset":
        """Create dataset from CSV file.

        Args:
            csv_path: Path to CSV file
            sequence_column: Name of sequence column
            label_column: Name of label column
            value_column: Name of value column (for regression)
            id_column: Name of ID column
            task_type: Either "classification" or "regression"

        Returns:
            SequenceDataset instance
        """
        df = pd.read_csv(csv_path)

        sequences = df[sequence_column].tolist()
        labels = df[label_column].tolist()

        values = None
        if task_type == "regression" and value_column is not None:
            values = df[value_column].tolist()

        ids = None
        if id_column is not None:
            ids = df[id_column].tolist()

        return cls(
            sequences=sequences,
            labels=labels,
            values=values,
            ids=ids,
            task_type=task_type
        )