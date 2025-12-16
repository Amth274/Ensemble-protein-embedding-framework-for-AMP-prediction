"""Data preprocessing utilities for AMP prediction."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import logging

logger = logging.getLogger(__name__)


class AMPDataProcessor:
    """Data processor for antimicrobial peptide datasets."""

    def __init__(self):
        """Initialize data processor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def load_data(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        sequence_column: str = "Sequence",
        label_column: str = "label",
        value_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load training and test data from CSV files.

        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file (optional)
            sequence_column: Name of sequence column
            label_column: Name of label column
            value_column: Name of value column (for regression)

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = None
        test_df = None

        if train_path:
            train_df = pd.read_csv(train_path)
            logger.info(f"Loaded training data: {len(train_df)} samples")

        if test_path:
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded test data: {len(test_df)} samples")

        return train_df, test_df

    def preprocess_sequences(
        self,
        df: pd.DataFrame,
        sequence_column: str = "Sequence",
        min_length: int = 5,
        max_length: int = 200
    ) -> pd.DataFrame:
        """Preprocess protein sequences.

        Args:
            df: DataFrame containing sequences
            sequence_column: Name of sequence column
            min_length: Minimum sequence length
            max_length: Maximum sequence length

        Returns:
            Processed DataFrame
        """
        logger.info("Preprocessing sequences...")

        # Remove sequences with invalid characters
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        df = df[df[sequence_column].apply(
            lambda seq: all(aa in valid_aas for aa in seq.upper())
        )].copy()

        # Filter by length
        df['seq_length'] = df[sequence_column].str.len()
        df = df[
            (df['seq_length'] >= min_length) &
            (df['seq_length'] <= max_length)
        ].copy()

        # Convert to uppercase
        df[sequence_column] = df[sequence_column].str.upper()

        logger.info(f"After preprocessing: {len(df)} sequences")
        return df

    def add_sequence_ids(
        self,
        df: pd.DataFrame,
        id_prefix: str = "pep",
        id_column: str = "ID"
    ) -> pd.DataFrame:
        """Add unique sequence IDs to DataFrame.

        Args:
            df: DataFrame to add IDs to
            id_prefix: Prefix for generated IDs
            id_column: Name of ID column

        Returns:
            DataFrame with added IDs
        """
        df = df.copy()
        df[id_column] = [f'{id_prefix}_{i:05d}' for i in range(1, len(df) + 1)]
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Split data into train, validation, and test sets.

        Args:
            df: DataFrame to split
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            stratify_column: Column to stratify on

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        stratify = df[stratify_column] if stratify_column else None

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        # Second split: train vs val
        val_df = None
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            stratify_train_val = (
                train_val_df[stratify_column] if stratify_column else None
            )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_train_val
            )
        else:
            train_df = train_val_df

        logger.info(f"Data split - Train: {len(train_df)}, "
                   f"Val: {len(val_df) if val_df is not None else 0}, "
                   f"Test: {len(test_df)}")

        return train_df, val_df, test_df

    def normalize_values(
        self,
        train_values: np.ndarray,
        test_values: Optional[np.ndarray] = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Normalize regression target values.

        Args:
            train_values: Training values to normalize
            test_values: Test values to normalize (optional)
            fit_scaler: Whether to fit the scaler

        Returns:
            Tuple of (normalized_train_values, normalized_test_values)
        """
        if fit_scaler:
            train_values_norm = self.scaler.fit_transform(
                train_values.reshape(-1, 1)
            ).flatten()
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming")
            train_values_norm = self.scaler.transform(
                train_values.reshape(-1, 1)
            ).flatten()

        test_values_norm = None
        if test_values is not None:
            test_values_norm = self.scaler.transform(
                test_values.reshape(-1, 1)
            ).flatten()

        return train_values_norm, test_values_norm

    def inverse_normalize_values(self, values: np.ndarray) -> np.ndarray:
        """Inverse normalize values back to original scale.

        Args:
            values: Normalized values

        Returns:
            Values in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")

        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    def prepare_classification_data(
        self,
        df: pd.DataFrame,
        sequence_column: str = "Sequence",
        label_column: str = "label"
    ) -> Dict[str, Union[List, np.ndarray]]:
        """Prepare data for classification task.

        Args:
            df: DataFrame containing sequences and labels
            sequence_column: Name of sequence column
            label_column: Name of label column

        Returns:
            Dictionary with sequences and labels
        """
        sequences = df[sequence_column].tolist()
        labels = df[label_column].values

        return {
            'sequences': sequences,
            'labels': labels
        }

    def prepare_regression_data(
        self,
        df: pd.DataFrame,
        sequence_column: str = "Sequence",
        label_column: str = "label",
        value_column: str = "value"
    ) -> Dict[str, Union[List, np.ndarray]]:
        """Prepare data for regression task.

        Args:
            df: DataFrame containing sequences, labels, and values
            sequence_column: Name of sequence column
            label_column: Name of label column
            value_column: Name of value column

        Returns:
            Dictionary with sequences, labels, and values
        """
        sequences = df[sequence_column].tolist()
        labels = df[label_column].values
        values = df[value_column].values

        return {
            'sequences': sequences,
            'labels': labels,
            'values': values
        }

    def get_sequence_statistics(self, df: pd.DataFrame, sequence_column: str = "Sequence") -> Dict:
        """Get statistics about sequence lengths and composition.

        Args:
            df: DataFrame containing sequences
            sequence_column: Name of sequence column

        Returns:
            Dictionary with statistics
        """
        sequences = df[sequence_column]
        lengths = sequences.str.len()

        # Amino acid composition
        all_sequences = ''.join(sequences)
        aa_counts = {aa: all_sequences.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        total_aa = sum(aa_counts.values())
        aa_frequencies = {aa: count/total_aa for aa, count in aa_counts.items()}

        stats = {
            'num_sequences': len(sequences),
            'length_stats': {
                'min': lengths.min(),
                'max': lengths.max(),
                'mean': lengths.mean(),
                'median': lengths.median(),
                'std': lengths.std()
            },
            'amino_acid_frequencies': aa_frequencies
        }

        return stats