"""Demo utilities for AMP prediction app."""

import pandas as pd
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Union
import logging
from pathlib import Path
import sys

# Add src to path
app_dir = Path(__file__).parent.parent
sys.path.append(str(app_dir.parent / "src"))

logger = logging.getLogger(__name__)


class DemoPredictor:
    """Demo predictor for AMP sequences.

    This is a simplified version for demonstration purposes.
    In production, this would load the actual trained ensemble models.
    """

    def __init__(self):
        """Initialize demo predictor with mock models."""
        self.models = {
            'CNN': MockModel('CNN'),
            'BiLSTM': MockModel('BiLSTM'),
            'GRU': MockModel('GRU'),
            'Transformer': MockModel('Transformer'),
            'BiCNN': MockModel('BiCNN'),
            'BiRNN': MockModel('BiRNN')
        }
        self.is_loaded = True

    def predict_single(self, sequence: str) -> Dict:
        """Predict AMP activity for a single sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            Dictionary with prediction results
        """
        # Get individual model predictions
        individual_results = {}
        for model_name, model in self.models.items():
            prediction, confidence = model.predict(sequence)
            individual_results[model_name] = {
                'prediction': prediction,
                'confidence': confidence
            }

        # Ensemble prediction (soft voting)
        ensemble_confidence = np.mean([r['confidence'] for r in individual_results.values()])
        ensemble_prediction = 1 if ensemble_confidence > 0.5 else 0

        return {
            'sequence': sequence,
            'ensemble': {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence
            },
            'individual': individual_results
        }

    def predict_batch(self, sequences: List[str]) -> List[Dict]:
        """Predict AMP activity for multiple sequences.

        Args:
            sequences: List of protein sequence strings

        Returns:
            List of prediction results
        """
        results = []
        for sequence in sequences:
            result = self.predict_single(sequence)
            results.append(result)
        return results

    def predict_with_uncertainty(self, sequence: str, n_samples: int = 100) -> Dict:
        """Predict with uncertainty estimation using bootstrap sampling.

        Args:
            sequence: Protein sequence string
            n_samples: Number of bootstrap samples

        Returns:
            Dictionary with predictions and uncertainty
        """
        predictions = []
        for _ in range(n_samples):
            # Simulate bootstrap sampling by adding noise
            result = self.predict_single(sequence)
            predictions.append(result['ensemble']['confidence'])

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        return {
            'sequence': sequence,
            'prediction': 1 if mean_pred > 0.5 else 0,
            'confidence': mean_pred,
            'uncertainty': std_pred,
            'confidence_interval': {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
        }


class MockModel:
    """Mock model for demonstration purposes."""

    def __init__(self, model_name: str):
        """Initialize mock model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        # Set different biases for different models to simulate diversity
        self.bias = {
            'CNN': 0.1,
            'BiLSTM': -0.05,
            'GRU': 0.05,
            'Transformer': 0.0,
            'BiCNN': 0.08,
            'BiRNN': -0.02
        }.get(model_name, 0.0)

    def predict(self, sequence: str) -> Tuple[int, float]:
        """Make prediction for a sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            Tuple of (prediction, confidence)
        """
        # Simple heuristic-based prediction for demo
        # In reality, this would use the trained neural network

        # Calculate basic features
        length = len(sequence)
        hydrophobic_ratio = sum(1 for aa in sequence if aa in 'AILMFPWV') / length
        charged_ratio = sum(1 for aa in sequence if aa in 'KRDE') / length
        aromatic_ratio = sum(1 for aa in sequence if aa in 'FWY') / length

        # Simple scoring function (mock)
        score = (
            0.3 * hydrophobic_ratio +
            0.2 * charged_ratio +
            0.1 * aromatic_ratio +
            0.1 * (1 / (1 + abs(length - 20))) +  # Prefer ~20 aa length
            self.bias +
            np.random.normal(0, 0.1)  # Add some noise
        )

        # Apply sigmoid to get probability
        confidence = 1 / (1 + np.exp(-5 * score))

        # Add model-specific variation
        confidence = np.clip(confidence + np.random.normal(0, 0.05), 0, 1)

        prediction = 1 if confidence > 0.5 else 0

        return prediction, confidence


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """Validate protein sequence.

    Args:
        sequence: Protein sequence string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, "Sequence cannot be empty"

    if len(sequence) < 5:
        return False, "Sequence too short (minimum 5 amino acids)"

    if len(sequence) > 200:
        return False, "Sequence too long (maximum 200 amino acids)"

    # Check for invalid characters
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_chars = set(sequence.upper()) - valid_aas

    if invalid_chars:
        return False, f"Invalid amino acid codes: {', '.join(invalid_chars)}"

    return True, ""


def load_example_data() -> pd.DataFrame:
    """Load example dataset for batch analysis.

    Returns:
        DataFrame with example sequences
    """
    # Example sequences for demonstration
    example_sequences = [
        "GLFDIVKKVVGALCS",  # Magainin-like
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin-2
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37
        "GIGAVLKVLTTGLPALISWIKRKRQQ",  # Melittin
        "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQ",  # Non-AMP (Albumin)
        "GIVEQCCTSICSLYQLENYCN",  # Non-AMP (Insulin)
        "FKWLKKIEKACALLPELCSWIKRKRQQ",  # Synthetic AMP
        "SGRGKQGGKARAKAKTRSSRAGLQFPVGRVHRLLRKGNYAE",  # Non-AMP (Histone)
        "ATKQFNCVQTVSLPGGCRAHPHIAICPPSQKY",  # Defensin
    ]

    # Add some metadata
    labels = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1]  # 1 = AMP, 0 = Non-AMP
    sources = [
        "Synthetic", "Frog", "Insect", "Human", "Bee venom",
        "Human serum", "Human pancreas", "Synthetic", "Human cell", "Human"
    ]

    df = pd.DataFrame({
        'Sequence': example_sequences,
        'Known_Label': labels,
        'Source': sources,
        'Length': [len(seq) for seq in example_sequences]
    })

    return df


def analyze_sequence_composition(sequence: str) -> Dict[str, float]:
    """Analyze amino acid composition of a sequence.

    Args:
        sequence: Protein sequence string

    Returns:
        Dictionary with composition analysis
    """
    sequence = sequence.upper()
    length = len(sequence)

    # Amino acid groups
    aa_groups = {
        'hydrophobic': 'AILMFPWV',
        'polar': 'NQST',
        'charged_positive': 'KR',
        'charged_negative': 'DE',
        'aromatic': 'FWY',
        'small': 'AGS',
        'sulfur': 'CM',
        'amide': 'NQ',
        'hydroxyl': 'ST',
        'basic': 'KRH',
        'acidic': 'DE'
    }

    composition = {}
    for group, amino_acids in aa_groups.items():
        count = sum(1 for aa in sequence if aa in amino_acids)
        composition[f'{group}_ratio'] = count / length if length > 0 else 0

    # Additional properties
    composition['length'] = length
    composition['unique_aas'] = len(set(sequence))

    # Calculate charge at pH 7 (simplified)
    positive_charge = sum(1 for aa in sequence if aa in 'KR')
    negative_charge = sum(1 for aa in sequence if aa in 'DE')
    composition['net_charge'] = positive_charge - negative_charge
    composition['charge_density'] = abs(composition['net_charge']) / length if length > 0 else 0

    return composition


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """Calculate simple sequence similarity.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Similarity score (0-1)
    """
    if len(seq1) != len(seq2):
        # For different lengths, use longest common subsequence approach
        return calculate_lcs_similarity(seq1, seq2)

    # For same length, calculate identity
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def calculate_lcs_similarity(seq1: str, seq2: str) -> float:
    """Calculate similarity using longest common subsequence.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Normalized LCS similarity score
    """
    m, n = len(seq1), len(seq2)

    # Create LCS table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])

    # Normalize by average length
    avg_length = (m + n) / 2
    return lcs[m][n] / avg_length if avg_length > 0 else 0


def generate_sequence_variants(sequence: str, n_variants: int = 5) -> List[str]:
    """Generate sequence variants for optimization demonstration.

    Args:
        sequence: Original sequence
        n_variants: Number of variants to generate

    Returns:
        List of sequence variants
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    variants = []

    for _ in range(n_variants):
        variant = list(sequence)

        # Make 1-3 random substitutions
        n_mutations = np.random.randint(1, min(4, len(sequence) + 1))
        positions = np.random.choice(len(sequence), size=n_mutations, replace=False)

        for pos in positions:
            # Avoid substituting with the same amino acid
            original_aa = variant[pos]
            available_aas = [aa for aa in amino_acids if aa != original_aa]
            variant[pos] = np.random.choice(available_aas)

        variants.append(''.join(variant))

    return variants


def estimate_mic_value(sequence: str) -> float:
    """Estimate MIC value for demonstration.

    Args:
        sequence: Protein sequence

    Returns:
        Estimated MIC value (μg/mL)
    """
    # Simple heuristic for demo purposes
    composition = analyze_sequence_composition(sequence)

    # Factors that typically correlate with lower MIC (better antimicrobial activity)
    length_factor = 1 / (1 + abs(len(sequence) - 20) / 10)  # Optimal around 20 aa
    charge_factor = min(composition['charge_density'] * 2, 1)  # Higher charge density helps
    hydrophobic_factor = composition['hydrophobic_ratio']  # Some hydrophobicity needed

    # Combine factors (this is a simplified model)
    activity_score = (length_factor * 0.3 + charge_factor * 0.4 + hydrophobic_factor * 0.3)

    # Convert to MIC (lower values = better activity)
    # Typical range: 1-100 μg/mL
    mic_value = 100 * (1 - activity_score) + np.random.normal(0, 5)
    mic_value = max(1, min(100, mic_value))  # Clamp to reasonable range

    return round(mic_value, 2)