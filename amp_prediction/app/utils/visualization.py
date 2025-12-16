"""Visualization utilities for the AMP prediction demo."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


def create_prediction_plot(predictions: List[Dict]) -> go.Figure:
    """Create prediction visualization for multiple sequences.

    Args:
        predictions: List of prediction results

    Returns:
        Plotly figure
    """
    # Extract data
    sequences = [p['sequence'][:10] + '...' if len(p['sequence']) > 10 else p['sequence'] for p in predictions]
    confidences = [p['ensemble']['confidence'] for p in predictions]
    pred_labels = ['AMP' if p['ensemble']['prediction'] == 1 else 'Non-AMP' for p in predictions]

    # Create scatter plot
    fig = px.scatter(
        x=range(len(sequences)),
        y=confidences,
        color=pred_labels,
        hover_data={'Sequence': sequences},
        title="Prediction Confidence by Sequence",
        labels={'x': 'Sequence Index', 'y': 'Confidence Score'},
        color_discrete_map={'AMP': '#2E86C1', 'Non-AMP': '#E74C3C'}
    )

    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Decision Threshold (0.5)")

    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def create_confidence_plot(result: Dict) -> go.Figure:
    """Create confidence visualization for individual models.

    Args:
        result: Prediction result dictionary

    Returns:
        Plotly figure
    """
    models = list(result['individual'].keys())
    confidences = [result['individual'][model]['confidence'] for model in models]
    predictions = [result['individual'][model]['prediction'] for model in models]

    colors = ['#2E86C1' if pred == 1 else '#E74C3C' for pred in predictions]

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker_color=colors,
            text=[f"{conf:.2%}" for conf in confidences],
            textposition='auto',
        )
    ])

    # Add ensemble line
    ensemble_conf = result['ensemble']['confidence']
    fig.add_hline(
        y=ensemble_conf,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text=f"Ensemble: {ensemble_conf:.2%}"
    )

    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Model",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400
    )

    return fig


def create_sequence_logo(sequences: List[str], title: str = "Sequence Logo") -> go.Figure:
    """Create a sequence logo visualization.

    Args:
        sequences: List of aligned sequences
        title: Plot title

    Returns:
        Plotly figure
    """
    if not sequences:
        return go.Figure()

    # Ensure all sequences have the same length (truncate or pad)
    max_length = max(len(seq) for seq in sequences)
    aligned_sequences = [seq.ljust(max_length, '-') for seq in sequences]

    # Calculate position weight matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY-'
    position_counts = []

    for pos in range(max_length):
        pos_counts = {aa: 0 for aa in amino_acids}
        for seq in aligned_sequences:
            if pos < len(seq):
                pos_counts[seq[pos]] += 1
        position_counts.append(pos_counts)

    # Create heatmap data
    heatmap_data = []
    for aa in amino_acids:
        row = [pos_counts[aa] for pos_counts in position_counts]
        heatmap_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        y=list(amino_acids),
        x=list(range(1, max_length + 1)),
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Amino Acid",
        height=400
    )

    return fig


def create_performance_comparison(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create performance comparison visualization.

    Args:
        metrics: Dictionary of model metrics

    Returns:
        Plotly figure
    """
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_names,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    colors = px.colors.qualitative.Set3[:len(models)]

    for i, metric in enumerate(metric_names):
        row = (i // 2) + 1
        col = (i % 2) + 1

        values = [metrics[model][metric] for model in models]

        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors,
                showlegend=False,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="Model Performance Comparison",
        height=600,
        showlegend=False
    )

    return fig


def create_embedding_heatmap(
    embeddings: np.ndarray,
    sequence: str,
    title: str = "Sequence Embeddings"
) -> go.Figure:
    """Create embedding heatmap visualization.

    Args:
        embeddings: Embedding matrix [seq_len, embedding_dim]
        sequence: Protein sequence
        title: Plot title

    Returns:
        Plotly figure
    """
    # Sample every nth dimension for visualization
    sample_rate = max(1, embeddings.shape[1] // 50)
    sampled_embeddings = embeddings[:, ::sample_rate]

    fig = go.Figure(data=go.Heatmap(
        z=sampled_embeddings.T,
        x=list(sequence),
        y=[f"Dim_{i}" for i in range(0, embeddings.shape[1], sample_rate)],
        colorscale='RdBu',
        zmid=0,
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Amino Acid Position",
        yaxis_title="Embedding Dimension",
        height=500
    )

    return fig


def create_roc_curve(
    y_true: List[int],
    y_scores: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """Create ROC curve visualization.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        model_name: Name of the model

    Returns:
        Plotly figure
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(width=2)
    ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=500,
        height=500,
        showlegend=True
    )

    return fig


def create_precision_recall_curve(
    y_true: List[int],
    y_scores: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """Create Precision-Recall curve visualization.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        model_name: Name of the model

    Returns:
        Plotly figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'{model_name} (AP = {avg_precision:.3f})',
        line=dict(width=2)
    ))

    # Add baseline
    baseline = sum(y_true) / len(y_true)
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline (Random): {baseline:.3f}"
    )

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=500,
        height=500,
        showlegend=True
    )

    return fig


def create_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None
) -> go.Figure:
    """Create confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels

    Returns:
        Plotly figure
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ['Non-AMP', 'AMP']

    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{cm[i, j]}<br>({cm_normalized[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
        width=400,
        height=400
    )

    return fig


def create_sequence_length_distribution(sequences: List[str]) -> go.Figure:
    """Create sequence length distribution plot.

    Args:
        sequences: List of protein sequences

    Returns:
        Plotly figure
    """
    lengths = [len(seq) for seq in sequences]

    fig = go.Figure(data=[go.Histogram(
        x=lengths,
        nbinsx=20,
        marker_color='lightblue',
        opacity=0.7
    )])

    fig.add_vline(
        x=np.mean(lengths),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(lengths):.1f}"
    )

    fig.update_layout(
        title="Sequence Length Distribution",
        xaxis_title="Sequence Length (amino acids)",
        yaxis_title="Count",
        height=400
    )

    return fig


def create_amino_acid_composition(sequences: List[str]) -> go.Figure:
    """Create amino acid composition plot.

    Args:
        sequences: List of protein sequences

    Returns:
        Plotly figure
    """
    # Count amino acids
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    for sequence in sequences:
        for aa in sequence.upper():
            if aa in aa_counts:
                aa_counts[aa] += 1

    total_count = sum(aa_counts.values())
    aa_frequencies = {aa: count/total_count for aa, count in aa_counts.items()}

    # Sort by frequency
    sorted_aas = sorted(aa_frequencies.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure(data=[go.Bar(
        x=[aa for aa, _ in sorted_aas],
        y=[freq for _, freq in sorted_aas],
        marker_color='lightgreen'
    )])

    fig.update_layout(
        title="Amino Acid Composition",
        xaxis_title="Amino Acid",
        yaxis_title="Frequency",
        height=400
    )

    return fig


def create_model_ensemble_weights(weights: Dict[str, float]) -> go.Figure:
    """Create ensemble weights visualization.

    Args:
        weights: Dictionary of model weights

    Returns:
        Plotly figure
    """
    models = list(weights.keys())
    weight_values = list(weights.values())

    fig = go.Figure(data=[go.Pie(
        labels=models,
        values=weight_values,
        hole=0.3
    )])

    fig.update_layout(
        title="Ensemble Model Weights",
        height=400
    )

    return fig