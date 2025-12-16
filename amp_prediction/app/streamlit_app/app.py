"""
Streamlit Web Application for AMP Prediction Demo

This application provides an interactive interface for demonstrating
the Enhanced Antimicrobial Peptide Prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import io
import time

# Add src to path
app_dir = Path(__file__).parent.parent
sys.path.append(str(app_dir.parent / "src"))

from src.embeddings import ESMSequenceEmbedding, ESMAminoAcidEmbedding
from src.data import SequenceDataset
from app.utils.demo_utils import DemoPredictor, load_example_data, validate_sequence
from app.utils.visualization import create_prediction_plot, create_confidence_plot, create_sequence_logo

# Page configuration
st.set_page_config(
    page_title="AMP Prediction Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #148F77;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86C1;
        margin: 0.5rem 0;
    }
    .sequence-input {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">🧬 Enhanced AMP Prediction System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #566573;">Predict antimicrobial activity using ESM-650M embeddings and ensemble deep learning</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["Demo Model (Fast)", "Full Ensemble (Accurate)"],
            help="Demo model for quick testing, Full ensemble for accurate predictions"
        )

        # Task type
        task_type = st.selectbox(
            "Task Type",
            ["Classification", "Regression"],
            help="Classification: AMP vs non-AMP, Regression: MIC prediction"
        )

        # Confidence threshold
        if task_type == "Classification":
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.78,
                step=0.01,
                help="Threshold for AMP classification"
            )

        st.markdown("---")

        # About section
        st.header("📖 About")
        st.markdown("""
        This demo showcases the Enhanced AMP Prediction system that uses:

        - **ESM-650M embeddings** for rich protein representation
        - **Ensemble of 6 models**: CNN, LSTM, GRU, BiLSTM, BiCNN, Transformer
        - **Multiple voting strategies** for robust predictions
        - **EvoGradient optimization** for sequence improvement
        """)

        # Performance metrics
        st.header("📊 Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "93.57%")
            st.metric("Precision", "99.01%")
        with col2:
            st.metric("ROC-AUC", "99.39%")
            st.metric("F1-Score", "93.06%")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "🧪 Examples", "📈 Model Insights"])

    with tab1:
        single_prediction_tab()

    with tab2:
        batch_analysis_tab()

    with tab3:
        examples_tab()

    with tab4:
        model_insights_tab()

def single_prediction_tab():
    """Single sequence prediction tab."""
    st.markdown('<h2 class="sub-header">Single Sequence Prediction</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Sequence input
        sequence_input = st.text_area(
            "Enter Protein Sequence",
            placeholder="GLFDIVKKVVGALCS",
            height=100,
            help="Enter a protein sequence using standard amino acid codes (A-Z)"
        )

        # Prediction button
        predict_button = st.button("🔮 Predict", type="primary")

        if predict_button and sequence_input:
            with st.spinner("Generating predictions..."):
                # Validate sequence
                is_valid, error_msg = validate_sequence(sequence_input.strip().upper())

                if not is_valid:
                    st.error(f"Invalid sequence: {error_msg}")
                    return

                # Initialize predictor if needed
                if st.session_state.predictor is None:
                    st.session_state.predictor = DemoPredictor()

                # Make prediction
                result = st.session_state.predictor.predict_single(sequence_input.strip().upper())

                # Display results
                display_single_prediction_results(result, sequence_input.strip().upper())

    with col2:
        # Sequence information
        if sequence_input:
            seq = sequence_input.strip().upper()
            st.markdown("### Sequence Info")
            st.info(f"""
            **Length**: {len(seq)} amino acids
            **Composition**: {len(set(seq))} unique amino acids
            **Hydrophobic**: {sum(1 for aa in seq if aa in 'AILMFPWV')/len(seq)*100:.1f}%
            **Charged**: {sum(1 for aa in seq if aa in 'KRDE')/len(seq)*100:.1f}%
            """)

def batch_analysis_tab():
    """Batch analysis tab."""
    st.markdown('<h2 class="sub-header">Batch Sequence Analysis</h2>', unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with sequences",
        type=['csv'],
        help="CSV file should have a 'Sequence' column"
    )

    # Sample data option
    if st.button("📊 Use Sample Data"):
        sample_data = load_example_data()
        st.session_state.batch_data = sample_data
        st.success("Sample data loaded!")

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Sequence' not in df.columns:
                st.error("CSV file must contain a 'Sequence' column")
                return
            st.session_state.batch_data = df
            st.success(f"Loaded {len(df)} sequences")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return

    # Display and analyze data
    if 'batch_data' in st.session_state:
        df = st.session_state.batch_data

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔮 Analyze All Sequences", type="primary"):
                with st.spinner("Analyzing sequences..."):
                    if st.session_state.predictor is None:
                        st.session_state.predictor = DemoPredictor()

                    results = st.session_state.predictor.predict_batch(df['Sequence'].tolist())
                    st.session_state.batch_results = results

                    display_batch_results(results, df)

        with col2:
            # Data statistics
            st.markdown("### Dataset Info")
            st.info(f"""
            **Total sequences**: {len(df)}
            **Avg length**: {df['Sequence'].str.len().mean():.1f}
            **Min length**: {df['Sequence'].str.len().min()}
            **Max length**: {df['Sequence'].str.len().max()}
            """)

def examples_tab():
    """Examples and tutorials tab."""
    st.markdown('<h2 class="sub-header">Example Sequences</h2>', unsafe_allow_html=True)

    # Known AMP examples
    st.markdown("### 🦠 Known Antimicrobial Peptides")

    amp_examples = {
        "Magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
        "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "Cecropin A": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "Defensin": "ATKQFNCVQTVSLPGGCRAHPHIAICPPSQKY"
    }

    for name, sequence in amp_examples.items():
        with st.expander(f"{name} - {len(sequence)} amino acids"):
            st.code(sequence, language=None)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Sequence**: `{sequence}`")
            with col2:
                if st.button(f"Test {name}", key=f"test_{name}"):
                    st.session_state.test_sequence = sequence

    # Non-AMP examples
    st.markdown("### 🧪 Non-Antimicrobial Sequences")

    non_amp_examples = {
        "Albumin fragment": "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQ",
        "Insulin chain": "GIVEQCCTSICSLYQLENYCN",
        "Histone H4": "SGRGKQGGKARAKAKTRSSRAGLQFPVGRVHRLLRKGNYAE"
    }

    for name, sequence in non_amp_examples.items():
        with st.expander(f"{name} - {len(sequence)} amino acids"):
            st.code(sequence, language=None)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Sequence**: `{sequence}`")
            with col2:
                if st.button(f"Test {name}", key=f"test_non_{name}"):
                    st.session_state.test_sequence = sequence

    # If a test sequence was selected, show it in the input
    if 'test_sequence' in st.session_state:
        st.success(f"Selected sequence: {st.session_state.test_sequence}")
        st.info("Go to the 'Single Prediction' tab to analyze this sequence!")

def model_insights_tab():
    """Model insights and explanations tab."""
    st.markdown('<h2 class="sub-header">Model Architecture & Insights</h2>', unsafe_allow_html=True)

    # Model architecture overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Ensemble Architecture")

        model_info = {
            "CNN1D": {"params": "~2M", "features": "Local patterns"},
            "BiLSTM": {"params": "~1.5M", "features": "Sequential dependencies"},
            "GRU": {"params": "~1.2M", "features": "Gated recurrence"},
            "Transformer": {"params": "~25M", "features": "Self-attention"},
            "BiCNN": {"params": "~3M", "features": "Hybrid CNN-LSTM"},
            "BiRNN": {"params": "~1.8M", "features": "Bidirectional RNN"}
        }

        for model, info in model_info.items():
            st.markdown(f"**{model}**: {info['params']} parameters - {info['features']}")

    with col2:
        st.markdown("### 📊 Performance Comparison")

        # Create performance comparison chart
        models = list(model_info.keys())
        accuracies = [92.98, 90.28, 94.20, 93.15, 95.30, 94.53]  # Example values

        fig = px.bar(
            x=models,
            y=accuracies,
            title="Individual Model Accuracies",
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            color=accuracies,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ESM embeddings explanation
    st.markdown("### 🔬 ESM-650M Embeddings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Evolutionary Scale Modeling (ESM)** captures:
        - Evolutionary patterns from millions of protein sequences
        - Contextual amino acid representations
        - Structural and functional relationships
        - 650M parameters trained on protein databases
        """)

    with col2:
        # Embedding visualization (mock data)
        fig = px.imshow(
            np.random.randn(20, 10),  # Mock embedding visualization
            title="Sample Amino Acid Embeddings",
            labels={'x': 'Embedding Dimension', 'y': 'Amino Acid Position'},
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Voting strategies
    st.markdown("### 🗳️ Ensemble Voting Strategies")

    voting_strategies = {
        "Soft Voting": "Averages predicted probabilities from all models",
        "Hard Voting": "Uses majority vote from binary predictions",
        "Weighted Voting": "Weights models based on validation performance"
    }

    for strategy, description in voting_strategies.items():
        st.markdown(f"**{strategy}**: {description}")

def display_single_prediction_results(result, sequence):
    """Display results for single sequence prediction."""
    st.markdown("### 🔮 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        # Main prediction
        prediction = result['ensemble']['prediction']
        confidence = result['ensemble']['confidence']

        if prediction == 1:
            st.success(f"🦠 **Antimicrobial Peptide** (Confidence: {confidence:.2%})")
        else:
            st.error(f"🚫 **Non-Antimicrobial** (Confidence: {confidence:.2%})")

        # Individual model predictions
        st.markdown("#### Individual Model Predictions")
        for model_name, model_result in result['individual'].items():
            pred_text = "AMP" if model_result['prediction'] == 1 else "Non-AMP"
            conf_text = f"{model_result['confidence']:.2%}"
            st.markdown(f"**{model_name}**: {pred_text} ({conf_text})")

    with col2:
        # Confidence visualization
        fig = create_confidence_plot(result)
        st.plotly_chart(fig, use_container_width=True)

        # Sequence properties
        st.markdown("#### Sequence Properties")
        properties = analyze_sequence_properties(sequence)
        for prop, value in properties.items():
            st.metric(prop, value)

def display_batch_results(results, df):
    """Display results for batch analysis."""
    st.markdown("### 📊 Batch Analysis Results")

    # Summary statistics
    predictions = [r['ensemble']['prediction'] for r in results]
    confidences = [r['ensemble']['confidence'] for r in results]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sequences", len(results))
    with col2:
        st.metric("Predicted AMPs", sum(predictions))
    with col3:
        st.metric("Predicted Non-AMPs", len(predictions) - sum(predictions))
    with col4:
        st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")

    # Results dataframe
    results_df = pd.DataFrame({
        'Sequence': df['Sequence'].tolist(),
        'Prediction': ['AMP' if p == 1 else 'Non-AMP' for p in predictions],
        'Confidence': confidences,
        'Length': df['Sequence'].str.len()
    })

    st.dataframe(results_df, use_container_width=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Prediction distribution
        fig1 = px.pie(
            values=[sum(predictions), len(predictions) - sum(predictions)],
            names=['AMP', 'Non-AMP'],
            title="Prediction Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Confidence distribution
        fig2 = px.histogram(
            x=confidences,
            title="Confidence Distribution",
            labels={'x': 'Confidence', 'y': 'Count'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def analyze_sequence_properties(sequence):
    """Analyze basic sequence properties."""
    aa_groups = {
        'Hydrophobic': 'AILMFPWV',
        'Polar': 'NQST',
        'Charged': 'KRDE',
        'Aromatic': 'FWY'
    }

    properties = {}
    for group, amino_acids in aa_groups.items():
        count = sum(1 for aa in sequence if aa in amino_acids)
        properties[f"{group} %"] = f"{count/len(sequence)*100:.1f}%"

    properties["Length"] = f"{len(sequence)} aa"
    properties["MW (kDa)"] = f"{len(sequence) * 0.11:.1f}"  # Rough estimate

    return properties

if __name__ == "__main__":
    main()