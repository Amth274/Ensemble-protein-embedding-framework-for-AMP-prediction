"""
Flask Web Application for AMP Prediction Demo

A modern, responsive web interface for demonstrating the Enhanced
Antimicrobial Peptide Prediction system using Flask.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging

# Add src to path
app_dir = Path(__file__).parent.parent
sys.path.append(str(app_dir.parent / "src"))

try:
    # Add utils path for imports
    utils_path = app_dir / "utils"
    sys.path.append(str(utils_path))

    from demo_utils import DemoPredictor, validate_sequence, load_example_data
    from demo_utils import analyze_sequence_composition, generate_sequence_variants
except ImportError as e:
    # Fallback if imports fail
    DemoPredictor = None
    print(f"Warning: Demo utilities not available. Some features may be limited. Error: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'amp-prediction-demo-key-2024'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor instance."""
    global predictor
    try:
        if DemoPredictor is not None:
            predictor = DemoPredictor()
            logger.info("Demo predictor initialized successfully")
        else:
            logger.warning("Demo predictor not available")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")

# Initialize predictor when module is loaded
initialize_predictor()

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Single sequence prediction page."""
    return render_template('predict.html')

@app.route('/batch')
def batch_page():
    """Batch analysis page."""
    return render_template('batch.html')

@app.route('/examples')
def examples_page():
    """Examples page with known AMPs."""
    try:
        example_data = load_example_data()
        examples = example_data.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to load examples: {str(e)}")
        examples = []

    return render_template('examples.html', examples=examples)

@app.route('/test-modal')
def test_modal_page():
    """Test page for debugging modal issues."""
    return render_template('test_modal.html')

@app.route('/about')
def about_page():
    """About page with model information."""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single sequence prediction."""
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        # Validate sequence
        is_valid, error_msg = validate_sequence(sequence)
        if not is_valid:
            return jsonify({'error': f'Invalid sequence: {error_msg}'}), 400

        if predictor is None:
            return jsonify({'error': 'Predictor not available'}), 500

        # Make prediction
        result = predictor.predict_single(sequence)

        # Add sequence analysis
        composition = analyze_sequence_composition(sequence)
        result['composition'] = composition

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """API endpoint for batch prediction."""
    try:
        data = request.get_json()
        sequences = data.get('sequences', [])

        if not sequences:
            return jsonify({'error': 'No sequences provided'}), 400

        if len(sequences) > 100:  # Limit batch size
            return jsonify({'error': 'Too many sequences (max 100)'}), 400

        if predictor is None:
            return jsonify({'error': 'Predictor not available'}), 500

        # Validate all sequences
        valid_sequences = []
        for seq in sequences:
            seq = seq.strip().upper()
            is_valid, _ = validate_sequence(seq)
            if is_valid:
                valid_sequences.append(seq)

        if not valid_sequences:
            return jsonify({'error': 'No valid sequences found'}), 400

        # Make predictions
        results = predictor.predict_batch(valid_sequences)

        # Add summary statistics
        predictions = [r['ensemble']['prediction'] for r in results]
        confidences = [r['ensemble']['confidence'] for r in results]

        summary = {
            'total_sequences': len(results),
            'predicted_amps': sum(predictions),
            'predicted_non_amps': len(predictions) - sum(predictions),
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }

        return jsonify({
            'results': results,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed'}), 500

@app.route('/api/analyze_composition', methods=['POST'])
def api_analyze_composition():
    """API endpoint for sequence composition analysis."""
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        # Validate sequence
        is_valid, error_msg = validate_sequence(sequence)
        if not is_valid:
            return jsonify({'error': f'Invalid sequence: {error_msg}'}), 400

        # Analyze composition
        composition = analyze_sequence_composition(sequence)

        return jsonify(composition)

    except Exception as e:
        logger.error(f"Composition analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/generate_variants', methods=['POST'])
def api_generate_variants():
    """API endpoint for generating sequence variants."""
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        n_variants = data.get('n_variants', 5)

        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400

        # Validate sequence
        is_valid, error_msg = validate_sequence(sequence)
        if not is_valid:
            return jsonify({'error': f'Invalid sequence: {error_msg}'}), 400

        # Generate variants
        variants = generate_sequence_variants(sequence, n_variants)

        # Predict for all variants
        if predictor is not None:
            all_sequences = [sequence] + variants
            all_results = predictor.predict_batch(all_sequences)

            variant_data = []
            for i, (seq, result) in enumerate(zip(all_sequences, all_results)):
                variant_data.append({
                    'type': 'Original' if i == 0 else f'Variant {i}',
                    'sequence': seq,
                    'prediction': 'AMP' if result['ensemble']['prediction'] == 1 else 'Non-AMP',
                    'confidence': result['ensemble']['confidence']
                })
        else:
            variant_data = [{'sequence': seq, 'type': f'Variant {i+1}'}
                          for i, seq in enumerate(variants)]

        return jsonify({'variants': variant_data})

    except Exception as e:
        logger.error(f"Variant generation error: {str(e)}")
        return jsonify({'error': 'Variant generation failed'}), 500

@app.route('/api/examples')
def api_examples():
    """API endpoint to get example sequences."""
    try:
        example_data = load_example_data()
        examples = example_data.to_dict('records')
        return jsonify(examples)
    except Exception as e:
        logger.error(f"Failed to load examples: {str(e)}")
        return jsonify({'error': 'Failed to load examples'}), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'predictor_available': predictor is not None,
        'version': '1.0.0'
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    return render_template('500.html'), 500

# Static file serving for development
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Initialize predictor
    initialize_predictor()

    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )