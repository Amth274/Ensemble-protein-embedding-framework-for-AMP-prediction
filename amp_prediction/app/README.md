# 🧬 AMP Prediction Demo Application

This directory contains interactive demo applications for the Enhanced Antimicrobial Peptide Prediction system.

## 📁 Directory Structure

```
app/
├── flask_app/              # Interactive web application
│   ├── app.py              # Main Flask application
│   ├── templates/          # HTML templates
│   │   ├── base.html       # Base template
│   │   ├── index.html      # Homepage
│   │   ├── predict.html    # Single prediction page
│   │   ├── batch.html      # Batch analysis page
│   │   └── examples.html   # Examples page
│   └── static/             # Static assets
│       ├── css/            # Stylesheets
│       ├── js/             # JavaScript files
│       └── images/         # Images
├── notebooks/              # Jupyter notebook demos
│   └── 01_Quick_Start_Demo.ipynb
├── utils/                  # Demo utilities
│   ├── demo_utils.py       # Prediction and validation utilities
│   └── visualization.py   # Plotting and visualization functions
├── examples/               # Example datasets
│   └── sample_sequences.csv
├── run_flask_app.py        # Flask app launcher
└── static/                 # Legacy static assets
```

## 🚀 Quick Start

### Option 1: Flask Web App (Recommended)

Launch the interactive web application:

```bash
# From the project root directory
cd amp_prediction/app
python run_flask_app.py
```

Or directly with Flask:

```bash
cd amp_prediction/app/flask_app
python app.py
```

The app will open in your browser at `http://127.0.0.1:5000`

### Option 2: Jupyter Notebooks

Explore the interactive notebooks:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook app/notebooks/

# Open 01_Quick_Start_Demo.ipynb
```

## 🎯 Features

### Flask Web App

The web application provides four main sections:

#### 🔍 Single Prediction
- Enter a protein sequence and get instant AMP predictions
- View individual model confidences and ensemble results
- Analyze sequence properties (length, composition, charge)
- Real-time sequence validation and interactive charts
- Adjustable prediction threshold

#### 📊 Batch Analysis
- Upload CSV files with multiple sequences
- Drag-and-drop file upload interface
- Analyze entire datasets with progress tracking
- Download results with predictions and confidences
- Sample data loading for quick testing

#### 🧪 Examples
- Pre-loaded examples of known AMPs and non-AMPs
- Famous peptides like Magainin-2, LL-37, Melittin, Cecropin A
- One-click testing of example sequences
- Detailed peptide information and activity data

#### 📈 Homepage & Architecture
- Modern responsive design with Bootstrap
- Ensemble architecture overview
- Individual model performance metrics
- Feature showcase and quick start guide

### Jupyter Notebooks

#### 01_Quick_Start_Demo.ipynb
- Complete walkthrough of the prediction system
- Single and batch prediction examples
- Visualization tutorials
- Sequence analysis and composition studies
- Uncertainty estimation
- Sequence variant exploration

## 🛠️ Dependencies

Required packages for the demo applications:

```bash
# Core dependencies
pip install flask pandas numpy torch

# Optional for enhanced features
pip install jupyter seaborn matplotlib scikit-learn plotly
```

Or install all at once:

```bash
pip install -r ../requirements.txt
```

## 📊 Example Usage

### Single Sequence Prediction

```python
from app.utils import DemoPredictor

# Initialize predictor
predictor = DemoPredictor()

# Predict AMP activity
result = predictor.predict_single("GIGKFLHSAKKFGKAFVGEIMNS")

print(f"Prediction: {'AMP' if result['ensemble']['prediction'] == 1 else 'Non-AMP'}")
print(f"Confidence: {result['ensemble']['confidence']:.2%}")
```

### Batch Analysis

```python
from app.utils import load_example_data

# Load example dataset
df = load_example_data()

# Analyze all sequences
results = predictor.predict_batch(df['Sequence'].tolist())

# Process results
predictions = [r['ensemble']['prediction'] for r in results]
confidences = [r['ensemble']['confidence'] for r in results]
```

### Visualization

```python
from app.utils.visualization import create_prediction_plot

# Create interactive plot
fig = create_prediction_plot(results)
fig.show()
```

## 🎨 Customization

### Adding New Examples

Edit `examples/sample_sequences.csv`:

```csv
Sequence,Known_Label,Source,Description,Length,MIC_uM
YOURSEQUENCE,1,Source,Description,15,25.0
```

### Custom Visualizations

Extend `utils/visualization.py`:

```python
def create_custom_plot(data):
    # Your custom visualization
    fig = go.Figure()
    # ... plotting code ...
    return fig
```

### Model Integration

Replace the `DemoPredictor` with real trained models by:

1. Loading actual model weights
2. Implementing real ESM embedding generation
3. Using the ensemble classes from `src/ensemble/`

## 🔧 Troubleshooting

### Common Issues

**App won't start:**
- Check that all dependencies are installed
- Ensure you're in the correct directory
- Try `pip install --upgrade flask`

**Predictions seem unrealistic:**
- The demo uses mock models for demonstration
- Replace `DemoPredictor` with real trained models for actual predictions

**Visualizations not showing:**
- Ensure Chart.js is loading properly (included via CDN)
- Try refreshing the browser page
- Check browser console for JavaScript errors

**Memory issues:**
- The demo is designed to be lightweight
- For production use, consider GPU acceleration for ESM embeddings

## 📝 Demo Limitations

This demo application includes:

- ✅ **Mock models** for fast demonstration
- ✅ **Simplified predictions** based on heuristics
- ✅ **Example datasets** with known AMPs
- ✅ **Interactive visualizations**
- ✅ **Educational content**

For production use, you'll need:

- 🔄 **Real trained models** (see training scripts)
- 🔄 **ESM embedding generation** (computationally intensive)
- 🔄 **GPU support** for faster inference
- 🔄 **Larger datasets** for comprehensive analysis

## 🚀 Next Steps

1. **Explore the demos** to understand the system capabilities
2. **Train real models** using the training scripts
3. **Generate embeddings** for your own datasets
4. **Customize the interface** for your specific needs
5. **Deploy for production** with proper model weights

## 📞 Support

- Check the main README for detailed documentation
- Review the source code in `src/` for implementation details
- Open issues on GitHub for bugs or feature requests

---

**Happy exploring! 🧬✨**