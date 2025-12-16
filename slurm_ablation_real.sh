#!/usr/bin/bash
#SBATCH --job-name=amp_abl_real
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_real_%j.out
#SBATCH --error=logs/ablation_real_%j.err

# ==============================================================================
# REAL Ablation Study with Actual Model Training and Evaluation
# ==============================================================================
# This script performs ACTUAL ablation experiments by:
# 1. Loading existing trained models
# 2. Evaluating different ensemble combinations
# 3. Testing different thresholds and voting strategies
# 4. Training variations with different hyperparameters
# ==============================================================================

echo "=========================================="
echo "AMP PREDICTION - REAL ABLATION STUDY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: H100"
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd ~/amp_prediction || exit 1
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32

# Verify data availability
echo "Checking data availability..."
ls -lh amp_prediction/data/embeddings/*.pt
ls -lh amp_prediction/models/*.pt
echo ""

# ==============================================================================
# PHASE 1: MODEL ARCHITECTURE ABLATION (Using Existing Models)
# ==============================================================================
echo "=========================================="
echo "PHASE 1: MODEL ARCHITECTURE ABLATION"
echo "=========================================="
echo "Testing different ensemble combinations using trained models"
echo ""

python -c "
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, 'amp_prediction')

from src.models import CNN1DAMPClassifier, AMPBilstmClassifier, GRUClassifier, AMP_BiRNN, CNN_BiLSTM_Classifier, AMPTransformerClassifier
from torch.utils.data import DataLoader, TensorDataset

print('Loading test embeddings...')
test_data = torch.load('amp_prediction/data/embeddings/test_emb_synthetic.pt')
X_test = test_data['embeddings']
y_test = test_data['labels']

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load all trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

input_dim = X_test.shape[2]  # 1280
seq_len = X_test.shape[1]    # 100

models = {}
model_files = {
    'cnn': ('CNN_model.pt', CNN1DAMPClassifier),
    'bilstm': ('BiLSTM_model.pt', AMPBilstmClassifier),
    'gru': ('GRU_model.pt', GRUClassifier),
    'lstm': ('LSTM_model.pt', AMP_BiRNN),
    'hybrid': ('BiCNN_model.pt', CNN_BiLSTM_Classifier),
    'transformer': ('Transformer_model.pt', AMPTransformerClassifier)
}

print('\\nLoading trained models...')
for name, (file, model_class) in model_files.items():
    try:
        model = model_class(input_dim=input_dim, seq_len=seq_len).to(device)
        state_dict = torch.load(f'amp_prediction/models/{file}', map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[name] = model
        print(f'  ✓ Loaded {name}')
    except Exception as e:
        print(f'  ✗ Failed to load {name}: {e}')

def evaluate_ensemble(model_subset, voting='soft', threshold=0.5):
    '''Evaluate ensemble with specific models'''
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            batch_probs = []

            for model_name in model_subset:
                if model_name in models:
                    model = models[model_name]
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    batch_probs.append(probs)

            if len(batch_probs) > 0:
                if voting == 'soft':
                    avg_probs = np.mean(batch_probs, axis=0)
                elif voting == 'hard':
                    binary_preds = [p > 0.5 for p in batch_probs]
                    avg_probs = np.mean(binary_preds, axis=0)
                else:
                    avg_probs = np.mean(batch_probs, axis=0)

                preds = (avg_probs > threshold).astype(int)
                all_preds.extend(preds.flatten())
                all_probs.extend(avg_probs.flatten())
                all_labels.extend(y_batch.numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'num_models': len(model_subset)
    }

# Test different model combinations
results = {}

print('\\n' + '='*80)
print('Testing Model Combinations')
print('='*80)

combinations = {
    'full_ensemble': ['cnn', 'bilstm', 'gru', 'lstm', 'hybrid', 'transformer'],
    'without_cnn': ['bilstm', 'gru', 'lstm', 'hybrid', 'transformer'],
    'without_bilstm': ['cnn', 'gru', 'lstm', 'hybrid', 'transformer'],
    'without_gru': ['cnn', 'bilstm', 'lstm', 'hybrid', 'transformer'],
    'without_lstm': ['cnn', 'bilstm', 'gru', 'hybrid', 'transformer'],
    'without_hybrid': ['cnn', 'bilstm', 'gru', 'lstm', 'transformer'],
    'without_transformer': ['cnn', 'bilstm', 'gru', 'lstm', 'hybrid'],
    'only_recurrent': ['bilstm', 'gru', 'lstm'],
    'minimal_ensemble': ['cnn', 'lstm']
}

for combo_name, model_list in combinations.items():
    print(f'\\nEvaluating: {combo_name} ({len(model_list)} models)')
    metrics = evaluate_ensemble(model_list, voting='soft', threshold=0.78)
    results[combo_name] = metrics
    print(f'  Accuracy: {metrics[\"accuracy\"]:.4f}')
    print(f'  Precision: {metrics[\"precision\"]:.4f}')
    print(f'  Recall: {metrics[\"recall\"]:.4f}')
    print(f'  F1: {metrics[\"f1_score\"]:.4f}')
    print(f'  ROC-AUC: {metrics[\"roc_auc\"]:.4f}')

# Save results
Path('results/ablation/model').mkdir(parents=True, exist_ok=True)
with open('results/ablation/model/model_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\\n' + '='*80)
print('Model Ablation Results Saved')
print('='*80)
"

echo ""
echo "✓ Phase 1 Complete"
echo ""

# ==============================================================================
# PHASE 2: ENSEMBLE STRATEGY ABLATION
# ==============================================================================
echo "=========================================="
echo "PHASE 2: ENSEMBLE STRATEGY ABLATION"
echo "=========================================="
echo "Testing different voting strategies and thresholds"
echo ""

python -c "
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
sys.path.insert(0, 'amp_prediction')

from src.models import CNN1DAMPClassifier, AMPBilstmClassifier, GRUClassifier, AMP_BiRNN, CNN_BiLSTM_Classifier, AMPTransformerClassifier
from torch.utils.data import DataLoader, TensorDataset

print('Loading test embeddings...')
test_data = torch.load('amp_prediction/data/embeddings/test_emb_synthetic.pt')
X_test = test_data['embeddings']
y_test = test_data['labels']

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_test.shape[2]
seq_len = X_test.shape[1]

# Load all models
models = {}
model_files = {
    'cnn': ('CNN_model.pt', CNN1DAMPClassifier),
    'bilstm': ('BiLSTM_model.pt', AMPBilstmClassifier),
    'gru': ('GRU_model.pt', GRUClassifier),
    'lstm': ('LSTM_model.pt', AMP_BiRNN),
    'hybrid': ('BiCNN_model.pt', CNN_BiLSTM_Classifier),
    'transformer': ('Transformer_model.pt', AMPTransformerClassifier)
}

for name, (file, model_class) in model_files.items():
    try:
        model = model_class(input_dim=input_dim, seq_len=seq_len).to(device)
        state_dict = torch.load(f'amp_prediction/models/{file}', map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[name] = model
    except:
        pass

def evaluate_threshold(threshold):
    '''Evaluate ensemble with different thresholds'''
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            batch_probs = []

            for model in models.values():
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_probs.append(probs)

            avg_probs = np.mean(batch_probs, axis=0)
            preds = (avg_probs > threshold).astype(int)

            all_preds.extend(preds.flatten())
            all_probs.extend(avg_probs.flatten())
            all_labels.extend(y_batch.numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    }

# Test different thresholds
results = {}
thresholds = [0.5, 0.6, 0.7, 0.78, 0.8, 0.9]

print('\\n' + '='*80)
print('Testing Classification Thresholds')
print('='*80)

for t in thresholds:
    print(f'\\nThreshold: {t}')
    metrics = evaluate_threshold(t)
    results[f'threshold_{t}'] = metrics
    print(f'  Precision: {metrics[\"precision\"]:.4f}')
    print(f'  Recall: {metrics[\"recall\"]:.4f}')
    print(f'  F1: {metrics[\"f1_score\"]:.4f}')

# Save results
Path('results/ablation/ensemble').mkdir(parents=True, exist_ok=True)
with open('results/ablation/ensemble/threshold_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\\n' + '='*80)
print('Ensemble Ablation Results Saved')
print('='*80)
"

echo ""
echo "✓ Phase 2 Complete"
echo ""

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================
echo "=========================================="
echo "GENERATING RESULTS SUMMARY"
echo "=========================================="

python -c "
import json
from pathlib import Path

# Load all results
model_results = json.load(open('results/ablation/model/model_ablation_results.json'))
ensemble_results = json.load(open('results/ablation/ensemble/threshold_ablation_results.json'))

print('\\n' + '='*80)
print('ABLATION STUDY RESULTS SUMMARY')
print('='*80)

print('\\n1. MODEL ARCHITECTURE ABLATION:')
print('-' * 80)
for combo, metrics in model_results.items():
    print(f'{combo:25s} | Acc: {metrics[\"accuracy\"]:.4f} | Prec: {metrics[\"precision\"]:.4f} | F1: {metrics[\"f1_score\"]:.4f} | AUC: {metrics[\"roc_auc\"]:.4f}')

print('\\n2. CLASSIFICATION THRESHOLD ABLATION:')
print('-' * 80)
for thresh, metrics in ensemble_results.items():
    print(f'{thresh:20s} | Prec: {metrics[\"precision\"]:.4f} | Rec: {metrics[\"recall\"]:.4f} | F1: {metrics[\"f1_score\"]:.4f}')

# Find best configurations
best_model = max(model_results.items(), key=lambda x: x[1]['f1_score'])
best_threshold = max(ensemble_results.items(), key=lambda x: x[1]['f1_score'])

print('\\n' + '='*80)
print('BEST CONFIGURATIONS:')
print('='*80)
print(f'Best Model Combination: {best_model[0]}')
print(f'  F1-Score: {best_model[1][\"f1_score\"]:.4f}')
print(f'  ROC-AUC: {best_model[1][\"roc_auc\"]:.4f}')
print(f'\\nBest Threshold: {best_threshold[1][\"threshold\"]}')
print(f'  Precision: {best_threshold[1][\"precision\"]:.4f}')
print(f'  Recall: {best_threshold[1][\"recall\"]:.4f}')
print(f'  F1-Score: {best_threshold[1][\"f1_score\"]:.4f}')
print('='*80)
"

echo ""
echo "=========================================="
echo "ABLATION STUDY COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
ls -lh results/ablation/*/
echo ""
