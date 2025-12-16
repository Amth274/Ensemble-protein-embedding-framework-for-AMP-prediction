# Model Ablation Script Analysis

## Overview

The `model_ablation.py` script performs a comprehensive ablation study by testing **6 different ESM protein language models** (ranging from 8M to 15B parameters) across **6 different classifier architectures** in a single execution. This creates a total of **36 model combinations** to systematically evaluate the impact of embedding model size on downstream classification performance.

## Script Location

- **File**: `amp_prediction/scripts/model_ablation.py`
- **Original Source**: `C:\Users\alpine\Downloads\model_abelation.py` (note: typo in filename)

---

## Key Features

### 1. **Comprehensive ESM Model Coverage**

Tests all major ESM-2 model variants:

| Model Name | Parameters | Hidden Dim | Description |
|------------|-----------|------------|-------------|
| `esm2_t48_15B_UR50D` | 15B | 5120 | Largest, highest capacity |
| `esm2_t36_3B_UR50D` | 3B | 2560 | Large model |
| `esm2_t33_650M_UR50D` | 650M | 1280 | **Current default** |
| `esm2_t30_150M_UR50D` | 150M | 640 | Lightweight option |
| `esm2_t12_35M_UR50D` | 35M | 480 | Very lightweight |
| `esm2_t6_8M_UR50D` | 8M | 320 | Smallest, fastest |

### 2. **Six Classifier Architectures**

Each ESM embedding is tested with all six models from the ensemble:

1. **CNN1DAMPClassifier**: 1D CNN with multi-scale convolutions
2. **AMPBilstmClassifier**: Bidirectional LSTM (3 layers)
3. **AMPTransformerClassifier**: Transformer encoder with self-attention
4. **GRUClassifier**: Bidirectional GRU
5. **AMP_BiRNN**: Bidirectional LSTM-RNN (2 layers)
6. **CNN_BiLSTM_Classifier**: Hybrid CNN + BiLSTM architecture

### 3. **Two-Stage Pipeline**

**Stage 1: Embedding Extraction (On-the-fly)**
- Loads ESM model and tokenizer from HuggingFace
- Processes sequences in batches (EMB_BATCH_SIZE=16)
- Extracts per-residue embeddings from `last_hidden_state`
- Handles special tokens (BOS/EOS) properly
- Pads/truncates to fixed sequence length (SEQ_MAX)
- Stores embeddings in memory for training

**Stage 2: Classifier Training**
- Trains each classifier architecture on the extracted embeddings
- 50 epochs with early stopping based on precision
- Uses custom ConservativeBCELoss (penalizes false positives)
- Saves best model checkpoint per combination
- Evaluates on test set and reports comprehensive metrics

---

## Architecture Details

### Embedding Extraction Process

```python
def extract_embeddings(df):
    # Tokenizes sequences with padding
    inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True)

    # Extracts embeddings from ESM model
    outputs = esm_model(**inputs)
    last_hidden = outputs.last_hidden_state  # [B, L, H]

    # Removes special tokens (BOS/EOS)
    aa_embeds = last_hidden[j, start:end, :]

    # Pads/truncates to SEQ_MAX
    aa_embeds = pad_truncate_embeddings(aa_embeds, SEQ_MAX, hidden_dim)
```

**Key Differences from Existing Pipeline**:
- **On-the-fly extraction**: Doesn't pre-compute embeddings to disk like `generate_embeddings.py`
- **Memory-based**: Keeps all embeddings in RAM for immediate use
- **Dynamic hidden_dim**: Automatically adapts to each ESM model's embedding dimension
- **Proper special token handling**: Uses attention mask to exclude BOS/EOS tokens

### Model Architectures

All models are **dynamically instantiated** with the correct `embedding_dim` for each ESM model:

```python
ensemble_models = [
    CNN1DAMPClassifier(embedding_dim=hidden_dim),    # hidden_dim varies: 320 to 5120
    AMPBilstmClassifier(embedding_dim=hidden_dim),
    AMPTransformerClassifier(embedding_dim=hidden_dim),
    GRUClassifier(embedding_dim=hidden_dim),
    AMP_BiRNN(input_dim=hidden_dim),
    CNN_BiLSTM_Classifier(input_dim=hidden_dim)
]
```

**Note**: This is a major improvement over fixed-dimension architectures, allowing testing with embeddings of different sizes.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 50 | Fixed, no early stopping on patience |
| **Train Batch Size** | 32 | For classifier training |
| **Test Batch Size** | 64 | For evaluation |
| **Optimizer** | AdamW | lr=2e-4, weight_decay=1e-5 |
| **Scheduler** | StepLR | step_size=3, gamma=0.7 |
| **Loss Function** | ConservativeBCELoss | Custom loss with false positive penalty |
| **Best Model Selection** | Precision | Saves model with highest precision |

### ConservativeBCELoss

Custom loss function that heavily penalizes false positives:

```python
base_loss = BCE(logits, targets)
probs = sigmoid(logits)
diff = probs - targets
penalty = relu(diff)^3  # Cubic penalty for overprediction
total_loss = base_loss + alpha * penalty
```

**Why this matters**: For AMP prediction, false positives (predicting non-AMPs as AMPs) are more costly than false negatives, so this loss function prioritizes high precision.

---

## Comparison with Existing Ablation Framework

### `scripts/ablation/run_ablation.py` (Existing)

**Pros**:
- Systematic configuration via YAML
- Supports multiple ablation types (embeddings, models, ensembles, training)
- Integrated with config management system
- Results saved to structured JSON files
- Multiple random seed support

**Cons**:
- Requires pre-computed embeddings
- Fixed to one ESM model at a time
- More complex setup (config files, CLI arguments)

### `scripts/model_ablation.py` (New)

**Pros**:
- Tests 6 ESM models in a single run (comprehensive embedding ablation)
- On-the-fly embedding extraction (no preprocessing needed)
- Self-contained script (no config files required)
- Dynamic architecture adaptation to different embedding dimensions
- Optimized for large-scale comparative analysis

**Cons**:
- Hardcoded file paths (needs modification for different data)
- Memory-intensive (keeps all embeddings in RAM)
- No ensemble evaluation (only individual models)
- Longer total runtime (embedding extraction + training × 36 combinations)

---

## Computational Requirements

### Memory Requirements (Estimated)

| ESM Model | GPU VRAM (Inference) | Embedding Storage | Classifier Training |
|-----------|----------------------|-------------------|---------------------|
| 15B params | ~60GB | ~500MB/dataset | ~4-8GB |
| 3B params | ~12GB | ~300MB/dataset | ~4-8GB |
| 650M params | ~4GB | ~200MB/dataset | ~4-8GB |
| 150M params | ~2GB | ~100MB/dataset | ~4-8GB |
| 35M params | ~0.5GB | ~80MB/dataset | ~4-8GB |
| 8M params | ~0.2GB | ~50MB/dataset | ~4-8GB |

**Total memory for single ESM model**: Embedding VRAM + Storage + Training VRAM

**Critical Consideration**: The 15B model requires **high-memory GPUs** (A100 80GB or H100). For other models, V100 32GB or A100 40GB is sufficient.

### Runtime Estimates (on H100 GPU)

Assuming dataset size of ~10K train + 2K test sequences:

| ESM Model | Embedding Extraction | Training 6 Models (50 epochs) | Total per ESM |
|-----------|---------------------|-------------------------------|---------------|
| 15B | ~45-60 min | ~3-4 hours | **4-5 hours** |
| 3B | ~15-20 min | ~3-4 hours | **3.5-4.5 hours** |
| 650M | ~5-8 min | ~3-4 hours | **3-4 hours** |
| 150M | ~3-5 min | ~3-4 hours | **3-4 hours** |
| 35M | ~2-3 min | ~3-4 hours | **3-4 hours** |
| 8M | ~1-2 min | ~3-4 hours | **3-4 hours** |

**Total Runtime for All 6 ESM Models**: ~20-26 hours on H100 GPU

**Recommendation**: For HPC/SLURM users, set job time limit to **12 hours per ESM model** or **72 hours for complete ablation** (with safety margin).

---

## Key Findings Expected

### Research Questions This Script Answers:

1. **Does larger ESM model size improve downstream classification?**
   - Compare ROC-AUC/Precision across 8M → 650M → 3B → 15B progression
   - Identify point of diminishing returns

2. **Which classifier architecture benefits most from richer embeddings?**
   - Compare performance deltas (e.g., Transformer may benefit more from 15B vs 8M)
   - Identify architecture-embedding synergies

3. **Is the 650M model truly optimal, or can we do better with larger/smaller models?**
   - Current baseline uses 650M (1280-dim embeddings)
   - Test if 3B (2560-dim) or 15B (5120-dim) improves performance
   - Test if lightweight models (150M, 35M) maintain acceptable performance

4. **Can we reduce computational cost without sacrificing performance?**
   - If 150M achieves similar metrics to 650M, recommend switching to lighter model
   - Quantify performance vs. cost tradeoffs

---

## Integration with Existing Codebase

### Similarities

1. **Model Architectures**: Uses identical classes from `src/models/`
2. **Metrics**: Same evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
3. **Data Format**: Expects same CSV structure (Sequence, Length, label columns)

### Differences

1. **No Config Files**: Hardcoded hyperparameters (vs. YAML-based config system)
2. **No CLI Interface**: Not integrated with `amp-embed` or `amp-train` commands
3. **No Ensemble Evaluation**: Tests individual models, not ensemble voting
4. **Custom Loss**: Uses ConservativeBCELoss instead of standard BCEWithLogitsLoss

### Potential Integration Path

To integrate this script into the existing framework:

1. **Add command-line arguments** for data paths, output directory, ESM model selection
2. **Save embeddings to disk** (optional) for reuse (like `generate_embeddings.py`)
3. **Use config.yaml** for hyperparameters instead of hardcoding
4. **Add ensemble evaluation** after individual model training
5. **Generate ablation report** (JSON/CSV) compatible with existing results format

---

## Recommendations for Usage

### When to Use This Script

✅ **Use `model_ablation.py` when**:
- You want to compare multiple ESM models systematically
- You have access to high-memory GPUs (especially for 15B/3B models)
- You want a comprehensive embedding ablation study
- You can afford long runtimes (20-26 hours for full sweep)

❌ **Don't use this script when**:
- You only need to test one ESM model (use `generate_embeddings.py` + `train_amp_models.py`)
- You have limited GPU memory (<16GB VRAM)
- You need quick experiments (use existing ablation framework with pre-computed embeddings)
- You want ensemble performance (this only tests individual models)

### Recommended Modifications for Production Use

1. **Add command-line arguments**:
   ```python
   parser = argparse.ArgumentParser()
   parser.add_argument('--train_data', required=True)
   parser.add_argument('--test_data', required=True)
   parser.add_argument('--esm_models', nargs='+', default=['facebook/esm2_t33_650M_UR50D'])
   parser.add_argument('--output_dir', default='models')
   parser.add_argument('--num_epochs', type=int, default=50)
   ```

2. **Save embeddings for reuse**:
   ```python
   torch.save({
       'train_embeddings': train_embeddings,
       'test_embeddings': test_embeddings,
       'esm_model': esm_name,
       'hidden_dim': hidden_dim
   }, f'data/embeddings/{esm_name.split("/")[-1]}_embeddings.pt')
   ```

3. **Add result logging**:
   ```python
   results = {
       'esm_model': esm_name,
       'classifier': model_name_short,
       'metrics': {
           'accuracy': acc,
           'precision': prec,
           'recall': rec,
           'f1': f1,
           'roc_auc': roc_auc,
           'ap': prc_auc
       }
   }
   with open(f'results/ablation/{model_name_short}_results.json', 'w') as f:
       json.dump(results, f, indent=2)
   ```

4. **Add SLURM job script template**:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=amp_ablation
   #SBATCH --time=12:00:00
   #SBATCH --gres=gpu:h100:1
   #SBATCH --mem=64G

   python model_ablation.py \
       --train_data data/train.csv \
       --test_data data/test.csv \
       --esm_models facebook/esm2_t33_650M_UR50D facebook/esm2_t36_3B_UR50D \
       --output_dir models/ablation
   ```

---

## File Paths to Update

Current hardcoded paths in the script:

```python
train_df = pd.read_csv('/home/aum-thaker/Desktop/VSC/AMP/Data/cls/train.csv')
test_df  = pd.read_csv('/home/aum-thaker/Desktop/VSC/AMP/Data/cls/test.csv')
```

**Recommended change** (use relative paths):

```python
train_df = pd.read_csv('amp_prediction/data/train.csv')
test_df  = pd.read_csv('amp_prediction/data/test.csv')
```

Or add as command-line arguments.

---

## Comparison with CLAUDE.md Guidance

### Alignment with Best Practices

✅ **Follows best practices**:
- Uses same model architectures from codebase
- Proper GPU device handling
- Batch processing for efficiency
- Best model checkpointing
- Comprehensive metrics reporting

⚠️ **Needs improvement**:
- Hardcoded paths (violates config-based approach)
- No random seed setting (reduces reproducibility)
- No integration with existing CLI tools
- Custom loss function not documented in config

### CLAUDE.md Updates Needed

Add section to CLAUDE.md about this script:

```markdown
### Large-Scale Embedding Ablation
```bash
# Test multiple ESM models systematically (requires high VRAM)
cd amp_prediction/scripts
python model_ablation.py  # Tests 6 ESM models × 6 classifiers = 36 combinations

# For HPC users: Set time limit to 12 hours per ESM model
# Full ablation (6 models): ~72 hours total
```

**When to use**: Comprehensive embedding ablation across ESM model sizes
**GPU requirements**: 16GB+ VRAM (60GB+ for 15B model)
**Runtime**: ~4 hours per ESM model on H100 GPU
```

---

## Summary Table: Script Outputs

Each run produces 36 model checkpoints:

| ESM Model | Classifiers | Checkpoints | Naming Pattern |
|-----------|-------------|-------------|----------------|
| esm2_t48_15B_UR50D | 6 | 6 files | `esm2_t48_15B_UR50D__CNN1DAMPClassifier_best.pt` |
| esm2_t36_3B_UR50D | 6 | 6 files | `esm2_t36_3B_UR50D__AMPBilstmClassifier_best.pt` |
| esm2_t33_650M_UR50D | 6 | 6 files | `esm2_t33_650M_UR50D__GRUClassifier_best.pt` |
| esm2_t30_150M_UR50D | 6 | 6 files | ... |
| esm2_t12_35M_UR50D | 6 | 6 files | ... |
| esm2_t6_8M_UR50D | 6 | 6 files | ... |

**Total**: 36 checkpoint files in `models/` directory

---

## Conclusion

The `model_ablation.py` script is a **powerful tool for comprehensive embedding ablation studies** that systematically evaluates the impact of ESM model size (8M to 15B parameters) on AMP classification performance across six different neural architectures.

**Key Value**: Answers the critical research question: *"Is the 650M ESM model optimal, or should we use larger (3B/15B) or smaller (150M/35M) models?"*

**Main Limitation**: Requires significant computational resources (high-memory GPUs, 20-26 hours runtime) and has hardcoded paths that need modification before use.

**Recommended Action**: Integrate this script's functionality into the existing ablation framework with proper CLI arguments, config file support, and result logging for production use.
