import os 
import numpy as np 
import sklearn 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from tqdm import tqdm
from torch.utils.data import Dataset,dataloader,DataLoader
from transformers import AutoModel, AutoTokenizer
## import all model
model_path = {
    'CNN':'cnn_regression copy.pt',
    'biLSTM':'bilstm_regression.pt',
    'Bi-CNN':'bi-cnn_reg.pt',
    'GRU':'gru_regression copy.pt',
    # 'Logistic':'logreg_regression.pt',
    'LSTM':'lstm_regression.pt',
    'Transformer':'transformer_regression.pt'
}
# train_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/reg/train_emb.pt'
test_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/reg/test_esm_regression.pt'
# train_dataset = torch.load(train_dataset)
test_dataset = torch.load(test_dataset)
test_dataset
# train_dataset[0]['label']
# train_embeddings = [d['embeddings'] for d in train_dataset]  # list of [seq_len, 1280] tensors
# train_labels = [d['label'] for d in train_dataset]

test_embeddings = [d['embeddings'] for d in test_dataset]
test_labels = [d['label'] for d in test_dataset]
test_values = [d['value'] for d in test_dataset]

def pad_embedding(tensor, max_len=100):
    if tensor.size(0) >= max_len:
        return tensor[:max_len]
    else:
        return F.pad(tensor, (0, 0, 0, max_len - tensor.size(0)))  # pad on seq_len axis

class EnsembeDataset(Dataset):
    def __init__(self, data, max_len=100):
        self.data = data
        self.max_len = max_len

    def pad_embedding(self, tensor):
        if tensor.size(0) >= self.max_len:
            return tensor[:self.max_len]
        else:
            return F.pad(tensor, (0, 0, 0, self.max_len - tensor.size(0)))  # pad along seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # if isinstance(entry, dict):
        emb = self.pad_embedding(entry['embeddings'])  # [max_len, 1280]
        label = entry['label']
        value = entry['value']
        return emb, label, value
        # else:
        #     raise TypeError(f"Expected dict, got {type(entry)}: {entry}")

test_dataset[1]['value']
test_dataset = EnsembeDataset(test_dataset)
test_loader = DataLoader(test_dataset,shuffle=True,batch_size=1)
for emb,label,value in test_loader:
    print(value.shape)
    break
class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, hidden_dim=256, num_layers=1, dropout=0.3, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_directions * hidden_dim),
            nn.Linear(self.num_directions * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Final output logit
        )

    def forward(self, x):  # x shape: [B, L=100, D=1280]
        output, hidden = self.gru(x)  # output: [B, L, num_directions*H], hidden: [num_layers*num_directions, B, H]
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2*H]
        else:
            hidden_cat = hidden[-1]  # [B, H]
        return self.classifier(hidden_cat).squeeze(1)  # [B]

class CNN1DAMPClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, seq_len=100, num_classes=1, dropout=0.3):
        super(CNN1DAMPClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.conv2 = nn.Conv1d(512, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveMaxPool1d(1)  # Output: [B, C, 1]
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                # [B, C]
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)  # [B, 1]
        )

    def forward(self, x):  # x: [B, L, D] = [B, 100, 1280]
        x = x.permute(0, 2, 1)  # -> [B, D, L] for Conv1D
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 512, L]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 256, L]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, L]
        x = self.pool(x)                     # [B, 128, 1]
        out = self.classifier(x)             # [B, 1]
        return out.squeeze(1)                # [B]

class CNN_BiLSTM_Classifier(nn.Module):
    def __init__(self, input_dim=1280, cnn_out_channels=256, lstm_hidden_size=128, lstm_layers=1, dropout=0.5):
        super(CNN_BiLSTM_Classifier, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.bilstm = nn.LSTM(input_size=cnn_out_channels,
                              hidden_size=lstm_hidden_size,
                              num_layers=lstm_layers,
                              dropout=dropout if lstm_layers > 1 else 0,
                              batch_first=True,
                              bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: [B, L, 1280]
        x = x.transpose(1, 2)  # [B, 1280, L]
        x = self.conv1d(x)     # [B, C, L]
        x = self.relu(x)
        x = self.maxpool(x)    # [B, C, L//2]

        x = x.transpose(1, 2)  # [B, L//2, C]
        output, _ = self.bilstm(x)  # [B, L//2, 2H]
        out = output[:, -1, :]      # Take last time step

        out = self.dropout(out)
        out = self.classifier(out)  # [B, 1]
        return out.squeeze(1)       # [B]

class AMPBilstmClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, hidden_dim=256, num_layers=1, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # No sigmoid
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # [B, L, 2*H]
        pooled = torch.mean(lstm_out, dim=1)  # Mean pooling
        return self.classifier(pooled).squeeze(1)  # [B]

class AMP_BiRNN(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)              # [B, T, 2*H]
        out = out[:, -1, :]               # [B, 2*H] — use last hidden state
        out = self.classifier(out)        # [B, 1]
        return out.squeeze(1)             # [B]

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=1280):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # Output logits (no sigmoid)

class AMPTransformerClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, seq_len=100, num_heads=1, num_layers=1, dropout=0.3):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            # nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(512, 1)
        )

    def forward(self, x):  # x: [B, L, D]
        x = self.transformer(x)              # → [B, L, D]
        x = x.mean(dim=1)                    # Mean pooling over L → [B, D]
        return self.classifier(x).squeeze(1) # → [B]

torch.cuda.is_available()
device='cuda'
models = {
    'CNN': CNN1DAMPClassifier().to(device),
    'biLSTM': AMPBilstmClassifier().to(device),
    'Bi-CNN': CNN_BiLSTM_Classifier().to(device),
    'GRU': GRUClassifier().to(device),
    # 'Logistic': LogisticRegression().to(device),
    'LSTM': AMP_BiRNN().to(device),
    'Transformer': AMPTransformerClassifier().to(device)
}
for name, model in models.items():
    model.load_state_dict(torch.load(model_path[name], map_location=device))
    model.eval()

model_outputs = {}
all_targets = []

with torch.no_grad():
    for x, _, values in tqdm(test_loader, desc="Evaluating"):
        x = x.to(device)
        values = values.to(device).float().view(-1)  # [B]
        all_targets.extend(values.cpu().numpy())

        for name, model in models.items():
            if name not in model_outputs:
                model_outputs[name] = []

            if name.lower() == 'logistic':
                x_input = x.mean(dim=1)  # For logistic regression
            else:
                x_input = x

            preds = model(x_input).squeeze()  # [B]
            model_outputs[name].append(preds.cpu())

len(model_outputs['CNN'])
weightage_mse = {
    'CNN':0.4259,
    'biLSTM':0.4548,
    'Bi-LSTM':0.4761,
    'GRU':0.4127,
    'LSTM':0.4433,
    'Transformer':0.5167


}
weightage_r2 = {
    'CNN':0.8090,
    'biLSTM':0.7960,
    'Bi-CNN':0.7864,
    'GRU':0.8149,
    'LSTM':0.8011,
    'Transformer':0.7682


}
for name in model_outputs:
    print(name)
    # break
mse_per_model = {
    'CNN': 0.4259,
    'biLSTM': 0.4548,
    'Bi-CNN': 0.4761,   # Correcting 'Bi-LSTM' to 'Bi-CNN'
    'GRU': 0.4127,
    'LSTM': 0.4433,
    'Transformer': 0.5167
}

# Compute inverse MSE
inv_mse = {k: 1 / (v + 1e-8) for k, v in mse_per_model.items()}

# Normalize
total_inv = sum(inv_mse.values())
weights_mse = {k: v / total_inv for k, v in inv_mse.items()}

print("🔧 Weights based on inverse MSE:")
for k, w in weights_mse.items():
    print(f"{k}: {w:.4f}")

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 🔁 Ensure all model outputs are tensors
for k in model_outputs:
    if isinstance(model_outputs[k], list):
        model_outputs[k] = torch.tensor(model_outputs[k])

# 📌 Step 1: Define MSE-based weights
mse_per_model = {
    'CNN': 0.4259,
    'biLSTM': 0.4548,
    'Bi-CNN': 0.4761,
    'GRU': 0.4127,
    'LSTM': 0.4433,
    'Transformer': 0.5167
}

# 📌 Step 2: Compute weights (inverse MSE)
epsilon = 1e-8
inv_mse = {k: 1 / (v + epsilon) for k, v in mse_per_model.items()}
total_inv = sum(inv_mse.values())
weights = {k: v / total_inv for k, v in inv_mse.items()}

print("🎯 Computed Weights:")
for k, w in weights.items():
    print(f"{k}: {w:.4f}")

# 📌 Step 3: Weighted ensemble prediction
ensemble_preds = None
for k in weights:
    pred_tensor = model_outputs[k]
    weight = weights[k]

    if ensemble_preds is None:
        ensemble_preds = pred_tensor * weight
    else:
        ensemble_preds += pred_tensor * weight

# Convert to numpy
ensemble_preds = ensemble_preds.numpy()

# 📌 Step 4: Evaluate
mse = mean_squared_error(all_targets, ensemble_preds)
mae = mean_absolute_error(all_targets, ensemble_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_targets, ensemble_preds)

print("\n📊 Weighted Ensemble Regression Performance:")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"R²    : {r2:.4f}")

# Convert predictions and targets to numpy
all_targets = np.array(all_targets)  # [N]

# Ensemble prediction via average (soft ensemble for regression)
all_preds_ensemble = torch.stack([model_outputs[name] for name in model_outputs], dim=0).mean(dim=0).numpy()  # [N]

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# Compute metrics
mse = mean_squared_error(all_targets, all_preds_ensemble)
mae = mean_absolute_error(all_targets, all_preds_ensemble)
rmse = np.sqrt(mse)
r2 = r2_score(all_targets, all_preds_ensemble)

print("\n📊 Ensemble Regression Performance:")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"R²    : {r2:.4f}")
# ✅
len(all_targets)
import numpy as np
from scipy.stats import pearsonr

# Convert to numpy
y_true = np.array(all_targets)
y_pred = np.array(all_preds_ensemble)

# Keep only positive MIC values
mask = (y_true > 0) & (y_pred > 0)
log_y_true = np.log10(y_true[mask])
log_y_pred = np.log10(y_pred[mask])

R, _ = pearsonr(log_y_true, log_y_pred)
print("Pearson R:", R)

len(mask)
model_outputs
import numpy as np
from scipy.stats import pearsonr

# Convert PyTorch tensors in model_outputs dict to numpy arrays
predictions = {k: v.detach().cpu().numpy() for k, v in model_outputs.items()}

# all_targets is already a numpy array (make sure it's log10 transformed if needed)
y_true = all_targets  

# Compute Pearson R for each model
r_values = {}
for model, y_pred in predictions.items():
    r, _ = pearsonr(y_true, y_pred)
    r_values[model] = r

print("📊 Pearson R values:")
for model, r in r_values.items():
    print(f"{model}: {r:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Your computed R values
# ==============================
my_r_values = {
    "CNN": 0.9034,
    "biLSTM": 0.8878,
    "Bi-CNN": 0.8970,
    "GRU": 0.9035,
    "LSTM": 0.8975,
    "Transformer": 0.8794,
    "Ensemble": 0.834,
}

# Original paper values
original_r_values = {
    "CNN": 0.39,
    "LSTM": 0.40,
    "Transformer": 0.37,
    "Ensemble": 0.42,
    "Attention": 0.33,
}

# ==============================
# Align models
# ==============================
all_models = sorted(set(my_r_values.keys()) | set(original_r_values.keys()))

my_vals = [my_r_values.get(m, np.nan) for m in all_models]
orig_vals = [original_r_values.get(m, np.nan) for m in all_models]

x = np.arange(len(all_models))  # positions
width = 0.35  # width of bars

# ==============================
# Plot
# ==============================
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, my_vals, width, label="Proposed Method", color="steelblue")
bars2 = ax.bar(x + width/2, orig_vals, width, label="Original Paper", color="darkorange")

ax.set_xlabel("Models")
ax.set_ylabel("Pearson R")
ax.set_title("Comparison of Pearson R: Proposed Method vs Original Paper")
ax.set_xticks(x)
ax.set_xticklabels(all_models, rotation=30)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add numeric labels above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("pearson_r_comparison.pdf")
plt.savefig("pearson_r_comparison.png", dpi=600)
plt.show()
 
print("\n📊 Individual Model Performances:")
for name in model_outputs:
    preds = model_outputs[name].numpy()
    mse = mean_squared_error(all_targets, preds)
    mae = mean_absolute_error(all_targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, preds)

    print(f"\n🔹 {name}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"R²    : {r2:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["CNN", "biLSTM", "Bi-CNN", "GRU", "LSTM", "Transformer"]

# Metrics
mse = [0.4138, 0.4813, 0.4473, 0.4127, 0.4433, 0.5167]
r2 = [0.8144, 0.7841, 0.7993, 0.8149, 0.8011, 0.7682]

x = np.arange(len(models))
width = 0.6

# Plot MSE
plt.figure(figsize=(10, 5))
plt.bar(x, mse, width, color="skyblue", edgecolor="black")
plt.xticks(x, models, rotation=30)
plt.ylabel("MSE")
plt.title("Mean Squared Error (MSE) of Models(Proposed)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("mse_comparison.pdf")
plt.savefig("mse_comparison.png", dpi=600)
plt.show()

# Plot R²
plt.figure(figsize=(10, 5))
plt.bar(x, r2, width, color="lightgreen", edgecolor="black")
plt.xticks(x, models, rotation=30)
plt.ylabel("R² Score")
plt.title("R² Values of Models(Proposed)")
plt.ylim(0, 1)  # since R² is between 0 and 1
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("r2_comparison.pdf")
plt.savefig("r2_comparison.png", dpi=600)
plt.show()

# ✅ You’ve already:
# - Loaded all models
# - Got predictions from each model into `model_outputs`
# - Calculated both weighted ensemble (inverse-MSE based) and unweighted average ensemble

# 🔁 Now we continue by selecting the TOP-K predictions with the lowest predicted MIC
# And optionally store results for further use
import pandas as pd

# 📌 Step 5: Rank top predicted peptides (lowest MIC = better)
k_top = 50  # you can change this

# Sort by predicted MIC from ensemble
top_indices = np.argsort(ensemble_preds)[:k_top]

# Build result DataFrame
results = []
for i in top_indices:
    entry = {
        'index': i,
        'true_MIC': all_targets[i],
        'pred_MIC': ensemble_preds[i],
    }
    for name in model_outputs:
        entry[f'{name}_pred'] = model_outputs[name][i].item()
    results.append(entry)

results_df = pd.DataFrame(results)

print("\n📌 Top {} peptides with lowest predicted MIC:".format(k_top))
print(results_df.head(10))

# Optionally save to CSV
# results_df.to_csv("top_k_predicted_peptides.csv", index=False)

# 📌 Step 6 (Optional): Compare ensemble types
# Weighted ensemble vs unweighted

print("\n📊 Comparison:")
print("Unweighted Ensemble:")
print(f"MSE: {mean_squared_error(all_targets, all_preds_ensemble):.4f}")
print(f"R2 : {r2_score(all_targets, all_preds_ensemble):.4f}")

print("\nWeighted Ensemble:")
print(f"MSE: {mean_squared_error(all_targets, ensemble_preds):.4f}")
print(f"R2 : {r2_score(all_targets, ensemble_preds):.4f}")

import warnings
warnings.filterwarnings("ignore")

# ✅ You’ve already:
# - Loaded all models
# - Got predictions from each model into `model_outputs`
# - Calculated both weighted ensemble (inverse-MSE based) and unweighted average ensemble

import pandas as pd
import torch.nn.functional as F


# 🔁 Step 5: Rank top predicted peptides (lowest MIC = better)
k_top = 50  # you can change this

# Sort by predicted MIC from ensemble
top_indices = np.argsort(ensemble_preds)[:k_top]

# Build result DataFrame
results = []
for i in top_indices:
    entry = {
        'index': i,
        'true_MIC': all_targets[i],
        'pred_MIC': ensemble_preds[i],
    }
    for name in model_outputs:
        entry[f'{name}_pred'] = model_outputs[name][i].item()
    results.append(entry)

results_df = pd.DataFrame(results)

print("\n📌 Top {} peptides with lowest predicted MIC:".format(k_top))
print(results_df.head(10))

# Optionally save to CSV
# results_df.to_csv("top_k_predicted_peptides.csv", index=False)

# 📌 Step 6 (Optional): Compare ensemble types
# Weighted ensemble vs unweighted

print("\n📊 Comparison:")
print("Unweighted Ensemble:")
print(f"MSE: {mean_squared_error(all_targets, all_preds_ensemble):.4f}")
print(f"R2 : {r2_score(all_targets, all_preds_ensemble):.4f}")

print("\nWeighted Ensemble:")
print(f"MSE: {mean_squared_error(all_targets, ensemble_preds):.4f}")
print(f"R2 : {r2_score(all_targets, ensemble_preds):.4f}")

# ✅ Step 7: EvoGradient Optimization on GRU (lowest MSE model)
def evo_gradient_esm(embeddings, model, aa_embed_matrix, max_iters=100, max_changes=0.5):
    device = next(model.parameters()).device
    embeddings = embeddings.clone().detach().unsqueeze(0).to(device).requires_grad_(True)  # [1, L, 1280]

    was_training = model.training
    model.train()  # CuDNN RNN backward requires training mode

    L = embeddings.shape[1]
    changed_positions = set()
    max_change_count = int(max_changes * L)

    for step in range(max_iters):
        model.zero_grad()
        mic_score = model(embeddings).squeeze()
        mic_score.backward()

        if embeddings.grad is None:
            raise RuntimeError("Gradient computation failed. Ensure the model and input require grad.")

        grad = embeddings.grad.detach().clone()[0]  # [L, 1280]
        importance = -grad.norm(dim=1)
        candidates = [i for i in range(L) if i not in changed_positions]
        if not candidates:
            break

        idx = max(candidates, key=lambda i: importance[i].item())
        changed_positions.add(idx)

        token_grad = grad[idx]  # [1280]
        best_score, best_token = float('inf'), None
        for aa_emb in aa_embed_matrix.to(device):
            delta = aa_emb - embeddings[0, idx]
            score = (delta * token_grad).sum().item()
            if score < best_score:
                best_score = score
                best_token = aa_emb

        embeddings.data[0, idx] = best_token
        embeddings.grad.zero_()

        if len(changed_positions) >= max_change_count:
            break

    if not was_training:
        model.eval()

    return embeddings[0].detach()

# ✅ Step 8: Run EvoGradient on top 5 sequences
print("\n🔄 Running EvoGradient optimization on top 5 test peptides...")

model_gru = models['GRU']  # Lowest MSE model
model_gru.eval()

# Load ESM-650M embeddings via HuggingFace
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_esm = AutoModel.from_pretrained(model_name).eval().to(device)

# Build AA embedding matrix (20 AAs)
aa_tokens = "ACDEFGHIKLMNPQRSTVWY"
aa_embeddings = {}

with torch.no_grad():
    for aa in aa_tokens:
        tokens = tokenizer(aa, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        outputs = model_esm(input_ids)
        rep = outputs.last_hidden_state[0, 1]  # token at position 1
        aa_embeddings[aa] = rep.cpu()

aa_embed_matrix = torch.stack([aa_embeddings[aa] for aa in aa_tokens])  # [20, 1280]

optimized_preds = []
original_preds = []
true_values = []

for idx in top_indices[:10]:
    emb = test_embeddings[idx]  # [L, 1280]
    true_val = test_values[idx]

    evolved = evo_gradient_esm(emb, model_gru, aa_embed_matrix)
    with torch.no_grad():
        pred_orig = model_gru(emb.unsqueeze(0).to(device)).item()
        pred_evolved = model_gru(evolved.unsqueeze(0)).item()

    print(f"Sequence {idx} | MIC: {true_val:.2f} | Original Pred: {pred_orig:.2f} → Optimized: {pred_evolved:.2f}")
    optimized_preds.append(pred_evolved)
    original_preds.append(pred_orig)
    true_values.append(true_val)