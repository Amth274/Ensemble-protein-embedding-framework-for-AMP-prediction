import os 
import numpy as np 
import sklearn 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from tqdm import tqdm
from torch.utils.data import Dataset,dataloader,DataLoader
## import all model
model_path = {
    'CNN':'cnn.pt',
    'biLSTM':'bilstm.pt',
    'Bi-CNN':'bi-cnn.pt',
    'GRU':'gru_best_model.pt',
    'Logistic':'logreg_best.pt',
    'LSTM':'lstm.pt',
    'Transformer':'best_transformer.pt'
}
# train_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/cls/train_emb.pt'
test_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/cls/test_esm.pt'
# train_dataset = torch.load(train_dataset)
test_dataset = torch.load(test_dataset)
test_dataset
# train_dataset[0]['label']
# train_embeddings = [d['embeddings'] for d in train_dataset]  # list of [seq_len, 1280] tensors
# train_labels = [d['label'] for d in train_dataset]

test_embeddings = [d['embeddings'] for d in test_dataset]
test_labels = [d['label'] for d in test_dataset]

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
        emb = self.pad_embedding(entry['embeddings'])   # [max_len, 1280]
        label = entry['label']
        return emb, label

test_dataset = EnsembeDataset(test_dataset)
test_loader = DataLoader(test_dataset,shuffle=True,batch_size=128)
for emb,label in test_loader:
    print(label.shape)
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

model_outputs = {
    'logits': {},        # model_name: [tensor(logits)]
    'probs': {},         # model_name: [tensor(probabilities)]
    'preds': {},         # model_name: [tensor(predictions)]
}
all_labels = []          # Ground truth labels
threshold = 0.7  # or set your preferred threshold for binarization

for name in models:
    model_outputs['logits'][name] = []
    model_outputs['probs'][name] = []
    model_outputs['preds'][name] = []

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        all_labels.extend(y.cpu().numpy())

        for name, model in models.items():
            model.eval()

            if name.lower() == 'logistic':
                # Logistic model takes mean pooled embeddings
                x_input = x.mean(dim=1)  # [B, 1280]
            else:
                x_input = x  # [B, 100, 1280]

            logits = model(x_input)             # [B]
            probs = torch.sigmoid(logits)       # [B]
            preds = (probs >= threshold).int()  # [B]

            # Save
            model_outputs['logits'][name].append(logits.cpu())
            model_outputs['probs'][name].append(probs.cpu())
            model_outputs['preds'][name].append(preds.cpu())

# Stack all batches together
for key in ['logits', 'probs', 'preds']:
    for name in models:
        model_outputs[key][name] = torch.cat(model_outputs[key][name], dim=0)  # [N]

all_labels = torch.tensor(all_labels)  # [N]
probs_stack = torch.stack([model_outputs['probs'][m] for m in models], dim=1)  # [N, 6]
ensemble_probs = probs_stack.mean(dim=1)  # average probs across models

# Get final preds at threshold=0.5
ensemble_preds_soft = (ensemble_probs >= 0.78).int()

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

y_true = all_labels.numpy()
y_score = ensemble_probs.numpy()
y_pred  = ensemble_preds_soft.numpy()

print("ROC-AUC:", roc_auc_score(y_true, y_score))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print(accuracy_score(y_true,y_pred))
# ----- Majority Voting (Hard) -----
stacked_preds = torch.stack([model_outputs['preds'][m] for m in models], dim=0)  # [7, N]
y_pred = (stacked_preds.sum(dim=0) >= (len(models) // 2 + 1)).int()              # [N]


# ----- Soft Voting (Average Probabilities) -----
stacked_probs = torch.stack([model_outputs['probs'][m] for m in models], dim=0)  # [7, N]
all_logits = stacked_probs.mean(dim=0)                                           # [N] averaged probs

all_logits[0]
for i in range(len(all_logits)):
    if all_logits[i]>0.7:
        all_logits[i] = 1
  
all_logits=all_logits.int()
# ----- Final Evaluation -----
print("📊 Ensemble Evaluation")
print(f"Accuracy  : {accuracy_score(all_labels, y_pred):.4f}")
print(f"Precision : {precision_score(all_labels, y_pred):.4f}")
print(f"Recall    : {recall_score(all_labels, y_pred):.4f}")
print(f"F1 Score  : {f1_score(all_labels, y_pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(all_labels, all_logits):.4f}")
# ----- Final Evaluation ----- soft voting 
print("📊 Ensemble Evaluation")
print(f"Accuracy  : {accuracy_score(all_labels, all_logits):.4f}")
print(f"Precision : {precision_score(all_labels, all_logits):.4f}")
print(f"Recall    : {recall_score(all_labels, all_logits):.4f}")
print(f"F1 Score  : {f1_score(all_labels, all_logits):.4f}")
print(f"ROC AUC   : {roc_auc_score(all_labels, all_logits):.4f}")
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, y_pred)
acc=roc_auc_score(all_labels,y_pred)

# Extract TP, FP, TN, FN
TN, FP, FN, TP = cm.ravel()

print("📊 Confusion Matrix:")
print(f"TP (True Positives) : {TP}")
print(f"TN (True Negatives) : {TN}")
print(f"FP (False Positives): {FP}")
print(f"FN (False Negatives): {FN}")
print(acc)

from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_logits)

# Extract TP, FP, TN, FN
TN, FP, FN, TP = cm.ravel()

print("📊 Confusion Matrix:")
print(f"TP (True Positives) : {TP}")
print(f"TN (True Negatives) : {TN}")
print(f"FP (False Positives): {FP}")
print(f"FN (False Negatives): {FN}")

from sklearn.metrics import confusion_matrix

print("\n📊 Individual Model Evaluations")
for name in models:
    print(f"\n🔹 Model: {name}")
    
    preds = model_outputs['preds'][name]       # binary predictions [N]
    probs = model_outputs['probs'][name]       # probabilities [N]
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    roc = roc_auc_score(all_labels, probs)
    
    print(f"TP (True Positives) : {tp}")
    print(f"TN (True Negatives) : {tn}")
    print(f"FP (False Positives): {fp}")
    print(f"FN (False Negatives): {fn}")
    
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

# Suppose you have:
# model_outputs['probs'][model_name] = [N] torch tensor of probabilities
# all_labels = [N] ground truth labels (already saved)

import numpy as np

# Convert model outputs to shape: [N, num_models]
X_meta = torch.stack([
    model_outputs['probs']['CNN'],
    model_outputs['probs']['biLSTM'],
    model_outputs['probs']['Bi-CNN'],
    model_outputs['probs']['GRU'],
    model_outputs['probs']['LSTM'],
    model_outputs['probs']['Transformer'],
], dim=1).numpy()  # [N, 6]

y_meta = all_labels.numpy()  # [N]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optionally split for tuning
X_train, X_val, y_train, y_val = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)

meta_clf = LogisticRegression()
meta_clf.fit(X_train, y_train)

# Evaluate
y_val_probs = meta_clf.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_probs >= 0.5).astype(int)

print(classification_report(y_val, y_val_pred))

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-AMP", "AMP"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Validation Set)")
plt.show()

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Original paper results (ROC AUC)
# ==============================
original_results = {
    "CNN": 0.9806,
    "Attention": 0.9731,
    "LSTM": 0.9814,
    "Transformer": 0.9765,
}

# Your results (ROC AUC)
my_results = {
    "CNN": 0.9889,
    "LSTM": 0.9932,
    "Transformer": 0.9905,
    # I assume Attention ≈ biLSTM or GRU in your case (need mapping)
    "Attention": 0.9869,  
}

# ==============================
# Match models in both dicts
# ==============================
common_models = list(set(my_results.keys()) & set(original_results.keys()))
x = np.arange(len(common_models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Bars
bars1 = ax.bar(x - width/2, [my_results[m] for m in common_models],
               width, label="Proposed Method", color="#1f77b4")
bars2 = ax.bar(x + width/2, [original_results[m] for m in common_models],
               width, label="Original Paper", color="#ff7f0e")

# Labels
ax.set_xlabel("Models")
ax.set_ylabel("ROC AUC")
ax.set_title("Comparison of ROC AUC: Proposed Method vs Original Paper")
ax.set_xticks(x)
ax.set_xticklabels(common_models, rotation=30)
ax.legend()
ax.set_ylim(0, 1.05)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save
plt.tight_layout()
plt.savefig("roc_auc_bar_comparison.pdf")
plt.savefig("roc_auc_bar_comparison.png", dpi=600)
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
ap_score = average_precision_score(y_val, y_val_probs)

plt.figure()
plt.plot(recall, precision, color='purple', lw=2, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()

X_test_meta = torch.stack([
    model_outputs['probs']['CNN'],
    model_outputs['probs']['biLSTM'],
    model_outputs['probs']['Bi-CNN'],
    model_outputs['probs']['GRU'],
    model_outputs['probs']['LSTM'],
    model_outputs['probs']['Transformer'],
], dim=1).numpy()

y_test_probs = meta_clf.predict_proba(X_test_meta)[:, 1]
y_test_pred = (y_test_probs >= 0.8).astype(int)

# Final performance
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

print("F1:", f1_score(all_labels, y_test_pred))
print("ROC AUC:", roc_auc_score(all_labels, y_test_probs))
print("Confusion Matrix:\n", confusion_matrix(all_labels, y_test_pred))
