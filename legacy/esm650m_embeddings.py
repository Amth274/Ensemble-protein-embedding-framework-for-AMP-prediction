import os 
import numpy as np 
import sklearn 
import torch 
import pandas as pd
from transformers import AutoModel,AutoTokenizer
import peft
from tqdm import tqdm 
model_id = r'facebook/esm1b_t33_650M_UR50S'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id,)
train_path = r'/home/aum-thaker/Desktop/VSC/AMP/Data/train.csv'
test_path  = r'/home/aum-thaker/Desktop/VSC/AMP/Data/test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_seq = train_df['Sequence'][0]
torch.cuda.is_available()
model.eval().to('cuda')
esm_650m = []
for seq in tqdm(train_df['Sequence']):
    input = tokenizer(seq,return_tensors='pt',add_special_tokens=True).to('cuda')
    with torch.no_grad():
        output = model(**input)
        attention_mask = input['attention_mask']
        token_embeddings = output.last_hidden_state  # shape: [1, seq_len, hidden_dim]

        # Compute masked mean
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embedding = (summed / counts).squeeze(0).cpu()  # shape: [hidden_dim]

        esm_650m.append(mean_embedding)

# train_df['esm_650m'] = esm_650m

embeddings = torch.stack(esm_650m).to(dtype=torch.float16)
labels = torch.tensor(train_df['label'].values)
sequences = train_df['Sequence'].tolist()
torch.save({
    'embeddings': embeddings,      # shape: [45000, 1280], float16
    'labels': labels,              # shape: [45000]
    'sequences': sequences         # list of strings
}, 'train.pt')
train_df['esm_650m'][123]==esmp[123]
esm_650m = []
for seq in tqdm(test_df['Sequence']):
    input = tokenizer(seq,return_tensors='pt',add_special_tokens=True).to('cuda')
    with torch.no_grad():
        output = model(**input)
        attention_mask = input['attention_mask']
        token_embeddings = output.last_hidden_state  # shape: [1, seq_len, hidden_dim]

        # Compute masked mean
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embedding = (summed / counts).squeeze(0).cpu()  # shape: [hidden_dim]

        esm_650m.append(mean_embedding)

test_df['esm_650m'] = esm_650m

# esm_650m[0]
embeddings_1 = torch.stack(esm_650m).to(dtype=torch.float16)
labels_1 = torch.tensor(test_df['label'].values)
sequences_1 = test_df['Sequence'].tolist()
torch.save({
    'embeddings': embeddings_1,      # shape: [45000, 1280], float16
    'labels': labels_1,              # shape: [45000]
    'sequences': sequences_1         # list of strings
}, 'test.pt')
import pandas as pd
import numpy as np

# # Convert each tensor to float32 NumPy array
# embeddings_np = np.stack([emb.cpu().numpy().astype(np.float32) for emb in esm_650m])

# # Create a new DataFrame with 1280 columns
# embedding_df = pd.DataFrame(embeddings_np, columns=[f'esm_{i}' for i in range(embeddings_np.shape[1])])

# # Concatenate with original DataFrame
# train_df = pd.concat([train_df, embedding_df], axis=1)

import torch
!mkdir otherModels5_2Output
!touch otherModels5_2Output/__init__.py

# otherModels5_2Output/__init__.py
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define layers (even placeholders)
    def forward(self, x):
        # dummy forward; real weights will load
        return x

model = torch.load('/home/aum-thaker/Downloads/CNN.pth', weights_only=False)
