import os 
import numpy as np 
import sklearn 
import torch 
import pandas as pd 
from transformers import AutoModel,AutoTokenizer
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
torch.cuda.is_available()
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModel.from_pretrained(model_name)
model.eval()
X_train = pd.read_csv('/home/aum-thaker/Desktop/VSC/AMP/Data/train.csv')
X_test = pd.read_csv('/home/aum-thaker/Desktop/VSC/AMP/Data/test.csv')
X_train.head()
X_test.head()
X_train['ID'] = ['pep__train{:05d}'.format(i) for i in range(1,len(X_train)+1)]
X_test['ID'] = ['pep_test{:05d}'.format(i) for i in range(1,len(X_test)+1)]
X_train = X_train[['ID','Sequence','Length','label']]
X_test = X_test[['ID','Sequence','Length','label']]
X_train.head()
class peptide_dataset(Dataset):
    def __init__(self,df):
        self.seq=df['Sequence'].tolist()
        self.label=df['label'].tolist()
        self.id=df['ID'].tolist()

    # def __getattribute__(self, name):
    def __len__(self):
        return len(self.seq)


    def __getitem__(self,idx):
        seq = self.seq[idx]
        id=self.id[idx]
        label=torch.tensor(self.label[idx],dtype=torch.float)

        return id,seq,label
dataset = peptide_dataset(X_train)
loader = DataLoader(dataset,batch_size=16,shuffle=True)
dataset_test = peptide_dataset(X_test)
test_loader = DataLoader(dataset_test,batch_size=256,shuffle=True)
for batch in test_loader:
    print(batch[0])
    print(len(test_loader))
    break
model.to('cuda')
all_embeddings = []

# model = model.to('cuda')  # Make sure model is on GPU

for ids, seqs, labels in tqdm(test_loader):
    for id_, seq, label in zip(ids, seqs, labels):
        # Tokenize and move to GPU
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            token_embs = outputs.last_hidden_state  # shape: [1, seq_len+2, 1280]

        # Remove special tokens
        aa_embeds = token_embs[0, 1:len(seq)+1, :].cpu()  # Move to CPU for storage

        all_embeddings.append({
            'ID': id_,
            'embeddings': aa_embeds,     # shape: [seq_len, 1280], on CPU
            'label': label.float()       # still on CPU, safe
        })

all_embeddings = []

# model = model.to('cuda')  # Make sure model is on GPU

for ids, seqs, labels in tqdm(loader):
    for id_, seq, label in zip(ids, seqs, labels):
        # Tokenize and move to GPU
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            token_embs = outputs.last_hidden_state  # shape: [1, seq_len+2, 1280]

        # Remove special tokens
        aa_embeds = token_embs[0, 1:len(seq)+1, :].cpu()  # Move to CPU for storage

        all_embeddings.append({
            'ID': id_,
            'embeddings': aa_embeds,     # shape: [seq_len, 1280], on CPU
            'label': label.float()       # still on CPU, safe
        })

all_embeddings[0]
torch.save(all_embeddings, "test_esm.pt")

## AA embeddings for Regression task
train_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/reg/train (1).csv'
test_dataset = '/home/aum-thaker/Desktop/VSC/AMP/Data/reg/test (1).csv'
X_train = pd.read_csv(train_dataset)
X_test = pd.read_csv(test_dataset)
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModel.from_pretrained(model_name)
model.eval()
X_train.head()
X_train['ID'] = ['pep__train{:05d}'.format(i) for i in range(1,len(X_train)+1)]
X_test['ID'] = ['pep_test{:05d}'.format(i) for i in range(1,len(X_test)+1)]
X_train = X_train[['ID','Sequence','value','label']]
X_test = X_test[['ID','Sequence','value','label']]
X_train.head()
X_test.head()
class peptide_dataset(Dataset):
    def __init__(self,df):
        self.seq=df['Sequence'].tolist()
        self.label=df['label'].tolist()
        self.id=df['ID'].tolist()
        self.value=df['value'].tolist()

    # def __getattribute__(self, name):
    def __len__(self):
        return len(self.seq)


    def __getitem__(self,idx):
        seq = self.seq[idx]
        id=self.id[idx]
        label=torch.tensor(self.label[idx],dtype=torch.float)
        value=torch.tensor(self.value[idx],dtype=torch.float)

        return id,seq,value,label
dataset_train = peptide_dataset(X_train)
train_loader = DataLoader(dataset_train,batch_size=16,shuffle=True)
dataset_test = peptide_dataset(X_test)
test_loader = DataLoader(dataset_test,batch_size=16,shuffle=True)
for batch in train_loader:
    print(batch)
    break
model.to('cuda')
for batch in test_loader:
    print(batch)
    break
all_embeddings = []

for ids, seqs, values, labels in tqdm(test_loader):
    inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        token_embs = outputs.last_hidden_state  # [B, seq_len+2, 1280]

    for i in range(len(seqs)):
        seq = seqs[i]
        length = len(seq)
        # remove special tokens
        aa_embeds = token_embs[i, 1:length+1].cpu()  # [L, 1280]

        all_embeddings.append({
            "ID": ids[i],
            "embeddings": aa_embeds,
            "label": labels[i].float(),
            "value": values[i].float()
        })

torch.save(all_embeddings, "test_esm_regression.pt")
# torch.save(all_embeddings, "test_esm_regression.pt")

all_embeddings = []

for ids, seqs, values, labels in tqdm(train_loader):
    inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        token_embs = outputs.last_hidden_state  # [B, seq_len+2, 1280]

    for i in range(len(seqs)):
        seq = seqs[i]
        length = len(seq)
        # remove special tokens
        aa_embeds = token_embs[i, 1:length+1].cpu()  # [L, 1280]

        all_embeddings.append({
            "ID": ids[i],
            "embeddings": aa_embeds,
            "label": labels[i].float(),
            "value": values[i].float()
        })

torch.save(all_embeddings, "train_esm_regression.pt")
# torch.save(all_embeddings, "train_esm_regression.pt")