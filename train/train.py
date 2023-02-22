import dill
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.data import Iterator

from network import LSTMClassifier
import copy

RANDOM_SEED = 2022
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_PATH = "/home/hash/python_test/model/practice/LSTM_english/data"

TEXT = Field(
    sequential=True,
    use_vocab=True,
    tokenize=word_tokenize,
    lower=True,
    batch_first=True,
)

LABEL = Field(
    sequential=False,
    use_vocab=False,
    batch_first=True,
)

cola_train_data, cola_valid_data, cola_test_data = \
    TabularDataset.splits(
        path=DATA_PATH+"/processed/",
        train="cola_train.tsv",
        validation="cola_valid.tsv",
        test="cola_test.tsv",
        format="tsv",
        fields=[("text",TEXT),("label",LABEL)],
        skip_header=1,
    )
TEXT.build_vocab(cola_train_data, min_freq=2)

cola_train_iterator, cola_valid_iterator, cola_test_iterator = \
    BucketIterator.splits(
        (cola_train_data, cola_valid_data, cola_test_data),
        batch_size=256,
        device=None,
        sort=False,
    )

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss=0
    for batch in train_loader:
        optimizer.zero_grad()
        text=batch.text
        if text.shape[0]>1:
            label = batch.label.type(torch.FloatTensor)
            text = text.to(device)
            label = label.to(device)
            output = model(text).flatten()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    return epoch_loss/len(train_loader)

def evaluate(model, valid_loader, criterion, device):
    model.eval()
    epoch_loss=0
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            text = batch.text
            label = batch.label.type(torch.FloatTensor)
            text = text.to(device)
            label = label.to(device)
            output = model(text).flatten()
            loss = criterion(output, label)

            epoch_loss += loss.item()
    
    return epoch_loss/len(valid_loader)


PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
N_EPOCHS = 20

lstm_classifier = LSTMClassifier(
    num_embeddings=len(TEXT.vocab),
    embedding_dim=100,
    hidden_size=200,
    num_layers=4,
    pad_idx=PAD_IDX,
)

if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"
_ = lstm_classifier.to(device)

optimizer = torch.optim.Adam(lstm_classifier.parameters())
bce_loss_fn = nn.BCELoss() # Binary Cross Entropy

for epoch in range(N_EPOCHS):
    train_loss = train(
        lstm_classifier,
        cola_train_iterator,
        optimizer,
        bce_loss_fn,
        device
    )
    valid_loss = evaluate(
        lstm_classifier,
        cola_valid_iterator,
        bce_loss_fn,
        device
    )
    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\t Val. Loss: {valid_loss}")

bf_lstm_classifier = copy.deepcopy(lstm_classifier)

with open("/home/hash/python_test/model/practice/LSTM_english/model/before_baseline_model.dill","wb") as f:
    model = {
        "TEXT": TEXT,
        "LABEL": LABEL,
        "classifier": bf_lstm_classifier
    }
    dill.dump(model,f)

sat_train_data, sat_valid_data, sat_test_data = \
    TabularDataset.splits(
        path=DATA_PATH+"/processed/",
        train="sat_train.tsv",
        validation="sat_valid.tsv",
        test="sat_test.tsv",
        format="tsv",
        fields=[("text",TEXT),("label",LABEL)],
        skip_header=1,
    )

sat_train_iterator, sat_valid_iterator, sat_test_iterator = \
    BucketIterator.splits(
        (sat_train_data, sat_valid_data, sat_test_data),
        batch_size=8,
        device=None,
        sort=False,
    )

for epoch in range(N_EPOCHS):
    train_loss = train(
        lstm_classifier,
        sat_train_iterator,
        optimizer,
        bce_loss_fn,
        device
    )
    valid_loss = evaluate(
        lstm_classifier,
        sat_valid_iterator,
        bce_loss_fn,
        device
    )
    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\t Val. Loss: {valid_loss}")


with open("/home/hash/python_test/model/practice/LSTM_english/model/after_baseline_model.dill","wb") as f:
    model = {
        "TEXT": TEXT,
        "LABEL": LABEL,
        "classifier": lstm_classifier
    }
    dill.dump(model,f)