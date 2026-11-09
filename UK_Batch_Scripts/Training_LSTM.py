import pickle
import os
import itertools
import joblib
from tqdm import tqdm
import time
import random

from datetime import datetime
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, DataLoader

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))

from ML_functions import TensorCatchmentDataset

import ML_Models
from ML_Models import LSTM_With_Embedding
import ML_Training

res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/res_tensor_data.pt")

# Create dataloaders
BATCH_SIZE = 128
NUM_WORKERS = 1


res_dataset = TensorCatchmentDataset(res_tensor_data)
# non_res_dataset = TensorCatchmentDataset(non_res_tensor_data)

reservoir_dataloader = DataLoader(
    res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

# non_reservoir_dataloader = DataLoader(non_reservoir_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# all_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) # Need to do full_dataset


print("\nTesting dataloader...")
start_time = time.time()

for i, batch in enumerate(reservoir_dataloader):
    if i == 2:
        print(f"Static shape: {batch['static'].shape}")
        print(f"Temporal shape: {batch['temporal'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"\nProcessing 100 batches...")

    if i >= 99:  # 0-99 is 100 batches
        break
# 0.07 seconds per batch without making into tensors first
# 0.0065 seconds per batch with making into tensors first


end_time = time.time()
elapsed_time = end_time - start_time


# Training parameters
num_epochs = 2
static_input_size = batch["static"].shape[1]
temporal_input_size = batch["temporal"].shape[-1]
learning_rate = 1e-3
embedding_hidden_size = 64
warmup = 10  # Number of initial time steps to ignore in loss calculation
loss_interval = 500  # Perform optimizer step after this many samples
embedding_output_size = 20
lstm_hidden_size = 128
lstm_num_layers = 1
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Embedded_LSTM = ML_Models.LSTM_With_Embedding(
    static_input_size=static_input_size,
    embedding_hidden_size=embedding_hidden_size,
    embedding_output_size=embedding_output_size,
    lstm_input_size=temporal_input_size + embedding_output_size,
    lstm_hidden_size=lstm_hidden_size,
    lstm_num_layers=lstm_num_layers,
    lstm_output_size=1,
    dropout=DROPOUT,
    bidirectional=False,
    Sequence_Target=True,
)

Embedded_LSTM.to(DEVICE)
# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(Embedded_LSTM.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


# --- 5. Training Loop ---
print("\n--- Starting Training ---")

ML_Training.train_model(
    Embedded_LSTM,
    reservoir_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    warmup,
    loss_interval,
    DEVICE,
    train_mode=True,
)


# ML_Training.train_model(Embedded_LSTM, val_dataloader, criterion, optimizer, scheduler, 1,
#                 warmup, loss_interval, DEVICE, train_mode = False)


print("--- Training Finished ---")

# You can add code here to save the trained model
model_save_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Lr{learning_rate}_Embhs{embedding_hidden_size}_Lhs{lstm_hidden_size}.pt"
torch.save(Embedded_LSTM, model_save_path)
print(f"Model saved to {model_save_path}")
