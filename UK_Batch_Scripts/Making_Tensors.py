import pickle 
import os
import gc
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

from ML_functions import CatchmentDataset, preprocess_to_tensors, TensorCatchmentDataset
from ML_Models import LSTM_With_Embedding


# Download Data
# with open('/home/mokr/Reservoir_Modelling/Sample_Pairs/catchment_date_pairs.pkl', 'rb') as f:
#     catchment_pairings = pickle.load(f)


with open('/home/mokr/Reservoir_Modelling/Sample_Pairs/catchment_date_pairs_splits.pkl', 'rb') as f:
    catchment_splits = pickle.load(f)

# Access each split directly
train_reservoir_pairs = catchment_splits['reservoir_train']
train_non_reservoir_pairs = catchment_splits['non_reservoir_train']

val_reservoir_pairs = catchment_splits['reservoir_val']
val_non_reservoir_pairs = catchment_splits['non_reservoir_val']

test_reservoir_pairs = catchment_splits['reservoir_test']
test_non_reservoir_pairs = catchment_splits['non_reservoir_test']



with open('/home/mokr/Reservoir_Modelling/Normalised_Temporal_Data.pkl', 'rb') as f:
    temporal_data = pickle.load(f)

for gauge_id in temporal_data.keys():
    temporal_data[gauge_id]['date'] = pd.to_datetime(temporal_data[gauge_id]['date'])
    temporal_data[gauge_id] = temporal_data[gauge_id].set_index('date')
    temporal_data[gauge_id] = temporal_data[gauge_id].select_dtypes(include=[np.number])

static_variables = pd.read_csv('/perm/mokr/Scaled_CamelsGB_V2/normalised_static_data.csv', 
                                dtype={'gauge_id': str})

static_variables = static_variables.set_index('gauge_id')
static_variables = static_variables.select_dtypes(include=[np.number])


# Create datasets
goal = 'discharge_spec_x'

# Data Loader
print("Preprocessing training data...")
train_res_tensor_data = preprocess_to_tensors(train_reservoir_pairs, temporal_data, static_variables, goal)
train_non_res_tensor_data = preprocess_to_tensors(train_non_reservoir_pairs, temporal_data, static_variables, goal)
torch.save(train_res_tensor_data, '/perm/mokr/CamelsGB_V2/train_res_tensor_data.pt')
torch.save(train_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/train_non_res_tensor_data.pt')
del train_res_tensor_data, train_non_res_tensor_data
gc.collect()

print("Preprocessing validation data...")
val_res_tensor_data = preprocess_to_tensors(val_reservoir_pairs, temporal_data, static_variables, goal)
val_non_res_tensor_data = preprocess_to_tensors(val_non_reservoir_pairs, temporal_data, static_variables, goal)
torch.save(val_res_tensor_data, '/perm/mokr/CamelsGB_V2/val_res_tensor_data.pt')
torch.save(val_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/val_non_res_tensor_data.pt')
del val_res_tensor_data, val_non_res_tensor_data
gc.collect()

print("Preprocessing testing data...")
test_res_tensor_data = preprocess_to_tensors(test_reservoir_pairs, temporal_data, static_variables, goal)
test_non_res_tensor_data = preprocess_to_tensors(test_non_reservoir_pairs, temporal_data, static_variables, goal)
torch.save(test_res_tensor_data, '/perm/mokr/CamelsGB_V2/test_res_tensor_data.pt')
torch.save(test_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/test_non_res_tensor_data.pt')
del test_non_res_tensor_data
gc.collect()

print("\nAll tensor data saved!")
# print(f"Train: {len(train_res_tensor_data)} reservoir, {len(train_non_res_tensor_data)} non-reservoir")
# print(f"Val: {len(val_res_tensor_data)} reservoir, {len(val_non_res_tensor_data)} non-reservoir")
# print(f"Test: {len(test_res_tensor_data)} reservoir, {len(test_non_res_tensor_data)} non-reservoir")

# Combine all pairs
# reservoir_pairs = catchment_pairings['paired_reservoir']
# non_reservoir_pairs = catchment_pairings['paired_non_reservoir']
# all_pairs = (catchment_pairings['paired_reservoir'] + 
#              catchment_pairings['paired_non_reservoir']) 

# sample = reservoir_pairs[0]
# gauge_id = str(sample['gauge_id'])
# init_date = sample['init_date']
# end_date = init_date + pd.Timedelta(days=89)


# Preprocess all splits
# print("Preprocessing reservoir data...")
# res_tensor_data = preprocess_to_tensors(reservoir_pairs, temporal_data, static_variables, goal)
# print("Preprocessing non_reservoir data...")
# non_res_tensor_data = preprocess_to_tensors(non_reservoir_pairs, temporal_data, static_variables, goal)

# # Save to disk
# torch.save(res_tensor_data, '/perm/mokr/CamelsGB_V2/res_tensor_data.pt')
# torch.save(non_res_tensor_data, '/perm/mokr/CamelsGB_V2/non_res_tensor_data.pt') # Not Done Yet

# print("Saved tensor data to disk")


# res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/res_tensor_data.pt')
test_res_dataset = TensorCatchmentDataset(test_non_res_tensor_data)
# Create dataloaders
BATCH_SIZE = 128
NUM_WORKERS = 1



# non_res_dataset = TensorCatchmentDataset(non_res_tensor_data)

test_reservoir_dataloader = DataLoader(test_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True


# non_reservoir_dataloader = DataLoader(non_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              # num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True


# reservoir_dataset = CatchmentDataset(reservoir_pairs, temporal_data, static_variables, goal=goal)
# non_reservoir_dataset = CatchmentDataset(non_reservoir_pairs, temporal_data, static_variables, goal=goal)
# full_dataset = CatchmentDataset(all_pairs, temporal_data, static_variables, goal=goal)

# reservoir_dataloader = DataLoader(reservoir_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= NUM_WORKERS)
# non_reservoir_dataloader = DataLoader(non_reservoir_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# all_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


print("\nTesting dataloader...")
start_time = time.time()

for i, batch in enumerate(test_reservoir_dataloader):
    if i == 0:
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

print(f"\nCompleted 100 batches")
print(f"Total time: {elapsed_time:.2f} seconds")
print(f"Average time per batch: {elapsed_time/100:.4f} seconds")
print(f"Batches per second: {100/elapsed_time:.2f}")

