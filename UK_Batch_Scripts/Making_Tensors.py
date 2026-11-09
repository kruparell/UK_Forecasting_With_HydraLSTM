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


CLOSEST_YEAR = True

with open(
    "/home/mokr/Reservoir_Modelling/Sample_Pairs/2003_catchment_date_pairs_splits.pkl", "rb"
) as f:
    catchment_splits = pickle.load(f)

# Access each split directly
train_reservoir_pairs = catchment_splits["reservoir_train"]
train_non_reservoir_pairs = catchment_splits["non_reservoir_train"]

val_reservoir_pairs = catchment_splits["reservoir_val"]
val_non_reservoir_pairs = catchment_splits["non_reservoir_val"]

test_reservoir_pairs = catchment_splits["reservoir_test"]
test_non_reservoir_pairs = catchment_splits["non_reservoir_test"]


with open("/home/mokr/Reservoir_Modelling/Normalised_Temporal_Data.pkl", "rb") as f:
    temporal_data = pickle.load(f)

for gauge_id in temporal_data.keys():
    temporal_data[gauge_id]["date"] = pd.to_datetime(temporal_data[gauge_id]["date"])
    temporal_data[gauge_id] = temporal_data[gauge_id].set_index("date")
    temporal_data[gauge_id] = temporal_data[gauge_id].select_dtypes(include=[np.number])

static_variables = pd.read_csv(
    "/perm/mokr/Scaled_CamelsGB_V2/normalised_static_data.csv", dtype={"gauge_id": str}
)

static_variables = static_variables.set_index("gauge_id")
static_variables = static_variables.select_dtypes(include=[np.number])

# EXCLUDE_STATIC = {

#     "gauge_lon", "gauge_northing", "hourly_flow_perc_complete",
#     "gauge_easting", "tawc_missing", 
#     "monthly_gwlevel_period_start", "monthly_gwlevel_period_end",
#     "gw_well_easting", "gw_well_datum", "gw_well_northing",
#     "daily_gwlevel_period_end", "gw_well_id", "daily_gwlevel_period_start",
#     "gw_well_name",
# }

# "tawc"
EXCLUDE_STATIC = {

    "gauge_lat", "gauge_lon", "gauge_northing", "hourly_flow_perc_complete",
    "gauge_easting",  "monthly_gwlevel_perc_complete", "daily_gwlevel_perc_complete",
    "monthly_gwlevel_period_start", "monthly_gwlevel_period_end", "reservoir_year_last", "reservoir_year_first",  
    "gw_well_easting", "gw_well_datum", "gw_well_northing", "gw_well_name", 
    "daily_gwlevel_period_end", "gw_well_id", "daily_gwlevel_period_start",
    'q5_uncert_upper', 'q25_uncert_upper', 'q50_uncert_upper', 'q75_uncert_upper', 'q95_uncert_upper', 'q99_uncert_upper',
    'q5_uncert_lower', 'q25_uncert_lower', 'q50_uncert_lower', 'q75_uncert_lower', 'q95_uncert_lower', 'q99_uncert_lower',
    'q_mean', 'Q5', 'Q95', 'stream_elas', 'slope_fdc', 'baseflow_index', 'baseflow_index_ceh', 'hfd_mean',
    'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq', 
    'runoff_ratio', 'daily_flow_perc_complete', 'daily_flow_extrap_dur',
    'max_hourly_flow', 'max_gauging_flow', 'max_daily_flow', 'hourly_flow_extrap_dur', 
    
}


# "max_gauging_flow_date", "gauge_id", "benchmark_catch", "gauge_name"

static_variables = (
    static_variables
    .drop(columns=list(EXCLUDE_STATIC))
)

print(list(static_variables.columns))


# Create datasets
goal = "discharge_spec_x"

# Data Loader


# print("Preprocessing training data...")
# train_res_tensor_data = preprocess_to_tensors(
#     train_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
# )
# train_non_res_tensor_data = preprocess_to_tensors(
#     train_non_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
# )
# torch.save(train_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_res_tensor_data.pt")
# torch.save(
#     train_non_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"
# )
# del train_res_tensor_data, train_non_res_tensor_data
# gc.collect()

# print("Preprocessing validation data...")
# val_res_tensor_data = preprocess_to_tensors(
#     val_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
# )
# val_non_res_tensor_data = preprocess_to_tensors(
#     val_non_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
# )
# torch.save(val_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_res_tensor_data.pt")
# torch.save(val_non_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt")
# del val_res_tensor_data, val_non_res_tensor_data
# gc.collect()

print("Preprocessing testing data...")
test_res_tensor_data = preprocess_to_tensors(
    test_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
)

torch.save(test_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_test_res_tensor_data.pt")


del test_res_tensor_data
gc.collect()

test_non_res_tensor_data = preprocess_to_tensors(
    test_non_reservoir_pairs, temporal_data, static_variables, goal, closest_year = CLOSEST_YEAR, window = 365
)
torch.save(
    test_non_res_tensor_data, "/perm/mokr/CamelsGB_V2/365_0209_no_lat_test_non_res_tensor_data.pt"
)

del test_non_res_tensor_data
gc.collect()

# Combine all pairs
# reservoir_pairs = catchment_pairings['paired_reservoir']
# non_reservoir_pairs = catchment_pairings['paired_non_reservoir']
# all_pairs = (catchment_pairings['paired_reservoir'] +
#              catchment_pairings['paired_non_reservoir'])

# sample = reservoir_pairs[0]
# gauge_id = str(sample['gauge_id'])
# init_date = sample['init_date']
# end_date = init_date + pd.Timedelta(days=89)


# # Preprocess all splits
# print("Preprocessing reservoir data...")
# res_tensor_data = preprocess_to_tensors(reservoir_pairs, temporal_data, static_variables, goal)
# print("Preprocessing non_reservoir data...")
# non_res_tensor_data = preprocess_to_tensors(non_reservoir_pairs, temporal_data, static_variables, goal)

# # Save to disk
# torch.save(res_tensor_data, '/perm/mokr/CamelsGB_V2/trimmed_landcover_res_tensor_data.pt')
# torch.save(non_res_tensor_data, '/perm/mokr/CamelsGB_V2/trimmed_landcover_non_res_tensor_data.pt') 

# print("Saved tensor data to disk")


# res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/res_tensor_data.pt')
# test_res_dataset = TensorCatchmentDataset(test_non_res_tensor_data)
# # Create dataloaders
# BATCH_SIZE = 128
# NUM_WORKERS = 1


# non_res_dataset = TensorCatchmentDataset(non_res_tensor_data)

# test_reservoir_dataloader = DataLoader(
#     test_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
# )  # , pin_memory=True, persistent_workers=True


# non_reservoir_dataloader = DataLoader(non_res_dataset, batch_size= BATCH_SIZE, shuffle=True,
# num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True


# reservoir_dataset = CatchmentDataset(reservoir_pairs, temporal_data, static_variables, goal=goal)
# non_reservoir_dataset = CatchmentDataset(non_reservoir_pairs, temporal_data, static_variables, goal=goal)
# full_dataset = CatchmentDataset(all_pairs, temporal_data, static_variables, goal=goal)

# reservoir_dataloader = DataLoader(reservoir_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= NUM_WORKERS)
# non_reservoir_dataloader = DataLoader(non_reservoir_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# all_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# print("\nTesting dataloader...")
# start_time = time.time()

# for i, batch in enumerate(test_reservoir_dataloader):
#     if i == 0:
#         print(f"Static shape: {batch['static'].shape}")
#         print(f"Temporal shape: {batch['temporal'].shape}")
#         print(f"Target shape: {batch['target'].shape}")
#         print(f"\nProcessing 100 batches...")

#     if i >= 99:  # 0-99 is 100 batches
#         break

# 0.07 seconds per batch without making into tensors first
# 0.0065 seconds per batch with making into tensors first


# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"\nCompleted 100 batches")
# print(f"Total time: {elapsed_time:.2f} seconds")
# print(f"Average time per batch: {elapsed_time/100:.4f} seconds")
# print(f"Batches per second: {100/elapsed_time:.2f}")
