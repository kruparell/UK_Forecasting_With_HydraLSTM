import pickle
import os
import gc
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch



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


def get_metadata(pairs, temporal_data):
    """Pre-convert all data to PyTorch tensors and save to disk"""
    
    meta_data = {
        'gauge_id': [],
        'init_date': []
    }
    skipped = 0
    print(f"Processing {len(pairs)} pairs...")
    for i, sample in enumerate(pairs):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(pairs)}")
        
        gauge_id = str(sample['gauge_id'])
        init_date = sample['init_date']
        end_date = init_date + pd.Timedelta(days=89)
        
        # Get data
        temporal_values = temporal_data[gauge_id].loc[init_date:end_date]
        
        if len(temporal_values) != 90:
            skipped += 1
            continue

        meta_data['gauge_id'].append(gauge_id)
        meta_data['init_date'].append(str(init_date))

    return meta_data

    
# Data Loader
print("Preprocessing training data...")
train_res_tensor_data = get_metadata(train_reservoir_pairs, temporal_data)
train_non_res_tensor_data = get_metadata(train_non_reservoir_pairs, temporal_data)
torch.save(train_res_tensor_data, '/perm/mokr/CamelsGB_V2/train_res_tensor_metadata.pt')
torch.save(train_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/train_non_res_tensor_metadata.pt')
del train_res_tensor_data, train_non_res_tensor_data
gc.collect()

print("Preprocessing validation data...")
val_res_tensor_data = get_metadata(val_reservoir_pairs, temporal_data)
val_non_res_tensor_data = get_metadata(val_non_reservoir_pairs, temporal_data)
torch.save(val_res_tensor_data, '/perm/mokr/CamelsGB_V2/val_res_tensor_metadata.pt')
torch.save(val_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/val_non_res_tensor_metadata.pt')
del val_res_tensor_data, val_non_res_tensor_data
gc.collect()

print("Preprocessing testing data...")
test_res_tensor_data = get_metadata(test_reservoir_pairs, temporal_data)
test_non_res_tensor_data = get_metadata(test_non_reservoir_pairs, temporal_data)
torch.save(test_res_tensor_data, '/perm/mokr/CamelsGB_V2/test_res_tensor_metadata.pt')
torch.save(test_non_res_tensor_data, '/perm/mokr/CamelsGB_V2/test_non_res_tensor_metadata.pt')
del test_non_res_tensor_data
gc.collect()
