import time
import itertools
import joblib
import sys
import gc
import pickle 
from tqdm import tqdm
import os


import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

import warnings


import sys
from pathlib import Path


# At the top of your script or just before the line causing the warning
warnings.filterwarnings('ignore', category=FutureWarning, message='Series.__getitem__ treating keys as positions is deprecated')

def sample_batch(all_pairs, batch_size, random_seed=None):
    """Sample a batch of catchment-date pairs uniformly."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.random.choice(len(all_pairs), size=batch_size, replace=False)
    return [all_pairs[i] for i in indices]
    
def load_scalers(output_path = 'scalers'):
    """Load saved scalers"""
    scalers = joblib.load(f'{output_path}.joblib')
    return scalers


def preprocess_to_tensors(pairs, temporal_data, static_variables, goal='discharge_spec_x'):
    """Pre-convert all data to PyTorch tensors and save to disk"""
    goals = ['discharge_spec_x', 'discharge_spec_y', 'discharge_vol_x', 'discharge_vol_y']
    
    tensor_data = {
        'static': [],
        'temporal': [],
        'target': [],
        # 'gauge_id': [],
        # 'init_date': []
    }
    
    print(f"Processing {len(pairs)} pairs...")
    for i, sample in enumerate(pairs):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(pairs)}")
        
        gauge_id = str(sample['gauge_id'])
        init_date = sample['init_date']
        end_date = init_date + pd.Timedelta(days=89)
        
        # Get data
        static_features = static_variables.loc[gauge_id].values.astype(np.float32)
        temporal_values = temporal_data[gauge_id].loc[init_date:end_date]
        true_discharge = temporal_values[goal].values.astype(np.float32)
        temporal_input = temporal_values.drop(columns=goals).values.astype(np.float32)

        # Convert to tensors immediately
        static_tensor = torch.FloatTensor(static_features)
        temporal_tensor = torch.FloatTensor(temporal_input)
        target_tensor = torch.FloatTensor(true_discharge)
        
        # Print tensor shapes for first sample
        # if i == 2:
        #     print(f"temporal_values shape: {temporal_values.shape}")
        #     print(f"true_discharge shape: {true_discharge.shape}")
        #     print(f"static_tensor shape: {static_tensor.shape}")
        #     print(f"temporal_tensor shape: {temporal_tensor.shape}")
        #     print(f"target_tensor shape: {target_tensor.shape}")
        #     print("--- End first sample ---\n")

        skipped = 0
        if len(temporal_values) != 90:
            if skipped < 20:  # Print first few issues
                print(f"Warning: Sample {i} (gauge {gauge_id}, date {init_date}) has {len(temporal_values)} days instead of 90, skipping")
            skipped += 1
            continue
            
        tensor_data['static'].append(static_tensor)
        tensor_data['temporal'].append(temporal_tensor)
        tensor_data['target'].append(target_tensor)
        # tensor_data['gauge_id'].append(gauge_id)
        # tensor_data['init_date'].append(str(init_date))
    
    return tensor_data


class TensorCatchmentDataset(Dataset):
    def __init__(self, tensor_data):
        """
        Parameters:
        -----------
        tensor_data : dict
            Dictionary with pre-converted tensors from torch.load()
        """
        self.static = tensor_data['static']
        self.temporal = tensor_data['temporal']
        self.target = tensor_data['target']
        # self.gauge_id = tensor_data['gauge_id']
        # self.init_date = tensor_data['init_date']
        
    def __len__(self):
        return len(self.static)
    
    def __getitem__(self, idx):
        return {
            'static': self.static[idx],
            'temporal': self.temporal[idx],
            'target': self.target[idx],
            # 'gauge_id': self.gauge_id[idx],
            # 'init_date': self.init_date[idx]
        }
        
# Define the Dataset class
class CatchmentDataset(Dataset):
    def __init__(self, pairs, temporal_data, static_variables, goal='discharge_spec_x'):
        """
        Parameters:
        -----------
        pairs : list of dict
            List of catchment-date pairs from the pickle file
        temporal_data : dict
            Dictionary of temporal data keyed by gauge_id
        static_variables : DataFrame
            DataFrame with static variables, with 'Gauge_Id' column
        goal : str
            Target variable name
        """
        self.pairs = pairs
        self.temporal_data = temporal_data
        self.static_variables = static_variables
        self.goal = goal
        self.goals = ['discharge_spec_x', 'discharge_spec_y', 'discharge_vol_x', 'discharge_vol_y']
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sample = self.pairs[idx]
        gauge_id = sample['gauge_id']
        init_date = sample['init_date']
        end_date = init_date + pd.Timedelta(days=89)  # (days + 1) days total
        
        # Get static features
        static_features = self.static_variables.loc[gauge_id].values.astype(np.float32)

        
        # Get temporal data
        temporal_values = self.temporal_data[gauge_id].loc[init_date:end_date]
    
        # Get target discharge
        true_discharge = temporal_values[self.goal].values
        
        # Get temporal input (excluding goal variables)
        temporal_input = temporal_values.drop(columns=self.goals).values
        
        # Convert to tensors
        static_tensor = torch.FloatTensor(static_features)
        temporal_tensor = torch.FloatTensor(temporal_input)
        target_tensor = torch.FloatTensor(true_discharge)
        
        return {
            'static': static_tensor,
            'temporal': temporal_tensor,
            'target': target_tensor,
            # 'gauge_id': gauge_id,
            # 'init_date': init_date
        }

class NSELoss(nn.Module):
    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = eps
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds and target shape: (batch_size, seq_len)
        numerator = torch.mean( (target - preds) ** 2, dim=1)
        denominator = torch.mean((target - torch.mean(target, dim=1, keepdim=True) ) ** 2, dim=1)

        nse = 1 - numerator / (denominator + self.eps)
        return 1 - nse.mean()  # minimize 1 - NSE


class NNSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds and target shape: (batch_size, seq_len)
        numerator = torch.mean( (target - preds) ** 2, dim=1)
        denominator = torch.mean((target - torch.mean(target, dim=1, keepdim=True) ) ** 2, dim=1)

        nse = 1 - numerator / (denominator + self.eps)
        return 1 -  (1/(2 - nse.mean()) )  # minimize 1 - NSE

class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency loss for batched predictions with multiple timesteps.
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
    r = correlation coefficient
    alpha = std(sim)/std(obs)
    beta = mean(sim)/mean(obs)
    
    Handles inputs of shape [batch_size, timesteps]
    """
    def __init__(self, reduction='mean'):
        super(KGELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, simulated, observed):
        """
        Args:
            simulated (torch.Tensor): Simulated values of shape [batch_size, timesteps]
            observed (torch.Tensor): Observed values of shape [batch_size, timesteps]
        Returns:
            torch.Tensor: KGE loss
        """
        # Ensure the tensors are floating point
        sim = simulated.float()
        obs = observed.float()
        
        # Handle potential NaN values
        eps = 1e-4
        
        # Calculate means along timestep dimension
        sim_mean = torch.mean(sim, dim=1)  # Shape: [batch_size]
        obs_mean = torch.mean(obs, dim=1)  # Shape: [batch_size]
        
        # Calculate standard deviations along timestep dimension
        sim_std = torch.std(sim, dim=1)  # Shape: [batch_size]
        obs_std = torch.std(obs, dim=1)  # Shape: [batch_size]
        
        # Calculate correlation coefficient for each sample in the batch
        sim_norm = sim - sim_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]
        obs_norm = obs - obs_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]
        
        # Calculate numerator and denominator for correlation
        numerator = torch.mean(sim_norm * obs_norm, dim=1)  # Shape: [batch_size]
        sim_norm_sum = torch.mean(sim_norm ** 2, dim=1)  # Shape: [batch_size]
        obs_norm_sum = torch.mean(obs_norm ** 2, dim=1)  # Shape: [batch_size]
        
        # Add epsilon for numerical stability
        denominator = torch.sqrt(sim_norm_sum * obs_norm_sum + eps)
        r = numerator / denominator  # Shape: [batch_size]
        
        # Calculate alpha (ratio of standard deviations)
        alpha = sim_std / (obs_std + eps)  # Shape: [batch_size]
        
        # Calculate beta (ratio of means)
        # Handle the case where observed mean is zero or very small
        beta = sim_mean / (obs_mean + eps)
        # Calculate KGE for each sample in the batch
        kge = 1 - ( (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 ) # Shape: [batch_size]
        
        # Convert to loss (1 - KGE, since KGE's optimal value is 1)
        loss = 1 - kge  # Shape: [batch_size]
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss


def calculate_kge(observed, simulated):
    """
    Calculate Kling-Gupta Efficiency for timeseries data.
    
    Parameters:
    -----------
    observed : array-like
        Observed timeseries values
    simulated : array-like
        Simulated timeseries values
        
    Returns:
    --------
    float
        KGE value (range: -∞ to 1, where 1 is perfect)
    """
    # Ensure inputs are numpy arrays
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Add small constant to avoid division by zero
    eps = 1e-3
    
    # Calculate components
    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)
    
    # Correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]
    
    # Ratio of means (beta)
    beta = mean_sim / (mean_obs + eps)
    
    # Ratio of standard deviations (alpha)
    alpha = std_sim / (std_obs + eps)
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)
    
    return kge