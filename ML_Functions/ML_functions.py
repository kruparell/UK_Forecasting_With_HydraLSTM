import time
import itertools
import joblib
import sys
import gc
import pickle
from tqdm import tqdm
import os
from typing import Dict, Iterable, List, Tuple


import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
from math import radians, sin, cos, sqrt, atan2
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
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Series.__getitem__ treating keys as positions is deprecated",
)


    
LANDCOVER_PREFIXES = [
    'dwood_perc', 'ewood_perc', 'grass_perc', 'shrub_perc',
    'crop_perc', 'urban_perc', 'inwater_perc', 'bares_perc'
]
AVAILABLE_LANDCOVER_YEARS = [1990, 2015, 2017, 2018, 2019, 2020, 2021, 2022]

def resolve_landcover_features(
    static_df: pd.DataFrame,
    gauge_id: str,
    year: int,
    prefixes: list[str] = LANDCOVER_PREFIXES,
    available_years: list[int] = AVAILABLE_LANDCOVER_YEARS,
) -> np.ndarray:
    """Selects static features with the closest temporally aligned landcover columns."""
    closest_yr = min(available_years, key=lambda y: abs(y - year))
    landcover_cols = [f"{p}_{closest_yr}" for p in prefixes]
    non_landcover_cols = [
        col for col in static_df.columns 
        if not any(col.startswith(p) for p in prefixes)
    ]
    return pd.concat([
        static_df.loc[gauge_id, non_landcover_cols],
        static_df.loc[gauge_id, landcover_cols],
    ]).values.astype(np.float32)


def inverse_transform_and_clip(
    tensor: torch.Tensor,
    scaler,
    min_val: float | None = 0.0,
) -> torch.Tensor:
    """Transfers tensor to CPU, unscales via scikit-learn scaler, clips, and returns tensor."""
    unscaled_np = scaler.inverse_transform(tensor.detach().cpu().numpy())
    if min_val is not None:
        unscaled_np = np.clip(unscaled_np, a_min=min_val, a_max=None)
    return torch.from_numpy(unscaled_np)


def generate_rolling_windows(
    series: torch.Tensor,
    window_size: int,
):
    """Yields consecutive sub-tensors of length window_size."""
    num_windows = len(series) - window_size + 1
    for i in range(num_windows):
        yield series[i : i + window_size]
        


def sample_batch(all_pairs, batch_size, random_seed=None):
    """Sample a batch of catchment-date pairs uniformly."""
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.random.choice(len(all_pairs), size=batch_size, replace=False)
    return [all_pairs[i] for i in indices]


def load_scalers(output_path="scalers"):
    """Load saved scalers"""
    scalers = joblib.load(f"{output_path}.joblib")
    return scalers

def slice_warmup_window(
    output: torch.Tensor,
    warmup: int
) -> torch.Tensor:
    """Removes warmup timesteps from predictions and squeezes the singleton feature dimension.

    Args:
        output: Model output tensor of shape (batch_size, timesteps, 1) or (batch_size, timesteps).
        warmup: Number of initial timesteps to drop.

    Returns:
        Tensor of shape (batch_size, timesteps - warmup).
    """
    sliced = output[:, warmup:]
    if sliced.dim() == 3 and sliced.size(-1) == 1:
        sliced = sliced.squeeze(-1)
    return sliced


def pair_batch_predictions_with_ground_truth(
    batch_stream: Iterable[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Restructures a streamed batch iterator of model results into grouped model lists.

    Args:
        batch_stream: Iterable yielding ({model_name: prediction_tensor}, ground_truth_tensor).

    Returns:
        Dictionary structured as {model_name: [(prediction_batch, target_batch), ...]}.
    """
    collected: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for batch_results, target in batch_stream:
        for model_name, output in batch_results.items():
            if model_name not in collected:
                collected[model_name] = []
            collected[model_name].append((output, target))
    return collected
    
def append_autoregressive_lag(
    temporal_input: torch.Tensor,
    target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Appends previous day's streamflow to temporal features and shifts target by one timestep.

    Args:
        temporal_input: Tensor of shape (batch_size, timesteps, features).
        target: Tensor of shape (batch_size, timesteps).

    Returns:
        Tuple of (shifted_temporal_tensor, shifted_target_tensor):
            - shifted_temporal: Shape (batch_size, timesteps - 1, features + 1)
            - shifted_target: Shape (batch_size, timesteps - 1)
    """
    previous_flow = target[:, :-1].unsqueeze(-1)
    lagged_temporal = torch.cat([temporal_input[:, :-1, :], previous_flow], dim=-1)
    lagged_target = target[:, 1:]
    return lagged_temporal, lagged_target

    
def get_closest_landcover_year(year):
    return min(AVAILABLE_LANDCOVER_YEARS, key=lambda y: abs(y - year))


def preprocess_to_tensors(
    pairs, temporal_data, static_variables, goal="discharge_spec_x", closest_year = True, window = 89
):
    """Pre-convert all data to PyTorch tensors and save to disk"""
    goals = [
        "discharge_spec_x",
        "discharge_spec_y",
        "discharge_vol_x",
        "discharge_vol_y",
    ]

    tensor_data = {
        "static": [],
        "temporal": [],
        "target": [],
        # 'gauge_id': [],
        # 'init_date': []
    }

    # Pre-compute non-landcover columns (constant across all samples)
    landcover_cols = [c for c in static_variables.columns 
                      if any(c.startswith(p) for p in LANDCOVER_PREFIXES)]
    non_landcover_cols = [c for c in static_variables.columns if c not in landcover_cols]

    print(f"Processing {len(pairs)} pairs...")
    skipped = 0
    for i, sample in enumerate(pairs):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(pairs)}")

        gauge_id = str(sample["gauge_id"])
        init_date = sample["init_date"]
        # end_date = init_date + pd.Timedelta(days=window)
        end_date = init_date + pd.Timedelta(days=window - 1)

        if closest_year:
                    static_features = resolve_landcover_features(
                        static_variables, gauge_id, end_date.year
                    )
        else:
            static_features = static_variables.loc[gauge_id].values.astype(np.float32)

        # Get data
        temporal_values = temporal_data[gauge_id].loc[init_date:end_date]
        true_discharge = temporal_values[goal].values.astype(np.float32)
        temporal_input = temporal_values.drop(columns=goals).values.astype(np.float32)

        # Convert to tensors immediately
        static_tensor = torch.FloatTensor(static_features)
        temporal_tensor = torch.FloatTensor(temporal_input)
        target_tensor = torch.FloatTensor(true_discharge)

        # Print tensor shapes for first sample
        if i == 2:
            print(f"temporal_values shape: {temporal_values.shape}")
            print(f"true_discharge shape: {true_discharge.shape}")
            print(f"static_tensor shape: {static_tensor.shape}")
            print(f"temporal_tensor shape: {temporal_tensor.shape}")
            print(f"target_tensor shape: {target_tensor.shape}")
            print("--- End first sample ---\n")

        
        if len(temporal_values) != window:
            if skipped < 20:  # Print first few issues
                print(
                    f"Warning: Sample {i} (gauge {gauge_id}, date {init_date}) has {len(temporal_values)} days instead of {window}, skipping"
                )
            skipped += 1
            continue

        tensor_data["static"].append(static_tensor)
        tensor_data["temporal"].append(temporal_tensor)
        tensor_data["target"].append(target_tensor)
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
        valid_indices = [
            i for i, t in enumerate(tensor_data["target"])
            if not torch.isnan(t).any()
        ]
        
        dropped = len(tensor_data["target"]) - len(valid_indices)
        if dropped > 0:
            print(f"Dropped {dropped}/{len(tensor_data['target'])} samples containing NaN targets.")

        self.static = [tensor_data["static"][i] for i in valid_indices]
        self.temporal = [tensor_data["temporal"][i] for i in valid_indices]
        self.target = [tensor_data["target"][i] for i in valid_indices]
        # self.gauge_id = tensor_data['gauge_id']
        # self.init_date = tensor_data['init_date']

    def __len__(self):
        return len(self.static)

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "temporal": self.temporal[idx],
            "target": self.target[idx],
            # 'gauge_id': self.gauge_id[idx],
            # 'init_date': self.init_date[idx]
        }


# Define the Dataset class
class CatchmentDataset(Dataset):
    def __init__(self, pairs, temporal_data, static_variables, goal="discharge_spec_x", window = 89):
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
        self.goals = [
            "discharge_spec_x",
            "discharge_spec_y",
            "discharge_vol_x",
            "discharge_vol_y",
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        gauge_id = sample["gauge_id"]
        init_date = sample["init_date"]
        # end_date = init_date + pd.Timedelta(days=window)  # (days + 1) days total
        end_date = init_date + pd.Timedelta(days=window-1) 
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
            "static": static_tensor,
            "temporal": temporal_tensor,
            "target": target_tensor,
            # 'gauge_id': gauge_id,
            # 'init_date': init_date
        }


class NSELoss(nn.Module):
    def __init__(self, eps=1e-1):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.mean((target - preds) ** 2, dim=1)
        target_mean = torch.mean(target, dim=1, keepdim=True)
        denominator = torch.mean((target - target_mean) ** 2, dim=1)

        # Check for non-finite intermediate values
        if torch.isnan(numerator).any():
            print("\n[!] NaN in NSE numerator (target - preds squared)")
        if torch.isnan(denominator).any():
            print("\n[!] NaN in NSE denominator (target variance)")
        if (denominator + self.eps <= 0).any():
            print("\n[!] Zero/Negative value in NSE denominator + eps")

        nse = 1 - numerator / (denominator + self.eps)
        loss = 1 - nse.mean()

        if torch.isnan(loss):
            print(f"\n[!] NSE value: mean={nse.mean()}, min={nse.min()}, max={nse.max()}")

        return loss

class NNSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds and target shape: (batch_size, seq_len)
        numerator = torch.mean((target - preds) ** 2, dim=1)
        denominator = torch.mean(
            (target - torch.mean(target, dim=1, keepdim=True)) ** 2, dim=1
        )

        nse = 1 - numerator / (denominator + self.eps)
        return 1 - (1 / (2 - nse.mean()))  # minimize 1 - NSE


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

    def __init__(self, reduction="mean"):
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
        sim_norm_sum = torch.mean(sim_norm**2, dim=1)  # Shape: [batch_size]
        obs_norm_sum = torch.mean(obs_norm**2, dim=1)  # Shape: [batch_size]

        # Add epsilon for numerical stability
        denominator = torch.sqrt(sim_norm_sum * obs_norm_sum + eps)
        r = numerator / denominator  # Shape: [batch_size]

        # Calculate alpha (ratio of standard deviations)
        alpha = sim_std / (obs_std + eps)  # Shape: [batch_size]

        # Calculate beta (ratio of means)
        # Handle the case where observed mean is zero or very small
        beta = sim_mean / (obs_mean + eps)
        # Calculate KGE for each sample in the batch
        kge = 1 - (
            (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
        )  # Shape: [batch_size]

        # Convert to loss (1 - KGE, since KGE's optimal value is 1)
        loss = 1 - kge  # Shape: [batch_size]

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
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
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (alpha - 1) ** 2)

    return kge


def generate_all_catchment_date_pairs(
    catchments_df, forecast_days=90, catchment_type=None, pair_id_col=None
):
    """
    Generate all valid catchment-date pairs for a given DataFrame.

    Parameters:
    -----------
    catchments_df : DataFrame
        DataFrame with catchment information and date columns
    forecast_days : int
        Number of days of data required after the initialization date
    catchment_type : str, optional
        Type label for these catchments (e.g., 'paired_reservoir')
    pair_id_col : str, optional
        Name of the pair_id column if it exists

    Returns:
    --------
    all_pairs : list of dict
        List of all valid catchment-date pairs
    """
    all_pairs = []

    # Determine if this is paired or unpaired data
    has_pair_dates = (
        "pair_start" in catchments_df.columns and "pair_end" in catchments_df.columns
    )

    forecast_buffer = pd.Timedelta(days=forecast_days)

    for _, row in catchments_df.iterrows():
        # Get the date range for this catchment
        if has_pair_dates:
            if pd.notna(row["pair_start"]) and pd.notna(row["pair_end"]):
                start_date = pd.Timestamp(row["pair_start"])
                end_date = pd.Timestamp(row["pair_end"])
                pair_id = row.get(pair_id_col, None) if pair_id_col else None
            else:
                continue
        else:
            if pd.notna(row["valid_start"]) and pd.notna(row["valid_end"]):
                start_date = pd.Timestamp(row["valid_start"])
                end_date = pd.Timestamp(row["valid_end"])
                pair_id = None
            else:
                continue

        # Calculate the latest valid initialization date
        latest_init_date = end_date - forecast_buffer

        if latest_init_date <= start_date:
            # Not enough data for this forecast length, skip
            print(
                f"Warning: Catchment {row['gauge_id']} doesn't have {forecast_days} days of data"
            )
            continue

        # Add all valid dates for this catchment
        num_valid_days = (latest_init_date - start_date).days + 1

        for day_offset in range(num_valid_days):
            init_date = start_date + pd.Timedelta(days=day_offset)
            pair_dict = {
                "gauge_id": f"{int(row['gauge_id'])}",
                "pair_id": pair_id,
                "init_date": init_date,
            }
            if catchment_type:
                pair_dict["catchment_type"] = catchment_type
            all_pairs.append(pair_dict)

    return all_pairs


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of earth in kilometers
    R = 6371.0

    distance = R * c
    return distance
