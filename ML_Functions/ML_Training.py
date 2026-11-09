from tqdm import tqdm
import os
import itertools
import joblib
from datetime import datetime
from tqdm import tqdm
import gc
from collections import defaultdict
import warnings

import numpy as np
import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))
import ML_functions


from ML_functions import (
    append_autoregressive_lag,
    pair_batch_predictions_with_ground_truth,
    inverse_transform_and_clip,
    generate_rolling_windows,
)
from ML_Metrics import (
    evaluate_all_hydrological_metrics,
    prepare_models_for_eval,
    calculate_nse, 
    calculate_nnse, 
    calculate_kge_components, 
    calculate_bias,
)



def load_data_tensors(data_dir, data_group_name, device):
    """
    Loads raw temporal, static, and target tensors from .pt files.

    Args:
        data_dir (str): Directory where the data tensors are stored.
        data_group_name (str): Name of the dataset group (e.g., 'Reservoirs_Train').
        device (torch.device): The device to load tensors onto ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the temporal, static, and targets tensors.
    """
    print("--- 1. Loading data tensors ---")
    try:
        temporal_tensor = torch.load(
            os.path.join(data_dir, f"temporal_inputs_{data_group_name}.pt"),
            map_location=device,
        )
        static_tensor = torch.load(
            os.path.join(data_dir, f"static_inputs_{data_group_name}.pt"),
            map_location=device,
        )
        targets_tensor = torch.load(
            os.path.join(data_dir, f"targets_{data_group_name}.pt"), map_location=device
        )

        print(f"Temporal tensor shape: {temporal_tensor.shape}")
        print(f"Static tensor shape: {static_tensor.shape}")
        print(f"Targets tensor shape: {targets_tensor.shape}")

        return temporal_tensor, static_tensor, targets_tensor
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your data directory and file names.")
        return None, None, None


def create_dataloader(
    temporal_data, static_data, targets_data, batch_size, shuffle=True
):
    """
    Creates a DataLoader from tensors and cleans up original tensor variables to save memory.

    Args:
        temporal_data (torch.Tensor): The scaled temporal data.
        static_data (torch.Tensor): The scaled static data.
        targets_data (torch.Tensor): The scaled target data.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: The configured DataLoader.
    """
    print("\n---  Creating DataLoader and cleaning memory ---")

    # Create a TensorDataset from the scaled tensors
    dataset = TensorDataset(temporal_data, static_data, targets_data)

    # Create a DataLoader to handle batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"DataLoader created with {len(dataloader)} batches of size {batch_size}.")

    # The original tensor variables passed to this function can now be deleted
    # outside the function call, after they are no longer needed.
    # The 'dataset' object now holds the data for the DataLoader.

    total, used, free = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
    print("Percentage of memory used", round((used / total) * 100, 2))
    return dataloader


def free_memory(*tensors):
    """
    Deletes tensor variables and runs Python's garbage collector.
    """
    print("Deleting specified tensors to free memory...")
    for tensor in tensors:
        del tensor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory freed.")


def run_model_pipeline(models, static_input, temporal_input, static_step=True):
    """
    Runs data through a single model or a sequential pipeline of models.

    Args:
        models (torch.nn.Module or list):
            A single model or a list of models to be run in order.
        static_input (torch.Tensor): The static input tensor.
        temporal_input (torch.Tensor): The temporal input tensor.
        use_flow: If True, concatenate previous day's flow to temporal inputs for autoregressive prediction

    Returns:
        torch.Tensor: The final output from the model(s).
    """
    if isinstance(models, list):
        # Handle a list of models (pipeline)
        if not models:
            raise ValueError("The model list cannot be empty.")

        # 1. First model takes the initial two inputs
        x = models[0](static_input, temporal_input)

        # 2. Subsequent models take the output of the previous one
        # The loop starts from the second model (index 1)
        for model in models[1:]:
            # The .squeeze() can be applied here if needed consistently,
            # or handled within the models themselves.

            if static_step:
                x = model(static_input, x)  # .squeeze()

            else:
                x = model(x)  # .squeeze()

        return x
    else:
        # Handle a single model
        return models(static_input, temporal_input)


def train_model(
    model,
    train_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    warmup,
    loss_interval,
    device="cpu",
    train_mode=True,
    time_interval=360,
    static_step=True,
    use_flow=False,
):
    """
    Train a neural network model with learning rate scheduling.

    Args:
        model: The neural network model to train
        dataloader: DataLoader providing batches of (temporal_input, static_input, target) with shapes [batch_size, num_days, num_variables] (no days for static)
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        warmup: Number of warmup steps to exclude from loss calculation
        loss_interval: Number of batches to accumulate show loss over

    """
    for epoch in range(num_epochs):
        # Set the model to training mode
        if isinstance(model, list):
            for m in model:
                m.train(train_mode)
        else:
            model.train(train_mode)

        # print('Using Device:', device)
        epoch_total_loss = 0.0
        current_total_loss = 0.0
        n_accumulated = 0

        # Use tqdm for a progress bar over the dataloader
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            mininterval=time_interval,
        )

        for batch_idx, batch in enumerate(progress_bar):
            static_input = batch["static"]
            temporal_input = batch["temporal"]
            target = batch["target"]

            # --- DIAGNOSTIC 1: Check inputs ---
            if torch.isnan(static_input).any():
                print(f"\n[!] NaN detected in static_input at batch {batch_idx}")
            if torch.isnan(temporal_input).any():
                print(f"\n[!] NaN detected in temporal_input at batch {batch_idx}")
            if torch.isnan(target).any():
                print(f"\n[!] NaN detected in target at batch {batch_idx}")
                
            if use_flow:
                temporal_input, target = append_autoregressive_lag(temporal_input, target)
            
            # The dataloader adds a batch dimension, which we keep
            temporal_input = temporal_input.to(device)
            static_input = static_input.to(device)
            target = target.to(device)


            # Forward pass (disabling autograd caching when validating)
            with torch.set_grad_enabled(train_mode):
                output = run_model_pipeline(
                    model, static_input, temporal_input, static_step
                )
                output_warmup_excluded = output[:, warmup:].squeeze(-1)
                target_warmup_excluded = target[:, warmup:]


                # --- DIAGNOSTIC 2: Check model output ---
                if torch.isnan(output_warmup_excluded).any():
                    print(f"\n[!] NaN detected in model predictions at batch {batch_idx}")
                    print(f"    Predictions min: {output_warmup_excluded.min()}, max: {output_warmup_excluded.max()}")

                loss = criterion(output_warmup_excluded, target_warmup_excluded)

                # --- DIAGNOSTIC 3: Check scalar loss ---
                if torch.isnan(loss):
                    print(f"\n[!] NaN detected in loss at batch {batch_idx}")
                    print(f"    Target warmup min: {target_warmup_excluded.min()}, max: {target_warmup_excluded.max()}")
                    raise FloatingPointError("Loss became NaN.")
                    
                if train_mode:
                    loss.backward()

                    # --- DIAGNOSTIC 4: Check exploding gradients ---
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters() if not isinstance(model, list) else [p for m in model for p in m.parameters()],
                        max_norm=1.0
                    )
                    if torch.isnan(total_norm):
                        print(f"\n[!] NaN detected in gradients at batch {batch_idx}")
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_total_loss += loss.item()
            current_total_loss += loss.item()

            n_accumulated += 1

            # Update progress bar with the average loss for this chunk
            avg_chunk_loss = current_total_loss / n_accumulated

            if n_accumulated % loss_interval == 0:
                progress_bar.set_postfix(loss=f"{avg_chunk_loss:.6f} ")

        # Step the scheduler at the end of each epoch
        if train_mode:
            scheduler.step()
        # At the end of the epoch loop
        avg_epoch_loss = epoch_total_loss / n_accumulated
        current_lr = scheduler.get_last_lr()[0]

        # Get validation loss, and if validation loss hasn't decreased, stop the run
    # print(f"End of Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_epoch_loss:.6f} | Current LR: {current_lr:.6f}\n")

    return avg_epoch_loss


def get_avg_gradient_magnitude(model):
    """Calculates the average L1 norm of gradients for all model parameters."""
    total_grad_magnitude = 0.0
    num_params_with_grad = 0

    models_to_check = model if isinstance(model, list) else [model]

    for m in models_to_check:
        for p in m.parameters():
            if p.grad is not None:
                total_grad_magnitude += p.grad.data.abs().mean().item()
                num_params_with_grad += 1

    return (
        total_grad_magnitude / num_params_with_grad if num_params_with_grad > 0 else 0.0
    )


def process_validation_batches(
    dataloader,
    model_names,
    models,
    warmup,
    streamflow_scaler,
    device="cpu",
    time_interval=180,
    max_batches=None,
    use_flow=False,
):
    """
    Process validation batches and yield results incrementally.
    Returns per-forecast predictions and targets.
    """
    progress_bar = tqdm(
        dataloader, desc="Processing batches", unit="batch", mininterval=time_interval
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):            
            if max_batches is not None and batch_idx >= max_batches:
                print(f"Reached maximum number of batches ({max_batches}). Stopping.")
                break

            static_input = batch["static"].to(device)
            temporal_input = batch["temporal"]
            target = batch["target"]

            if use_flow:
                temporal_input, target = append_autoregressive_lag(temporal_input, target)

            temporal_input = temporal_input.to(device)
            target = target.to(device)

            target_warmup_excluded = target[:, warmup:]

            # Move to CPU and inverse transform immediately (on batch, not full dataset)
            target_warmup_excluded = inverse_transform_and_clip(
                target_warmup_excluded, streamflow_scaler, min_val=None
            )

            batch_results = {}
            for model_name, model in zip(model_names, models):
                output = run_model_pipeline(model, static_input, temporal_input)
                output_warmup_excluded = output[:, warmup:].squeeze(-1)
                batch_results[model_name] = inverse_transform_and_clip(
                    output_warmup_excluded, streamflow_scaler, min_val=0.0
                )

            # Yield batch results immediately
            yield batch_results, target_warmup_excluded

            # Clean up GPU memory
            del static_input, temporal_input, target, output
            if device != "cpu":
                torch.cuda.empty_cache()

def collect_validation_results(validation_generator):
    return pair_batch_predictions_with_ground_truth(validation_generator)
    
def validate_model_scores(
    model_names,
    models,
    dataloader,
    warmup,
    streamflow_scaler,
    device="cpu",
    time_interval=180,
    max_batches=None,
    use_flow=False,
):
    models = prepare_models_for_eval(models, freeze_parameters=False)

    # Initialize results dictionary to store lists of metrics for each forecast
    results = {
        model_name: {
            "NSE": [],
            "NNSE": [],
            "KGE_alpha": [],
            "KGE_beta": [],
            "KGE_correlation": [],
            "KGE": [],
            "bias": [],
        }
        for model_name in model_names
    }

    # Process batches as generator (memory efficient)
    batch_generator = process_validation_batches(
        dataloader,
        model_names,
        models,
        warmup,
        streamflow_scaler,
        device=device,
        time_interval=time_interval,
        max_batches=max_batches,
        use_flow=use_flow,
    )

    # Calculate metrics for each forecast in each batch
    for batch_results, target_warmup_excluded in batch_generator:
        # Iterate over each sample (forecast) in the batch
        batch_size = target_warmup_excluded.shape[0]

        for i in range(batch_size):
            target_forecast = target_warmup_excluded[i]

            for model_name in model_names:
                            output_forecast = batch_results[model_name][i]
                            metrics = evaluate_all_hydrological_metrics(output_forecast, target_forecast)
                            for metric_key, val in metrics.items():
                                results[model_name][metric_key].append(val)
    return results



def lagged_model_predictions(
    model_names,
    models,
    dataloader,
    warmup,
    streamflow_scaler,
    device="cpu",
    time_interval=180,
    max_batches=None,
    use_flow=False,
):
    models = prepare_models_for_eval(models, freeze_parameters=False)
    # Initialize results dictionary to store lists of metrics for each forecast
    results = {
        model_name: {"Prediction": [], "Truth": []}  # dict, not set
        for model_name in model_names
    }

    # Process batches as generator (memory efficient)
    batch_generator = process_validation_batches(
        dataloader,
        model_names,
        models,
        warmup,
        streamflow_scaler,
        device=device,
        time_interval=time_interval,
        max_batches=max_batches,
        use_flow=use_flow,
    )

    # Calculate metrics for each forecast in each batch
    for batch_results, target_warmup_excluded in batch_generator:
        # Iterate over each sample (forecast) in the batch
        batch_size = target_warmup_excluded.shape[0]

        for i in range(batch_size):
            target_forecast = target_warmup_excluded[i][0].detach().cpu().item()

            for model_name in model_names:
                output_forecast = batch_results[model_name][i][0].detach().cpu().item()
                results[model_name]["Prediction"].append(output_forecast)
                results[model_name]["Truth"].append(target_forecast)

    return results


def get_basin_name(basin_idx, csv_base_path="/perm/mokr/Scaled_CamelsGB_V2"):
    # Split by underscore and take the first part as location
    csv_filename = f"normalised_static_data.csv"
    csv_path = os.path.join(csv_base_path, csv_filename)
    df = pd.read_csv(csv_path, engine="python")
    gauge_name = df[df["gauge_id"] == basin_idx]["gauge_name"].values[0]

    return gauge_name


def calculate_rolling_metrics(results_dict, metadata, window_size=90):
    """
    Calculate rolling window metrics per gauge independently.
    
    Args:
        results_dict: dict [model_name]['Prediction'/'Truth'] -> list of tensors
        metadata:     dict with 'gauge_id' and 'init_date' lists (same length as predictions)
        window_size:  number of timesteps per rolling window (default 90)
    
    Returns:
        dict of {model_name: {metric_name: [values], 'gauge_id': [], 'init_date': []}}
    """
    metrics = defaultdict(lambda: defaultdict(list))
    model_names = list(results_dict.keys())

    # Group indices by gauge
    gauge_ids = metadata['gauge_id']
    init_dates = metadata['init_date']

    # gauge_to_indices = defaultdict(list)
    for model_name in model_names:
        predictions = results_dict[model_name]['Prediction']
        truths      = results_dict[model_name]['Truth']
        n = len(predictions)

        gauge_to_indices = defaultdict(list)
        for idx, gauge_id in enumerate(gauge_ids[:n]):  # Only index up to n
            gauge_to_indices[gauge_id].append(idx)
            
        for gauge_id, indices in gauge_to_indices.items():
            num_windows = len(indices) - window_size + 1

            if num_windows <= 0:
                continue  # Skip gauges with fewer timesteps than window

            preds_gauge = torch.as_tensor([predictions[j] for j in indices])
            truths_gauge = torch.as_tensor([truths[j] for j in indices])

            pred_windows = generate_rolling_windows(preds_gauge, window_size)
            truth_windows = generate_rolling_windows(truths_gauge, window_size)

            for i, (out_win, tgt_win) in enumerate(zip(pred_windows, truth_windows)):
                sample_metrics = evaluate_all_hydrological_metrics(out_win, tgt_win)
                for metric_key, val in sample_metrics.items():
                    metrics[model_name][metric_key].append(val)
                metrics[model_name]["gauge_id"].append(gauge_id)
                metrics[model_name]["init_date"].append(init_dates[indices[i]])
    return {k: dict(v) for k, v in metrics.items()}


