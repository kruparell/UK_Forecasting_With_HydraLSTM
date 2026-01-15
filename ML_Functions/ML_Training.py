from tqdm import tqdm
import os
import itertools
import joblib
from datetime import datetime
from tqdm import tqdm
import gc
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
        temporal_tensor = torch.load(os.path.join(data_dir, f'temporal_inputs_{data_group_name}.pt'), map_location=device)
        static_tensor = torch.load(os.path.join(data_dir, f'static_inputs_{data_group_name}.pt'), map_location=device)
        targets_tensor = torch.load(os.path.join(data_dir, f'targets_{data_group_name}.pt'), map_location=device)
        
        print(f"Temporal tensor shape: {temporal_tensor.shape}")
        print(f"Static tensor shape: {static_tensor.shape}")
        print(f"Targets tensor shape: {targets_tensor.shape}")
        
        return temporal_tensor, static_tensor, targets_tensor
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your data directory and file names.")
        return None, None, None

def create_dataloader(temporal_data, static_data, targets_data, batch_size, shuffle=True):
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

    total, used, free = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print('Percentage of memory used', round((used / total) * 100, 2))
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


def run_model_pipeline(models, static_input, temporal_input, static_step = False):
    """
    Runs data through a single model or a sequential pipeline of models.

    Args:
        models (torch.nn.Module or list):
            A single model or a list of models to be run in order.
        static_input (torch.Tensor): The static input tensor.
        temporal_input (torch.Tensor): The temporal input tensor.

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
                x = model(static_input, x.squeeze()) # Applying squeeze as in your example

            else:
                x = model(x.squeeze()) # Applying squeeze as in your example


        return x
    else:
        # Handle a single model
        return models(static_input, temporal_input)


        
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, 
                warmup, loss_interval, device = 'cpu', train_mode = True, time_interval = 360, static_step = True):
    """
    Train a neural network model with learning rate scheduling.
    
    Args:
        model: The neural network model to train
        dataloader: DataLoader providing batches of (temporal_input, static_input, target)
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


        
        print('Using Device:', device)
        epoch_total_loss = 0.0
        current_total_loss = 0.0
        n_accumulated = 0
        
        # Use tqdm for a progress bar over the dataloader
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", mininterval= time_interval)
        for batch in progress_bar:
            
            static_input = batch['static']
            temporal_input = batch['temporal']
            target = batch['target']
            
            # The dataloader adds a batch dimension, which we keep
            temporal_input = temporal_input.to(device)
            static_input = static_input.to(device)
            target = target.to(device)
            
            # Forward pass
            # The model output is sliced to exclude the warm-up period
            output = run_model_pipeline(model, static_input, temporal_input, static_step)
            # output = model(static_input, temporal_input)
                
            output_warmup_excluded = output[:, warmup:].squeeze()
            
            # Target is also sliced to match the output
            target_warmup_excluded = target[:, warmup:]
            # Calculate loss

            
            loss = criterion(output_warmup_excluded, target_warmup_excluded)

            # out_var = output_warmup_excluded.var(dim=1).mean().item()
            # tgt_var = target_warmup_excluded.var(dim=1).mean().item()
    


            if train_mode:
                loss.backward()
                    
                optimizer.step()
                optimizer.zero_grad()
            epoch_total_loss += loss.item() 
            current_total_loss += loss.item()
            
            n_accumulated += 1

            
            
            # Update progress bar with the average loss for this chunk
            avg_chunk_loss = current_total_loss / n_accumulated

            if n_accumulated % loss_interval == 0:
                progress_bar.set_postfix(loss=f"{avg_chunk_loss:.6f} ")
                
                # progress_bar.set_postfix(
                #     out_var=f"{out_var:.4f}",
                #     tgt_var=f"{tgt_var:.4f}"
                # )

        # Step the scheduler at the end of each epoch
        scheduler.step()
        # At the end of the epoch loop
        avg_epoch_loss = epoch_total_loss / n_accumulated
        current_lr = scheduler.get_last_lr()[0]
        print(f"End of Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_epoch_loss:.6f} | Current LR: {current_lr:.6f}\n")



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
                
    return total_grad_magnitude / num_params_with_grad if num_params_with_grad > 0 else 0.0
    

def process_validation_batches(dataloader, model_names, models, warmup, streamflow_scaler, 
                                device='cpu', time_interval=180, max_batches=None):
    """
    Process validation batches and return processed outputs and targets for each model.
    
    Args:
        dataloader: DataLoader for validation data
        model_names: List of model names
        models: List of models
        warmup: Number of warmup timesteps to exclude
        streamflow_scaler: Scaler for inverse transforming streamflow values
        device: Device to run models on ('cpu' or 'cuda')
        time_interval: Update interval for progress bar
        max_batches: Maximum number of batches to process (None for all)
    
    Returns:
        dict: Dictionary with model names as keys, each containing lists of 
              (output_warmup_excluded, target_warmup_excluded) tuples for each batch
    """
    # Initialize storage for outputs
    batch_outputs = {model_name: [] for model_name in model_names}
    
    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Processing batches", unit="batch", mininterval=time_interval)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if max_batches is not None and batch_idx >= max_batches:
                print(f"Reached maximum number of batches ({max_batches}). Stopping processing.")
                break
    
            static_input = batch['static']
            temporal_input = batch['temporal']
            target = batch['target']
            
            # Move inputs to device
            temporal_input = temporal_input.to(device)
            static_input = static_input.to(device)
            target = target.to(device)
            target_warmup_excluded = target[:, warmup:]
        
            target_warmup_excluded = torch.from_numpy(
                streamflow_scaler.inverse_transform(target_warmup_excluded.detach().numpy())
            )
            
            # Forward pass for each model
            for model_name, model in zip(model_names, models):
                output = run_model_pipeline(model, static_input, temporal_input)
                output_warmup_excluded = output[:, warmup:]
        
                output_warmup_excluded = torch.from_numpy(
                    streamflow_scaler.inverse_transform(
                        output_warmup_excluded.squeeze(-1).detach().cpu().numpy()
                    )
                ).to(output_warmup_excluded.device)
                output_warmup_excluded = np.clip(output_warmup_excluded, a_min=0, a_max=None)
                
                # Store the outputs and targets for this batch
                batch_outputs[model_name].append((output_warmup_excluded, target_warmup_excluded))
    
    return batch_outputs


def validate_model_scores(model_names, models, dataloader, warmup, streamflow_scaler, device='cpu', time_interval=180, max_batches=None):
    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]
    
    # Set all models to evaluation mode
    for model in models:
        if isinstance(model, list):
            # If the item is a list, loop through its sub-models
            for sub_model in model:
                sub_model.eval()
        else:
            # Otherwise, it's a single model
            model.eval()
    
    # Initialize results dictionary to store lists of metrics for each model
    results = {model_name: {
        'NSE': [],
        'NNSE': [],
        'KGE_alpha': [],
        'KGE_beta': [],
        'KGE_correlation': [],
        'KGE': [],
        'bias': []
    } for model_name in model_names}
    
    # Process all validation batches
    batch_outputs = process_validation_batches(
        dataloader, model_names, models, warmup, streamflow_scaler,
        device=device, time_interval=time_interval, max_batches=max_batches
    )
    
    # Calculate metrics for each model and batch
    for model_name in model_names:
        for output_warmup_excluded, target_warmup_excluded in batch_outputs[model_name]:
            nse = calculate_nse(output_warmup_excluded, target_warmup_excluded)
            nnse = calculate_nnse(output_warmup_excluded, target_warmup_excluded)
            bias = calculate_bias(output_warmup_excluded, target_warmup_excluded)
            correlation, alpha, beta = calculate_kge_components(output_warmup_excluded, target_warmup_excluded)
            kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            # Store results for this batch
            results[model_name]['NSE'].append(nse.item())
            results[model_name]['NNSE'].append(nnse.item())
            results[model_name]['KGE_alpha'].append(alpha.item())
            results[model_name]['KGE_beta'].append(beta.item())
            results[model_name]['KGE_correlation'].append(correlation.item())
            results[model_name]['KGE'].append(kge.item())
            results[model_name]['bias'].append(bias.item())
    
    return results


def get_basin_name(basin_idx, csv_base_path="/perm/mokr/Scaled_CamelsGB_V2"):
    # Split by underscore and take the first part as location
    csv_filename = f"normalised_static_data.csv"
    csv_path = os.path.join(csv_base_path, csv_filename)
    df = pd.read_csv(csv_path, engine='python')
    gauge_name = df[df['gauge_id'] == basin_idx]['gauge_name'].values[0]

    return gauge_name