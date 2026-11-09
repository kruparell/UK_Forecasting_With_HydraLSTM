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


def scale_data(temporal_tensor, static_tensor, targets_tensor, scaler_dir, device):
    """
    Normalizes temporal, static, and target tensors using saved scalers.

    Args:
        temporal_tensor (torch.Tensor): The raw temporal input data.
        static_tensor (torch.Tensor): The raw static input data.
        targets_tensor (torch.Tensor): The raw target data.
        scaler_dir (str): Directory containing the 'reservoir_scalers_dict.pkl' and
                          'reservoir_static_scalers_dict.pkl' files.
        device (torch.device): The device to store the scaled tensors on.

    Returns:
        tuple: A tuple containing scaled temporal, static, and targets tensors,
               plus the static input size.
    """
    print("\n--- 2. Loading scalers and normalizing data ---")

    # --- Load scalers ---
    temporal_scaler_dict = joblib.load(
        os.path.join(scaler_dir, "reservoir_scalers_dict.pkl")
    )
    static_scaler = joblib.load(
        os.path.join(scaler_dir, "reservoir_static_scalers_dict.pkl")
    )

    # --- Scale Static Data ---
    static_np = static_tensor.cpu().numpy()
    static_np_scaled = static_scaler.transform(static_np)
    static_tensor_scaled = torch.from_numpy(static_np_scaled).float().to(device)
    static_input_size = static_tensor_scaled.shape[1]
    print(f"Static data scaled. Input size: {static_input_size}")

    # --- Scale Targets (Streamflow) ---
    streamflow_scaler = temporal_scaler_dict["streamflow"]
    targets_np = targets_tensor.cpu().numpy()
    original_shape_targets = targets_np.shape
    scaled_targets = streamflow_scaler.transform(targets_np.reshape(-1, 1))
    targets_tensor_scaled = (
        torch.from_numpy(scaled_targets.reshape(original_shape_targets))
        .float()
        .to(device)
    )
    print("Target data (streamflow) scaled.")

    # --- Scale Temporal Features ---
    sample_nc_path = (
        "/ec/res4/scratch/ecm7072/caravan/timeseries/netcdf/hysets/hysets_06324500.nc"
    )
    with xr.open_dataset(sample_nc_path) as ds_sample:
        if "potential_evaporation_sum_ERA5_LAND" in ds_sample:
            ds_sample = ds_sample.rename(
                {"potential_evaporation_sum_ERA5_LAND": "potential_evaporation_sum"}
            )
        temporal_variable_names = sorted([var for var in ds_sample.data_vars])
    temporal_variable_names.remove("streamflow")  # We don't scale streamflow here

    temporal_tensor_scaled = torch.empty_like(temporal_tensor)
    for i, var_name in enumerate(
        tqdm(temporal_variable_names, desc="Scaling Temporal Vars")
    ):
        if var_name in temporal_scaler_dict:
            scaler = temporal_scaler_dict[var_name]
            feature_slice = temporal_tensor[:, :, i].cpu().numpy()
            original_shape_feat = feature_slice.shape

            flat_data = feature_slice.reshape(-1, 1)
            scaled_data = scaler.transform(flat_data)

            temporal_tensor_scaled[:, :, i] = torch.from_numpy(
                scaled_data.reshape(original_shape_feat)
            )
        else:
            print(
                f"Warning: No scaler for temporal var '{var_name}'. Using original data."
            )
            temporal_tensor_scaled[:, :, i] = temporal_tensor[:, :, i]

    return (
        temporal_tensor_scaled.to(device),
        static_tensor_scaled,
        targets_tensor_scaled,
        static_input_size,
    )


def scale_data_np(temporal_np, static_np, targets_np, scaler_dir):
    """
    Normalizes temporal, static, and target numpy arrays using saved scalers.

    Args:
        temporal_np (np.ndarray): The raw temporal input data.
        static_np (np.ndarray): The raw static input data.
        targets_np (np.ndarray): The raw target data.
        scaler_dir (str): Directory containing the scaler files.

    Returns:
        tuple: A tuple containing scaled temporal, static, and targets as numpy arrays,
               plus the static input size.
    """

    # --- Load scalers ---
    temporal_scaler_dict = joblib.load(
        os.path.join(scaler_dir, "reservoir_scalers_dict.pkl")
    )
    static_scaler = joblib.load(
        os.path.join(scaler_dir, "reservoir_static_scalers_dict.pkl")
    )

    # --- Scale Static Data ---
    static_np_scaled = static_scaler.transform(static_np)
    static_input_size = static_np_scaled.shape[1]

    # --- Scale Targets (Streamflow) ---
    streamflow_scaler = temporal_scaler_dict["streamflow"]
    original_shape_targets = targets_np.shape
    targets_np_scaled = streamflow_scaler.transform(targets_np.reshape(-1, 1)).reshape(
        original_shape_targets
    )

    # --- Scale Temporal Features ---
    # NOTE: This path is hardcoded as in your example.
    sample_nc_path = (
        "/ec/res4/scratch/ecm7072/caravan/timeseries/netcdf/hysets/hysets_06324500.nc"
    )
    with xr.open_dataset(sample_nc_path) as ds_sample:
        if "potential_evaporation_sum_ERA5_LAND" in ds_sample:
            ds_sample = ds_sample.rename(
                {"potential_evaporation_sum_ERA5_LAND": "potential_evaporation_sum"}
            )
        temporal_variable_names = sorted([var for var in ds_sample.data_vars])
        temporal_variable_names.remove("streamflow")

    temporal_np_scaled = np.empty_like(temporal_np, dtype=np.float32)
    for i, var_name in enumerate(
        tqdm(temporal_variable_names, desc="Scaling Temporal Vars")
    ):
        if var_name in temporal_scaler_dict:
            scaler = temporal_scaler_dict[var_name]
            feature_slice = temporal_np[:, :, i]
            original_shape_feat = feature_slice.shape

            flat_data = feature_slice.reshape(-1, 1)
            scaled_data = scaler.transform(flat_data)

            temporal_np_scaled[:, :, i] = scaled_data.reshape(original_shape_feat)
        else:
            temporal_np_scaled[:, :, i] = temporal_np[:, :, i]
    return temporal_np_scaled, static_np_scaled, targets_np_scaled, static_input_size


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


def run_model_pipeline(models, static_input, temporal_input, static_step=False):
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
                x = model(
                    static_input, x.squeeze()
                )  # Applying squeeze as in your example

            else:
                x = model(x.squeeze())  # Applying squeeze as in your example

        return x
    else:
        # Handle a single model
        return models(static_input, temporal_input)


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


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    warmup,
    loss_interval,
    device="cpu",
    train_mode=True,
    time_interval=180,
    static_step=True,
):
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

        print("Using Device:", device)
        epoch_total_loss = 0.0
        current_total_loss = 0.0
        n_accumulated = 0

        # Use tqdm for a progress bar over the dataloader
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="batch",
            mininterval=time_interval,
        )
        for batch in progress_bar:

            static_input = batch["static"]
            temporal_input = batch["temporal"]
            target = batch["target"]

            # The dataloader adds a batch dimension, which we keep
            temporal_input = temporal_input.to(device)
            static_input = static_input.to(device)
            target = target.to(device)

            # Forward pass
            # The model output is sliced to exclude the warm-up period
            output = run_model_pipeline(
                model, static_input, temporal_input, static_step
            )
            # output = model(static_input, temporal_input)

            output_warmup_excluded = output[:, warmup:].squeeze()

            # Target is also sliced to match the output
            target_warmup_excluded = target[:, warmup:]
            # Calculate loss

            loss = criterion(output_warmup_excluded, target_warmup_excluded)

            out_var = output_warmup_excluded.var(dim=1).mean().item()
            tgt_var = target_warmup_excluded.var(dim=1).mean().item()

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
        print(
            f"End of Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_epoch_loss:.6f} | Current LR: {current_lr:.6f}\n"
        )


def create_dataloaders(dataset_names, data_dir, scaler_dir, device, batch_size):
    """
    Processes a list of dataset names to create and return a dictionary of dataloaders.

    Args:
        dataset_names (list): A list of strings with the names of the datasets.
        data_dir (str): The directory where the raw data is stored.
        scaler_dir (str): The directory where the scalers are stored.
        device (torch.device): The device to load tensors onto (e.g., 'cpu' or 'cuda').
        batch_size (int): The batch size for the dataloaders.

    Returns:
        dict: A dictionary where keys are dataset names and values are the corresponding dataloaders.
    """
    dataloaders = {}
    print(f"--- Starting Dataloader Creation Process ---")
    for name in dataset_names:
        # Automatically determine if data should be shuffled.
        # Typically, training data is shuffled, while validation/test data is not.
        should_shuffle = "Train" in name

        # --- 1. Load Data Tensors ---
        temporal_raw, static_raw, targets_raw = load_data_tensors(
            data_dir, name, device
        )

        # --- 2. Load Scalers and Normalize Data ---
        temporal_scaled, static_scaled, targets_scaled, _ = scale_data(
            temporal_tensor=temporal_raw,
            static_tensor=static_raw,
            targets_tensor=targets_raw,
            scaler_dir=scaler_dir,
            device=device,
        )

        free_memory(temporal_raw, static_raw, targets_raw)

        # --- 3. Create a specific Dataloader ---
        dataloader = create_dataloader(
            temporal_data=temporal_scaled,
            static_data=static_scaled,
            targets_data=targets_scaled,
            batch_size=batch_size,
            shuffle=should_shuffle,
        )

        # The scaled tensors are now in the dataloader, so we can free their memory
        free_memory(temporal_scaled, static_scaled, targets_scaled)

        # --- 4. Store the final dataloader in the dictionary ---
        dataloaders[name] = dataloader
        print(f"Created dataloader for '{name}'.\n")

    print("--- All Dataloaders Created ---")
    return dataloaders


class NpzDataset(Dataset):
    def __init__(self, npz_path, scaler_dir):
        """
        Initializes the Dataset by loading arrays from a single .npz file.

        Args:
            npz_path (str): Path to the .npz file.
        """
        # Load the .npz file archive
        with np.load(npz_path) as archive:
            # Extract the arrays using their keys
            self.temporal = archive["temporal"]
            self.static = archive["static"]
            self.targets = archive["targets"]

        self.temporal_scalers = joblib.load(
            os.path.join(scaler_dir, "reservoir_scalers_dict.pkl")
        )
        self.static_scaler = joblib.load(
            os.path.join(scaler_dir, "reservoir_static_scalers_dict.pkl")
        )
        self.target_scaler = self.temporal_scalers[
            "streamflow"
        ]  # Assuming target is streamflow

        sample_nc_path = "/ec/res4/scratch/ecm7072/caravan/timeseries/netcdf/hysets/hysets_06324500.nc"
        with xr.open_dataset(sample_nc_path) as ds_sample:
            if "potential_evaporation_sum_ERA5_LAND" in ds_sample:
                ds_sample = ds_sample.rename(
                    {"potential_evaporation_sum_ERA5_LAND": "potential_evaporation_sum"}
                )
            temporal_variable_names = sorted([var for var in ds_sample.data_vars])
            temporal_variable_names.remove("streamflow")

        self.temporal_variable_names = (
            temporal_variable_names  # List of feature names in order
        )

        self.scaler_dir = scaler_dir

    def __getitem__(self, idx):
        """
        Fetches one sample from the arrays loaded in memory.
        """
        temporal_np = self.temporal[idx]
        static_np = self.static[idx]
        target_np = self.targets[idx]

        # 2. Scale the single sample
        # Scale temporal data feature by feature
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scaled_temporal = np.empty_like(temporal_np, dtype=np.float32)
            for i, var_name in enumerate(self.temporal_variable_names):
                if var_name in self.temporal_scalers:
                    scaler = self.temporal_scalers[var_name]
                    # Reshape for scaler, transform, and reshape back
                    scaled_temporal[:, i] = scaler.transform(
                        temporal_np[:, i].reshape(-1, 1)
                    ).flatten()
                else:
                    scaled_temporal[:, i] = temporal_np[
                        :, i
                    ]  # Pass-through if no scaler

            # Scale static data
            # Reshape to (1, n_features), transform, and remove the extra dimension
            scaled_static = self.static_scaler.transform(
                static_np.reshape(1, -1)
            ).squeeze()

            # Scale target data
            scaled_target = self.target_scaler.transform(
                target_np.reshape(-1, 1)
            ).flatten()

        # (temporal_np_scaled, static_np_scaled, targets_np_scaled, self.static_input_size) = scale_data_np(
        #     temporal_np=temporal_np, static_np=static_np, targets_np=targets_np, scaler_dir=self.scaler_dir)

        # 3. Convert final scaled NumPy arrays to PyTorch tensors and move to device
        temporal = torch.from_numpy(scaled_temporal)
        static = torch.from_numpy(scaled_static)
        targets = torch.from_numpy(scaled_target)

        return temporal, static, targets

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.targets)
