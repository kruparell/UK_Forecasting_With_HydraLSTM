"""
Reusable functions for no-flow hydrological model evaluation.

This module loads the configured models and data, runs the existing project
evaluation functions, and writes results to the configured shelve databases.
"""

import gc
import sys
from pathlib import Path

import joblib
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import evaluation_config as config


def _add_project_functions_to_path():
    path = str(config.FUNCTIONS_PATH)
    if path not in sys.path:
        sys.path.append(path)


def load_complete_model(path):
    """Load a complete serialized PyTorch model object onto the configured device."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    return torch.load(
        path,
        weights_only=False,
        map_location=torch.device(config.DEVICE),
    )


def load_selected_models(selected_names=None):
    """
    Load only the selected models.

    Returns parallel lists of model names and model objects, as expected by
    lagged_model_predictions().
    """
    _add_project_functions_to_path()
    from ML_Models import HydraModel

    if selected_names is None:
        selected_names = config.MODELS_TO_EVALUATE

    valid_names = set(config.MODEL_CHECKPOINTS) | {
        "Hydra_Main",
        "Hydra_ResHead",
        "Hydra_NonResHead",
    }
    unknown = set(selected_names) - valid_names
    if unknown:
        raise ValueError(f"Unknown model selection(s): {sorted(unknown)}")

    loaded = {}

    # Standard embedded LSTM models.
    for name in selected_names:
        if name in config.MODEL_CHECKPOINTS:
            loaded[name] = load_complete_model(config.MODEL_CHECKPOINTS[name])

    # Load shared Hydra body only if at least one Hydra model was selected.
    selected_hydra = set(selected_names) & {
        "Hydra_Main",
        "Hydra_ResHead",
        "Hydra_NonResHead",
    }

    if selected_hydra:
        full_base = config.HYDRA_FULL_BASE_PATH
        nonres_base = config.HYDRA_NONRES_BASE_PATH

        body = load_complete_model(Path(f"{full_base}_Body.pt"))

        if "Hydra_Main" in selected_hydra:
            main_head = load_complete_model(Path(f"{full_base}_Main_Head.pt"))
            loaded["Hydra_Main"] = HydraModel(body, main_head)

        if "Hydra_ResHead" in selected_hydra:
            res_head = load_complete_model(
                Path(f"{full_base}_Res_Embedding_Head.pt")
            )
            loaded["Hydra_ResHead"] = [body, res_head]

        if "Hydra_NonResHead" in selected_hydra:
            nonres_head = load_complete_model(
                Path(f"{nonres_base}_NonRes_Embedding_Head.pt")
            )
            loaded["Hydra_NonResHead"] = [body, nonres_head]

    # Retain the user's requested order, even when only a subset is selected.
    ordered_names = [name for name in selected_names if name in loaded]
    return ordered_names, [loaded[name] for name in ordered_names]


def load_test_dataloader(tensor_path):
    """Load one catchment group's tensor dataset and create its DataLoader."""
    _add_project_functions_to_path()
    from ML_functions import TensorCatchmentDataset

    tensor_data = torch.load(
        tensor_path,
        weights_only=False,
        map_location="cpu",
    )
    dataset = TensorCatchmentDataset(tensor_data)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )


def load_streamflow_scaler():
    """Reconstruct the streamflow scaler from the hydrometeorological scaler."""
    temporal_scalers = joblib.load(config.SCALER_PATH)

    scaler = StandardScaler()
    # Streamflow is the second-to-last temporal feature in the existing pipeline.
    scaler.mean_ = temporal_scalers.mean_[[-2]]
    scaler.var_ = temporal_scalers.var_[[-2]]
    scaler.scale_ = temporal_scalers.scale_[[-2]]
    return scaler


def evaluate_catchment_group(
    group_name,
    shelf_key,
    dataloader,
    metadata_path,
    model_names,
    models,
    scaler,
):
    """Run no-flow lagged predictions and rolling metrics for one catchment group."""
    import shelve

    _add_project_functions_to_path()
    from ML_Training import lagged_model_predictions, calculate_rolling_metrics

    print(f"\nEvaluating {group_name} catchments (no river flow)...")

    metadata = torch.load(
        metadata_path,
        weights_only=False,
        map_location="cpu",
    )

    lagged_results = lagged_model_predictions(
        model_names,
        models,
        dataloader,
        streamflow_scaler=scaler,
        warmup=config.NOFLOW_WARMUP,
        device=config.DEVICE,
        time_interval=config.TIME_INTERVAL,
        max_batches=config.MAX_BATCHES,
        use_flow=False,
    )

    print(f"Calculating rolling metrics for {group_name}...")
    rolling_metrics = calculate_rolling_metrics(
        lagged_results,
        metadata,
        window_size=config.WINDOW_SIZE,
    )

    with shelve.open(str(config.LAGGED_RESULTS_PATH)) as db:
        db[shelf_key] = lagged_results

    with shelve.open(str(config.METRICS_PATH)) as db:
        db[shelf_key] = rolling_metrics

    print(f"Saved '{shelf_key}' to both shelve databases.")

    del lagged_results, rolling_metrics, metadata
    gc.collect()
