import mlflow
import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))

from ML_functions import TensorCatchmentDataset

from ML_Models import LSTM_With_Embedding
from ML_Training import train_model
import ML_functions


mlflow.set_tracking_uri("http://127.0.0.1:2945")
mlflow.set_experiment('mlflow_reservoir')
mlflow.pytorch.autolog()


train_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/train_res_tensor_data.pt")
train_res_dataset = TensorCatchmentDataset(train_res_tensor_data)

# Make Scotland Test
val_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/ScotlandPlus_val_res_tensor_data.pt")
val_res_dataset = TensorCatchmentDataset(val_res_tensor_data)


## Experimnet to make things faster
# torch.set_num_threads(8)  # Adjust based on your CPU cores
# torch.set_num_interop_threads(2)

## Fixed Parameters
BATCH_SIZE = 256
NUM_WORKERS = 1
LSTM_OUTPUT_SIZE = 1
# GAMMA = 0.9
warmup = 10
loss_interval = 5e4  # Show loss after this many samples


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


train_reservoir_dataloader = DataLoader(
    train_res_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)  # , pin_memory=True , prefetch_factor=2

val_reservoir_dataloader = DataLoader(
    val_res_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=2,
)  # , pin_memory=True, persistent_workers=True

for i, batch in enumerate(train_reservoir_dataloader):
    if i == 0:
        print(f"Static shape: {batch['static'].shape}")
        print(f"Temporal shape: {batch['temporal'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"\nProcessing 100 batches...")
    break

STATIC_INPUT_SIZE = batch["static"].shape[1]
TEMPORAL_INPUT_SIZE = batch["temporal"].shape[-1]


## Currently Fixed Pararmeters
dropout = 0.4
num_epochs = 8
lstm_h_size = 256
lr = 1e-3
use_flow = True

# --- Hyperparameter Search Setup ---
hyperparam_grid = {
    "gamma": [0.8, 0.95],  # Checked 0.8, 0.95 is better almost always
    "lstm_num_layers": [1, 2],
    "embedding_output_size": [16, 32],
    "lstm_hidden_size": [128, 256],
    "embedding_hidden_size": [64, 128],
}


# Store results
results = []

# Generate all combinations of hyperparameters
param_names = list(hyperparam_grid.keys())
param_values = list(hyperparam_grid.values())
all_combinations = list(product(*param_values))

print(f"Total hyperparameter combinations to test: {len(all_combinations)}")
mlflow.enable_system_metrics_logging()
# Iterate through all hyperparameter combinations
for combo in all_combinations:
    params = dict(zip(param_names, combo))
    run_name = f"emb_o{params['embedding_output_size']}_lstm_h{params['lstm_hidden_size']}_layers{params['lstm_num_layers']}_gamma{params['gamma']}"
    
    with mlflow.start_run(run_name=run_name):    
        # Unpack parameters
        emb_o_size = params["embedding_output_size"]
        lstm_h_size = params["lstm_hidden_size"]
        lstm_n_layers = params["lstm_num_layers"]
        emb_h_size = params["embedding_hidden_size"]
        gamma_val = params["gamma"]
        
        # FIXED: Log all hyperparameters
        mlflow.log_param("embedding_output_size", emb_o_size)
        mlflow.log_param("lstm_hidden_size", lstm_h_size)
        mlflow.log_param("lstm_num_layers", lstm_n_layers)
        mlflow.log_param("embedding_hidden_size", emb_h_size)
        mlflow.log_param("gamma", gamma_val)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("use_flow", use_flow)
        try:
            # 1. Initialize Model
            lstm_input_size = TEMPORAL_INPUT_SIZE + emb_o_size + int(use_flow)
    
            Reservoir_LSTM = LSTM_With_Embedding(
                static_input_size=STATIC_INPUT_SIZE,
                embedding_hidden_size=emb_h_size,
                embedding_output_size=emb_o_size,
                lstm_input_size=lstm_input_size,
                lstm_hidden_size=lstm_h_size,
                lstm_num_layers=lstm_n_layers,
                lstm_output_size=LSTM_OUTPUT_SIZE,
                dropout=dropout,
                bidirectional=False,
                Sequence_Target=True,
            )
            Reservoir_LSTM.to(DEVICE)
    
            # 2. Define Loss Function and Optimizer
            criterion = ML_functions.NSELoss()
            optimizer = optim.Adam(Reservoir_LSTM.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_val)
    
            # 3. Train the model
            # print("  Starting training...")
            current_val_loss = np.inf
            for epoch in range(num_epochs):
                print(f"  Epoch {epoch+1}/{num_epochs}")
                train_loss = train_model(
                    Reservoir_LSTM,
                    train_reservoir_dataloader,
                    criterion,
                    optimizer,
                    scheduler,
                    1,
                    warmup,
                    loss_interval,
                    DEVICE,
                    train_mode=True,
                    time_interval=360,
                    static_step=True,
                    use_flow=use_flow,
                )
    
                # 5. Evaluate on Validation Set
                # print("  Starting validating...")
                val_loss = train_model(
                    Reservoir_LSTM,
                    val_reservoir_dataloader,
                    criterion,
                    optimizer,
                    scheduler,
                    1,
                    warmup,
                    loss_interval,
                    DEVICE,
                    train_mode=False,
                    time_interval=360,
                    static_step=True,
                    use_flow=use_flow,
                )
                print(f"  Validation Loss: {val_loss}")
                mlflow.log_metric("loss", val_loss, step=epoch)
                if val_loss < current_val_loss:
                    current_val_loss = val_loss
                    if use_flow:
                        model_save_path = f'/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Reservoir/ScotlandPlus_Reservoir_LSTM_Numlayers{lstm_n_layers}_LSTMhs{lstm_h_size}_Embhs{emb_h_size}_Embos{emb_o_size}_Epochs{epoch}_Gamma{gamma_val}.pt'
    
                    else:
                        model_save_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Reservoir/ScotlandPlus_Reservoir_LSTM_Numlayers{lstm_n_layers}_LSTMhs{lstm_h_size}_Embhs{emb_h_size}_Embos{emb_o_size}_Epochs{epoch}_Gamma{gamma_val}.pt"
                        
    
                    mlflow.log_artifact(model_save_path, artifact_path="models")
                    print(f"  Model saved to {model_save_path}")
                else:
                    print(f"  No improvement, stopping early at epoch {epoch+1}")
                    break
    
            # Store results
            result_dict = {
                "embedding_output_size": emb_o_size,
                "embedding_hidden_size": emb_h_size,
                "lstm_hidden_size": lstm_h_size,
                "lstm_num_layers": lstm_n_layers,
                "gamma": gamma_val,
                "learning_rate": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_path": model_save_path,
                "status": "success",
            }
            results.append(result_dict)
    
        except Exception as e:
            print(f"  ERROR: Training/Validation failed: {e}")
            # Store failed result
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            result_dict = {
                "embedding_output_size": emb_o_size,
                "embedding_hidden_size": emb_h_size,
                "lstm_hidden_size": lstm_h_size,
                "lstm_num_layers": lstm_n_layers,
                "gamma": gamma_val,
                "learning_rate": lr,
                "train_loss": None,
                "val_loss": None,
                "model_path": None,
                "status": "failed",
                "error": str(e),
            }
            results.append(result_dict)

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by validation loss (best first, NaN values last)
results_df = results_df.sort_values("val_loss", ascending=True, na_position="last")

# Display results
print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))

# Save to CSV
csv_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Reservoir/ScotlandPlus_{use_flow}_flow_hyperparameter_search_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Print best hyperparameters (only if there are successful runs)
successful_results = results_df[results_df["status"] == "success"]
if len(successful_results) > 0:
    best_idx = successful_results["val_loss"].idxmin()
    best_params = results_df.loc[best_idx]
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"Validation Loss: {best_params['val_loss']}")
    print(f"Train Loss: {best_params['train_loss']}")
    print(f"Embedding Output Size: {best_params['embedding_output_size']}")
    print(f"Embedding Hidden Size: {best_params['embedding_hidden_size']}")
    print(f"LSTM Hidden Size: {best_params['lstm_hidden_size']}")
    print(f"LSTM Num Layers: {best_params['lstm_num_layers']}")
    # print(f"Num Epochs: {best_params['num_epochs']}")
    print(f"Gamma: {best_params['gamma']}")
    print(f"Model Path: {best_params['model_path']}")
else:
    print("\n⚠ No successful runs to report best hyperparameters.")
