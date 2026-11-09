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

train_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"
)
train_non_res_dataset = TensorCatchmentDataset(train_non_res_tensor_data)

# Make scotland test set
val_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt"
)
val_non_res_dataset = TensorCatchmentDataset(val_non_res_tensor_data)
## Fixed Parameters
BATCH_SIZE = 256
NUM_WORKERS = 2
LSTM_OUTPUT_SIZE = 1
loss_interval = 5e4  # Show loss after this many samples

## Currently Fixed Pararmeters
dropout = 0.4
num_epochs = 8
lstm_h_size = 256
# lr = 1e-3
use_flow = False
warmup = 10 #89 #10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


train_non_reservoir_dataloader = DataLoader(
    train_non_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

val_non_reservoir_dataloader = DataLoader(
    val_non_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

for i, batch in enumerate(train_non_reservoir_dataloader):
    if i == 0:
        # print(f"Static shape: {batch['static'].shape}")
        print(f"Temporal shape: {batch['temporal'].shape}")
        # print(f"Target shape: {batch['target'].shape}")
        # print(f"\nProcessing 100 batches...")
    break

STATIC_INPUT_SIZE = batch["static"].shape[1]
TEMPORAL_INPUT_SIZE = batch["temporal"].shape[-1]



# --- Hyperparameter Search Setup ---
hyperparam_grid = {
    "gamma": [0.8, 0.95],  # Checked 0.8, 0.95 is better almost always
    "lstm_num_layers": [1, 2],
    "embedding_output_size": [16, 32],
    "lstm_hidden_size": [128, 256],
    "embedding_hidden_size": [64, 128],
    "dropout": [0.1, 0.4],
    "learning_rate": [1e-3, 1e-4],
}

# Store results
results = []

# Generate all combinations of hyperparameters
param_names = list(hyperparam_grid.keys())
param_values = list(hyperparam_grid.values())
all_combinations = list(product(*param_values))

print(f"Total hyperparameter combinations to test: {len(all_combinations)}")

# Iterate through all hyperparameter combinations
for combo in all_combinations:
    # Unpack parameters
    params = dict(zip(param_names, combo))
    emb_o_size = params["embedding_output_size"]
    lstm_h_size = params["lstm_hidden_size"]
    lstm_n_layers = params["lstm_num_layers"]
    emb_h_size = params["embedding_hidden_size"]
    gamma_val = params["gamma"]
    dropout = params["dropout"]
    lr = params["learning_rate"]
    
    try:
        # 1. Initialize Model
        lstm_input_size = TEMPORAL_INPUT_SIZE + emb_o_size + int(use_flow)

        Natural_LSTM = LSTM_With_Embedding(
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
        Natural_LSTM.to(DEVICE)

        # 2. Define Loss Function and Optimizer
        criterion = ML_functions.NSELoss()
        has_nan_temporal = any(torch.isnan(t).any() for t in val_non_res_tensor_data['temporal'])
        has_nan_target = any(torch.isnan(t).any() for t in val_non_res_tensor_data['target'])
        print(f"NaN in val temporal: {has_nan_temporal}, NaN in val target: {has_nan_target}")
        
        optimizer = optim.Adam(Natural_LSTM.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_val)

        # 3. Train the model
        # print("  Starting training...")

        current_val_loss = np.inf
        train_loss = None
        model_save_path = None
        epoch = 0

        for epoch in range(num_epochs):
            print(f"  Epoch {epoch+1}/{num_epochs}")
            train_loss = train_model(
                Natural_LSTM,
                train_non_reservoir_dataloader,
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
                Natural_LSTM,
                val_non_reservoir_dataloader,
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

            if val_loss < current_val_loss:
                current_val_loss = val_loss

                if use_flow:
                    ## With Flow
                    model_save_path = f'/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Natural/365_0209_No_Lat_Natural_LSTM_Numlayers{lstm_n_layers}_LSTMhs{lstm_h_size}_Embhs{emb_h_size}_Embos{emb_o_size}_Epochs{epoch+1}_Gamma{gamma_val}_{lr}_{dropout}.pt'

                else:
                    ## Without Flow
                    model_save_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Natural/365_0209_No_Lat_Natural_LSTM_Flow{use_flow}_Numlayers{lstm_n_layers}_LSTMhs{lstm_h_size}_Embhs{emb_h_size}_Embos{emb_o_size}_Epochs{epoch+1}_Gamma{gamma_val}_{lr}_{dropout}.pt"

                torch.save(Natural_LSTM, model_save_path)
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
            "num_epochs": epoch + 1,
            "gamma": gamma_val,
            "learning_rate": lr,
            "train_loss": train_loss,
            "val_loss": current_val_loss,
            "model_path": model_save_path,
        }
        results.append(result_dict)
        
    except Exception as e:
        print(f"  ERROR: Training/Validation failed: {e}")
        # Store failed result
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
            "error": str(e),
        }
        results.append(result_dict)

    finally:
            # Prevent VRAM buildup across combinations
            for var in ["Natural_LSTM", "optimizer", "scheduler", "criterion"]:
                if var in locals():
                    del locals()[var]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by validation loss (best first)
results_df = results_df.sort_values("val_loss", ascending=True, na_position="last")

# Display results
print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))

# Save to CSV
csv_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Natural/365_0209_{use_flow}_flow_hyperparameter_search_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Print best hyperparameters
best_idx = results_df["val_loss"].idxmin()
best_params = results_df.loc[best_idx]
print("\n" + "=" * 80)
print("BEST HYPERPARAMETERS")
print("=" * 80)
print(f"Validation Loss: {best_params['val_loss']}")
print(f"Embedding Output Size: {best_params['embedding_output_size']}")
print(f"Embedding Hidden Size: {best_params['embedding_hidden_size']}")
print(f"LSTM Hidden Size: {best_params['lstm_hidden_size']}")
print(f"LSTM Num Layers: {best_params['lstm_num_layers']}")
# print(f"Num Epochs: {best_params['num_epochs']}")
print(f"Gamma: {best_params['gamma']}")
print(f"Learning Rate: {best_params['learning_rate']}")
print(f"Model Path: {best_params['model_path']}")
