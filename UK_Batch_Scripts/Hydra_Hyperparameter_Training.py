import os
import numpy as np
import pandas as pd
from itertools import product
import copy

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))

from ML_functions import TensorCatchmentDataset

import ML_Models
from ML_Models import LSTM_With_Embedding, LSTMModel, HydraModel
from ML_Training import train_model

from ML_functions import NSELoss

QUICK_TEST = True            # set False for the real run
N_SAMPLES  = 500 if QUICK_TEST else 200_000
NUM_EPOCHS = 1     if QUICK_TEST else 60


# Create dataloaders
train_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"
)
train_non_res_dataset = TensorCatchmentDataset(train_non_res_tensor_data)

train_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_res_tensor_data.pt")
train_res_dataset = TensorCatchmentDataset(train_res_tensor_data)

val_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt"
)
val_non_res_dataset = TensorCatchmentDataset(val_non_res_tensor_data)

val_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_res_tensor_data.pt")
val_res_dataset = TensorCatchmentDataset(val_res_tensor_data)

train_full_dataset = ConcatDataset([train_res_dataset, train_non_res_dataset])
val_full_dataset = ConcatDataset([val_res_dataset, val_non_res_dataset])

## Fixed Parameters
BATCH_SIZE = 256
NUM_WORKERS = 1
LSTM_OUTPUT_SIZE = 1
GAMMA = 0.95
loss_interval = 500  # Perform optimizer step after this many samples
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Currently Fixed Pararmeters
dropout = 0.3
num_epochs = 8
lstm_n_layers = 1
warmup = 10 #89
lstm_h_size = 256
lr = 1e-3
use_flow = False


train_reservoir_dataloader = DataLoader(
    train_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

val_reservoir_dataloader = DataLoader(
    val_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True


train_non_reservoir_dataloader = DataLoader(
    train_non_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

val_non_reservoir_dataloader = DataLoader(
    val_non_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True



train_full_dataloader = DataLoader(
    train_full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

val_full_dataloader = DataLoader(
    val_full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

for i, batch in enumerate(train_reservoir_dataloader):
    break

STATIC_INPUT_SIZE = batch["static"].shape[1]
TEMPORAL_INPUT_SIZE = batch["temporal"].shape[-1]


hyperparam_grid = {
    "gamma": [0.95],  #0.8, 
    "embedding_output_size": [16],
    "lstm_hidden_size": [256], # 128, 
    "hydra_body_output_size": [128], # 64,
    "embedding_hidden_size": [128], # 64, 
    "res_head_learning_rate": [1e-3, 1e-4],
    "learning_rate": [1e-3], #, 1e-4
}


# Store results
results = []

full_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/365_0209_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp3_ResEp3_ResLR0.001_Gamma0.95_0.001_0.001"
 


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
    emb_h_size = params["embedding_hidden_size"]
    hydra_bo_size = params["hydra_body_output_size"]
    gamma_val = params["gamma"]
    res_lr = params["res_head_learning_rate"]
    lr = params["learning_rate"]

    try:
        # 1. Initialize Model
        body_input_size = TEMPORAL_INPUT_SIZE + emb_o_size + int(use_flow)

        Hydra_Body = LSTM_With_Embedding(
            static_input_size=STATIC_INPUT_SIZE,
            embedding_hidden_size=emb_h_size,
            embedding_output_size=emb_o_size,
            lstm_input_size=body_input_size,
            lstm_hidden_size=lstm_h_size,
            lstm_num_layers=lstm_n_layers,
            lstm_output_size=hydra_bo_size,
            dropout=dropout,
            bidirectional=False,
            Sequence_Target=True,
        )

        Hydra_Head = LSTMModel(
            input_size=hydra_bo_size,
            hidden_size=lstm_h_size,
            num_layers=lstm_n_layers,
            output_size=1,
            dropout=dropout,
            bidirectional=False,
            Sequence_Target=True,
        ) 

        # Non-Reservoir Head with Embedding
        Hydra_NonRes_Head_Embedding = LSTM_With_Embedding(
            static_input_size=STATIC_INPUT_SIZE,
            embedding_hidden_size=emb_h_size,
            embedding_output_size=emb_o_size,
            lstm_input_size=hydra_bo_size + emb_o_size,
            lstm_hidden_size=lstm_h_size,
            lstm_num_layers=lstm_n_layers,
            lstm_output_size=1,
            dropout=dropout,
            bidirectional=False,
            Sequence_Target=True,
        )

        # Reservoir Head with Embedding
        Hydra_Res_Head_Embedding = LSTM_With_Embedding(
            static_input_size=STATIC_INPUT_SIZE,
            embedding_hidden_size=emb_h_size,
            embedding_output_size=emb_o_size,
            lstm_input_size=hydra_bo_size + emb_o_size,
            lstm_hidden_size=lstm_h_size,
            lstm_num_layers=lstm_n_layers,
            lstm_output_size=1,
            dropout=dropout,
            bidirectional=False,
            Sequence_Target=True,
        )

        Hydra_Main = HydraModel(Hydra_Body, Hydra_Head)

        Hydra_Main.to(DEVICE)
        Hydra_Head.to(DEVICE)
        Hydra_NonRes_Head_Embedding.to(DEVICE)
        Hydra_Res_Head_Embedding.to(DEVICE)

        # 2. Define Loss Function and Optimizers
        criterion = NSELoss()

        # Optimizer for Main Hydra (Body + Default Head)
        optimizer_main = optim.Adam(
            list(Hydra_Main.body.parameters()) + list(Hydra_Main.head_default.parameters()),
            lr=lr,
        )

        # Optimizer for Non-Reservoir Head
        optimizer_nonres_embedding_head = optim.Adam(
            Hydra_NonRes_Head_Embedding.parameters(), lr=res_lr
        )

        # Optimizer for Reservoir Head
        optimizer_res_embedding_head = optim.Adam(
            Hydra_Res_Head_Embedding.parameters(), lr=res_lr
        )

        # Schedulers
        scheduler_main = optim.lr_scheduler.ExponentialLR(optimizer_main, gamma=gamma_val)
        scheduler_nonres_embedding_head = optim.lr_scheduler.ExponentialLR(optimizer_nonres_embedding_head, gamma=gamma_val)
        scheduler_res_embedding_head = optim.lr_scheduler.ExponentialLR(optimizer_res_embedding_head, gamma=gamma_val)
    

    #     Hydra_Body = torch.load(
    # f"{full_base_path}_Body.pt", weights_only=False, map_location=torch.device("cpu")
    #     )
    #     for param in Hydra_Body.parameters():
    #         param.requires_grad = False
    #     Hydra_Body.eval()

        # 3. Train Hydra_Main on full data
        current_val_loss_main = np.inf
        best_epoch_main = 0
        patience_counter = 0
        patience = 2

        for epoch in range(num_epochs):
            train_loss_main = train_model(
                Hydra_Main,
                train_full_dataloader, #train_non_reservoir_dataloader
                criterion,
                optimizer_main,
                scheduler_main,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=True,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            # Validate
            val_loss_main = train_model(
                Hydra_Main,
                val_full_dataloader, # val_non_reservoir_dataloader
                criterion,
                optimizer_main,
                scheduler_main,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=False,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            # Early stopping check
            if val_loss_main < current_val_loss_main:
                current_val_loss_main = val_loss_main
                best_epoch_main = epoch + 1
                # Save best model state
                best_model_state_main = {
                    "body": Hydra_Body.state_dict(),
                    "head": Hydra_Head.state_dict(),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    # Load best model
                    Hydra_Body.load_state_dict(best_model_state_main["body"])
                    Hydra_Head.load_state_dict(best_model_state_main["head"])
                    break

        epoch_main = best_epoch_main

        Hydra_Body.eval()
        for p in Hydra_Body.parameters():
            p.requires_grad = False
        
        # 4. Train Non-Reservoir Head on non-reservoir data
        current_val_loss_nonres = np.inf
        best_epoch_nonres = 0
        patience_counter = 0
        best_model_state_nonres = {}

        for epoch in range(num_epochs):
            train_loss_nonres = train_model(
                [Hydra_Body, Hydra_NonRes_Head_Embedding],
                train_non_reservoir_dataloader,
                criterion,
                optimizer_nonres_embedding_head,
                scheduler_nonres_embedding_head,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=True,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            val_loss_nonres = train_model(
                [Hydra_Body, Hydra_NonRes_Head_Embedding],
                val_non_reservoir_dataloader,
                criterion,
                optimizer_nonres_embedding_head,
                scheduler_nonres_embedding_head,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=False,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            if val_loss_nonres < current_val_loss_nonres:
                current_val_loss_nonres = val_loss_nonres
                best_epoch_nonres = epoch + 1
                best_model_state_nonres = {
                    "nonres_head": Hydra_NonRes_Head_Embedding.state_dict()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    [Non-Res Head] Early stopping at epoch {epoch+1}")
                    Hydra_NonRes_Head_Embedding.load_state_dict(
                        best_model_state_nonres["nonres_head"]
                    )
                    break

        epoch_nonres = best_epoch_nonres

    # 5. Train Reservoir Head on reservoir data
        current_val_loss_res = np.inf
        best_epoch_res = 0
        patience_counter = 0
        best_model_state_res = {}

        for epoch in range(num_epochs):
            train_loss_res = train_model(
                [Hydra_Body, Hydra_Res_Head_Embedding],
                train_reservoir_dataloader,
                criterion,
                optimizer_res_embedding_head,
                scheduler_res_embedding_head,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=True,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            val_loss_res = train_model(
                [Hydra_Body, Hydra_Res_Head_Embedding],
                val_reservoir_dataloader,
                criterion,
                optimizer_res_embedding_head,
                scheduler_res_embedding_head,
                1,
                warmup,
                loss_interval,
                DEVICE,
                train_mode=False,
                time_interval=360,
                static_step=True,
                use_flow=use_flow,
            )

            if val_loss_res < current_val_loss_res:
                current_val_loss_res = val_loss_res
                best_epoch_res = epoch + 1
                best_model_state_res = {
                    "res_head": Hydra_Res_Head_Embedding.state_dict()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    [Res Head] Early stopping at epoch {epoch+1}")
                    Hydra_Res_Head_Embedding.load_state_dict(
                        best_model_state_res["res_head"]
                    )
                    break

        epoch_res = best_epoch_res

        # Save checkpoints
        base_name = f"365_0209_Hydra_L{lstm_n_layers}_H{lstm_h_size}_EmbH{emb_h_size}_EmbO{emb_o_size}_Out{hydra_bo_size}_MainEp{epoch_main}_NonResEp{epoch_nonres}_ResEp{epoch_res}_LR{lr}_ResLR{res_lr}"
        body_path = f"{save_dir}{base_name}_Body.pt"
        head_path = f"{save_dir}{base_name}_Main_Head.pt"
        nonres_emb_path = f"{save_dir}{base_name}_NonRes_Head.pt"
        res_emb_path = f"{save_dir}{base_name}_Res_Head.pt"

        torch.save(Hydra_Body, body_path)
        torch.save(Hydra_Head, head_path)
        torch.save(Hydra_NonRes_Head_Embedding, nonres_emb_path)
        torch.save(Hydra_Res_Head_Embedding, res_emb_path)

        # Store results
        result_dict = {
            "embedding_output_size": emb_o_size,
            "embedding_hidden_size": emb_h_size,
            "lstm_hidden_size": lstm_h_size,
            "lstm_num_layers": lstm_n_layers,
            "hydra_body_output_size": hydra_bo_size,
            "gamma": gamma_val,
            "learning_rate": lr,
            "res_head_learning_rate": res_lr,
            "train_loss_main": train_loss_main,
            "val_loss_main": current_val_loss_main,
            "train_loss_nonres": train_loss_nonres,
            "val_loss_nonres": current_val_loss_nonres,
            "train_loss_res": train_loss_res,
            "val_loss_res": current_val_loss_res,
            "body_path": body_path,
            "head_path": head_path,
            "nonres_emb_path": nonres_emb_path,
            "res_emb_path": res_emb_path,
            "status": "success",
        }
        results.append(result_dict)

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by reservoir validation loss (best first, NaN values last)
results_df = results_df.sort_values("val_loss_res", ascending=True, na_position="last")

# Display results
print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))

# Save to CSV
csv_path = f"/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/{use_flow}_flow_hyperparameter_search_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Print best hyperparameters (only if there are successful runs)
successful_results = results_df[results_df["status"] == "success"]
if len(successful_results) > 0:
    best_idx = successful_results["val_loss_main"].idxmin()
    best_params = results_df.loc[best_idx]
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"Validation Loss (Main): {best_params['val_loss_main']}")
    print(f"Validation Loss (Reservoir): {best_params['val_loss_res']}")
    print(f"Train Loss (Main): {best_params['train_loss_main']}")
    print(f"Train Loss (Reservoir): {best_params['train_loss_res']}")
    print(f"Embedding Output Size: {best_params['embedding_output_size']}")
    print(f"Embedding Hidden Size: {best_params['embedding_hidden_size']}")
    print(f"LSTM Hidden Size: {best_params['lstm_hidden_size']}")
    print(f"LSTM Num Layers: {best_params['lstm_num_layers']}")
    print(f"Hydra Body Output Size: {best_params['hydra_body_output_size']}")
    # print(f"Num Epochs: {best_params['num_epochs']}")
    print(f"Gamma: {best_params['gamma']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Body Path: {best_params['body_path']}")
    print(f"Head Path: {best_params['head_path']}")
    print(f"Res Embedding Path: {best_params['res_emb_path']}")
else:
    print("\n⚠ No successful runs to report best hyperparameters.")
