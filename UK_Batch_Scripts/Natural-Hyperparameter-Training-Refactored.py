import torch
from torch.utils.data import DataLoader, RandomSampler

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))


from ML_functions import TensorCatchmentDataset
from ML_Training_Runner import run_lstm_grid_search

train_dataset = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"))
val_dataset = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt"))

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
N_WORKERS = 0
N_SAMPLES = int(2e5)
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=N_WORKERS,
                         sampler=RandomSampler(train_dataset, num_samples=N_SAMPLES))


val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=N_WORKERS)

grid = {
    "gamma": [0.95],
    "lstm_num_layers": [1, 2],
    "embedding_output_size": [16],
    "lstm_hidden_size": [128, 256],
    "embedding_hidden_size": [64, 128],
    "dropout": [0.1, 0.4],
    "learning_rate": [1e-3],
}

run_lstm_grid_search(
    model_name="Natural",
    train_loader=train_loader,
    val_loader=val_loader,
    hyperparam_grid=grid,
    save_dir="/home/mokr/Reservoir_Modelling/CamelsGB_Models/Natural",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)