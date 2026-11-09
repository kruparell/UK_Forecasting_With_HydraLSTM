import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler


import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
# Use insert(0) instead of append() so this folder always takes precedence
sys.path.insert(0, str(functions_path))
sys.path.append(str(functions_path))


import ML_functions
print(">>> Loaded ML_functions from:", ML_functions.__file__)

from ML_functions import TensorCatchmentDataset
from ML_Training_Runner import run_lstm_grid_search


N_WORKERS = 0 

def load_and_clean_tensor_data(filepath: str) -> dict:
    data = torch.load(filepath)
    valid_indices = [
        i for i, t in enumerate(data["target"]) 
        if not torch.isnan(t).any()
    ]
    dropped = len(data["target"]) - len(valid_indices)
    print(f"[{Path(filepath).name}] Kept {len(valid_indices)}/{len(data['target'])} samples (dropped {dropped} with NaN).")
    
    return {
        "static": [data["static"][i] for i in valid_indices],
        "temporal": [data["temporal"][i] for i in valid_indices],
        "target": [data["target"][i] for i in valid_indices],
    }

train_res = TensorCatchmentDataset(load_and_clean_tensor_data("/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_res_tensor_data.pt"))
train_non = TensorCatchmentDataset(load_and_clean_tensor_data("/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"))
val_res = TensorCatchmentDataset(load_and_clean_tensor_data("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_res_tensor_data.pt"))
val_non = TensorCatchmentDataset(load_and_clean_tensor_data("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt"))

# train_loader = DataLoader(ConcatDataset([train_res, train_non]), batch_size=256, shuffle=True, num_workers=N_WORKERS)

N_SAMPLES = int(2e5)
train_loader = DataLoader(ConcatDataset([train_res, train_non]), batch_size=256, num_workers=N_WORKERS,
                         sampler=RandomSampler(ConcatDataset([train_res, train_non]), num_samples=N_SAMPLES) )



val_loader = DataLoader(ConcatDataset([val_res, val_non]), batch_size=256, shuffle=True, num_workers=N_WORKERS)

grid = {
    "gamma": [0.95],
    "lstm_num_layers": [1, 2],
    "embedding_output_size": [16, 32],
    "lstm_hidden_size": [128, 256],
    "embedding_hidden_size": [128],
    "dropout": [0.0, 0.2],
    "learning_rate": [1e-3],
}

run_lstm_grid_search(
    model_name="Full",
    train_loader=train_loader,
    val_loader=val_loader,
    hyperparam_grid=grid,
    save_dir="/home/mokr/Reservoir_Modelling/CamelsGB_Models/Full",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    loss_interval=500,
)