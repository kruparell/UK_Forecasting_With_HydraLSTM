from itertools import product
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))


from ML_functions import NSELoss, TensorCatchmentDataset
from ML_Models import HydraModel, LSTMModel, LSTM_With_Embedding
from ML_Training_Runner import extract_input_dimensions, save_and_display_results, train_with_early_stopping

# 1. Datasets & Loaders
train_res = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_res_tensor_data.pt"))
train_non = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_train_non_res_tensor_data.pt"))
val_res = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_res_tensor_data.pt"))
val_non = TensorCatchmentDataset(torch.load("/perm/mokr/CamelsGB_V2/365_0209_no_lat_val_non_res_tensor_data.pt"))

N_WORKERS = 0
N_SAMPLES = int(2e5)
# res_train_loader = DataLoader(train_res, batch_size=256, shuffle=True, num_workers=N_WORKERS)
res_train_loader = DataLoader(train_res, batch_size=256, num_workers=N_WORKERS,
                         sampler=RandomSampler(train_res, num_samples=N_SAMPLES))

res_val_loader = DataLoader(val_res, batch_size=256, shuffle=True, num_workers=N_WORKERS)
# nonres_train_loader = DataLoader(train_non, batch_size=256, shuffle=True, num_workers=N_WORKERS)
nonres_train_loader = DataLoader(train_non, batch_size=256, num_workers=N_WORKERS,
                         sampler=RandomSampler(train_non, num_samples=N_SAMPLES))


nonres_val_loader = DataLoader(val_non, batch_size=256, shuffle=True, num_workers=N_WORKERS)
# full_train_loader = DataLoader(ConcatDataset([train_res, train_non]), batch_size=256, shuffle=True, num_workers=N_WORKERS)
full_train_loader = DataLoader(ConcatDataset([train_res, train_non]), batch_size=256, num_workers=N_WORKERS,
                         sampler=RandomSampler([train_res, train_non], num_samples=N_SAMPLES))


full_val_loader = DataLoader(ConcatDataset([val_res, val_non]), batch_size=256, shuffle=True, num_workers=N_WORKERS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_SIZE, TEMPORAL_SIZE = extract_input_dimensions(res_train_loader)

hyperparam_grid = {
    "gamma": [0.8, 0.95],
    "embedding_output_size": [16],
    "lstm_hidden_size": [256],
    "hydra_body_output_size": [128],
    "embedding_hidden_size": [128],
    "res_head_learning_rate": [1e-3, 1e-4],
    "learning_rate": [1e-3],
}

save_dir = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/"
results = []
combinations = [dict(zip(hyperparam_grid.keys(), v)) for v in product(*hyperparam_grid.values())]

for params in combinations:
    try:
        body = LSTM_With_Embedding(
            static_input_size=STATIC_SIZE,
            embedding_hidden_size=params["embedding_hidden_size"],
            embedding_output_size=params["embedding_output_size"],
            lstm_input_size=TEMPORAL_SIZE + params["embedding_output_size"],
            lstm_hidden_size=params["lstm_hidden_size"],
            lstm_num_layers=1,
            lstm_output_size=params["hydra_body_output_size"],
            dropout=0.3,
            Sequence_Target=True,
        ).to(DEVICE)

        head_main = LSTMModel(params["hydra_body_output_size"], params["lstm_hidden_size"], 1, 1, dropout=0.3, Sequence_Target=True).to(DEVICE)
        head_nonres = LSTM_With_Embedding(STATIC_SIZE, params["embedding_hidden_size"], params["embedding_output_size"], params["hydra_body_output_size"] + params["embedding_output_size"], params["lstm_hidden_size"], 1, 1, dropout=0.3, Sequence_Target=True).to(DEVICE)
        head_res = LSTM_With_Embedding(STATIC_SIZE, params["embedding_hidden_size"], params["embedding_output_size"], params["hydra_body_output_size"] + params["embedding_output_size"], params["lstm_hidden_size"], 1, 1, dropout=0.3, Sequence_Target=True).to(DEVICE)

        hydra_main = HydraModel(body, head_main).to(DEVICE)
        criterion = NSELoss()

        # Phase 1: Train Body + Main Head on full data
        opt_main = optim.Adam(list(hydra_main.body.parameters()) + list(hydra_main.head_default.parameters()), lr=params["learning_rate"])
        sched_main = optim.lr_scheduler.ExponentialLR(opt_main, gamma=params["gamma"])
        res_main = train_with_early_stopping(hydra_main, full_train_loader, full_val_loader, criterion, opt_main, sched_main, 8, 10, 500, DEVICE, model_tag="Hydra-Main")

        # Freeze body
        body.eval()
        for p in body.parameters():
            p.requires_grad = False

        # Phase 2: Train Non-Reservoir Head
        opt_nonres = optim.Adam(head_nonres.parameters(), lr=params["res_head_learning_rate"])
        sched_nonres = optim.lr_scheduler.ExponentialLR(opt_nonres, gamma=params["gamma"])
        res_nonres = train_with_early_stopping([body, head_nonres], nonres_train_loader, nonres_val_loader, criterion, opt_nonres, sched_nonres, 8, 10, 500, DEVICE, model_tag="NonRes-Head")

        # Phase 3: Train Reservoir Head
        opt_res = optim.Adam(head_res.parameters(), lr=params["res_head_learning_rate"])
        sched_res = optim.lr_scheduler.ExponentialLR(opt_res, gamma=params["gamma"])
        res_res = train_with_early_stopping([body, head_res], res_train_loader, res_val_loader, criterion, opt_res, sched_res, 8, 10, 500, DEVICE, model_tag="Res-Head")

        model_save_path = (
            f"{save_dir}/Hydra_Body_Main_NonRes_Res_Epochs{res_res['best_epoch']}"
            f"_LR{params['learning_rate']}_ResLR{params['res_head_learning_rate']}.pt"
        )
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save a dictionary containing the state dictionaries of all components
        base = (
            f"{save_dir}/365_0209_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs{params['lstm_hidden_size']}"
            f"_Embhs{params['embedding_hidden_size']}_Embos{params['embedding_output_size']}"
            f"_BodyOut{params['hydra_body_output_size']}_MainEp{res_main['best_epoch']}"
            f"_ResEp{res_res['best_epoch']}_ResLR{params['res_head_learning_rate']}"
            f"_Gamma{params['gamma']}_{params['learning_rate']}_{params['res_head_learning_rate']}"
        )
        Path(base).parent.mkdir(parents=True, exist_ok=True)

        def cpu_copy(m):
            return copy.deepcopy(m).cpu().eval()

        torch.save(cpu_copy(hydra_main.body),         f"{base}_Body.pt")
        torch.save(cpu_copy(hydra_main.head_default), f"{base}_Main_Head.pt")
        torch.save(cpu_copy(head_res),                f"{base}_Res_Embedding_Head.pt")
        torch.save(cpu_copy(head_nonres),             f"{base}_NonRes_Embedding_Head.pt")
        print(f"  ✓ Saved Hydra components with base path: {base}")
        
        model_save_path = base
        
        results.append({
            **params,
            "train_loss_main": res_main["last_train_loss"],
            "val_loss_main": res_main["best_val_loss"],
            "val_loss_res": res_res["best_val_loss"],
            "model_path": model_save_path, # Optional: track path in results
            "status": "success",
        })

    except Exception as e:
        print(f"Hydra trial failed: {e}")
        results.append({**params, "status": "failed", "error": str(e)})
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

save_and_display_results(results, sort_col="val_loss_res", csv_path=f"{save_dir}/hydra_hyperparameter_search_results.csv")