import os
import sys
import gc
import random
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import shelve

# --- Local Module Configuration ---
FUNCTIONS_PATH = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
if str(FUNCTIONS_PATH) not in sys.path:
    sys.path.append(str(FUNCTIONS_PATH))

# Importing from your custom files
import ML_functions
from ML_functions import TensorCatchmentDataset, NSELoss
from ML_Models import LSTM_With_Embedding, LSTMModel, HydraModel
from ML_Training import (
    validate_model_scores,
    run_model_pipeline,
    process_validation_batches,
    get_basin_name,
    lagged_model_predictions,
    calculate_rolling_metrics,
)
from ML_Metrics import (
    calculate_nse,
    calculate_nnse,
    calculate_kge_components,
    calculate_bias,
)
from ML_Plots import display_results_table, plot_score_cdfs

warmup = 10
max_batches = int(1e7)
# max_batches = int(1e2)

TIME_INTERVAL = 120
WINDOW_SIZE = 365
### Loading Data
test_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/no_lat_test_non_res_tensor_data.pt"
)
test_non_res_dataset = TensorCatchmentDataset(test_non_res_tensor_data)

BATCH_SIZE = 1
NUM_WORKERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_non_reservoir_dataloader = DataLoader(
    test_non_res_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

test_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/no_lat_test_res_tensor_data.pt")
test_res_dataset = TensorCatchmentDataset(test_res_tensor_data)


test_reservoir_dataloader = DataLoader(
    test_res_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

test_res_tensor_metadata = torch.load(
    "/perm/mokr/CamelsGB_V2/test_res_tensor_metadata.pt"
)
test_non_res_tensor_metadata = torch.load(
    "/perm/mokr/CamelsGB_V2/test_non_res_tensor_metadata.pt"
)


### Loading Models

# Not using river flow
No_Flow_Reservoir_Embedded_LSTM = torch.load("/home/mokr/Reservoir_Modelling/CamelsGB_Models/Reservoir/2003_No_Lat_Reservoir_LSTM_Numlayers1_LSTMhs256_Embhs64_Embos16_Epochs0_Gamma0.8_0.001_0.4.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)



No_Flow_NonReservoir_Embedded_LSTM = torch.load(    "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Natural/2003_No_Lat_Natural_LSTM_FlowFalse_Numlayers2_LSTMhs256_Embhs64_Embos16_Epochs2_Gamma0.95_0.001_0.4.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)



No_Flow_Full_Embedded_LSTM = torch.load("/home/mokr/Reservoir_Modelling/CamelsGB_Models/Full/89Warmup_No_Lat__Full_LSTM_Numlayers1_LSTMhs128_Embhs128_Embos16_Epochs2_Gamma0.95.pt_0.001_0",
    weights_only=False,
    map_location=torch.device("cpu"),
)



# base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/Hydra_LSTM_Numlayers1_LSTMhs128_Embhs64_Embos16_BodyOut128_MainEp1_ResEp2_ResLR0.001_Gamma0.8"
# No_Flow_Hydra_Body = torch.load(
#     f"{base_path}_Body.pt", weights_only=False, map_location=torch.device("cpu")
# )
# No_Flow_Hydra_Head = torch.load(
#     f"{base_path}_Main_Head.pt", weights_only=False, map_location=torch.device("cpu")
# )
# No_Flow_Hydra_ResHead = torch.load(
#     f"{base_path}_Res_Embedding_Head.pt",
#     weights_only=False,
#     map_location=torch.device("cpu"),
# )
# No_Flow_Hydra_Main = HydraModel(No_Flow_Hydra_Body, No_Flow_Hydra_Head)

full_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/2003_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp3_ResEp3_ResLR0.001_Gamma0.95_0.001_0.001"

 
# full_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/89Day_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs128_Embhs64_Embos16_BodyOut128_MainEp3_ResEp1_ResLR0.001_Gamma0.8_0.0001_0.001"

No_Flow_Full_Hydra_Body = torch.load(
    f"{full_base_path}_Body.pt", weights_only=False, map_location=torch.device("cpu")
)
No_Flow_Full_Hydra_Head = torch.load(
    f"{full_base_path}_Main_Head.pt", weights_only=False, map_location=torch.device("cpu")
)
No_Flow_Full_Hydra_ResHead = torch.load(
    f"{full_base_path}_Res_Embedding_Head.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)
No_Flow_Full_Hydra_Main = HydraModel(No_Flow_Full_Hydra_Body, No_Flow_Full_Hydra_Head)

nonres_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/2003_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp3_ResEp1_ResLR0.001_Gamma0.95_0.001_0.001"




No_Flow_Full_Hydra_NonResHead = torch.load(
    f"{nonres_base_path}_NonRes_Embedding_Head.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)

## Using River Flow
# Reservoir_Embedded_LSTM = torch.load(
#     "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Reservoir/2003_No_Lat_Reservoir_LSTM_Numlayers2_LSTMhs256_Embhs128_Embos16_Epochs3_Gamma0.95_0.001_0.1.pt",
#     weights_only=False,
#     map_location=torch.device("cpu"),
# )


# NonReservoir_Embedded_LSTM = torch.load(
#     "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Natural/2003_No_Lat_Natural_LSTM_Numlayers2_LSTMhs256_Embhs128_Embos32_Epochs5_Gamma0.95_0.001_0.1.pt",
#     weights_only=False,
#     map_location=torch.device("cpu"),
# )
# Full_Embedded_LSTM = torch.load(
#     "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Full/2003_No_Lat__Full_LSTM_Numlayers2_LSTMhs256_Embhs128_Embos16_Epochs1_Gamma0.95_0.001_0.1.pt",
#     weights_only=False,
#     map_location=torch.device("cpu"),
# )



# base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Using_Flow/Hydra/2003_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp8_ResEp1_ResLR0.001_Gamma0.95_0.001_0.001"
# Hydra_Body = torch.load(
#     f"{base_path}_Body.pt", weights_only=False, map_location=torch.device("cpu")
# )
# Hydra_Head = torch.load(
#     f"{base_path}_Main_Head.pt", weights_only=False, map_location=torch.device("cpu")
# )
# Hydra_ResHead = torch.load(
#     f"{base_path}_Res_Embedding_Head.pt",
#     weights_only=False,
#     map_location=torch.device("cpu"),
# )
# Hydra_Main = HydraModel(Hydra_Body, Hydra_Head)

## Setup scaler
temporal_scalers = joblib.load("/home/mokr/Reservoir_Modelling/scaler_hydromet.pkl")
scaler_river = StandardScaler()
scaler_river.mean_ = temporal_scalers.mean_[[-2]]
scaler_river.var_ = temporal_scalers.var_[[-2]]
scaler_river.scale_ = temporal_scalers.scale_[[-2]]


## Set up Model Lists
MODEL_NAME_MAP = {
    "Reservoir Embedded LSTM": "Reservoir",
    "NonReservoir Embedded LSTM": "Non-Reservoir",
    "Full Embedded LSTM": "Full LSTM",
    "Hydra-LSTM: Main": "Hydra: Main",
    "Hydra-LSTM: Reservoir Head": "Hydra: Res Head",
    "Hydra-LSTM: Non-Reservoir Head": "Hydra: Non-Res Head",

    
}

model_names = [
    "Reservoir_Embedded_LSTM",
    "NonReservoir_Embedded_LSTM",
    "Full_Embedded_LSTM",
    "Hydra_Main",
    "Hydra_ResHead",
    "Hydra_NonResHead",
    
]

no_flow_models = [
    No_Flow_Reservoir_Embedded_LSTM,
    No_Flow_NonReservoir_Embedded_LSTM,
    No_Flow_Full_Embedded_LSTM,
    No_Flow_Full_Hydra_Main,
    [No_Flow_Full_Hydra_Body, No_Flow_Full_Hydra_ResHead],
    [No_Flow_Full_Hydra_Body, No_Flow_Full_Hydra_NonResHead],
]

# flow_models = [
#     Reservoir_Embedded_LSTM,
#     NonReservoir_Embedded_LSTM,
#     Full_Embedded_LSTM,
#     Hydra_Main,
#     [Hydra_Body, Hydra_ResHead],
# ]  


# Create output directories
os.makedirs("/home/mokr/Reservoir_Modelling/results/non_reservoir", exist_ok=True)
os.makedirs("/home/mokr/Reservoir_Modelling/results/reservoir", exist_ok=True)


"""
print("Running no flow reservoir validation...")
no_flow_reservoir_validation_results = validate_model_scores(
    model_names,
    no_flow_models,
    test_reservoir_dataloader,
    warmup,
    streamflow_scaler=scaler_river,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=False,
)


print("Running flow reservoir validation...")
flow_reservoir_validation_results = validate_model_scores(
    model_names,
    flow_models,
    test_reservoir_dataloader,
    warmup,
    streamflow_scaler=scaler_river,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=True,
)


print("Running no flow non-reservoir validation...")
no_flow_nonreservoir_validation_results = validate_model_scores(
    model_names,
    no_flow_models,
    test_non_reservoir_dataloader,
    warmup,
    streamflow_scaler=scaler_river,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=False,
)

print("Running flow non-reservoir validation...")
flow_nonreservoir_validation_results = validate_model_scores(
    model_names,
    flow_models,
    test_non_reservoir_dataloader,
    warmup,
    streamflow_scaler=scaler_river,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=True,
)
"""

FLOW_WARMUP = 364
NOFLOW_WARMUP = FLOW_WARMUP + 1

# Helper paths
LAGGED_RESULTS_PATH = "/home/mokr/Reservoir_Modelling/results/0209_365_no_lat_lagged_results" # 2204
METRICS_PATH = "/home/mokr/Reservoir_Modelling/results/0209_365_no_lat_rolling_metrics"
# LAGGED_RESULTS_PATH = "/home/mokr/Reservoir_Modelling/results/89Day_lagged_results"
# METRICS_PATH = "/home/mokr/Reservoir_Modelling/results/89Day_rolling_metrics"



print("Running no flow reservoir lagged predictions...")
no_flow_reservoir_lagged_results = lagged_model_predictions(
    model_names,
    no_flow_models,
    test_reservoir_dataloader,
    streamflow_scaler=scaler_river,
    warmup=NOFLOW_WARMUP,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=False,
)


print("Calculating metrics for no-flow reservoir...")
no_flow_reservoir_metrics = calculate_rolling_metrics(
    no_flow_reservoir_lagged_results, test_res_tensor_metadata, window_size=WINDOW_SIZE
)



# no_flow_reservoir
with shelve.open(LAGGED_RESULTS_PATH) as db:
    db["no_flow_reservoir"] = no_flow_reservoir_lagged_results
with shelve.open(METRICS_PATH) as db:
    db["no_flow_reservoir"] = no_flow_reservoir_metrics
del no_flow_reservoir_lagged_results, no_flow_reservoir_metrics
gc.collect()
print("  no-flow reservoir saved & cleared.")



# print("Running flow reservoir lagged predictions...")
# flow_reservoir_lagged_results = lagged_model_predictions(
#     model_names,
#     flow_models,
#     test_reservoir_dataloader,
#     streamflow_scaler=scaler_river,
#     warmup=FLOW_WARMUP,
#     device="cpu",
#     time_interval=TIME_INTERVAL,
#     max_batches=max_batches,
#     use_flow=True,
# )

# print("Calculating metrics for flow reservoir...")
# flow_reservoir_metrics = calculate_rolling_metrics(
#     flow_reservoir_lagged_results, test_res_tensor_metadata, window_size=WINDOW_SIZE
# )


# flow_reservoir
# with shelve.open(LAGGED_RESULTS_PATH) as db:
#     db["flow_reservoir"] = flow_reservoir_lagged_results
# with shelve.open(METRICS_PATH) as db:
#     db["flow_reservoir"] = flow_reservoir_metrics
# del flow_reservoir_lagged_results, flow_reservoir_metrics
# gc.collect()
# print("  flow reservoir saved & cleared.")

print("Running no flow non-reservoir lagged predictions...")
no_flow_nonreservoir_lagged_results = lagged_model_predictions(
    model_names,
    no_flow_models,
    test_non_reservoir_dataloader,
    streamflow_scaler=scaler_river,
    warmup=NOFLOW_WARMUP,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=max_batches,
    use_flow=False,
)

print("Calculating metrics for no-flow non-reservoir...")
no_flow_nonreservoir_metrics = calculate_rolling_metrics(
    no_flow_nonreservoir_lagged_results, test_non_res_tensor_metadata, window_size=WINDOW_SIZE
)



with shelve.open(LAGGED_RESULTS_PATH) as db:
    db["no_flow_nonreservoir"] = no_flow_nonreservoir_lagged_results
with shelve.open(METRICS_PATH) as db:
    db["no_flow_nonreservoir"] = no_flow_nonreservoir_metrics


del no_flow_nonreservoir_lagged_results, no_flow_nonreservoir_metrics
gc.collect()
print(" no-flow non-reservoir saved & cleared.")


# print("Running flow non-reservoir lagged predictions...")
# flow_nonreservoir_lagged_results = lagged_model_predictions(
#     model_names,
#     flow_models,
#     test_non_reservoir_dataloader,
#     streamflow_scaler=scaler_river,
#     warmup=FLOW_WARMUP,
#     device="cpu",
#     time_interval=TIME_INTERVAL,
#     max_batches=max_batches,
#     use_flow=True,
# )

# print("Calculating metrics for flow non-reservoir...")
# flow_nonreservoir_metrics = calculate_rolling_metrics(
#     flow_nonreservoir_lagged_results, test_non_res_tensor_metadata, window_size=WINDOW_SIZE
# )


# with shelve.open(LAGGED_RESULTS_PATH) as db:
#     db["flow_nonreservoir"] = flow_nonreservoir_lagged_results
# with shelve.open(METRICS_PATH) as db:
#     db["flow_nonreservoir"] = flow_nonreservoir_metrics
    
# del flow_nonreservoir_lagged_results, flow_nonreservoir_metrics
# gc.collect()
# print("  flow non-reservoir saved & cleared.")


"""
print("Saving results...")
results_dict = {
    "no_flow_reservoir": no_flow_reservoir_validation_results,
    "flow_reservoir": flow_reservoir_validation_results,
    "no_flow_nonreservoir": no_flow_nonreservoir_validation_results,
    "flow_nonreservoir": flow_nonreservoir_validation_results,
}

with open("/home/mokr/Reservoir_Modelling/results/no_lat_validation_results.pkl", "wb") as f:
    pickle.dump(results_dict, f)

print(
    "Validation complete! Results saved to /home/mokr/Reservoir_Modelling/results/no_lat_validation_results.pkl"
)
"""
