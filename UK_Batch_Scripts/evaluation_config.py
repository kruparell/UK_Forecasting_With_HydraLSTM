"""
Central configuration for the no-flow hydrological model evaluation.

Edit this file to change paths, model selection, or evaluation settings.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project and data paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/home/mokr/Reservoir_Modelling")
FUNCTIONS_PATH = PROJECT_ROOT / "ML_Functions"
MODEL_ROOT = PROJECT_ROOT / "CamelsGB_Models"
DATA_ROOT = Path("/perm/mokr/CamelsGB_V2")
RESULTS_ROOT = PROJECT_ROOT / "results"

RESERVOIR_TEST_TENSORS = DATA_ROOT / "no_lat_test_res_tensor_data.pt"
NONRESERVOIR_TEST_TENSORS = DATA_ROOT / "no_lat_test_non_res_tensor_data.pt"
RESERVOIR_METADATA = DATA_ROOT / "test_res_tensor_metadata.pt"
NONRESERVOIR_METADATA = DATA_ROOT / "test_non_res_tensor_metadata.pt"
SCALER_PATH = PROJECT_ROOT / "scaler_hydromet.pkl"

# Existing shelve paths retained for downstream compatibility.
LAGGED_RESULTS_PATH = RESULTS_ROOT / "0209_365_no_lat_lagged_results"
METRICS_PATH = RESULTS_ROOT / "0209_365_no_lat_rolling_metrics"

# ---------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------

TIME_INTERVAL = 120
WINDOW_SIZE = 365
FLOW_WARMUP = 364
NOFLOW_WARMUP = FLOW_WARMUP + 1

BATCH_SIZE = 1
NUM_WORKERS = 1
MAX_BATCHES = int(1e7)
DEVICE = "cpu"

# Explicitly select the models to evaluate.
# Options:
#   Reservoir_Embedded_LSTM
#   NonReservoir_Embedded_LSTM
#   Full_Embedded_LSTM
#   Hydra_Main
#   Hydra_ResHead
#   Hydra_NonResHead
MODELS_TO_EVALUATE = (
    "Reservoir_Embedded_LSTM",
    "NonReservoir_Embedded_LSTM",
    "Full_Embedded_LSTM",
    "Hydra_Main",
    "Hydra_ResHead",
    "Hydra_NonResHead",
)

# ---------------------------------------------------------------------
# Complete PyTorch model checkpoints
# ---------------------------------------------------------------------

MODEL_CHECKPOINTS = {
    "Reservoir_Embedded_LSTM": (
        MODEL_ROOT
        / "Reservoir/365_0209_No_Lat_Reservoir_LSTM_Numlayers2_LSTMhs256_Embhs128_Embos16_Epochs2_Gamma0.95_0.001_0.4.pt"
    ),
    "NonReservoir_Embedded_LSTM": (
        MODEL_ROOT
        / "Natural/365_0209_No_Lat_Natural_LSTM_Numlayers2_LSTMhs128_Embhs64_Embos16_Epochs1_Gamma0.95_0.001_0.4.pt"
    ),
    "Full_Embedded_LSTM": (
        MODEL_ROOT
        / "Full/365_0209_No_Lat_Full_LSTM_Numlayers1_LSTMhs128_Embhs128_Embos32_Epochs1_Gamma0.95_0.001_0.2.pt"
    ),
}

# Hydra component paths: placeholders carried over from the previous script.
# Replace these when the updated Hydra checkpoints are available.
HYDRA_FULL_BASE_PATH = (
    MODEL_ROOT
    / "Hydra/Hydra_Body_Main_NonRes_Res_Epochs1_LR0.001_ResLR0.001.pt"
)
HYDRA_NONRES_BASE_PATH = (
    MODEL_ROOT
    / "Hydra/Hydra_Body_Main_NonRes_Res_Epochs1_LR0.001_ResLR0.001.pt"
)
