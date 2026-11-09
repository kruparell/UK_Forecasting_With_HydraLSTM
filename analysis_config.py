"""Shared constants for the analysis notebooks. Save as ML_Functions/analysis_config.py"""

RESULTS_DIR = "/home/mokr/Reservoir_Modelling/results"

# Existing results (produced by the run that created the 1505 shelves)
SHELVE_METRICS_PATH = f"{RESULTS_DIR}/365_0209_no_lat_rolling_metrics"
SHELVE_RESULTS_PATH = f"{RESULTS_DIR}/365_0209_no_lat_lagged_results"

# Hand-off file: notebook 02 writes it, notebooks 03 and 04 read it
ANALYSIS_CACHE_PATH = f"{RESULTS_DIR}/analysis_cache.pkl"

# Colours live in ML_Plots (single source of truth)
from ML_Plots import MODEL_COLORS, AVERAGE_MODEL_COLORS

MODEL_NAME_MAP = {
    "Reservoir Embedded LSTM": "Reservoir LSTM",
    "NonReservoir Embedded LSTM": "Non-Reservoir LSTM",
    "Full Embedded LSTM": "Full LSTM",
    "Hydra-LSTM: Main": "Hydra: Full",
    "Hydra-LSTM: Reservoir Head": "Hydra-LSTM: Res-Head",
    "Hydra-LSTM: Non-Reservoir Head": "Hydra-LSTM: NonRes-Head",
}

MODEL_NAMES = [
    "Reservoir_Embedded_LSTM",
    "NonReservoir_Embedded_LSTM",
    "Full_Embedded_LSTM",
    "Hydra_Main",
    "Hydra_ResHead",
    "Hydra_NonResHead",
]

