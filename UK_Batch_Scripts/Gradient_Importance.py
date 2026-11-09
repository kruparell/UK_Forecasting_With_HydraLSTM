import sys
import pickle
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc

# --- Local Module Configuration ---
FUNCTIONS_PATH = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
if str(FUNCTIONS_PATH) not in sys.path:
    sys.path.append(str(FUNCTIONS_PATH))

# Importing from your custom files
import ML_functions
# from ML_Metrics import compute_integrated_gradients

from ML_functions import TensorCatchmentDataset, NSELoss
from ML_Models import LSTM_With_Embedding, LSTMModel, HydraModel
from ML_Training import (
    run_model_pipeline,
    process_validation_batches,
)

warmup = 89
MAX_BATCHES = int(5e4) #2e4 usually

TIME_INTERVAL = 60*5
### Loading Data

BATCH_SIZE = 1
NUM_WORKERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_integrated_gradients(
    model_names,
    models,
    dataloader,
    warmup,
    feature_names_static,
    feature_names_temporal,
    device="cpu",
    time_interval=180,
    max_batches=None,
    use_flow=False,
    aggregation="mean_abs",
    chunk_size=5000,
    n_steps=50,
    baseline_type="zeros",
    output_path='/perm/mokr/reservoir_grad_importance.csv',
):

    header_written = False
    """
    Compute Integrated Gradients (IG) feature attributions.

    IG approximates the integral of gradients along a straight path from a
    baseline input x' to the actual input x:

        IG(x)_i = (x_i - x'_i) * (1/n_steps) * sum_{k=1}^{n_steps} dF/dx_i |_{x' + (k/n_steps)*(x-x')}

    This satisfies the Completeness axiom: sum(IG) == F(x) - F(x'), making
    attributions directly interpretable as contributions to the output.

    Args:
        model_names:            List of string identifiers for each model.
        models:                 List of models (or lists of sub-models) to evaluate.
        dataloader:             PyTorch DataLoader yielding dicts with keys
                                "static", "temporal", and "target".
        warmup:                 Number of leading timesteps to exclude from
                                the output before averaging.
        feature_names_static:   Column names for the static feature tensor.
        feature_names_temporal: Column names for the temporal feature tensor.
        device:                 Device string, e.g. "cpu" or "cuda".
        time_interval:          Minimum seconds between tqdm progress updates.
        max_batches:            If set, stop after this many batches.
        use_flow:               If True, append the lagged target as an extra
                                temporal feature and shift the target by one step.
        aggregation:            How to post-process the raw IG values per feature:
                                  "mean_abs"    -> |IG| (unsigned importance)
                                  "rms"         -> sqrt(IG^2)  (same sign as mean_abs
                                                   but emphasises large values)
                                  "mean_signed" -> IG as-is    (keeps direction)
        chunk_size:             Flush `records` to a DataFrame every this many
                                rows to keep peak RAM usage bounded.
        n_steps:                Number of Riemann steps used to approximate the
                                integral (default 50; 20-300 is a typical range).
        baseline_type:          Reference input x':
                                  "zeros"    -> all-zero tensor (most common)
                                  "gaussian" -> random N(0, 0.01) noise, seeded
                                               for reproducibility

    Returns:
        pd.DataFrame indexed by ("model", "sample_idx") with one column per
        static feature and one column per (temporal_feature, timestep) pair.
    """
    # ------------------------------------------------------------------ #
    # 0.  Normalise model list and freeze weights                         #
    # ------------------------------------------------------------------ #
    if not isinstance(models, list):
        models = [models]

    for model in models:
        if isinstance(model, list):
            for sub_model in model:
                sub_model.eval()
                for param in sub_model.parameters():
                    param.requires_grad = False
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    records = []
    df_list = []
    global_sample_idx = 0

    progress_bar = tqdm(
        dataloader, desc="Processing batches", unit="batch", mininterval=time_interval
    )

    for batch_idx, batch in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached maximum number of batches ({max_batches}). Stopping.")
            break

        static_input_raw   = batch["static"].to(device)   # (B, S)
        temporal_input_raw = batch["temporal"]             # (B, T, F)
        target             = batch["target"]

        if use_flow:
            previous_flow      = target[:, :-1].unsqueeze(-1)
            temporal_input_raw = temporal_input_raw[:, :-1, :]
            temporal_input_raw = torch.cat([temporal_input_raw, previous_flow], dim=-1)
            target             = target[:, 1:]

        temporal_input_raw = temporal_input_raw.to(device)
        batch_size         = temporal_input_raw.shape[0]

        for model_name, model in zip(model_names, models):
            for s in range(batch_size):
                # -------------------------------------------------------- #
                # 1.  Extract single sample and create its baseline         #
                # -------------------------------------------------------- #
                static_s   = static_input_raw[s]    # (S,)
                temporal_s = temporal_input_raw[s]  # (T, F)

                if baseline_type == "zeros":
                    baseline_static   = torch.zeros_like(static_s)
                    baseline_temporal = torch.zeros_like(temporal_s)
                elif baseline_type == "gaussian":
                    rng = torch.Generator(device=device)
                    rng.manual_seed(global_sample_idx + s)
                    baseline_static   = torch.normal(0.0, 0.01, size=static_s.shape,
                                                     generator=rng, device=device)
                    baseline_temporal = torch.normal(0.0, 0.01, size=temporal_s.shape,
                                                     generator=rng, device=device)
                else:
                    raise ValueError(
                        f"Unknown baseline_type '{baseline_type}'. "
                        "Expected 'zeros' or 'gaussian'."
                    )

                # -------------------------------------------------------- #
                # 2.  Build a batch of n_steps interpolated inputs          #
                # -------------------------------------------------------- #
                alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps,
                                        device=device)              # (n_steps,)

                delta_static   = static_s   - baseline_static      # (S,)
                delta_temporal = temporal_s - baseline_temporal     # (T, F)

                # Broadcast: (n_steps,1) * (1,S) -> (n_steps, S)
                interp_static = (baseline_static.unsqueeze(0)
                                 + alphas.unsqueeze(1) * delta_static.unsqueeze(0))

                # Broadcast: (n_steps,1,1) * (1,T,F) -> (n_steps, T, F)
                interp_temporal = (baseline_temporal.unsqueeze(0)
                                   + alphas.view(-1, 1, 1) * delta_temporal.unsqueeze(0))

                interp_static   = interp_static.detach().requires_grad_(True)
                interp_temporal = interp_temporal.detach().requires_grad_(True)

                # -------------------------------------------------------- #
                # 3.  Single forward + backward pass over all n_steps       #
                with torch.enable_grad():
                    raw_output = run_model_pipeline(
                        model, interp_static, interp_temporal
                    )
                    # raw_output: (n_steps, T_out, 1) or (n_steps, T_out)
                    output_warmup_excluded = raw_output[:, warmup:].squeeze(-1)
                    # Mean over time → (n_steps,); sum over steps → scalar
                    scalar_loss = output_warmup_excluded.mean(dim=1).sum()

                scalar_loss.backward()

                # -------------------------------------------------------- #
                # 4.  Integrated Gradients = delta * mean_gradient          #
                avg_grad_static   = interp_static.grad.mean(dim=0)     # (S,)
                avg_grad_temporal = interp_temporal.grad.mean(dim=0)    # (T, F)

                ig_static   = delta_static   * avg_grad_static          # (S,)
                ig_temporal = delta_temporal * avg_grad_temporal        # (T, F)

                # -------------------------------------------------------- #
                # 5.  Apply aggregation to final IG values                  #
                # -------------------------------------------------------- #
                if aggregation == "mean_abs":
                    static_feat_attr   = ig_static.abs()
                    temporal_feat_attr = ig_temporal.abs()
                elif aggregation == "rms":
                    static_feat_attr   = ig_static.pow(2).sqrt()
                    temporal_feat_attr = ig_temporal.pow(2).sqrt()
                elif aggregation == "mean_signed":
                    static_feat_attr   = ig_static
                    temporal_feat_attr = ig_temporal
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

                # -------------------------------------------------------- #
                # 6.  Store results                                          #
                # -------------------------------------------------------- #
                row_dict = {
                    "model":      model_name,
                    "sample_idx": global_sample_idx + s,
                }

                row_dict.update({
                    f"static__{k}": v.item()
                    for k, v in zip(feature_names_static,
                                    static_feat_attr.detach().cpu())
                })

                T_steps = temporal_feat_attr.shape[0]
                for i, k in enumerate(feature_names_temporal):
                    for t in range(T_steps):
                        row_dict[f"temporal__{k}_day_{t}"] = (
                            temporal_feat_attr[t, i].detach().item()
                        )

                records.append(row_dict)

                # -------------------------------------------------------- #
                # 7.  Clean up per-sample tensors                           #
                # -------------------------------------------------------- #
                del (interp_static, interp_temporal,
                     avg_grad_static, avg_grad_temporal,
                     ig_static, ig_temporal,
                     static_feat_attr, temporal_feat_attr,
                     scalar_loss, raw_output, output_warmup_excluded,
                     delta_static, delta_temporal)

            # Clean up model gradients and batch-level graph anchors
            if isinstance(model, list):
                for sub_model in model:
                    sub_model.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        global_sample_idx += batch_size

    #     Flush to a DataFrame chunk to bound peak RAM
    #     if len(records) >= chunk_size:
    #         df_list.append(pd.DataFrame(records))
    #         records.clear()
    #         gc.collect()

    # # Flush any remaining rows
    # if records:
    #     df_list.append(pd.DataFrame(records))
    #     records.clear()
    #     gc.collect()


        # Replace the chunk flush inside the loop:
        if len(records) >= chunk_size:
            chunk_df = pd.DataFrame(records)
            chunk_df.set_index(["model", "sample_idx"], inplace=True)
            chunk_df.to_csv(output_path, mode='a', header=not header_written)
            header_written = True
            records.clear()
            del chunk_df
            gc.collect()

    # Replace the entire tail:
    if records:
        chunk_df = pd.DataFrame(records)
        chunk_df.set_index(["model", "sample_idx"], inplace=True)
        chunk_df.to_csv(output_path, mode='a', header=not header_written)
        records.clear()
        del chunk_df
        gc.collect()
    print(f"Done. Results streamed to {output_path}")
    
    print("Concatenating chunks into final DataFrame...")
    # final_df = pd.concat(df_list, ignore_index=True)

    # del df_list
    # gc.collect()

    # final_df.set_index(["model", "sample_idx"], inplace=True)
    # return final_df.sort_index()
    

    
## Loading Models

## Not using river flow
No_Flow_Reservoir_Embedded_LSTM = torch.load("/home/mokr/Reservoir_Modelling/CamelsGB_Models/Reservoir/2003_No_Lat_Reservoir_LSTM_Numlayers1_LSTMhs256_Embhs64_Embos16_Epochs0_Gamma0.8_0.001_0.4.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)

No_Flow_NonReservoir_Embedded_LSTM = torch.load(    "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Natural/2003_No_Lat_Natural_LSTM_FlowFalse_Numlayers2_LSTMhs256_Embhs64_Embos16_Epochs2_Gamma0.95_0.001_0.4.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)

No_Flow_Full_Embedded_LSTM = torch.load(    "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Full/2003_No_Lat__Full_LSTM_Numlayers2_LSTMhs128_Embhs128_Embos32_Epochs0_Gamma0.95.pt_0.001_0.1",
    weights_only=False,
    map_location=torch.device("cpu"),
)

No_Flow_Full_Embedded_LSTM = torch.load("/home/mokr/Reservoir_Modelling/CamelsGB_Models/Full/89Warmup_No_Lat__Full_LSTM_Numlayers1_LSTMhs128_Embhs128_Embos16_Epochs2_Gamma0.95.pt_0.001_0
",
    weights_only=False,
    map_location=torch.device("cpu"),
)



full_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/2003_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp3_ResEp3_ResLR0.001_Gamma0.95_0.001_0.001"


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

full_base_path = "/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/2003_No_Lat_Full_Hydra_LSTM_Numlayers1_LSTMhs256_Embhs128_Embos16_BodyOut128_MainEp3_ResEp1_ResLR0.001_Gamma0.95_0.001_0.001"
No_Flow_Full_Hydra_NonResHead = torch.load(
    f"{full_base_path}_NonRes_Embedding_Head.pt",
    weights_only=False,
    map_location=torch.device("cpu"),
)


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


with open("/home/mokr/Reservoir_Modelling/Normalised_Temporal_Data.pkl", "rb") as f:
    temporal_data = pickle.load(f)

static_variables = pd.read_csv(
    "/perm/mokr/Scaled_CamelsGB_V2/normalised_static_data.csv", dtype={"gauge_id": str}
)



static_variables = static_variables.set_index("gauge_id")
static_variables = static_variables.select_dtypes(include=[np.number])

EXCLUDE_STATIC = {

    "gauge_lat", "gauge_lon", "gauge_northing", "hourly_flow_perc_complete",
    "gauge_easting",  "monthly_gwlevel_perc_complete", "daily_gwlevel_perc_complete",
    "monthly_gwlevel_period_start", "monthly_gwlevel_period_end", "reservoir_year_last", "reservoir_year_first",  
    "gw_well_easting", "gw_well_datum", "gw_well_northing", "gw_well_name", 
    "daily_gwlevel_period_end", "gw_well_id", "daily_gwlevel_period_start",
    'q5_uncert_upper', 'q25_uncert_upper', 'q50_uncert_upper', 'q75_uncert_upper', 'q95_uncert_upper', 'q99_uncert_upper',
    'q5_uncert_lower', 'q25_uncert_lower', 'q50_uncert_lower', 'q75_uncert_lower', 'q95_uncert_lower', 'q99_uncert_lower',
    'q_mean', 'Q5', 'Q95', 'stream_elas', 'slope_fdc', 'baseflow_index', 'baseflow_index_ceh', 'hfd_mean',
    'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq', 
    'runoff_ratio', 'daily_flow_perc_complete', 'daily_flow_extrap_dur',
    'max_hourly_flow', 'max_gauging_flow', 'max_daily_flow', 'hourly_flow_extrap_dur', 
    
}


static_variables = (
    static_variables
    .drop(columns=list(EXCLUDE_STATIC))
)

# ── Temporal feature names ──────────────────────────────────────────────────
goals = ["discharge_spec_x", "discharge_spec_y", "discharge_vol_x", "discharge_vol_y", 'date']
first_gauge = next(iter(temporal_data))
feature_names_temporal = [
    c for c in temporal_data[first_gauge].columns if c not in goals
]

feature_names_static = [c for c in static_variables.columns]



test_res_tensor_data = torch.load("/perm/mokr/CamelsGB_V2/no_lat_test_res_tensor_data.pt")
test_res_dataset = TensorCatchmentDataset(test_res_tensor_data)


test_reservoir_dataloader = DataLoader(
    test_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True




reservoir_df = compute_integrated_gradients(
    model_names=model_names,
    models=no_flow_models,
    dataloader=test_reservoir_dataloader,
    warmup=warmup,
    feature_names_static=feature_names_static,
    feature_names_temporal=feature_names_temporal,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=MAX_BATCHES,
    use_flow=False,
    aggregation="mean_abs", output_path = '/perm/mokr/1505_reservoir_gradient_importance.csv'#1505
)


# reservoir_df.to_csv("/perm/mokr/2204_reservoir_gradient_importance.csv")
print("Saved reservoir DataFrame to CSV successfully.")

# Remove from memory
# del reservoir_df
del test_res_tensor_data, test_reservoir_dataloader, test_res_dataset
gc.collect()

# del flow_models, Reservoir_Embedded_LSTM, NonReservoir_Embedded_LSTM, Full_Embedded_LSTM, Hydra_Body, Hydra_Head, Hydra_ResHead, Hydra_Main
# gc.collect()



test_non_res_tensor_data = torch.load(
    "/perm/mokr/CamelsGB_V2/no_lat_test_non_res_tensor_data.pt"
)
test_non_res_dataset = TensorCatchmentDataset(test_non_res_tensor_data)


test_non_reservoir_dataloader = DataLoader(
    test_non_res_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)  # , pin_memory=True, persistent_workers=True

print("Running non-reservoir processing")
nonreservoir_df = compute_integrated_gradients(
    model_names=model_names,
    models=no_flow_models,
    dataloader=test_non_reservoir_dataloader,
    warmup=warmup,
    feature_names_static=feature_names_static,
    feature_names_temporal=feature_names_temporal,
    device="cpu",
    time_interval=TIME_INTERVAL,
    max_batches=MAX_BATCHES,
    use_flow=False,
    aggregation="mean_abs",
    output_path = '/perm/mokr/1505_nonreservoir_gradient_importance.csv', #2204
)

# nonreservoir_df.to_csv("/perm/mokr/2204_nonreservoir_gradient_importance.csv")
print("Saved non-reservoir DataFrame to CSV successfully.")

# Remove from memory
# del nonreservoir_df
gc.collect()