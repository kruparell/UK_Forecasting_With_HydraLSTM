import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

import os
import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))


from ML_functions import append_autoregressive_lag
from ML_Gradient_Analysis import (
    interpolate_riemann_path,
    generate_attribution_baseline,
    aggregate_gradient_saliency,
    format_attribution_sample_record,
)

def calculate_nse(predictions, targets):
    """Calculate Nash-Sutcliffe Efficiency"""
    numerator = torch.mean((predictions - targets) ** 2)
    denominator = torch.mean((targets - torch.mean(targets)) ** 2)
    return 1 - ((numerator + 1e-1) / (denominator + 1e-1))


def calculate_nnse(predictions, targets):
    """Calculate Nash-Sutcliffe Efficiency"""
    numerator = torch.mean((predictions - targets) ** 2)
    denominator = torch.mean((targets - torch.mean(targets)) ** 2)
    nse = 1 - ((numerator + 1e-1) / (denominator + 1e-1))
    return 1 / (2 - nse)


def calculate_kge_components(predictions, targets):
    """Calculate KGE components (alpha, beta, and correlation)"""
    # Correlation coefficient
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    numerator = torch.mean((predictions - pred_mean) * (targets - target_mean))
    pred_std = torch.sqrt(torch.mean((predictions - pred_mean) ** 2))
    target_std = torch.sqrt(torch.mean((targets - target_mean) ** 2))
    correlation = numerator / ((pred_std * target_std) + 1e-3)

    # Gamma (variability ratio)
    gamma = (pred_std / (pred_mean + 1e-6) + 1e-6) / (
        (target_std / (target_mean + 1e-6)) + 1e-6
    )

    # Beta (bias ratio)
    beta = pred_mean / (target_mean + 1e-3)

    return correlation, gamma, beta

def compute_kge_from_components(
    correlation: torch.Tensor | float,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
) -> torch.Tensor | float:
    """Calculates overall KGE score from r, alpha, and beta components."""
    if isinstance(correlation, torch.Tensor):
        return 1.0 - torch.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return 1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def evaluate_all_hydrological_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Evaluates standard hydrological performance metrics between two series."""
    nse = calculate_nse(prediction, target).item()
    nnse = calculate_nnse(prediction, target).item()
    bias = calculate_bias(prediction, target).item()
    correlation, alpha, beta = calculate_kge_components(prediction, target)
    kge = compute_kge_from_components(correlation, alpha, beta).item()

    return {
        "NSE": nse,
        "NNSE": nnse,
        "bias": bias,
        "KGE": kge,
        "KGE_correlation": correlation.item(),
        "KGE_alpha": alpha.item(),
        "KGE_beta": beta.item(),
    }


def prepare_models_for_eval(
    models,
    freeze_parameters: bool = False,
) -> list:
    """Flattens model hierarchy, sets eval mode, and optionally freezes parameters."""
    model_list = [models] if not isinstance(models, list) else models
    flat_list = []
    for m in model_list:
        if isinstance(m, list):
            flat_list.extend(m)
        else:
            flat_list.append(m)

    for m in flat_list:
        m.eval()
        if freeze_parameters:
            for p in m.parameters():
                p.requires_grad = False
    return model_list

def calculate_bias(predictions, targets):
    """Calculate bias (mean difference)"""
    return torch.mean(predictions - targets)

def get_consecutive_gauge_indices(metadata):
    """
    Returns a dictionary mapping gauge_id to a slice (start, end)
    Assumes gauge_ids are grouped consecutively.
    """
    gauge_map = {}
    current_idx = 0

    # groupby returns the key and an iterator of the group
    for gauge_id, group in groupby(metadata["gauge_id"]):
        group_len = len(list(group))
        # Store as range or slice for efficiency
        gauge_map[str(gauge_id)] = range(current_idx, current_idx + group_len)
        current_idx += group_len

    return gauge_map


def get_basin_name(basin_idx, csv_base_path="/perm/mokr/Scaled_CamelsGB_V2"):
    # Split by underscore and take the first part as location
    csv_filename = f"normalised_static_data.csv"
    csv_path = os.path.join(csv_base_path, csv_filename)
    df = pd.read_csv(csv_path, engine="python")
    gauge_name = df[df["gauge_id"] == basin_idx]["gauge_name"].values[0]

    return gauge_name


def compute_gradient_importance(
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
    chunk_size=5000 
):
    models = prepare_models_for_eval(models, freeze_parameters=True)
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

        static_input_raw   = batch["static"].to(device)
        temporal_input_raw = batch["temporal"]
        target = batch["target"]

        if use_flow:
            temporal_input_raw, target = append_autoregressive_lag(temporal_input_raw, target)
            
        temporal_input_raw = temporal_input_raw.to(device)
        batch_size = temporal_input_raw.shape[0]

        for model_name, model in zip(model_names, models):
            static_input   = static_input_raw.detach().requires_grad_(True)
            temporal_input = temporal_input_raw.detach().requires_grad_(True)

            with torch.enable_grad():
                raw_output = run_model_pipeline(model, static_input, temporal_input)
                output_warmup_excluded = raw_output[:, warmup:].squeeze(-1)
                per_sample_output = output_warmup_excluded.mean(dim=1)

            for s in range(batch_size):
                if static_input.grad is not None:
                    static_input.grad.zero_()
                if temporal_input.grad is not None:
                    temporal_input.grad.zero_()

                retain = (s < batch_size - 1)
                per_sample_output[s].backward(retain_graph=retain)

                g_static   = static_input.grad[s].detach()    
                g_temporal = temporal_input.grad[s].detach()  

                static_feat_attr = aggregate_gradient_saliency(g_static, aggregation)
                temporal_feat_attr = aggregate_gradient_saliency(g_temporal, aggregation)
                
                records.append(
                    format_attribution_sample_record(
                        model_name,
                        global_sample_idx + s,
                        static_feat_attr,
                        temporal_feat_attr,
                        feature_names_static,
                        feature_names_temporal,
                    )
                )
                # MEMORY FIX 2: Explicitly delete the intermediate gradient tensors
                del g_static, g_temporal, static_feat_grad, temporal_feat_grad

            # MEMORY FIX 3: Force wipe model gradients just in case, and clear graph anchors
            if isinstance(model, list):
                for sub_model in model:
                    sub_model.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
                
            del raw_output, per_sample_output, static_input, temporal_input

        # Empty the CUDA cache if you are using a GPU
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        global_sample_idx += batch_size

        # --- NEW: Chunking Logic ---
        if len(records) >= chunk_size:
            df_list.append(pd.DataFrame(records))
            records.clear() # Dump the heavy dictionaries
            gc.collect()    # Force garbage collection

    # Catch any remaining rows after the loop finishes
    if records:
        df_list.append(pd.DataFrame(records))
        records.clear()
        gc.collect()

    print("Concatenating chunks into final DataFrame...")
    # Concat all the small DataFrames at once (highly optimized in C)
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Clean up the list to free RAM before the final sort
    del df_list 
    gc.collect()

    final_df.set_index(["model", "sample_idx"], inplace=True)
    return final_df.sort_index()


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
):
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
    models = prepare_models_for_eval(models, freeze_parameters=True)

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
            temporal_input_raw, target = append_autoregressive_lag(temporal_input_raw, target)
            
        temporal_input_raw = temporal_input_raw.to(device)
        batch_size         = temporal_input_raw.shape[0]

        for model_name, model in zip(model_names, models):
            for s in range(batch_size):
                # -------------------------------------------------------- #
                # 1.  Extract single sample and create its baseline         #
                # -------------------------------------------------------- #
                static_s   = static_input_raw[s]
                temporal_s = temporal_input_raw[s]
                
                baseline_static = generate_attribution_baseline(static_s, baseline_type, seed=global_sample_idx + s)
                baseline_temporal = generate_attribution_baseline(temporal_s, baseline_type, seed=global_sample_idx + s)
                
                delta_static = static_s - baseline_static
                delta_temporal = temporal_s - baseline_temporal
                
                interp_static = interpolate_riemann_path(baseline_static, static_s, n_steps)
                interp_temporal = interpolate_riemann_path(baseline_temporal, temporal_s, n_steps)
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
                static_feat_grad = aggregate_gradient_saliency(ig_static, aggregation)
                temporal_feat_grad = aggregate_gradient_saliency(ig_temporal, aggregation)
                
                records.append(
                    format_attribution_sample_record(
                        model_name,
                        global_sample_idx + s,
                        static_feat_grad,
                        temporal_feat_grad,
                        feature_names_static,
                        feature_names_temporal,
                    )
                )

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

        # Flush to a DataFrame chunk to bound peak RAM
        if len(records) >= chunk_size:
            df_list.append(pd.DataFrame(records))
            records.clear()
            gc.collect()

    # Flush any remaining rows
    if records:
        df_list.append(pd.DataFrame(records))
        records.clear()
        gc.collect()

    print("Concatenating chunks into final DataFrame...")
    final_df = pd.concat(df_list, ignore_index=True)

    del df_list
    gc.collect()

    final_df.set_index(["model", "sample_idx"], inplace=True)
    return final_df.sort_index()
    
    