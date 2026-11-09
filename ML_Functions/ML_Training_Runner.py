from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))


from ML_functions import NSELoss
from ML_Models import LSTM_With_Embedding
from ML_Training import train_model


def extract_input_dimensions(dataloader: DataLoader) -> Tuple[int, int]:
    """Inspects the first batch of a dataloader to determine static and temporal sizes."""
    for batch in dataloader:
        static_size = batch["static"].shape[1]
        temporal_size = batch["temporal"].shape[-1]
        break
    return static_size, temporal_size

def _snapshot(model):
    ms = model if isinstance(model, list) else [model]
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in ms]

def _restore(model, snap):
    ms = model if isinstance(model, list) else [model]
    for m, s in zip(ms, snap):
        m.load_state_dict(s)
        
def train_with_early_stopping(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    num_epochs: int,
    warmup: int,
    loss_interval: int,
    device: torch.device,
    patience: int = 2,
    use_flow: bool = False,
    static_step: bool = True,
    time_interval: int = 360,
    model_tag: str = "",
) -> Dict[str, Any]:
    """Canonical training and validation loop with patience-based early stopping."""
    current_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    last_train_loss = None

    for epoch in range(num_epochs):
        tag_prefix = f"[{model_tag}] " if model_tag else ""
        print(f"  {tag_prefix}Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            1,
            warmup,
            loss_interval,
            device,
            train_mode=True,
            time_interval=time_interval,
            static_step=static_step,
            use_flow=use_flow,
        )
        last_train_loss = train_loss

        val_loss = train_model(
            model,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            1,
            warmup,
            loss_interval,
            device,
            train_mode=False,
            time_interval=time_interval,
            static_step=static_step,
            use_flow=use_flow,
        )
        print(f"  {tag_prefix}Validation Loss: {val_loss:.6f}")

        if val_loss < current_val_loss:
            current_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = _snapshot(model)
                
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {tag_prefix}Early stopping triggered at epoch {epoch + 1}")
                break
        
    # after the for loop, before the return:
    if best_model_state is not None:
        _restore(model, best_model_state)
        
    return {
        "best_epoch": best_epoch,
        "best_val_loss": current_val_loss,
        "last_train_loss": last_train_loss,
        "best_state": best_model_state,
    }


def save_and_display_results(
    results: List[Dict[str, Any]],
    sort_col: str,
    csv_path: str,
) -> pd.DataFrame:
    """Formats hyperparameter search results, logs best trial, and saves to CSV."""
    df = pd.DataFrame(results).sort_values(sort_col, ascending=True, na_position="last")
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    successful = df[df.get("status", "success") == "success"] if "status" in df.columns else df
    if not successful.empty and successful[sort_col].notna().any():
        best_row = successful.loc[successful[sort_col].idxmin()]
        print("\n" + "=" * 80)
        print("BEST HYPERPARAMETERS")
        print("=" * 80)
        for col, val in best_row.items():
            print(f"{col}: {val}")
    return df


def run_lstm_grid_search(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyperparam_grid: Dict[str, List[Any]],
    save_dir: str,
    device: torch.device,
    use_flow: bool = False,
    warmup: int = 10,
    loss_interval: int = 500,
    num_epochs: int = 8,
    patience: int = 2,
    csv_output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Unified grid search runner for standard LSTM With Embedding models."""
    static_size, temporal_size = extract_input_dimensions(train_loader)
    combinations = [dict(zip(hyperparam_grid.keys(), v)) for v in product(*hyperparam_grid.values())]
    print(f"[{model_name}] Total combinations to test: {len(combinations)}")

    results = []
    for params in combinations:
        emb_o_size = params["embedding_output_size"]
        emb_h_size = params["embedding_hidden_size"]
        lstm_h_size = params["lstm_hidden_size"]
        lstm_n_layers = params["lstm_num_layers"]
        gamma_val = params["gamma"]
        dropout = params["dropout"]
        lr = params["learning_rate"]

        model_save_path = None
        try:
            model = LSTM_With_Embedding(
                static_input_size=static_size,
                embedding_hidden_size=emb_h_size,
                embedding_output_size=emb_o_size,
                lstm_input_size=temporal_size + emb_o_size + int(use_flow),
                lstm_hidden_size=lstm_h_size,
                lstm_num_layers=lstm_n_layers,
                lstm_output_size=1,
                dropout=dropout,
                bidirectional=False,
                Sequence_Target=True,
            ).to(device)

            criterion = NSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_val)

            metrics = train_with_early_stopping(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=num_epochs,
                warmup=warmup,
                loss_interval=loss_interval,
                device=device,
                patience=patience,
                use_flow=use_flow,
            )

            model_save_path = (
                f"{save_dir}/365_0209_No_Lat_{model_name}_LSTM_Numlayers{lstm_n_layers}"
                f"_LSTMhs{lstm_h_size}_Embhs{emb_h_size}_Embos{emb_o_size}_Epochs{metrics['best_epoch']}"
                f"_Gamma{gamma_val}_{lr}_{dropout}.pt"
            )
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

            results.append({
                **params,
                "num_epochs": metrics["best_epoch"],
                "train_loss": metrics["last_train_loss"],
                "val_loss": metrics["best_val_loss"],
                "model_path": model_save_path,
                "status": "success",
            })
        except Exception as e:
            print(f"  ERROR: Trial failed: {e}")
            results.append({
                **params,
                "num_epochs": None,
                "train_loss": None,
                "val_loss": None,
                "model_path": None,
                "status": "failed",
                "error": str(e),
            })
        finally:
            if "model" in locals():
                del model, optimizer, scheduler, criterion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = csv_output_path or f"{save_dir}/365_0209_{use_flow}_flow_hyperparameter_search_results.csv"
    return save_and_display_results(results, sort_col="val_loss", csv_path=csv_path)