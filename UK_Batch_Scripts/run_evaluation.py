"""
Entry point for the no-flow hydrological model evaluation.

Run from the directory containing these three files:
    python run_evaluation.py
"""

import torch

import evaluation_config as config
from evaluation_functions import (
    load_selected_models,
    load_test_dataloader,
    load_streamflow_scaler,
    evaluate_catchment_group,
)


def main():
    if not config.MODELS_TO_EVALUATE:
        raise ValueError(
            "MODELS_TO_EVALUATE is empty. Select at least one model in config.py."
        )

    config.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading selected models...")
    model_names, models = load_selected_models(config.MODELS_TO_EVALUATE)
    print("Models selected:", ", ".join(model_names))

    print("Loading test datasets...")
    reservoir_loader = load_test_dataloader(config.RESERVOIR_TEST_TENSORS)
    nonreservoir_loader = load_test_dataloader(config.NONRESERVOIR_TEST_TENSORS)
    scaler = load_streamflow_scaler()

    # Evaluate the two catchment groups independently.
    evaluate_catchment_group(
        group_name="reservoir",
        shelf_key="no_flow_reservoir",
        dataloader=reservoir_loader,
        metadata_path=config.RESERVOIR_METADATA,
        model_names=model_names,
        models=models,
        scaler=scaler,
    )

    evaluate_catchment_group(
        group_name="non-reservoir",
        shelf_key="no_flow_nonreservoir",
        dataloader=nonreservoir_loader,
        metadata_path=config.NONRESERVOIR_METADATA,
        model_names=model_names,
        models=models,
        scaler=scaler,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
