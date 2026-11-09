import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import joblib
import random

import sys
from pathlib import Path

functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
from ML_Losses import compute_log_likelihood


def plot_ensemble_predictions(
    ensemble_predictions,
    true_discharge=None,
    title="Ensemble Predictions",
    plot_type="both",
    unnormalize=True,
    scaler_path="/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib",
):
    """
    Plot ensemble predictions with multiple visualization options.

    Args:
        ensemble_predictions (torch.Tensor): Tensor of ensemble predictions
        true_discharge (torch.Tensor, optional): True discharge values for comparison
        title (str, optional): Plot title
        plot_type (str, optional): Type of plot to generate
        normalize (bool, optional): Whether to normalize using the scaler
        scaler_path (str, optional): Path to the joblib scaler file
    """
    # Normalize if requested
    if unnormalize:
        ensemble_predictions, true_discharge = load_and_unnormalize(
            ensemble_predictions, true_discharge, scaler_path
        )

    # Convert to numpy if tensor
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.numpy()

    # Convert true_discharge to numpy if tensor
    if true_discharge is not None and isinstance(true_discharge, torch.Tensor):
        true_discharge = true_discharge.numpy().squeeze()

    # Validate plot_type
    valid_plot_types = ["ensemble", "violin", "both"]
    if plot_type not in valid_plot_types:
        raise ValueError(f"plot_type must be one of {valid_plot_types}")

    # Create subplots based on plot type
    if plot_type == "ensemble":
        fig, ax = plt.subplots(figsize=(14, 7))
        _plot_ensemble(ax, ensemble_predictions, true_discharge, title)
    elif plot_type == "violin":
        fig, ax = plt.subplots(figsize=(14, 7))
        _plot_violin(ax, ensemble_predictions, true_discharge, title)
    else:  # both
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        _plot_ensemble(ax1, ensemble_predictions, true_discharge, f"{title} - Ensemble")
        _plot_violin(
            ax2, ensemble_predictions, true_discharge, f"{title} - Violin Plot"
        )

    plt.tight_layout()
    plt.show()


def _plot_ensemble(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Plot ensemble predictions on a given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Calculate statistics
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    lower_percentile = np.percentile(ensemble_predictions, 10, axis=0)
    upper_percentile = np.percentile(ensemble_predictions, 90, axis=0)

    # Plot ensemble members (light blue, low opacity)
    for prediction in ensemble_predictions:
        random_alpha = random.uniform(0.2, 0.7)
        ax.plot(prediction, color="blue", alpha=random_alpha, linewidth=1)

    # Plot mean prediction
    ax.plot(mean_predictions, color="blue", linewidth=2, label="Ensemble Mean")

    # Plot confidence interval
    ax.fill_between(
        range(len(mean_predictions)),
        lower_percentile,
        upper_percentile,
        color="grey",
        alpha=0.2,
        label="80% Confidence Interval",
    )

    # Plot true_discharge if provided
    if true_discharge is not None:
        ax.plot(true_discharge, color="black", linewidth=2, label="Observations")

    if true_discharge is not None:
        ax.legend()

    ax.set_title(title or "Ensemble Predictions")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Normalised River Discharge")
    ax.legend()


def _plot_violin(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Create violin plot of ensemble predictions on a given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Prepare data for violin plot
    # Transpose to get distribution for each time step
    ensemble_T = ensemble_predictions

    # Create violin plot
    sns.violinplot(data=ensemble_T, ax=ax, color="lightcoral", cut=0, log_scale=False)

    # Plot true discharge if provided
    if true_discharge is not None:
        ax.plot(
            range(len(true_discharge)),
            true_discharge,
            color="black",
            linewidth=2,
            label="Observations",
            marker="o",
        )

    ax.set_title(title or "Ensemble Predictions - Violin Plot")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Normalised River Discharge")

    if true_discharge is not None:
        ax.legend()


def kl_divergence(p_samples, q_samples, bins=10):
    # Create histograms for both distributions
    p_counts, bin_edges = np.histogram(p_samples, bins=bins, density=True)
    q_counts, _ = np.histogram(q_samples, bins=bin_edges, density=True)

    # Add small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    p_counts = p_counts + epsilon
    q_counts = q_counts + epsilon

    # Normalize
    p_counts = p_counts / np.sum(p_counts)
    q_counts = q_counts / np.sum(q_counts)

    # Calculate KL divergence using scipy.special.rel_entr
    kl_div = np.sum(rel_entr(p_counts, q_counts))
    return kl_div


def CMAL_quantile(alpha, parameters):
    m = parameters[0]
    b = parameters[1]
    tau = parameters[2]

    if 0 < alpha <= tau:
        return m + (b / (1 - tau)) * np.log(alpha / tau)
    elif tau < alpha < 1:
        return m - (b / tau) * np.log((1 - alpha) / (1 - tau))
    else:
        return "Alpha must be between 0 and 1"


def transform_CMAL_parameters(tensor):
    """
    Transform a tensor by applying:
    - Softplus to the second element (index 1) of each row
    - Sigmoid to the third element (index 2) of each row

    Args:
        tensor: Input tensor of shape [batch_size, features]

    Returns:
        Transformed tensor of the same shape
    """
    result = tensor.clone()

    # Apply softplus to the second column (index 1)
    result[:, 1] = F.softplus(tensor[:, 1])

    # Apply sigmoid to the third column (index 2)
    result[:, 2] = torch.sigmoid(tensor[:, 2])

    return result


def draw_from_CMAL_distribution(param_predictions, num_samples=1000):
    """
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions

    Parameters:
    -----------
    prediction : dict
        Dictionary containing tensors for mu, b, tau, and pi
        Each tensor has shape (batch_size, num_components)

    num_samples : int, optional
        Number of samples to generate (default: 1000)

    Returns:
    --------
    numpy.ndarray
        Samples drawn from the mixture distribution
    """

    mu = param_predictions["mu"]
    b = param_predictions["b"]
    tau = param_predictions["tau"]
    pi = param_predictions["pi"]

    pi = pi / torch.sum(pi, dim=1, keepdim=True)

    batch_size, num_components = mu.shape

    # Prepare output array
    samples = torch.zeros((batch_size, num_samples))

    # Sample for each batch
    for batch in range(batch_size):
        # Select component for each sample based on mixing weights
        component_indices = torch.multinomial(pi[batch], num_samples, replacement=True)

        # Draw samples from selected components
        for sample_idx, component in enumerate(component_indices):
            mu_comp = mu[batch, component]
            b_comp = b[batch, component]
            tau_comp = tau[batch, component]

            # Asymmetric Laplacian sampling
            u = np.random.uniform(0, 1)

            if u <= tau_comp:
                sample = mu_comp + b_comp * torch.log(u / tau_comp) / (1 - tau_comp)
            else:
                sample = (
                    mu_comp - b_comp * torch.log((1 - u) / (1 - tau_comp)) / tau_comp
                )

            samples[batch, sample_idx] = sample
    # Squeeze to 1D if batch size is 1
    return samples.squeeze()


def draw_from_CMAL_distribution_np(param_predictions, num_samples=1000):
    """
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions

    Parameters:
    -----------
    prediction : dict
        Dictionary containing tensors for mu, b, tau, and pi
        Each tensor has shape (batch_size, num_components)

    num_samples : int, optional
        Number of samples to generate (default: 1000)

    Returns:
    --------
    numpy.ndarray
        Samples drawn from the mixture distribution
    """

    mu = param_predictions["mu"]
    b = param_predictions["b"]
    tau = param_predictions["tau"]
    pi = param_predictions["pi"]

    pi = pi / np.sum(pi, axis=1, keepdims=True)
    batch_size, num_components = mu.shape

    # Prepare output array
    samples = np.zeros((batch_size, num_samples))

    # Sample for each batch
    for batch in range(batch_size):
        # Select component for each sample based on mixing weights
        component_indices = np.random.choice(
            len(pi[batch]), size=num_samples, p=pi[batch]
        )

        # Draw samples from selected components
        for sample_idx, component in enumerate(component_indices):
            mu_comp = mu[batch, component]
            b_comp = b[batch, component]
            tau_comp = tau[batch, component]

            # Asymmetric Laplacian sampling
            u = np.random.uniform(0, 1)

            if u <= tau_comp:
                sample = mu_comp + b_comp * np.log(u / tau_comp) / (1 - tau_comp)
            else:
                sample = mu_comp - b_comp * np.log((1 - u) / (1 - tau_comp)) / tau_comp

            samples[batch, sample_idx] = sample
    # Squeeze to 1D if batch size is 1
    return samples.squeeze()


def load_and_unnormalize(
    ensemble_predictions,
    true_discharge,
    scaler_path="/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib",
):
    """
    Load scaler and normalize ensemble predictions and true discharge.

    Args:
        ensemble_predictions (torch.Tensor): Ensemble predictions to normalize
        true_discharge (torch.Tensor): True discharge values to normalize
        scaler_path (str): Path to the joblib scaler file

    Returns:
        tuple: Normalized ensemble predictions and true discharge
    """
    # Load the scaler
    try:
        scaler = joblib.load(scaler_path)["streamflow"]
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}")
        return ensemble_predictions, true_discharge
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return ensemble_predictions, true_discharge

    # Convert to numpy if tensor
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.numpy()

    if isinstance(true_discharge, torch.Tensor):
        true_discharge = true_discharge.numpy()

    # Normalize ensemble predictions
    # Assuming the scaler is a scipy.stats.StandardScaler
    unnormalized_ensemble = (ensemble_predictions * scaler.scale_) + scaler.mean_

    # Normalize true discharge
    unnormalized_true_discharge = (true_discharge * scaler.scale_) + scaler.mean_

    return unnormalized_ensemble, unnormalized_true_discharge


def get_ensemble_members_vectorized(
    model,
    Hist_X_Chunk_Torch,
    Fore_X_Chunk_Torch,
    num_members=1,
    keep_original=False,
    device="cpu",
):
    # CHECK IF THE RESHAPING RETURNS THNGS BACK WHERE THEY BELONG
    """
    Generate ensemble predictions for discharge.

    Args:
        model (torch.nn.Module): The trained machine learning model
        Hist_X_Chunk_Torch (torch.Tensor): Historical input chunk (batch size, num_steps, num_variables)
        Fore_X_Chunk_Torch (torch.Tensor): Forecast input chunk
        num_members (int, optional): Number of ensemble members to generate. Defaults to 1.

    Returns:
        torch.Tensor: Ensemble predictions of discharge
    """

    num_steps = Fore_X_Chunk_Torch.shape[1]
    batch_size = Fore_X_Chunk_Torch.shape[0]

    hist_steps = Hist_X_Chunk_Torch.shape[1]

    # Prepare containers for predictions and masks
    Pred_Discharge = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
    Discharge_Mask = torch.zeros(num_members, batch_size, num_steps, 1, device=device)

    # Repeat historical input for all ensemble members
    Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch.repeat(num_members, 1, 1, 1)
    Fore_X_Chunk_Torch_repeated = Fore_X_Chunk_Torch.repeat(num_members, 1, 1, 1)

    # Iterate through steps
    for step in range(num_steps):
        # Prepare input by concatenating forecast chunk, current predictions, and mask
        Binary_Fore_X_Chunk_Torch = torch.cat(
            [
                Fore_X_Chunk_Torch_repeated,
                Pred_Discharge[:, :, :],
                Discharge_Mask[:, :, :],
            ],
            dim=-1,
        )

        # treat the ensembles and batch_size as one big batch set so the dimension is 3 again
        Binary_Fore_X_Chunk_Torch = Binary_Fore_X_Chunk_Torch.reshape(
            num_members * batch_size, num_steps, -1
        )
        Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch_repeated.reshape(
            num_members * batch_size, hist_steps, -1
        )

        # Draw samples from the model
        with torch.no_grad():
            param_predictions = model(
                Hist_X_Chunk_Torch_repeated.to(torch.float32),
                Binary_Fore_X_Chunk_Torch.to(torch.float32),
            )[1]

        # Transform parameters and draw samples

        param_predictions_step = ML_functions.transform_CMAL_parameters_multi(
            param_predictions
        )[:, step]

        param_prediction = {
            "mu": param_predictions_step[:, ::4],  # mu is at indices 0 mod 4
            "b": param_predictions_step[
                :, 1::4
            ],  # b is at indices 1 mod 4 (ensure positive)
            "tau": param_predictions_step[
                :, 2::4
            ],  # tau is at indices 2 mod 4 (ensure 0-1)
            "pi": param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
        }

        # Draw samples for each ensemble member
        samples = (
            draw_from_CMAL_distribution(param_prediction, num_samples=1)
            .clone()
            .detach()
        )
        # Update predictions and masks
        Pred_Discharge[:, :, step, :] = (
            samples.clone().detach().reshape(num_members, batch_size, 1)
        )
        Discharge_Mask[:, :, step, :] = 1

    if keep_original:
        Original_Predictions = torch.zeros(
            num_members, batch_size, num_steps, 1, device=device
        )
        Original_Predictions_Saved = torch.zeros(
            num_members, batch_size, num_steps, 1, device=device
        )
        Discharge_Mask = torch.zeros(
            num_members, batch_size, num_steps, 1, device=device
        )

        Binary_Fore_X_Chunk_Torch = torch.cat(
            [
                Fore_X_Chunk_Torch_repeated,
                Original_Predictions[:, :, :],
                Discharge_Mask[:, :, :],
            ],
            dim=-1,
        )

        # treat the ensembles and batch_size as one big batch set so the dimension is 3 again
        Binary_Fore_X_Chunk_Torch = Binary_Fore_X_Chunk_Torch.reshape(
            num_members * batch_size, num_steps, -1
        )
        Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch_repeated.reshape(
            num_members * batch_size, hist_steps, -1
        )

        with torch.no_grad():
            param_predictions = model(
                Hist_X_Chunk_Torch_repeated.to(torch.float32),
                Binary_Fore_X_Chunk_Torch.to(torch.float32),
            )[1]

        for step in range(num_steps):
            # Transform parameters and draw samples
            param_predictions_step = ML_functions.transform_CMAL_parameters_multi(
                param_predictions
            )[:, step]

            param_prediction = {
                "mu": param_predictions_step[:, ::4],  # mu is at indices 0 mod 4
                "b": param_predictions_step[
                    :, 1::4
                ],  # b is at indices 1 mod 4 (ensure positive)
                "tau": param_predictions_step[
                    :, 2::4
                ],  # tau is at indices 2 mod 4 (ensure 0-1)
                "pi": param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
            }

            # Draw samples for each ensemble member
            samples = draw_from_CMAL_distribution(param_prediction, num_samples=1)

            # Update predictions and masks
            Original_Predictions_Saved[:, :, step, :] = (
                samples.clone().detach().reshape(num_members, batch_size, 1)
            )

        return Pred_Discharge.squeeze(), Original_Predictions_Saved.squeeze()
    else:
        return Pred_Discharge.squeeze()


def get_ensemble_members(model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, num_members=1):
    """
    Generate ensemble predictions for discharge.

    Args:
        model (torch.nn.Module): The trained machine learning model
        Hist_X_Chunk_Torch (torch.Tensor): Historical input chunk
        Fore_X_Chunk_Torch (torch.Tensor): Forecast input chunk
        num_members (int, optional): Number of ensemble members to generate. Defaults to 1.

    Returns:
        torch.Tensor: Ensemble predictions of discharge
    """

    num_steps = len(Fore_X_Chunk_Torch[0])
    ensemble_predictions = []

    for member in range(num_members):
        # Initialize tensors for predictions and mask
        Pred_Discharge = torch.zeros(num_steps)
        Discharge_Mask = torch.zeros(num_steps)

        for step in range(num_steps):
            # Prepare input by concatenating forecast chunk, current predictions, and mask
            Binary_Fore_X_Chunk_Torch = torch.cat(
                [
                    Fore_X_Chunk_Torch,
                    Pred_Discharge.unsqueeze(0).unsqueeze(-1),
                    Discharge_Mask.unsqueeze(0).unsqueeze(-1),
                ],
                dim=-1,
            )

            # Draw a sample from the model
            with torch.no_grad():
                param_predictions = model(
                    Hist_X_Chunk_Torch.to(torch.float32),
                    Binary_Fore_X_Chunk_Torch.to(torch.float32),
                )[1]

            # Transform parameters and draw sample (assuming external functions)
            param_predictions_step = ML_functions.transform_CMAL_parameters_multi(
                param_predictions
            )[:, step]

            param_prediction = {
                "mu": param_predictions_step[:, ::4],  # mu is at indices 0 mod 4
                "b": param_predictions_step[
                    :, 1::4
                ],  # b is at indices 1 mod 4 (ensure positive)
                "tau": param_predictions_step[
                    :, 2::4
                ],  # tau is at indices 2 mod 4 (ensure 0-1)
                "pi": param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
            }

            sample = draw_from_CMAL_distribution_np(param_prediction, num_samples=1)

            # Update predictions and mask
            Pred_Discharge[step] = torch.tensor(sample)
            Discharge_Mask[step] = 1

        ensemble_predictions.append(Pred_Discharge)

    return torch.stack(ensemble_predictions)


def generate_greedy_trajectories(
    ensemble_predictions,
    param_predictions,
    model,
    Fore_X_Chunk_Torch,
    Hist_X_Chunk_Torch,
):
    """
    Generate greedy trajectories by selecting the next timestep value that maximizes the PDF.
    Optimized for GPU performance.

    Args:
        ensemble_predictions (torch.Tensor): Shape (num_members, batch_size, num_steps)
        param_predictions (dict): Contains 'mu', 'b', 'tau', 'pi' for each timestep
        model: The model used for predictions
        Fore_X_Chunk_Torch (torch.Tensor): Forecast features tensor
        Hist_X_Chunk_Torch (torch.Tensor): Historical features tensor

    Returns:
        torch.Tensor: Shape (num_members, batch_size, num_steps), greedy trajectories
    """
    num_members, batch_size, num_steps = ensemble_predictions.shape
    device = ensemble_predictions.device

    # Pre-allocate memory for trajectories and selected indices
    trajectories = torch.zeros((num_members, batch_size, num_steps), device=device)
    selected_indices = torch.full(
        (num_members, batch_size, num_steps), -1, dtype=torch.int64, device=device
    )

    # Process all batches in parallel for the first timestep
    if num_steps > 0:
        step_values = ensemble_predictions[:, :, 0]  # (num_members, batch_size)

        # Generate permutations for all batches in one go
        # For each batch, we need a different permutation of indices
        batch_permutations = torch.stack(
            [torch.randperm(num_members, device=device) for _ in range(batch_size)],
            dim=1,
        )  # (num_members, batch_size)

        # Use advanced indexing to assign values efficiently
        batch_indices = torch.arange(batch_size, device=device)
        for m in range(num_members):
            member_indices = batch_permutations[m]  # (batch_size,)
            trajectories[m, batch_indices, 0] = step_values[
                member_indices, batch_indices
            ]
            selected_indices[m, batch_indices, 0] = member_indices

    # Pre-allocate the modified forecast features for all members at once
    all_fore_chunks = (
        Fore_X_Chunk_Torch.clone()
        .unsqueeze(0)
        .expand(num_members, -1, -1, -1)
        .contiguous()
    )
    batched_hist_chunks = Hist_X_Chunk_Torch.repeat(num_members, 1, 1)

    # For steps beyond the first one
    for step in range(1, num_steps):
        step_values = ensemble_predictions[:, :, step]  # (num_members, batch_size)

        # Create batch input for the model - much more efficient than running per member
        # Update forecast chunks for all members simultaneously
        for m in range(num_members):
            all_fore_chunks[m, :, 0:step, -1] = 1
            all_fore_chunks[m, :, 0:step, -2] = trajectories[m, :, 0:step]

        # Reshape for batch processing
        batched_fore_chunks = all_fore_chunks.reshape(
            -1, all_fore_chunks.shape[2], all_fore_chunks.shape[3]
        )
        # batched_hist_chunks = Hist_X_Chunk_Torch.repeat(num_members, 1, 1)

        # Run a single forward pass with all members batched together
        with torch.no_grad():
            batched_predictions = model(
                batched_hist_chunks.to(torch.float32),
                batched_fore_chunks.to(torch.float32),
            )[1][:, step, :]

        # Reshape back to per-member format
        new_predictions = batched_predictions.reshape(num_members, batch_size, -1)

        # Transform parameters once for all members
        all_param_predictions = ML_functions.transform_CMAL_parameters_multi(
            new_predictions
        )

        all_param_dictionaries = {
            "mu": all_param_predictions[:, :, ::4],  # mu is at indices 0 mod 4
            "b": all_param_predictions[:, :, 1::4],  # b is at indices 1 mod 4
            "tau": all_param_predictions[:, :, 2::4],  # tau is at indices 2 mod 4
            "pi": all_param_predictions[:, :, 3::4],  # pi is at indices 3 mod 4
        }

        # Create mask to avoid selecting duplicate indices
        available_mask = torch.ones(
            (num_members, batch_size), dtype=torch.bool, device=device
        )

        # Process each member sequentially (this part is harder to parallelize due to dependencies)
        for m in range(num_members):
            member_params = {
                "mu": all_param_dictionaries["mu"][m].unsqueeze(0),
                "b": all_param_dictionaries["b"][m].unsqueeze(0),
                "tau": all_param_dictionaries["tau"][m].unsqueeze(0),
                "pi": all_param_dictionaries["pi"][m].unsqueeze(0),
            }

            # Compute log-likelihoods for all possible choices
            log_likelihoods = compute_log_likelihood(
                step_values, member_params
            )  # Shape: (num_members, batch_size)

            # Apply mask to prevent duplicate selections
            masked_log_likelihoods = log_likelihoods.clone()
            masked_log_likelihoods[~available_mask] = float("-inf")

            # Get the best indices
            chosen_indices = torch.argmax(
                masked_log_likelihoods, dim=0
            )  # (batch_size,)

            # Update available mask to mark indices as used
            for b in range(batch_size):
                available_mask[chosen_indices[b], b] = False

            # Store chosen values
            batch_indices = torch.arange(batch_size, device=device)
            trajectories[m, batch_indices, step] = step_values[
                chosen_indices, batch_indices
            ]
            selected_indices[m, batch_indices, step] = chosen_indices

    return trajectories


def _plot_boxplot(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Create a detailed boxplot with multiple quantiles for ensemble predictions.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Transpose to get distribution for each time step
    ensemble_T = ensemble_predictions.T

    # Calculate quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    boxplot_data = []

    # Compute quantiles for each time step
    for time_step in ensemble_T:
        step_quantiles = np.percentile(time_step, [q * 100 for q in quantiles])
        boxplot_data.append(step_quantiles)

    # Convert to numpy array for easier manipulation
    boxplot_data = np.array(boxplot_data)

    # Custom boxplot with multiple quantiles
    box_width = 0.6
    whisker_width = 0.3

    # Plot boxes
    for i, (q1, q2, median, q3, q4) in enumerate(boxplot_data):
        # Body of the box (50% interval)
        ax.add_patch(
            plt.Rectangle(
                (i - box_width / 2, q2),
                box_width,
                q3 - q2,
                facecolor="lightblue",
                edgecolor="blue",
                alpha=0.5,
            )
        )

        # Median line
        ax.plot(
            [i - box_width / 2, i + box_width / 2],
            [median, median],
            color="blue",
            linewidth=2,
        )

        # Lower whisker (5%)
        ax.plot([i, i], [q1, q2], color="blue", linewidth=1)
        ax.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [q1, q1],
            color="blue",
            linewidth=1,
        )

        # Upper whisker (95%)
        ax.plot([i, i], [q3, q4], color="blue", linewidth=1)
        ax.plot(
            [i - whisker_width / 2, i + whisker_width / 2],
            [q4, q4],
            color="blue",
            linewidth=1,
        )

    # Plot true discharge if provided
    if true_discharge is not None:
        ax.plot(
            range(len(true_discharge)),
            true_discharge,
            color="black",
            linewidth=2,
            label="Observations",
            marker="o",
        )

    ax.set_title(title or "Ensemble Predictions - Detailed Boxplot")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Normalised River Discharge")
    ax.set_xticks(range(len(ensemble_T)))

    if true_discharge is not None:
        ax.legend()


def plot_dispersion_histograms(dispersion_df):
    """
    Plots histogram of truth percentile ranks for each statistic.

    Parameters:
    dispersion_df: DataFrame as returned by get_dispersion_calculations
                   Rows are percentile bins, columns are statistics
    """
    ax = dispersion_df.plot(kind="bar", figsize=(12, 6), width=0.8)

    expected_value = 100 / len(dispersion_df)
    ax.axhline(
        y=expected_value,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Expected (uniform): {expected_value:.2f}%",
    )

    ax.set_xlabel("Percentile Bin")
    ax.set_ylabel("Percentage of Samples")
    ax.set_title("Truth Percentile Rank Distribution per Statistic")
    ax.legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_uniform_qq(dispersion_df, title="Q-Q Plot: All Statistics vs Uniform"):
    """
    Creates Quantile-Quantile plots comparing each statistic's distribution
    to a Uniform distribution.

    Parameters:
    dispersion_df: DataFrame where rows are percentile bins, columns are statistics
                   Values should be percentage of samples in each bin
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    from matplotlib.gridspec import GridSpec

    # Get list of statistics (column names)
    statistics = dispersion_df.columns
    n_stats = len(statistics)

    # For each statistic
    for i, stat in enumerate(statistics):
        # Get the data - convert percentages to proportions
        data = dispersion_df[stat].values / 100.0

        # Convert from bin percentages to individual data points
        # We'll replicate each bin value according to its percentage
        bin_edges = np.linspace(0, 1, len(data) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Generate synthetic data points from bin frequencies
        synthetic_data = []
        for j, bin_center in enumerate(bin_centers):
            # Number of points to generate for this bin
            n_points = int(round(data[j] * 100))
            # Add points at bin center with small jitter
            if n_points > 0:
                jitter = np.random.uniform(-0.1 / len(data), 0.1 / len(data), n_points)
                synthetic_data.extend([bin_center + j for j in jitter])

        # Generate uniform theoretical quantiles
        uniform_quantiles = np.linspace(0, 1, len(synthetic_data))

        # Sort our data to get empirical quantiles
        empirical_quantiles = np.sort(synthetic_data)

    # Optional: Also create a single-plot version with all statistics
    plt.figure(figsize=(10, 10))
    for stat in statistics:
        data = dispersion_df[stat].values / 100.0

        # Generate synthetic data points from bin frequencies
        synthetic_data = []
        for j, bin_center in enumerate(bin_centers):
            n_points = int(round(data[j] * 100))
            if n_points > 0:
                jitter = np.random.uniform(-0.1 / len(data), 0.1 / len(data), n_points)
                synthetic_data.extend([bin_center + j for j in jitter])

        # Generate uniform theoretical quantiles
        uniform_quantiles = np.linspace(0, 1, len(synthetic_data))

        # Sort our data to get empirical quantiles
        empirical_quantiles = np.sort(synthetic_data)

        # Plot this statistic
        plt.scatter(uniform_quantiles, empirical_quantiles, alpha=0.5, label=stat)

    # Add diagonal line
    plt.plot([0, 1], [0, 1], "r--", linewidth=3, label="Uniform")  # Thicker line
    plt.title(f"{title}", fontsize=20)  # Larger title
    plt.xlabel("Uniform Theoretical Quantiles", fontsize=18)
    plt.ylabel("Empirical Quantiles", fontsize=18)
    plt.grid(True, alpha=0.3)

    # Make tick marks bigger
    plt.tick_params(axis="both", which="major", width=2, length=8, labelsize=16)

    # Make legend bigger with larger markers
    legend = plt.legend(prop={"size": 14}, frameon=True, markerscale=2)

    plt.tight_layout()

    plt.show()
