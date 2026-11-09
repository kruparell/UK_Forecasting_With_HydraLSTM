import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


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


class CRPSLoss(nn.Module):
    """
    PyTorch module implementing CRPS loss for a mixture of asymmetric Laplacian distributions

    This loss can be used directly in standard PyTorch training loops.
    """

    def __init__(self, num_integration_points=100):
        """
        Initialize the CRPS loss function

        Args:
            num_integration_points (int): Number of points to use for numerical integration
        """
        super(CRPSLoss, self).__init__()
        self.num_integration_points = num_integration_points

    def forward(self, param_predictions, observations):
        """
        Calculate CRPS loss for a batch of predictions

        Args:
            param_predictions (dict): Dictionary with distribution parameters
                                     'mu': location parameters (batch_size, num_components)
                                     'b': scale parameters (batch_size, num_components)
                                     'tau': asymmetry parameters (batch_size, num_components)
                                     'pi': mixture weights (batch_size, num_components)
            observations (torch.Tensor): Observed values (shape: (batch_size,))

        Returns:
            torch.Tensor: CRPS loss (shape: (batch_size,))
        """
        observations = observations["y"]
        batch_size = observations.shape[0]

        # Extract parameters for asymmetric Laplacian mixture
        m = param_predictions["mu"]  # (batch_size, num_components)
        b = param_predictions["b"]  # (batch_size, num_components)
        t = param_predictions["tau"]  # (batch_size, num_components)
        pi = param_predictions["pi"]  # (batch_size, num_components)

        # Ensure pi sums to 1 along component dimension
        pi = pi / torch.sum(pi, dim=1, keepdim=True)

        # Calculate proper mean and standard deviation for each component
        # Mean = mu + b * (1 - 2*tau) / (tau * (1 - tau))
        component_means = m + b * (1 - 2 * t) / (t * (1 - t))

        # Variance = b^2 * (1 - 2*tau + 2*tau^2) / (tau^2 * (1 - tau)^2)
        component_vars = b**2 * (1 - 2 * t + 2 * t**2) / (t**2 * (1 - t) ** 2)
        component_stds = torch.sqrt(component_vars)

        # Calculate mixture mean (weighted average of component means)
        mixture_mean = torch.sum(pi * component_means, dim=1)  # (batch_size,)

        # Calculate mixture variance: E[(X-μ)²] = Σπ_i[(μ_i-μ)² + σ_i²]
        mean_variance = torch.sum(
            pi * (component_means - mixture_mean.unsqueeze(1)) ** 2, dim=1
        )
        avg_variance = torch.sum(pi * component_vars, dim=1)
        mixture_var = mean_variance + avg_variance
        mixture_std = torch.sqrt(mixture_var)

        mean_avg = torch.mean(mixture_mean)
        std_avg = torch.mean(mixture_std)

        # Use mean±5*std
        lower_bound = (mean_avg - 4 * std_avg).item()
        upper_bound = (mean_avg + 6 * std_avg).item()

        # Create integration points
        integration_points = torch.linspace(
            lower_bound,
            upper_bound,
            self.num_integration_points,
            device=observations.device,
        )
        integration_points = integration_points.repeat(
            batch_size, 1
        )  # (batch_size, num_integration_points)

        # Compute CDF at integration points
        cdf_values = self._compute_mixture_cdf(integration_points, param_predictions)

        # Calculate heaviside step function (1 if integration_point >= observation, 0 otherwise)
        observations_expanded = observations.unsqueeze(1).expand_as(integration_points)
        heaviside = (integration_points >= observations_expanded).float()

        # Calculate squared difference
        squared_diff = (cdf_values - heaviside) ** 2

        # Numerical integration using trapezoidal rule
        dx = (upper_bound - lower_bound) / (self.num_integration_points - 1)
        integrated_values = 0.5 * dx * (squared_diff[:, :-1] + squared_diff[:, 1:])
        crps = torch.sum(integrated_values, dim=1)  # Sum over integration points
        crps = torch.sum(crps)
        return crps

    def _compute_mixture_cdf(self, integration_points, param_predictions):
        """
        Compute CDF of a mixture of asymmetric Laplacian distributions

        Args:
            x (torch.Tensor): Points to evaluate CDF at (shape: (batch_size, num_points))
            params (dict): Dictionary with distribution parameters

        Returns:
            torch.Tensor: CDF values (shape: (batch_size, num_points))
        """
        # Get parameters
        m = param_predictions["mu"]  # (batch_size, num_components)
        b = param_predictions["b"]  # (batch_size, num_components)
        t = param_predictions["tau"]  # (batch_size, num_components)
        pi = param_predictions["pi"]  # (batch_size, num_components)

        # Ensure pi sums to 1 along component dimension
        pi = pi / torch.sum(pi, dim=1, keepdim=True)

        # Expand dimensions for broadcasting
        # y: (batch_size, num_points, 1)
        # m: (batch_size, 1, num_components)
        integration_points_expanded = integration_points.unsqueeze(-1)
        m_expanded = m.unsqueeze(1)
        b_expanded = b.unsqueeze(1)
        t_expanded = t.unsqueeze(1)
        pi_expanded = pi.unsqueeze(1)

        # Calculate normalized error
        error = (
            integration_points_expanded - m_expanded
        )  # (batch_size, num_points, num_components)

        # Calculate CDF value based on error sign
        greater_mask = error > 0
        less_mask = ~greater_mask

        # Initialize CDF tensor
        component_cdfs = torch.zeros_like(error)

        # Handle positive errors
        if torch.any(less_mask):
            component_cdfs[less_mask] = t_expanded.expand_as(error)[
                less_mask
            ] * torch.exp(
                (1 - t_expanded.expand_as(error)[less_mask])
                * error[less_mask]
                / b_expanded.expand_as(error)[less_mask]
            )

        # Handle negative errors
        if torch.any(greater_mask):
            component_cdfs[greater_mask] = 1 - (
                1 - t_expanded.expand_as(error)[greater_mask]
            ) * torch.exp(
                (-t_expanded.expand_as(error)[greater_mask])
                * error[greater_mask]
                / b_expanded.expand_as(error)[greater_mask]
            )

        # Apply mixture weights and sum over components
        cdf_values = torch.sum(
            pi_expanded.expand_as(component_cdfs) * component_cdfs, dim=2
        )

        return cdf_values


class CMALLoss(nn.Module):
    # Negative log-likelihood for a single asymmetric Laplacian distribution
    # Taken from Neural Hydrology, uses Alternative parameterization for Bayesian quantile regression (wikipedia)
    def __init__(self, eps=1e-8):
        super(CMALLoss, self).__init__()
        self.eps = eps  # Small constant for numerical stability

    def forward(
        self, prediction: dict[str, torch.Tensor], ground_truth: dict[str, torch.Tensor]
    ):
        mask = ~torch.isnan(ground_truth["y"])

        # Apply mask to all relevant tensors
        y = ground_truth["y"][mask]
        m = prediction["mu"][mask]
        b = prediction["b"][mask]
        t = prediction["tau"][mask]
        p = prediction["pi"][mask]
        error = y.unsqueeze(-1) - m
        log_like = (
            torch.log(t + self.eps)
            + torch.log(1.0 - t + self.eps)
            - torch.log(b + self.eps)
            - torch.max(t * error, (t - 1.0) * error) / (b + self.eps)
        )

        log_weights = torch.log(p + self.eps)  # Prevent log(0) issues

        result = torch.logsumexp(
            log_weights + log_like, dim=-1
        )  # Use dim=-1 for flexibility

        result = -torch.mean(result, dim=-1)
        return result


def compute_log_likelihood(y, param_predictions):
    """
    Compute the log likelihood for each ensemble member at a given timestep.

    Args:
        y (np.ndarray): Target value (shape: (num_trajectories, ))
        param_predictions (dict): Dictionary with 'mu', 'b', 'tau', 'pi' arrays (shape: (num_members, ))

    Returns:
        np.ndarray: Log-likelihood values for each ensemble member (shape: (num_trajectories, num_members))
    """
    m = param_predictions["mu"]  # Mean
    b = param_predictions["b"]  # Scale (>0)
    t = param_predictions["tau"]  # Asymmetry (0-1)
    p = param_predictions["pi"]  # Mixture weights

    y = y.unsqueeze(-1)  # Shape (num_trajectories, 1)
    m = m.clone().detach()  # (num_members,)
    b = b.clone().detach()  # (num_members,)
    t = t.clone().detach()  # (num_members,)
    p = p.clone().detach()  # (num_members,)

    error = y - m
    log_like = (
        torch.log(t + 1e-8)
        + torch.log(1.0 - t + 1e-8)
        - torch.log(b + 1e-8)
        - torch.max(t * error, (t - 1.0) * error) / b
    )

    log_weights = torch.log(p + 1e-8)  # Prevent log(0)
    result = torch.logsumexp(
        log_weights + log_like, dim=-1
    )  # Shape: (num_trajectories, num_members)

    return result


def compute_CDF(y, param_predictions):
    """
    Compute CDF of predicted distribution

    Args:
        y (torch.Tensor): Points at which to evaluate the CDF (shape: (batch_size, num_points))
        param_predictions (dict): Dictionary with 'mu', 'b', 'tau', 'pi' arrays
            mu (torch.Tensor): Mean parameters (shape: (batch_size, num_components))
            b (torch.Tensor): Scale parameters (shape: (batch_size, num_components))
            tau (torch.Tensor): Shape parameters (shape: (batch_size, num_components))
            pi (torch.Tensor): Mixture weights (shape: (batch_size, num_components))

    Returns:
        torch.Tensor: CDF values at the points y (shape: (batch_size, num_points))
    """
    m = param_predictions["mu"]  # (batch_size, num_components)
    b = param_predictions["b"]  # (batch_size, num_components)
    t = param_predictions["tau"]  # (batch_size, num_components)
    pi = param_predictions["pi"]  # (batch_size, num_components)

    # Ensure pi sums to 1 along component dimension
    pi = pi / torch.sum(pi, dim=1, keepdim=True)

    # Expand dimensions for broadcasting
    # y: (batch_size, num_points, 1)
    # m: (batch_size, 1, num_components)
    y_expanded = y.unsqueeze(-1)
    m_expanded = m.unsqueeze(1)
    b_expanded = b.unsqueeze(1)
    t_expanded = t.unsqueeze(1)
    pi_expanded = pi.unsqueeze(1)

    # Calculate normalized error
    error = y_expanded - m_expanded  # (batch_size, num_points, num_components)

    # Calculate CDF value based on error sign
    positive_mask = error > 0
    negative_mask = ~positive_mask

    # Initialize CDF tensor
    component_cdfs = torch.zeros_like(error)

    # Handle positive errors
    if torch.any(positive_mask):
        component_cdfs[positive_mask] = t_expanded.expand_as(error)[
            positive_mask
        ] * torch.exp(
            (1 - t_expanded.expand_as(error)[positive_mask])
            * error[positive_mask]
            / b_expanded.expand_as(error)[positive_mask]
        )

    # Handle negative errors
    if torch.any(negative_mask):
        component_cdfs[negative_mask] = 1 - (
            1 - t_expanded.expand_as(error)[negative_mask]
        ) * torch.exp(
            (-t_expanded.expand_as(error)[negative_mask])
            * error[negative_mask]
            / b_expanded.expand_as(error)[negative_mask]
        )

    # Apply mixture weights and sum over components
    cdf_values = torch.sum(
        pi_expanded.expand_as(component_cdfs) * component_cdfs, dim=2
    )

    return cdf_values


def crps_loss(param_predictions, observations, num_integration_points=100):
    """
    Calculate CRPS loss for a batch of predictions

    Args:
        param_predictions (dict): Dictionary with distribution parameters
        observations (torch.Tensor): Observed values (shape: (batch_size,))
        num_integration_points (int): Number of points to use for numerical integration

    Returns:
        torch.Tensor: CRPS loss (shape: (batch_size,))
    """
    batch_size = observations.shape[0]

    # Define integration range
    # We need to consider the support of the distribution
    std_estimate = torch.mean(param_predictions["b"])
    mean_estimate = torch.mean(param_predictions["mu"])

    # Define integration range: mean ± 5*std should cover most of the distribution
    lower_bound = (mean_estimate - 5 * std_estimate).item()
    upper_bound = (mean_estimate + 5 * std_estimate).item()

    # Create integration points
    integration_points = torch.linspace(
        lower_bound, upper_bound, num_integration_points
    )
    integration_points = integration_points.repeat(
        batch_size, 1
    )  # (batch_size, num_integration_points)

    # Compute CDF at integration points
    cdf_values = compute_CDF(
        integration_points, param_predictions
    )  # (batch_size, num_integration_points)

    # Calculate heaviside step function (1 if integration_point >= observation, 0 otherwise)
    # First, expand observations to match integration points shape
    observations_expanded = observations.unsqueeze(1).expand_as(integration_points)
    heaviside = (integration_points >= observations_expanded).float()

    # Calculate squared difference
    squared_diff = (cdf_values - heaviside) ** 2

    # Numerical integration using trapezoidal rule
    dx = (upper_bound - lower_bound) / (num_integration_points - 1)
    integrated_values = 0.5 * dx * (squared_diff[:, :-1] + squared_diff[:, 1:])
    crps = torch.sum(integrated_values, dim=1)  # Sum over integration points

    return crps


# def compute_crps(forecasts, observation):
#     """
#     Calculate the Continuous Ranked Probability Score (CRPS).

#     Parameters:
#     - forecasts: numpy array of ensemble forecasts, shape (n_members,)
#     - observation: scalar true observation

#     Returns:
#     - CRPS score (lower is better)
#     """
#     # Sort the forecasts
#     forecasts[forecasts < -0.26787253] = -0.2678724
#     n_members, batch_size, num_steps = forecasts.shape
#     forecasts_reshaped = forecasts.permute(1, 2, 0).reshape(batch_size * num_steps, n_members)
#     sorted_forecasts, _ = torch.sort(forecasts_reshaped, dim=1)


#    # Reshape observations
#     observations_flat = observation.reshape(batch_size * num_steps, 1)
#     observations_expanded = observations_flat.expand(-1, n_members)

#     # Calculate heaviside function
#     heaviside = (sorted_forecasts >= observations_expanded).float()
#     # Create ranks and normalize
#     ranks = torch.arange(1, n_members + 1, device=forecasts.device) / n_members
#     # Calculate CRPS
#     crps_flat = torch.mean((heaviside - ranks)**2, dim=1)

#     # Reshape back to (batch_size, num_steps)
#     crps_scores = crps_flat.reshape(batch_size, num_steps)

#     return crps_scores


def compute_crps(forecasts, observation, epsilon=0.0):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).

    Parameters:
    - forecasts: numpy array of ensemble forecasts, shape (n_members,)
    - observation: scalar true observation

    Returns:
    - CRPS score (lower is better)
    """
    # Sort the forecasts
    forecasts[forecasts < -0.26787253] = -0.2678724
    n_members, batch_size, num_steps = forecasts.shape
    forecasts_reshaped = forecasts.permute(1, 2, 0)

    # Reshape observations

    observations_expanded = observation.unsqueeze(-1)

    term1 = torch.abs(forecasts_reshaped - observations_expanded).mean(dim=-1)

    x_j = forecasts_reshaped.unsqueeze(3)  # (batch_size, num_steps, n_members, 1)
    x_k = forecasts_reshaped.unsqueeze(2)  # (batch_size, num_steps, 1, n_members)
    pairwise_diffs = torch.abs(x_j - x_k)
    term2 = pairwise_diffs.sum(dim=(2, 3)) / (2 * n_members * (n_members - 1))
    crps_scores = term1 - ((1 - epsilon) * term2)

    return crps_scores


def calculate_overall_crps(
    all_ensemble_predictions,
    true_flow,
    features=["total_flow", "variance", "gradient", "autoregression"],
    unsqueeze=False,
):
    """
    Calculate CRPS for each feature across all prediction instances

    Parameters:
    - all_ensemble_predictions: List of dictionaries with ensemble summaries
    - true_flow: List of dictionaries with true flow values

    Returns:
    - Dictionary with CRPS scores for each feature
    """
    # Features to compute CRPS for
    crps_scores = {feature: [] for feature in features}

    # Loop through each prediction/observation pair
    for ensemble_pred, true in zip(all_ensemble_predictions, true_flow):
        for feature in features:
            # For each prediction instance and feature, compute CRPS
            forecasts = ensemble_pred[feature]
            observation = true[feature]

            # Compute CRPS for this instance and feature
            if unsqueeze == True:
                forecasts, observation = forecasts.unsqueeze(-1), observation.unsqueeze(
                    -1
                )

            crps_value = compute_crps(forecasts, observation)
            crps_scores[feature].append(crps_value)

    # Compute mean CRPS for each feature
    for feature in features:
        crps_scores[feature] = np.mean(crps_scores[feature])

    return crps_scores


class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency loss for batched predictions with multiple timesteps.
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
    r = correlation coefficient
    alpha = std(sim)/std(obs)
    beta = mean(sim)/mean(obs)

    Handles inputs of shape [batch_size, timesteps]
    """

    def __init__(self, reduction="mean"):
        super(KGELoss, self).__init__()
        self.reduction = reduction

    def forward(self, simulated, observed):
        """
        Args:
            simulated (torch.Tensor): Simulated values of shape [batch_size, timesteps]
            observed (torch.Tensor): Observed values of shape [batch_size, timesteps]
        Returns:
            torch.Tensor: KGE loss
        """
        # Ensure the tensors are floating point
        sim = simulated.float()
        obs = observed.float()

        # Handle potential NaN values
        eps = 1e-12

        # Calculate means along timestep dimension
        sim_mean = torch.mean(sim, dim=1)  # Shape: [batch_size]
        obs_mean = torch.mean(obs, dim=1)  # Shape: [batch_size]

        # Calculate standard deviations along timestep dimension
        sim_std = torch.std(sim, dim=1)  # Shape: [batch_size]
        obs_std = torch.std(obs, dim=1)  # Shape: [batch_size]

        # Calculate correlation coefficient for each sample in the batch
        sim_norm = sim - sim_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]
        obs_norm = obs - obs_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]

        # Calculate numerator and denominator for correlation
        numerator = torch.sum(sim_norm * obs_norm, dim=1)  # Shape: [batch_size]
        sim_norm_sum = torch.sum(sim_norm**2, dim=1)  # Shape: [batch_size]
        obs_norm_sum = torch.sum(obs_norm**2, dim=1)  # Shape: [batch_size]

        # Add epsilon for numerical stability
        denominator = torch.sqrt(sim_norm_sum * obs_norm_sum + eps)
        r = numerator / denominator  # Shape: [batch_size]

        # Calculate alpha (ratio of standard deviations)
        alpha = sim_std / (obs_std + eps)  # Shape: [batch_size]

        # Calculate beta (ratio of means)
        # Handle the case where observed mean is zero or very small
        beta = sim_mean / (obs_mean + eps)
        # Calculate KGE for each sample in the batch
        kge = 1 - torch.sqrt(
            (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
        )  # Shape: [batch_size]

        # Convert to loss (1 - KGE, since KGE's optimal value is 1)
        loss = 1 - kge  # Shape: [batch_size]

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


def get_member_summaries_torch(ensembles):
    """Returns statistical summaries for the hydrographs generated by each of the ensemble members"""
    import torch

    # Assuming ensembles is already a torch tensor with shape [num_members, num_batches, num_steps]
    num_members, num_batches, num_steps = ensembles.shape

    # I'm assuming this function exists elsewhere in your code
    # You'll need to implement load_and_unnormalize in PyTorch or adapt it
    # ensembles, * = load_and_unnormalize(ensembles, ensembles, scaler_path)

    ensembles_summaries = {}

    # Total flow - sum along the time dimension
    ensembles_summaries["total_flow"] = torch.sum(ensembles, dim=-1)

    # Variance along the time dimension
    ensembles_summaries["variance"] = torch.var(ensembles, dim=-1)

    # For gradient calculation with polyfit, you might want to use a PyTorch-compatible method
    # or fall back to numpy for this specific calculation

    # Autoregression calculation
    current = ensembles[:, :, 1:]  # Shape: (num_members, num_batches, num_steps-1)
    lagged = ensembles[:, :, :-1]  # Shape: (num_members, num_batches, num_steps-1)

    # Calculate means along time dimension for each member and batch
    current_mean = torch.mean(
        current, dim=-1, keepdim=True
    )  # Shape: (num_members, num_batches, 1)
    lagged_mean = torch.mean(
        lagged, dim=-1, keepdim=True
    )  # Shape: (num_members, num_batches, 1)

    # Calculate cross-covariance between current and lagged for each member and batch
    cross_cov = torch.mean(
        (current - current_mean) * (lagged - lagged_mean), dim=2
    )  # Shape: (num_members, num_batches)

    # Calculate variance of current and lagged for each member and batch
    var_current = torch.mean(
        (current - current_mean) ** 2, dim=-1
    )  # Shape: (num_members, num_batches)
    var_lagged = torch.mean(
        (lagged - lagged_mean) ** 2, dim=-1
    )  # Shape: (num_members, num_batches)

    # Initialize ar_coeffs tensor with zeros
    ar_coeffs = torch.zeros((num_members, num_batches), device=ensembles.device)

    # Calculate correlation coefficients, handling zero variance cases
    nonzero_mask = (var_current > 0) & (var_lagged > 0)
    ar_coeffs[nonzero_mask] = cross_cov[nonzero_mask] / torch.sqrt(
        var_current[nonzero_mask] * var_lagged[nonzero_mask]
    )

    ensembles_summaries["autoregression"] = ar_coeffs

    # FDC curve calculation
    sorted_ensembles, _ = torch.sort(ensembles, dim=-1)
    ensembles_summaries["fdc_curve"] = (
        torch.log(sorted_ensembles[:, :, 7] + 1e-8)
        - torch.log(sorted_ensembles[:, :, 2] + 1e-8)
    ) / 5

    return ensembles_summaries


def get_member_summaries(
    ensembles,
    scaler_path="/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib",
):
    """Returns statistical summaries for the hydrographs generated by each of the ensemmble members"""

    ensembles, _ = load_and_unnormalize(ensembles, ensembles, scaler_path)
    ensembles_summaries = {}
    ensembles_summaries["total_flow"] = np.sum(ensembles, axis=1)
    ensembles_summaries["variance"] = np.var(ensembles, axis=1)
    x = np.arange(ensembles.shape[1])  # Lead times
    ensembles_summaries["gradient"] = np.array(
        [np.polyfit(x, ensemble, 1)[0] for ensemble in ensembles]
    )

    ar_coeffs = []  # Create a list to store the coefficients

    for i in range(len(ensembles)):
        # Calculate covariance matrix
        cov_matrix = np.cov(ensembles[i, 1:], ensembles[i, 0:-1])

        if cov_matrix[0, 0] > 0 and cov_matrix[1, 1] > 0:
            ar_coeff = cov_matrix[0, 1] / (np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1]))
        else:
            ar_coeff = 0
        ar_coeffs.append(ar_coeff)

    ensembles_summaries["autoregression"] = np.vstack(ar_coeffs)

    ensembles_summaries["fdc_curve"] = (
        np.log(np.sort(ensembles, axis=-1)[:, 7] + 0.2678726)
        - np.log(np.sort(ensembles, axis=-1)[:, 2] + 0.2678726)
    ) / 5
    return ensembles_summaries


def get_dispersion_calculations(ensemble_summaries, truth, interval=0.1):
    """Returns a dataframe showing how often X% of ensemble members underpredict each statistic, 10-20%, 20-30%... 90-100%
    Parameters:
    ensemble_summaries: list of dictionaries with statistics from get_member_summaries
                       Each key contains an array of shape (num_members)
    truth: observed data statistics matching the keys in ensemble_summaries
           Should be a list of dictionares with the same keys as ensemble_summaries

    Returns:
    DataFrame with percentage distribution of ensemble members in each dispersion bin
    """
    # Define bins for percentiles
    bins = np.arange(0, 1 + interval, interval)  # 0, 0.1, 0.2, ..., 1.0
    bin_labels = [
        f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins) - 1)
    ]

    # Initialize dictionary to store all ranks for each stat
    all_ranks = {}

    for ens_dict, truth_dict in zip(ensemble_summaries, truth):
        for stat_name in ens_dict.keys():
            ensemble_stat = np.asarray(ens_dict[stat_name])  # shape: (num_members,)
            truth_val = truth_dict[stat_name]  # scalar

            rank = np.mean(ensemble_stat < truth_val)  # single float
            if stat_name not in all_ranks:
                all_ranks[stat_name] = []
            all_ranks[stat_name].append(rank)

    # Now compute histogram for each stat
    dispersion_results = {}
    for stat_name, ranks in all_ranks.items():
        hist, _ = np.histogram(ranks, bins=bins)
        dispersion_results[stat_name] = (
            hist / len(ranks) * 100
        )  # Convert to percentages

    return pd.DataFrame(dispersion_results, index=bin_labels)


# KGE and NSE computation functions
def compute_nse(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    return 1 - np.sum((observed - simulated) ** 2) / np.sum(
        (observed - np.mean(observed)) ** 2
    )


def compute_kge(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    cc = np.corrcoef(simulated, observed)[0, 1]
    alpha = np.std(simulated) / np.std(observed)
    beta = np.mean(simulated) / np.mean(observed)
    return 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
