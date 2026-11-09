from typing import Any, Dict, Sequence
import torch

def interpolate_riemann_path(
    baseline: torch.Tensor,
    sample: torch.Tensor,
    n_steps: int
) -> torch.Tensor:
    """Generates linear interpolation points between a baseline and an input sample.

    Args:
        baseline: Baseline tensor of shape (D1, D2, ...).
        sample: Input tensor of shape (D1, D2, ...).
        n_steps: Number of integration intervals along the path.

    Returns:
        Tensor of shape (n_steps, D1, D2, ...) containing path-interpolated points.
    """
    device = sample.device
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device)
    delta = sample - baseline
    
    # Broadcast along step dimension for any arbitrary input shape
    shape_expansion = (n_steps,) + (1,) * delta.dim()
    return baseline.unsqueeze(0) + alphas.view(shape_expansion) * delta.unsqueeze(0)
    

def generate_attribution_baseline(
    sample: torch.Tensor,
    baseline_type: str = "zeros",
    seed: int | None = None
) -> torch.Tensor:
    """Constructs a reference baseline matching the shape and device of the input tensor.

    Args:
        sample: Reference input tensor.
        baseline_type: 'zeros' for null baseline or 'gaussian' for small noise.
        seed: Optional manual seed for gaussian reproducibility.

    Returns:
        Baseline tensor matching sample shape, dtype, and device.
    """
    if baseline_type == "zeros":
        return torch.zeros_like(sample)
    elif baseline_type == "gaussian":
        rng = torch.Generator(device=sample.device)
        if seed is not None:
            rng.manual_seed(seed)
        return torch.normal(0.0, 0.01, size=sample.shape, generator=rng, device=sample.device)
    else:
        raise ValueError(f"Unknown baseline_type '{baseline_type}'. Expected 'zeros' or 'gaussian'.")
        
def aggregate_gradient_saliency(
    gradients: torch.Tensor,
    aggregation: str = "mean_abs"
) -> torch.Tensor:
    """Applies reduction/scaling to raw gradient or attribution tensors.

    Args:
        gradients: Gradient tensor of any shape.
        aggregation: 'mean_abs', 'rms', or 'mean_signed'.

    Returns:
        Aggregated importance tensor of the same shape.
    """
    if aggregation == "mean_abs":
        return gradients.abs()
    elif aggregation == "rms":
        return gradients.pow(2).sqrt()
    elif aggregation == "mean_signed":
        return gradients
    else:
        raise ValueError(f"Unknown aggregation: '{aggregation}'. Expected 'mean_abs', 'rms', or 'mean_signed'.")
        
def format_attribution_sample_record(
    model_name: str,
    sample_idx: int,
    static_attr: torch.Tensor,
    temporal_attr: torch.Tensor,
    feature_names_static: Sequence[str],
    feature_names_temporal: Sequence[str]
) -> Dict[str, Any]:
    """Flattens per-sample multidimensional attributions into a columnar schema dictionary.

    Args:
        model_name: Identifier name of the model evaluated.
        sample_idx: Global index identifier of the sample.
        static_attr: 1D tensor of shape (num_static_features,).
        temporal_attr: 2D tensor of shape (timesteps, num_temporal_features).
        feature_names_static: Column names matching static dimension.
        feature_names_temporal: Column names matching temporal feature dimension.

    Returns:
        Dictionary mapping column names to scalar float values.
    """
    row: Dict[str, Any] = {
        "model": model_name,
        "sample_idx": sample_idx,
    }

    for name, value in zip(feature_names_static, static_attr.detach().cpu()):
        row[f"static__{name}"] = value.item()

    timesteps, _ = temporal_attr.shape
    temporal_cpu = temporal_attr.detach().cpu()
    for feat_idx, name in enumerate(feature_names_temporal):
        for t in range(timesteps):
            row[f"temporal__{name}_day_{t}"] = temporal_cpu[t, feat_idx].item()

    return row