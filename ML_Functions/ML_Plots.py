"""
ML_Plots - figures and tables for model results.

Catchment-split and gauge-network code now lives in ML_Catchment_Splits.py.
Old imports such as `from ML_Plots import assign_splits_greedy` still work (see __getattr__ at the bottom).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import ks_2samp
from itertools import combinations


# ---------------------------------------------------------------------------
# Shared constants (single source of truth; analysis_config.py imports these)
# ---------------------------------------------------------------------------
# Default legend names used by plot_score_cdfs when no map is passed. Pass your own map for different labels.
MODEL_NAME_MAP = {
    "Reservoir Embedded LSTM": "Reservoir",
    "NonReservoir Embedded LSTM": "Non-Reservoir",
    "Full Embedded LSTM": "Full LSTM",
    "Hydra-LSTM: Main": "Hydra: Main",
    "Hydra-LSTM: Reservoir Head": "Hydra: Res Head",
}

# Keys are the display names the plotting functions look colours up by.
MODEL_COLORS = {
    "Reservoir":           "#0F6E56",  # reservoir family
    "Hydra: Res Head":     "#5DCAA5",
    "Non-Reservoir":       "#534AB7",  # non-reservoir family
    "Hydra: Non-Res Head": "#AFA9EC",
    "Full LSTM":           "#993C1D",  # whole-network family
    "Hydra: Main":         "#F0997B",
}

AVERAGE_MODEL_COLORS = {
    "Specialist":       "#2166ac",
    "Full LSTM":        "#d6191b",
    "Hydra Specialist": "#6baed6",
    "Hydra: Main":      "#d61582",
}


# ---------------------------------------------------------------------------
# Tables and statistics
# ---------------------------------------------------------------------------
def display_results_table(results, use_median=False, confidence_interval=None, n_bootstrap=1000, to_list = False):
    """
    Display validation results as a formatted table.
    
    Parameters:
    results (dict): Dictionary containing model names as keys and dicts of scores as values.
    use_median (bool): If True, calculates the median instead of the mean. Default is False.
    confidence_interval (float | None): If provided, compute a confidence interval at this level
        (e.g. 0.95 for 95%). Uses bootstrapping for the median and a t-distribution for the mean.
    n_bootstrap (int): Number of bootstrap resamples when computing CI for the median. Default 1000.
    """
    agg_func = np.median if use_median else np.mean
    metrics = ["NSE", "NNSE", "KGE", "KGE_alpha", "KGE_beta", "KGE_correlation", "bias"]

    def compute_ci_bootstrap(values, level, n_resamples):
        """Bootstrap percentile CI."""
        values = np.asarray(values)
        rng = np.random.default_rng()
        boot_stats = np.array([
            agg_func(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_resamples)
        ])
        alpha = 1 - level
        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return lower, upper

    def compute_ci_t(values, level):
        """t-distribution CI for the mean."""
        from scipy import stats
        n = len(values)
        se = stats.sem(values)
        interval = stats.t.interval(level, df=n - 1, loc=np.mean(values), scale=se)
        return interval[0], interval[1]

    summary = {}
    for model_name, scores in results.items():
        summary[model_name] = {}
        for metric in metrics:

            if to_list:
                vals = np.asarray(scores[metric].to_list())
            else:
                vals = np.asarray(scores[metric])
            point = agg_func(vals)

            if confidence_interval is not None:
                if use_median:
                    lo, hi = compute_ci_bootstrap(vals, confidence_interval, n_bootstrap)
                else:
                    lo, hi = compute_ci_t(vals, confidence_interval)
                summary[model_name][metric] = (point, lo, hi)
            else:
                summary[model_name][metric] = point

    # Build a DataFrame of point estimates for ranking, and a display DataFrame
    point_df = pd.DataFrame(
        {model: {m: v[0] if confidence_interval else v for m, v in scores.items()}
         for model, scores in summary.items()}
    ).T

    if confidence_interval is not None:
        display_df = pd.DataFrame(
            {model: {m: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
                     for m, v in scores.items()}
             for model, scores in summary.items()}
        ).T
    else:
        display_df = point_df.copy()

    # Determine best model for each metric using point estimates
    best_models = {
        "NSE":             point_df["NSE"].idxmax(),
        "NNSE":            point_df["NNSE"].idxmax(),
        "KGE":             point_df["KGE"].idxmax(),
        "KGE_correlation": point_df["KGE_correlation"].idxmax(),
        "KGE_alpha":       point_df["KGE_alpha"].apply(lambda x: abs(x - 1)).idxmin(),
        "KGE_beta":        point_df["KGE_beta"].apply(lambda x: abs(x - 1)).idxmin(),
        "bias":            point_df["bias"].apply(abs).idxmin(),
    }

    def highlight_best(col):
        best_model = best_models[col.name]
        return ["font-weight: bold" if idx == best_model else "" for idx in col.index]

    fmt = "{}" if confidence_interval else "{:.2f}"
    styled_df = display_df.style.apply(highlight_best, axis=0).format(fmt)
    return styled_df


def per_catchment_ks(models_dict, model_names, metric = 'KGE'):
    # List to store the results of each comparison
    ks_results = []
    
    # combinations gets all unique pairs of models
    for model1, model2 in combinations(model_names, 2):
        
        # Extract the KGE series for both models, dropping any NaN values 
        kge1 = models_dict[model1][metric].dropna()
        kge2 = models_dict[model2][metric].dropna()
        
        # Perform the two-sample Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(kge1, kge2)
        
        # Store the results
        ks_results.append({
            'Model 1': model1,
            'Model 2': model2,
            'KS Statistic': ks_stat,
            'p-value': p_value
        })
    
    # Convert the results list into a DataFrame
    ks_df = pd.DataFrame(ks_results)
    ks_df = ks_df.sort_values(by='p-value').reset_index(drop=True)
    
    # Display the resulting DataFrame
    return ks_df


# ---------------------------------------------------------------------------
# CDF plots
# ---------------------------------------------------------------------------
def plot_score_cdfs(
    validation_results,
    trim_percent=None,
    model_name_map=None,
    catchment_type="Reservoir",
    flow_type="Without Flow",
    score_types=None,  
):
    """
    Plot cumulative distribution functions for each score type with paper-ready formatting.

    Parameters:
    -----------
    validation_results : dict
        Dictionary with model names as keys and score dictionaries as values
    trim_percent : float, optional
        If provided, trim the top and bottom trim_percent of data
    model_name_map : dict, optional
        Dictionary to rename models in the legend. If None, uses MODEL_NAME_MAP
    """

    TITLE_FONTSIZE = 20
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 16
    LEGEND_FONTSIZE = 14

    if model_name_map is None:
        model_name_map = MODEL_NAME_MAP

    _default_score_types = [
        "NSE",
        "NNSE",
        "KGE",
        "KGE_correlation",
        "KGE_alpha",
        "KGE_beta",
        "bias",
    ]
    score_types = score_types if score_types is not None else _default_score_types


    for score_type in score_types:
        plt.figure(figsize=(10, 8))

        for model_name, scores in validation_results.items():
            values = np.array(scores[score_type])

            # Apply trimming if specified
            if trim_percent is not None:
                lower = np.percentile(values, trim_percent)
                upper = np.percentile(values, 100 - trim_percent)
                values = values[(values >= lower) & (values <= upper)]

            # Sort values for CDF
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

            # Use mapped name if available, otherwise use original
            display_name = model_name_map.get(model_name, model_name)

            plt.plot(sorted_values, cumulative, linewidth=2.5, label=display_name)

        # Title and labels with larger fonts
        title_suffix = f" (trimmed {trim_percent}%)" if trim_percent is not None else ""
        plt.title(
            f"{catchment_type} Gauges CDF of {score_type}{title_suffix}: {flow_type}",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel(f"{score_type} Value", fontsize=LABEL_FONTSIZE, fontweight="bold")
        plt.ylabel("Cumulative Probability", fontsize=LABEL_FONTSIZE, fontweight="bold")

        # Larger tick labels
        plt.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

        # Larger legend
        plt.legend(
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
            shadow=True,
            loc="best",
            framealpha=0.9,
        )

        # Grid with better visibility
        plt.grid(True, linestyle="--", alpha=0.5, linewidth=1)

        plt.tight_layout()
        plt.show()


def plot_cdfs(
    avg_metrics_by_results,
    metrics=["KGE"],
    figsize_per_plot=(9, 6),
    title_names=None,
    model_names=None,
    model_colors=None,
    xlim = -0.5,
):
    # ── font sizes ────────────────────────────────────────────────────────────
    fs_title  = 20
    fs_xlabel = 24
    fs_ylabel = fs_xlabel
    fs_tick   = 16
    fs_legend = 18
    # ─────────────────────────────────────────────────────────────────────────
    if model_colors is None:
        model_colors = MODEL_COLORS
    results_names = list(avg_metrics_by_results.keys())
    n_cols = len(results_names)
    n_rows = len(metrics)
    if title_names is None:
        title_names = results_names
    if len(title_names) != n_cols:
        raise ValueError(
            f"title_names has {len(title_names)} entries but there are {n_cols} results dicts."
        )
    fallback_colors = cm.tab10(np.linspace(0, 0.9, 10))
    fallback_idx = 0
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    legend_handles = None

    for col_idx, (results_name, display_title) in enumerate(
        zip(results_names, title_names)
    ):
        model_dict = avg_metrics_by_results[results_name]
        models = list(model_dict.keys())
        n_models = len(models)

        if model_names is None:
            display_model_names = models
        else:
            if len(model_names) != len(models):
                raise ValueError(
                    f"model_names has {len(model_names)} entries but '{results_name}' "
                    f"has {len(models)} models."
                )
            display_model_names = model_names

        # ── linestyle: last 3 models get dashed, rest solid ──────────────────
        linestyles = ["solid"] * n_models
        for i in range(max(0, int(n_models/2)), n_models):
            linestyles[i] = (0, (5, 4))  # dashed
        # ─────────────────────────────────────────────────────────────────────

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for model_idx, (model, display_name) in enumerate(
                zip(models, display_model_names)
            ):
                df = model_dict[model]
                if metric not in df.columns:
                    print(
                        f"Warning: '{metric}' not found for model '{model}' "
                        f"in '{results_name}'. Skipping."
                    )
                    continue
                if display_name in model_colors:
                    color = model_colors[display_name]
                else:
                    color = fallback_colors[fallback_idx % 10]
                    fallback_idx += 1
                values = df[metric].dropna().values
                sorted_vals = np.sort(values)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(
                    sorted_vals, cdf,
                    label=display_name,
                    color=color,
                    linewidth=1.8,
                    linestyle=linestyles[model_idx],  # ← dashed for last 3
                )
            ax.set_xlabel(metric, fontsize=fs_xlabel)
            ax.set_ylabel("CDF" if col_idx == 0 else "", fontsize=fs_ylabel)
            # ax.set_title(display_title, fontsize=fs_title, fontweight="bold")
            ax.set_xlim(xlim, 1)
            ax.tick_params(labelsize=fs_tick)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    # ── move 3rd model (index 2) to the final legend slot ────────────────────
    if legend_handles is not None and len(legend_handles) > 3:
        item_h = legend_handles.pop(3)
        item_l = legend_labels.pop(3)
        legend_handles.append(item_h)
        legend_labels.append(item_l)
    # ─────────────────────────────────────────────────────────────────────────

    n_models = len(legend_handles)
    fig.legend(
        legend_handles, legend_labels,
        fontsize=fs_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=n_models,
        framealpha=0.7,
    )
    fig.tight_layout(pad=2.0)
    return fig


# ---------------------------------------------------------------------------
# Skill vs catchment attributes
# ---------------------------------------------------------------------------
def plot_models_vs_attributes(df, attributes, target_metric_cols, title_prefix, ylabel):
    # Set up a distinct color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_metric_cols)))
    
    fig, axes = plt.subplots(
        len(attributes), 1, 
        figsize=(10, 5 * len(attributes)), 
        squeeze=False
    )

    for i, attr in enumerate(attributes):
        ax = axes[i, 0]
        
        for col, color in zip(target_metric_cols, colors):
            # Drop NaNs for this specific pair
            valid_data = df[[attr, col]].dropna()
            x = valid_data[attr].values
            y = valid_data[col].values
            
            model_name = col.replace('NSE_', '').replace('Delta_NSE_', '')
            
            if len(x) > 2:
                # Linear regression
                res = stats.linregress(x, y)
                slope = res.slope
                
                # 95% Confidence Interval for the slope: slope +/- t_crit * stderr
                # degrees of freedom = n - 2
                t_crit = stats.t.ppf(0.975, df=len(x) - 2)
                ci_half_width = t_crit * res.stderr
                
                # Format label with slope and 95% CI
                label = f"{model_name}: m = {slope:.3f} ± {ci_half_width:.3f} (95% CI)"
                
                # Scatter points
                ax.scatter(x, y, color=color, alpha=0.7, edgecolors='k', linewidth=0.5)
                
                # Line of best fit
                x_vals = np.linspace(x.min(), x.max(), 100)
                y_vals = res.intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label)
            else:
                ax.scatter(x, y, color=color, alpha=0.7, label=model_name)

        if 'Delta' in title_prefix:
            ax.axhline(0, color='grey', linestyle=':', linewidth=1)

        ax.set_xlabel(attr, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title_prefix}: vs {attr}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Hydrographs
# ---------------------------------------------------------------------------
def plot_hydrograph_grid(
    panels, model_names, model_colors=None,
    start_date=None, end_date=None,
    zoom_start=None, zoom_days=90, zoom_end=None,
    save_path=None, dpi=300, panel_size=(5.4, 3.5), font_scale=1.0,
    legend_ncols=4, ylabel=r"Streamflow $(mm\ day^{-1})$",
    nse_fn=None, show=True,
):
    """
    One column per gauge. Top row: hydrograph over [start_date, end_date].
    Optional bottom row: zoom on `zoom_days` days from `zoom_start` (or up to `zoom_end`).

    panels      : list of dicts, one per gauge, each with
                  gauge_id, name, models (lagged results dict), gauge_ids, dates (the matching metadata).
                  Reservoir and natural gauges use different results/metadata, hence per-panel.
    zoom_start  : date string for all gauges, or {gauge_id: date} if the peak season differs per gauge.
    save_path   : e.g. 'figs/hydrographs.png' (.pdf also works).
    nse_fn      : defaults to plain NSE. Pass your notebook's calculate_nse to match the old legend values.
    Returns (fig, nse_df). nse_df has NSE per gauge, window ('full'/'zoom') and model.
    """
    if model_colors is None:
        model_colors = MODEL_COLORS
    if nse_fn is None:
        nse_fn = lambda p, t: 1 - np.sum((p - t) ** 2) / np.sum((t - np.mean(t)) ** 2)

    fs = lambda x: x * font_scale
    TITLE, LABEL, TICK, LEGEND = fs(14), fs(13), fs(11), fs(12)
    fallback_colors = cm.tab10(np.linspace(0, 0.9, 10))
    colors, fb = {}, 0
    for name in model_names:
        if name in model_colors:
            colors[name] = model_colors[name]
        else:
            colors[name] = fallback_colors[fb % 10]; fb += 1

    def get_zoom(gid):
        if zoom_start is None:
            return None
        zs = zoom_start.get(gid, zoom_start.get(str(gid))) if isinstance(zoom_start, dict) else zoom_start
        if zs is None:
            return None
        zs = pd.Timestamp(zs)
        ze = pd.Timestamp(zoom_end) if zoom_end is not None else zs + pd.Timedelta(days=zoom_days - 1)
        return zs, ze

    def get_series(panel):
        ids = np.asarray(panel["gauge_ids"]).astype(str)
        mask = ids == str(panel["gauge_id"])
        if not mask.any():
            raise ValueError(f"gauge_id {panel['gauge_id']} not found in this panel's metadata")
        dts = pd.to_datetime(pd.Series(list(panel["dates"])))[mask].values
        keys = list(panel["models"].keys())
        def mk(arr):
            s = pd.Series(np.asarray(arr)[mask], index=pd.DatetimeIndex(dts))
            return s[~s.index.duplicated()].sort_index()
        truth = mk(panel["models"][keys[0]]["Truth"])
        preds = [mk(panel["models"][k]["Prediction"]) for k in keys]
        return truth, preds

    n_cols = len(panels)
    n_rows = 2 if zoom_start is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows), squeeze=False)

    rows, letter = [], (lambda r, c: "abcdefghijklmnopqrstuvwxyz"[r * n_cols + c])   # a, b, c across the top row, then d, e, f
    for c, panel in enumerate(panels):
        gid = panel["gauge_id"]
        truth, preds = get_series(panel)
        a = pd.Timestamp(start_date) if start_date is not None else truth.index.min()
        b = pd.Timestamp(end_date) if end_date is not None else truth.index.max()
        windows = [("full", a, b)]
        z = get_zoom(gid)
        if z is not None:
            windows.append(("zoom", *z))

        for r, (wname, w0, w1) in enumerate(windows):
            ax = axes[r][c]
            idx = pd.date_range(w0, w1)           # daily axis: missing days show as gaps, not joined lines
            t = truth.reindex(idx)
            if t.notna().sum() == 0:
                ax.text(0.5, 0.5, f"No data\n{w0.date()} to {w1.date()}", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.plot(idx, t.values, color="black", lw=2 if wname == "full" else 2.2, ls="--", label="Observed", zorder=10)
            for name, p in zip(model_names, preds):
                ps = p.reindex(idx)
                ok = t.notna() & ps.notna()
                score = nse_fn(ps[ok].values, t[ok].values) if ok.sum() > 2 else np.nan
                rows.append({"gauge_id": gid, "name": panel.get("name", gid), "window": wname,
                             "start": w0.date(), "end": w1.date(), "model": name, "NSE": score})
                ax.plot(idx, ps.values, color=colors[name], alpha=0.85, lw=1.5,
                        ls=(0, (5, 4)) if "Hydra" in name else "solid", label=name)

            if wname == "full":
                if z is not None:
                    ax.axvspan(z[0], z[1], color="0.5", alpha=0.18, lw=0, zorder=0)
                ax.set_title(f"({letter(r, c)}) {panel.get('name', gid)}", fontsize=TITLE, fontweight="bold", loc="left")
            else:
                ax.set_title(f"({letter(r, c)}) {w0:%d %b %Y} – {w1:%d %b %Y}", fontsize=TITLE, loc="left")
                for sp in ax.spines.values():          # box the zoom panels, matching the grey band above
                    sp.set_visible(True); sp.set_color("0.5")
            loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ax.set_xlim(w0, w1)
            ax.tick_params(labelsize=TICK)
            ax.grid(True, ls=":", alpha=0.7)
            if wname == "full":
                ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL)

    handles, labels = None, None
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(h) == len(model_names) + 1:
            handles, labels = h, l; break
    if handles is not None:
        leg_rows = -(-len(labels) // legend_ncols)
        fig_h = panel_size[1] * n_rows
        bottom = (0.25 + 0.30 * leg_rows * font_scale) / fig_h
        fig.legend(handles, labels, loc="lower center", ncol=legend_ncols, fontsize=LEGEND,
                   frameon=True, edgecolor="grey", bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backwards compatibility: catchment-split code moved to ML_Catchment_Splits.py.
# Loaded on first use, so importing ML_Plots does not pull in geopandas/networkx/sklearn.
# ---------------------------------------------------------------------------
_MOVED_TO_SPLITS = ['assign_gauges_to_catchments', 'assign_splits_greedy', 'count_gauges_per_catchment', 'create_geographic_split', 'get_component_stats', 'get_component_stats_distance', 'load_and_filter_catchments', 'plot_catchment_sets', 'plot_catchment_sets_distance', 'plot_catchments_only', 'plot_catchments_with_gauges', 'plot_geographic_network', 'prepare_gauge_data']


def __getattr__(name):
    if name in _MOVED_TO_SPLITS:
        import importlib
        return getattr(importlib.import_module("ML_Catchment_Splits"), name)
    raise AttributeError(f"module 'ML_Plots' has no attribute {name!r}")
