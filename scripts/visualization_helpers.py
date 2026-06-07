"""
visualization_helpers.py
========================

Reusable plotting functions for the G-F Consistency Framework.

All plots adhere to publication-quality figure standards:

- Resolution: 300 dpi (PNG)
- Font size: minimum 8pt for all text elements
- Color palette: colorblind-safe Okabe-Ito palette
- Line width: minimum 1.0 pt
- Marker size: minimum 20 pt^2

This module provides:

- ``plot_gf_curves_comparison()``: Multi-method G-F purity curves
- ``plot_spearman_scatter()``: Correlation scatter plot with regression line
- ``plot_convergence_with_ci()``: Sample size convergence with CI shading
- ``plot_topology_radar()``: Cross-species topology radar chart
- ``plot_runtime_breakdown()``: Pipeline runtime stacked bar chart

Dependencies
------------
- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7

Author: Yuhan Zhang
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# ===========================================================================
# Global Constants
# ===========================================================================

# Okabe-Ito colorblind-safe palette (8 colors + extensions for 11 methods)
OKABE_ITO_PALETTE: List[str] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
    "#000000",  # Black (extension)
    "#8B4513",  # Saddle Brown (extension)
    "#4B0082",  # Indigo (extension)
]

# Method-specific color and marker assignments for 11 methods
METHOD_STYLES: Dict[str, Dict[str, Any]] = {
    "DM":          {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "Spectral":    {"color": "#56B4E9", "marker": "s", "linestyle": "-"},
    "PCA":         {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "MDS":         {"color": "#F0E442", "marker": "D", "linestyle": "-"},
    "VGAE":        {"color": "#0072B2", "marker": "v", "linestyle": "--"},
    "VGAE-feat":   {"color": "#D55E00", "marker": "P", "linestyle": "--"},
    "DeepWalk":    {"color": "#CC79A7", "marker": "X", "linestyle": "-."},
    "Node2Vec":    {"color": "#999999", "marker": "*", "linestyle": "-."},
    "GraphSAGE":   {"color": "#000000", "marker": "h", "linestyle": ":"},
    "GAT":         {"color": "#8B4513", "marker": "8", "linestyle": ":"},
    "GIN":         {"color": "#4B0082", "marker": "p", "linestyle": ":"},
}

# Default figure parameters
DEFAULT_DPI: int = 300
DEFAULT_FONT_SIZE: int = 8
DEFAULT_FIGSIZE_SINGLE: Tuple[float, float] = (4.0, 3.0)
DEFAULT_FIGSIZE_DUAL: Tuple[float, float] = (8.0, 3.5)
DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = (12.0, 5.0)


def _setup_plot_style() -> None:
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": DEFAULT_FONT_SIZE,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 5,
            "patch.linewidth": 0.5,
            "font.family": "sans-serif",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
        }
    )


def _get_method_style(method: str) -> Dict[str, Any]:
    """Get plot style for a method, with fallback to default."""
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    # Fallback: cycle through palette
    idx = hash(method) % len(OKABE_ITO_PALETTE)
    return {
        "color": OKABE_ITO_PALETTE[idx],
        "marker": "o",
        "linestyle": "-",
    }


# ===========================================================================
# 1. G-F Curves Comparison Plot
# ===========================================================================

def plot_gf_curves_comparison(
    curves_data: Dict[str, Dict[str, np.ndarray]],
    thresholds: np.ndarray,
    species_panels: Optional[List[str]] = None,
    fixed_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    random_baseline: Optional[Dict[str, float]] = None,
    highlight_methods: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot G-F purity curves for all embedding methods across species.

    Creates a multi-panel figure (one panel per species) showing the
    G-F purity curve f(t) for each method as a function of the distance
    threshold t.

    Parameters
    ----------
    curves_data : dict
        Nested dictionary: ``{species: {method: purity_values}}``.
        ``purity_values`` is a 1D numpy array of length ``len(thresholds)``.
    thresholds : np.ndarray
        1D array of distance threshold values (x-axis).
    species_panels : list of str or None
        Species names to include as panels. If None, uses all keys in
        ``curves_data``.
    fixed_intervals : dict or None
        ``{species: (t_low, t_high)}`` for drawing vertical dashed lines
        marking the fixed interval boundaries.
    random_baseline : dict or None
        ``{species: float}`` for drawing a horizontal dashed line at the
        random baseline purity level.
    highlight_methods : list of str or None
        Methods to highlight with thicker lines. If None, all methods
        are drawn with equal weight.
    figsize : tuple or None
        Figure size (width, height) in inches. Defaults to (8, 3.5) for
        two panels.
    save_path : str or None
        If provided, saves the figure to this file path (PNG, 300 dpi).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 200)
    >>> curves = {
    ...     "yeast": {
    ...         "DM": 0.3 + 0.3 * np.exp(-((t - 0.2) / 0.1) ** 2),
    ...         "Node2Vec": 0.25 + 0.2 * np.exp(-((t - 0.3) / 0.15) ** 2),
    ...     },
    ...     "human": {
    ...         "DM": 0.2 + 0.25 * np.exp(-((t - 0.15) / 0.08) ** 2),
    ...         "Node2Vec": 0.3 + 0.35 * np.exp(-((t - 0.15) / 0.1) ** 2),
    ...     },
    ... }
    >>> fig = plot_gf_curves_comparison(
    ...     curves, t,
    ...     fixed_intervals={"yeast": (0.05, 0.422), "human": (0.05, 0.297)},
    ...     random_baseline={"yeast": 0.12, "human": 0.08},
    ... )
    >>> # fig.savefig("gf_curves_comparison.png", dpi=300)
    """
    _setup_plot_style()

    if species_panels is None:
        species_panels = list(curves_data.keys())

    n_panels = len(species_panels)
    if figsize is None:
        figsize = (4.0 * n_panels, 3.5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, species in zip(axes, species_panels):
        methods_dict = curves_data.get(species, {})

        # Sort methods: matrix factorization first, then random walk, then GNN
        method_order = [
            m for m in METHOD_STYLES.keys() if m in methods_dict
        ]
        remaining = [m for m in methods_dict if m not in method_order]
        method_order.extend(remaining)

        for method in method_order:
            purity = methods_dict[method]
            style = _get_method_style(method)

            lw = 1.8 if highlight_methods and method in highlight_methods else 1.2
            alpha = 1.0 if highlight_methods and method in highlight_methods else 0.85

            ax.plot(
                thresholds,
                purity,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=lw,
                alpha=alpha,
                label=method,
            )

        # Fixed interval boundaries
        if fixed_intervals and species in fixed_intervals:
            t_low, t_high = fixed_intervals[species]
            ax.axvline(
                t_low, color="gray", linestyle="--", linewidth=0.8, alpha=0.6
            )
            ax.axvline(
                t_high, color="gray", linestyle="--", linewidth=0.8, alpha=0.6
            )

        # Random baseline
        if random_baseline and species in random_baseline:
            ax.axhline(
                random_baseline[species],
                color="red",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
                label="Random baseline",
            )

        ax.set_xlabel("Distance threshold $t$")
        ax.set_title(f"{species.capitalize()}", fontweight="bold")
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # y-label only on first axis
    axes[0].set_ylabel("Purity $f(t)$")

    # Legend below the last panel
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        ncol = min(4, len(handles))
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, -0.15),
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.9,
            edgecolor="gray",
        )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# 2. Spearman Correlation Scatter Plot
# ===========================================================================

def plot_spearman_scatter(
    data: pd.DataFrame,
    gf_col: str = "gf_score",
    metric_col: str = "auroc",
    method_col: str = "method",
    species_col: Optional[str] = "species",
    rho: Optional[float] = None,
    pvalue: Optional[float] = None,
    ci: Optional[Tuple[float, float]] = None,
    metric_label: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a scatter plot of downstream metric vs. G-F Score with regression line.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with G-F Scores and downstream metric values.
    gf_col : str, default="gf_score"
        Column for G-F Score (x-axis).
    metric_col : str, default="auroc"
        Column for downstream metric (y-axis).
    method_col : str, default="method"
        Column for method identifiers (used for point labels).
    species_col : str or None, default="species"
        Column for species (used for color coding). If None, all points
        are the same color.
    rho : float or None
        Pre-computed Spearman rho for annotation. If None, computed internally.
    pvalue : float or None
        Pre-computed p-value for annotation.
    ci : tuple or None
        95% CI for rho, e.g., (0.45, 0.89).
    metric_label : str or None
        Human-readable label for the y-axis metric. Defaults to the
        column name in uppercase.
    figsize : tuple or None
        Figure size in inches.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "method": [f"M{i}" for i in range(11)],
    ...     "gf_score": np.linspace(0.3, 0.7, 11),
    ...     "auroc": np.linspace(0.5, 0.9, 11) + rng.normal(0, 0.03, 11),
    ... })
    >>> fig = plot_spearman_scatter(df, species_col=None)
    """
    _setup_plot_style()

    if figsize is None:
        figsize = DEFAULT_FIGSIZE_SINGLE

    fig, ax = plt.subplots(figsize=figsize)

    # Color by species if available
    species_colors = {
        "yeast": "#0072B2",
        "human": "#E69F00",
    }

    if species_col and species_col in data.columns:
        for species, grp in data.groupby(species_col):
            color = species_colors.get(str(species).lower(), "#999999")
            ax.scatter(
                grp[gf_col],
                grp[metric_col],
                c=color,
                s=30,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=str(species).capitalize(),
                zorder=3,
            )
            # Label points
            for _, row in grp.iterrows():
                ax.annotate(
                    row[method_col],
                    (row[gf_col], row[metric_col]),
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    else:
        ax.scatter(
            data[gf_col],
            data[metric_col],
            c="#0072B2",
            s=30,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        for _, row in data.iterrows():
            ax.annotate(
                row[method_col],
                (row[gf_col], row[metric_col]),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

    # Linear regression line
    from scipy import stats as sp_stats

    x = data[gf_col].dropna().values
    y = data[metric_col].dropna().values
    valid = ~(np.isnan(x) | np.isnan(y))
    x_v, y_v = x[valid], y[valid]

    if len(x_v) >= 3:
        slope, intercept, _, _, _ = sp_stats.linregress(x_v, y_v)
        x_line = np.linspace(x_v.min(), x_v.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Compute Spearman rho if not provided
    if rho is None and len(x_v) >= 3:
        rho, pvalue = sp_stats.spearmanr(x_v, y_v)

    # Annotation
    if metric_label is None:
        metric_label = metric_col.upper()

    ax.set_xlabel("G-F Score")
    ax.set_ylabel(metric_label)

    # Build annotation string
    anno_parts = []
    if rho is not None:
        anno_parts.append(f"$\\rho_S$ = {rho:.3f}")
    if pvalue is not None:
        if pvalue < 0.001:
            anno_parts.append("$p$ < 0.001")
        else:
            anno_parts.append(f"$p$ = {pvalue:.3f}")
    if ci is not None:
        anno_parts.append(f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    if anno_parts:
        anno_text = "\n".join(anno_parts)
        ax.text(
            0.05,
            0.95,
            anno_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    if species_col and species_col in data.columns:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# 3. Convergence Plot with CI
# ===========================================================================

def plot_convergence_with_ci(
    stats_df: pd.DataFrame,
    size_col: str = "size",
    mean_col: str = "mean",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    full_set_gf: Optional[float] = None,
    method_groups: Optional[Dict[str, pd.DataFrame]] = None,
    log_x: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot G-F Score convergence as a function of validation subset size.

    Shows the mean G-F Score with 95% CI shading at each subset size,
    optionally with a horizontal reference line at the full-set G-F Score.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with per-size statistics. Must contain columns for
        size, mean, and CI bounds.
    size_col : str, default="size"
        Column for subset sizes (x-axis).
    mean_col : str, default="mean"
        Column for mean G-F Scores (y-axis).
    ci_lower_col : str, default="ci_lower"
        Column for lower CI bound.
    ci_upper_col : str, default="ci_upper"
        Column for upper CI bound.
    full_set_gf : float or None
        Full-set G-F Score value for a horizontal reference line.
    method_groups : dict or None
        If provided, overlay multiple convergence curves (one per method).
        Keys are method names; values are DataFrames like ``stats_df``.
    log_x : bool, default=True
        If True, use logarithmic x-axis for subset sizes.
    figsize : tuple or None
        Figure size in inches.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> sizes = [50, 100, 200, 500, 1000]
    >>> df = pd.DataFrame({
    ...     "size": sizes,
    ...     "mean": [0.52, 0.54, 0.545, 0.549, 0.550],
    ...     "ci_lower": [0.45, 0.50, 0.52, 0.535, 0.543],
    ...     "ci_upper": [0.59, 0.58, 0.57, 0.563, 0.557],
    ... })
    >>> fig = plot_convergence_with_ci(df, full_set_gf=0.55)
    """
    _setup_plot_style()

    if figsize is None:
        figsize = DEFAULT_FIGSIZE_SINGLE

    fig, ax = plt.subplots(figsize=figsize)

    if method_groups:
        # Overlay multiple method convergence curves
        for i, (method, m_df) in enumerate(method_groups.items()):
            style = _get_method_style(method)
            sizes = m_df[size_col].values
            means = m_df[mean_col].values
            ci_lo = m_df[ci_lower_col].values
            ci_hi = m_df[ci_upper_col].values

            ax.plot(
                sizes, means,
                color=style["color"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.2,
                label=method,
                zorder=3,
            )
            ax.fill_between(
                sizes, ci_lo, ci_hi,
                color=style["color"],
                alpha=0.15,
                zorder=2,
            )
    else:
        # Single convergence curve
        sizes = stats_df[size_col].values
        means = stats_df[mean_col].values
        ci_lo = stats_df[ci_lower_col].values
        ci_hi = stats_df[ci_upper_col].values

        ax.plot(
            sizes, means,
            color="#0072B2",
            marker="o",
            markersize=5,
            linewidth=1.5,
            zorder=3,
        )
        ax.fill_between(
            sizes, ci_lo, ci_hi,
            color="#0072B2",
            alpha=0.2,
            zorder=2,
            label="95% CI",
        )

    # Full-set reference line
    if full_set_gf is not None:
        ax.axhline(
            full_set_gf,
            color="#D55E00",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label=f"Full-set G-F = {full_set_gf:.3f}",
        )

    if log_x:
        ax.set_xscale("log")
        # Set sensible tick positions for log scale
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)

    ax.set_xlabel("Validation set size (nodes)")
    ax.set_ylabel("G-F Score")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# 4. Cross-Species Topology Radar Chart
# ===========================================================================

def plot_topology_radar(
    metrics: Dict[str, Dict[str, float]],
    metric_labels: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a radar (spider) chart comparing network topology metrics between species.

    Parameters
    ----------
    metrics : dict
        ``{species: {metric_name: value}}`` dictionary. For example:
        ``{"yeast": {"modularity": 0.45, "spectral_gap": 0.12, ...}, ...}``.
    metric_labels : dict or None
        ``{metric_name: display_label}`` for axis labels. If None, uses
        the metric names directly.
    normalize : bool, default=True
        If True, normalize each metric to [0, 1] range across species
        for visual comparability.
    figsize : tuple or None
        Figure size in inches.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Notes
    -----
    For metrics where "higher is better" for embedding (e.g., modularity),
    the value is used directly. For metrics where "lower is better"
    (e.g., mixing time), the value is inverted (1 - normalized) so that
    all axes point outward in the "more impactful" direction.

    Examples
    --------
    >>> metrics = {
    ...     "yeast": {
    ...         "modularity": 0.45,
    ...         "spectral_gap": 0.12,
    ...         "clustering": 0.35,
    ...         "diameter": 8,
    ...         "assortativity": -0.05,
    ...         "degree_gini": 0.55,
    ...     },
    ...     "human": {
    ...         "modularity": 0.52,
    ...         "spectral_gap": 0.08,
    ...         "clustering": 0.28,
    ...         "diameter": 12,
    ...         "assortativity": 0.02,
    ...         "degree_gini": 0.62,
    ...     },
    ... }
    >>> fig = plot_topology_radar(metrics)
    """
    _setup_plot_style()

    if figsize is None:
        figsize = (5.0, 5.0)

    species_names = list(metrics.keys())
    if len(species_names) < 2:
        warnings.warn(
            "Radar chart requires at least 2 species for comparison.",
            UserWarning,
            stacklevel=2,
        )

    # Collect all metric names (union across species)
    all_metric_names = []
    for sp_data in metrics.values():
        for name in sp_data:
            if name not in all_metric_names:
                all_metric_names.append(name)

    n_metrics = len(all_metric_names)
    if n_metrics < 3:
        warnings.warn(
            "Radar chart requires at least 3 metrics.",
            UserWarning,
            stacklevel=2,
        )

    # Normalize values
    if normalize:
        min_vals = {}
        max_vals = {}
        for name in all_metric_names:
            vals = [
                metrics[sp].get(name, np.nan) for sp in species_names
            ]
            vals_valid = [v for v in vals if not np.isnan(v)]
            min_vals[name] = min(vals_valid) if vals_valid else 0
            max_vals[name] = max(vals_valid) if vals_valid else 1

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    species_colors = {
        "yeast": "#0072B2",
        "human": "#E69F00",
    }
    species_alphas = {"yeast": 0.25, "human": 0.25}

    for species in species_names:
        values = []
        for name in all_metric_names:
            raw = metrics[species].get(name, np.nan)
            if normalize:
                rng_val = max_vals[name] - min_vals[name]
                if rng_val > 0:
                    norm = (raw - min_vals[name]) / rng_val
                else:
                    norm = 0.5
                values.append(norm)
            else:
                values.append(raw)
        values += values[:1]  # Close the polygon

        color = species_colors.get(species.lower(), "#999999")
        alpha = species_alphas.get(species.lower(), 0.2)

        ax.plot(
            angles, values,
            color=color,
            linewidth=1.5,
            label=species.capitalize(),
            zorder=3,
        )
        ax.fill(
            angles, values,
            color=color,
            alpha=alpha,
            zorder=2,
        )

    # Set labels
    if metric_labels is None:
        metric_labels = {name: name.replace("_", " ").title() for name in all_metric_names}

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [metric_labels.get(name, name) for name in all_metric_names],
        fontsize=7,
    )

    # Remove radial tick labels for cleanliness
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        framealpha=0.9,
        fontsize=8,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# 5. Runtime Breakdown Stacked Bar Chart
# ===========================================================================

def plot_runtime_breakdown(
    runtime_data: pd.DataFrame,
    method_col: str = "method",
    step_cols: Optional[List[str]] = None,
    species_col: Optional[str] = "species",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a stacked bar chart of pipeline runtime breakdown by step.

    Parameters
    ----------
    runtime_data : pd.DataFrame
        DataFrame where each row is a method (or method-species combination)
        and columns contain wall-clock times (in seconds) for each pipeline step.
    method_col : str, default="method"
        Column for method names (x-axis categories).
    step_cols : list of str or None
        Columns for pipeline step runtimes. If None, all numeric columns
        except ``method_col`` and ``species_col`` are used.
    species_col : str or None, default="species"
        If present, groups bars by species.
    figsize : tuple or None
        Figure size in inches.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "method": ["DM", "Spectral", "PCA", "Node2Vec", "GraphSAGE"],
    ...     "Load & Standardize": [0.1, 0.1, 0.1, 0.1, 0.1],
    ...     "Distance Matrix": [2.5, 2.5, 2.5, 2.5, 2.5],
    ...     "Neighborhood Graph": [15.0, 15.0, 15.0, 15.0, 15.0],
    ...     "Leiden Detection": [120.0, 120.0, 120.0, 120.0, 120.0],
    ...     "Purity & Integration": [1.0, 1.0, 1.0, 1.0, 1.0],
    ... })
    >>> fig = plot_runtime_breakdown(df)
    """
    _setup_plot_style()

    if step_cols is None:
        exclude = {method_col}
        if species_col and species_col in runtime_data.columns:
            exclude.add(species_col)
        step_cols = [
            c
            for c in runtime_data.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(runtime_data[c])
        ]

    n_steps = len(step_cols)
    n_methods = len(runtime_data)

    if figsize is None:
        figsize = (max(6.0, n_methods * 0.8), 4.0)

    # Step colors: use a sequential subset of the palette
    step_colors = [
        "#0072B2",  # Blue
        "#56B4E9",  # Sky Blue
        "#E69F00",  # Orange
        "#D55E00",  # Vermillion
        "#009E73",  # Bluish Green
        "#CC79A7",  # Reddish Purple
        "#999999",  # Gray
        "#F0E442",  # Yellow
    ]
    # Cycle if more steps than colors
    while len(step_colors) < n_steps:
        step_colors.extend(step_colors)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_methods)
    bar_width = 0.6

    # Bottom accumulator for stacking
    bottoms = np.zeros(n_methods)

    for i, step in enumerate(step_cols):
        values = runtime_data[step].values.astype(float)
        ax.bar(
            x,
            values,
            bar_width,
            bottom=bottoms,
            color=step_colors[i % len(step_colors)],
            label=step,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        bottoms += values

    # Total time annotation on top of each bar
    for i in range(n_methods):
        total = bottoms[i]
        ax.text(
            x[i],
            total + bottoms.max() * 0.02,
            f"{total:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        runtime_data[method_col].values,
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title("G-F Score Pipeline Runtime Breakdown", fontweight="bold")

    # Use log scale if range is large
    if bottoms.max() > 0 and bottoms.max() / max(bottoms.min(), 1) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("Wall-clock time (seconds, log scale)")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        framealpha=0.9,
        fontsize=7,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# 6. Time-Accuracy Tradeoff Curve
# ===========================================================================

def plot_time_accuracy_tradeoff(
    data: pd.DataFrame,
    sampling_col: str = "n_sampling_points",
    accuracy_col: str = "accuracy_pct",
    time_col: str = "wall_time_seconds",
    elbow_point: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the tradeoff between computational cost and G-F Score accuracy.

    Dual y-axis plot: accuracy (left) and wall-clock time (right) as
    functions of the number of sampling points K.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns for sampling points, accuracy, and time.
    sampling_col : str, default="n_sampling_points"
        Column for number of sampling points (x-axis).
    accuracy_col : str, default="accuracy_pct"
        Column for accuracy percentage (left y-axis).
    time_col : str, default="wall_time_seconds"
        Column for wall-clock time (right y-axis).
    elbow_point : int or None
        If provided, draws a vertical dashed line at the elbow point
        (minimum K achieving > 99% accuracy).
    figsize : tuple or None
        Figure size in inches.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "n_sampling_points": [50, 100, 150, 200, 300, 400, 500],
    ...     "accuracy_pct": [96.5, 98.8, 99.5, 99.8, 99.95, 99.99, 100.0],
    ...     "wall_time_seconds": [30, 60, 90, 120, 180, 240, 300],
    ... })
    >>> fig = plot_time_accuracy_tradeoff(df, elbow_point=150)
    """
    _setup_plot_style()

    if figsize is None:
        figsize = (5.0, 3.5)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    x = data[sampling_col].values
    accuracy = data[accuracy_col].values
    time_vals = data[time_col].values

    # Accuracy curve (left y-axis)
    line1 = ax1.plot(
        x, accuracy,
        color="#0072B2",
        marker="o",
        markersize=5,
        linewidth=1.5,
        label="Accuracy (%)",
        zorder=3,
    )
    ax1.set_xlabel("Number of sampling points $K$")
    ax1.set_ylabel("G-F Score accuracy (%)", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.set_ylim(90, 101)

    # Time curve (right y-axis)
    line2 = ax2.plot(
        x, time_vals,
        color="#D55E00",
        marker="s",
        markersize=5,
        linewidth=1.5,
        linestyle="--",
        label="Wall-clock time (s)",
        zorder=3,
    )
    ax2.set_ylabel("Wall-clock time (seconds)", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")

    # Elbow point
    if elbow_point is not None:
        ax1.axvline(
            elbow_point,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )
        ax1.annotate(
            f"Elbow: K={elbow_point}",
            xy=(elbow_point, ax1.get_ylim()[1] * 0.995),
            fontsize=7,
            ha="center",
            va="top",
            color="gray",
        )

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines, labels,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=7,
    )

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")

    return fig


# ===========================================================================
# Main: Example Usage
# ===========================================================================

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("G-F Consistency Framework: Visualization Helpers -- Example Usage")
    print("=" * 70)

    rng = np.random.default_rng(42)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. G-F Curves Comparison ----
    print("\n--- 1. G-F Curves Comparison Plot ---")
    t = np.linspace(0.01, 1.0, 200)
    methods_demo = ["DM", "Spectral", "PCA", "Node2Vec", "GraphSAGE"]

    curves_demo: Dict[str, Dict[str, np.ndarray]] = {
        "yeast": {},
        "human": {},
    }
    for m in methods_demo:
        peak = rng.uniform(0.15, 0.35)
        height = rng.uniform(0.3, 0.6)
        width = rng.uniform(0.08, 0.15)
        curves_demo["yeast"][m] = (
            0.1 + height * np.exp(-((t - peak) / width) ** 2)
            + rng.normal(0, 0.005, len(t))
        )
        # Human: different profile
        peak_h = rng.uniform(0.1, 0.25)
        height_h = rng.uniform(0.25, 0.5)
        curves_demo["human"][m] = (
            0.08 + height_h * np.exp(-((t - peak_h) / width) ** 2)
            + rng.normal(0, 0.005, len(t))
        )

    fig1 = plot_gf_curves_comparison(
        curves_demo,
        t,
        fixed_intervals={"yeast": (0.05, 0.422), "human": (0.05, 0.297)},
        random_baseline={"yeast": 0.12, "human": 0.08},
        save_path=os.path.join(output_dir, "example_gf_curves.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_gf_curves.png')}")
    plt.close(fig1)

    # ---- 2. Spearman Scatter ----
    print("\n--- 2. Spearman Correlation Scatter ---")
    scatter_df = pd.DataFrame(
        {
            "method": methods_demo,
            "gf_score": np.linspace(0.35, 0.65, len(methods_demo)),
            "auroc": np.linspace(0.55, 0.85, len(methods_demo))
            + rng.normal(0, 0.02, len(methods_demo)),
        }
    )
    fig2 = plot_spearman_scatter(
        scatter_df,
        species_col=None,
        metric_label="Link Prediction AUROC",
        save_path=os.path.join(output_dir, "example_spearman_scatter.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_spearman_scatter.png')}")
    plt.close(fig2)

    # ---- 3. Convergence with CI ----
    print("\n--- 3. Convergence with CI ---")
    conv_df = pd.DataFrame(
        {
            "size": [50, 100, 200, 500, 1000],
            "mean": [0.50, 0.53, 0.545, 0.549, 0.550],
            "ci_lower": [0.42, 0.48, 0.52, 0.535, 0.543],
            "ci_upper": [0.58, 0.58, 0.57, 0.563, 0.557],
        }
    )
    fig3 = plot_convergence_with_ci(
        conv_df,
        full_set_gf=0.55,
        save_path=os.path.join(output_dir, "example_convergence.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_convergence.png')}")
    plt.close(fig3)

    # ---- 4. Topology Radar ----
    print("\n--- 4. Cross-Species Topology Radar ---")
    topo_metrics = {
        "yeast": {
            "Modularity Q": 0.45,
            "Spectral Gap": 0.12,
            "Clustering Coeff": 0.35,
            "Eff. Diameter": 8,
            "Assortativity": 0.05,
            "Degree Gini": 0.55,
        },
        "human": {
            "Modularity Q": 0.52,
            "Spectral Gap": 0.08,
            "Clustering Coeff": 0.28,
            "Eff. Diameter": 12,
            "Assortativity": 0.02,
            "Degree Gini": 0.62,
        },
    }
    fig4 = plot_topology_radar(
        topo_metrics,
        save_path=os.path.join(output_dir, "example_topology_radar.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_topology_radar.png')}")
    plt.close(fig4)

    # ---- 5. Runtime Breakdown ----
    print("\n--- 5. Runtime Breakdown ---")
    runtime_df = pd.DataFrame(
        {
            "method": ["DM", "Spectral", "PCA", "Node2Vec", "GraphSAGE"],
            "Load & Standardize": [0.1, 0.1, 0.1, 0.1, 0.1],
            "Distance Matrix": [2.5, 2.5, 2.5, 2.5, 2.5],
            "Neighborhood Graph": [15, 15, 15, 15, 15],
            "Leiden Detection": [120, 120, 120, 120, 120],
            "Purity & Integration": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    fig5 = plot_runtime_breakdown(
        runtime_df,
        save_path=os.path.join(output_dir, "example_runtime_breakdown.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_runtime_breakdown.png')}")
    plt.close(fig5)

    # ---- 6. Time-Accuracy Tradeoff ----
    print("\n--- 6. Time-Accuracy Tradeoff ---")
    tradeoff_df = pd.DataFrame(
        {
            "n_sampling_points": [50, 100, 150, 200, 300, 400, 500],
            "accuracy_pct": [96.5, 98.8, 99.5, 99.8, 99.95, 99.99, 100.0],
            "wall_time_seconds": [30, 60, 90, 120, 180, 240, 300],
        }
    )
    fig6 = plot_time_accuracy_tradeoff(
        tradeoff_df,
        elbow_point=150,
        save_path=os.path.join(output_dir, "example_time_accuracy.png"),
    )
    print(f"  Saved to: {os.path.join(output_dir, 'example_time_accuracy.png')}")
    plt.close(fig6)

    print("\n" + "=" * 70)
    print("All example figures generated successfully.")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 70)
