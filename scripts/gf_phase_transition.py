#!/usr/bin/env python3
"""
G-F Curve Phase Transition Analysis (Step 46 / Phase 17)
========================================================

Compute numerical derivatives of G-F purity curves d(purity)/dr and
d^2(purity)/dr^2 for all embedding methods.  Identify critical radii
where functional organisation undergoes sharp transitions.  Test
coincidence with Betti-curve topological transitions and extract
critical exponents.

Key questions
-------------
1. Does protein functional organisation have a characteristic length
   scale (critical radius) analogous to phase transitions in physical
   systems?
2. Do high-GF methods have sharper transitions?
3. Are critical radii conserved across species?
4. Do functional (purity) and topological (Betti) transitions coincide?

Output
------
- results/gf_phase_transition.json
- figures/FigS21_gf_derivatives.png
- figures/FigS22_critical_radii.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    get_results_dir, get_figures_dir, SEED,
)

# ============================================================
# Constants
# ============================================================

RESULTS = get_results_dir()
FIGURES = get_figures_dir()
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colourblind-safe palette
PALETTE = {
    "Spectral":   "#0072B2",
    "DM":         "#E69F00",
    "MDS":        "#009E73",
    "PCA":        "#56B4E9",
    "Node2Vec":   "#D55E00",
    "DeepWalk":   "#CC79A7",
    "VGAE":       "#F0E442",
    "VGAE-feat":  "#999999",
    "GraphSAGE":  "#0173B2",
    "GAT":        "#DE8F05",
    "GIN":        "#029E73",
    "Random":     "#CCCCCC",
}

# Methods available in gf_curves_200pts.json
YEAST_METHODS = ["Spectral", "DM", "MDS", "PCA", "Node2Vec",
                 "DeepWalk", "VGAE", "VGAE-feat"]

# Top-GF vs low-GF grouping (for sharpness comparison)
HIGH_GF = ["Spectral", "DM", "MDS"]
LOW_GF  = ["VGAE", "VGAE-feat", "GAT"]


# ============================================================
# Numerical derivatives with Savitzky-Golay smoothing
# ============================================================

def smooth_derivative(r, y, window=21, poly=3):
    """Compute smoothed first derivative via Savitzky-Golay filter.

    Parameters
    ----------
    r : array
        Independent variable (radius values, uniformly spaced).
    y : array
        Purity values.
    window : int
        Window length for SG filter (must be odd, > poly).
    poly : int
        Polynomial order for SG filter.

    Returns
    -------
    dy_dr : array
        First derivative (same length as y, edge effects present).
    """
    dr = r[1] - r[0]
    # Ensure window is odd and <= len(y)
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1
    if w < poly + 2:
        w = poly + 2
        if w % 2 == 0:
            w += 1
    dy = savgol_filter(y, window_length=w, polyorder=poly,
                       deriv=1, delta=dr)
    return dy


def smooth_second_derivative(r, y, window=21, poly=4):
    """Compute smoothed second derivative via Savitzky-Golay filter."""
    dr = r[1] - r[0]
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1
    if w < poly + 2:
        w = poly + 2
        if w % 2 == 0:
            w += 1
    d2y = savgol_filter(y, window_length=w, polyorder=poly,
                        deriv=2, delta=dr)
    return d2y


# ============================================================
# Critical-point detection
# ============================================================

def find_purity_peak(r, purity):
    """Find the radius of maximum purity."""
    idx = np.argmax(purity)
    return float(r[idx]), float(purity[idx]), int(idx)


def find_inflection_points(r, d2y):
    """Find radii where d^2(purity)/dr^2 crosses zero (inflection points).

    Returns list of (r_inflect, index) tuples.
    """
    inflections = []
    for i in range(1, len(d2y)):
        if d2y[i - 1] * d2y[i] < 0:
            # Linear interpolation for zero-crossing
            frac = abs(d2y[i - 1]) / (abs(d2y[i - 1]) + abs(d2y[i]))
            r_cross = r[i - 1] + frac * (r[i] - r[i - 1])
            inflections.append((float(r_cross), i))
    return inflections


def find_betti_half_merge(r, betti0):
    """Find the radius where B0(r) = n_initial / 2 (topological percolation).

    B0 is the number of connected components, monotonically decreasing.
    """
    if len(betti0) == 0:
        return None, None
    n_initial = betti0[0]
    half_n = n_initial / 2.0
    for i in range(len(betti0)):
        if betti0[i] <= half_n:
            return float(r[i]), i
    return None, None


def find_betti1_peak(r, betti1):
    """Find the radius of maximum B1 (loop count)."""
    if len(betti1) == 0 or max(betti1) == 0:
        return None, None, None
    idx = int(np.argmax(betti1))
    return float(r[idx]), int(betti1[idx]), idx


def compute_critical_exponent(r, purity, r_c, side="left", n_fit=20):
    """Estimate the critical exponent beta near r_c.

    Near the critical point: purity(r) ~ |r - r_c|^beta.
    Fit log(purity) ~ beta * log(|r - r_c|) in a window around r_c.

    Parameters
    ----------
    side : str
        "left" = approach from below r_c, "right" = above r_c,
        "both" = both sides.
    n_fit : int
        Number of points to use on each side.
    """
    idx_c = np.argmin(np.abs(r - r_c))

    if side in ("left", "both"):
        lo = max(0, idx_c - n_fit)
        hi = idx_c
        if hi - lo < 5:
            return None, None
        dr_left = np.abs(r[lo:hi] - r_c)
        p_left = purity[lo:hi]
        mask_l = (dr_left > 1e-10) & (p_left > 1e-10)
        if mask_l.sum() < 5:
            return None, None
        log_dr_l = np.log(dr_left[mask_l])
        log_p_l = np.log(p_left[mask_l])

    if side in ("right", "both"):
        lo = idx_c + 1
        hi = min(len(r), idx_c + n_fit + 1)
        if hi - lo < 5:
            return None, None
        dr_right = np.abs(r[lo:hi] - r_c)
        p_right = purity[lo:hi]
        mask_r = (dr_right > 1e-10) & (p_right > 1e-10)
        if mask_r.sum() < 5:
            return None, None
        log_dr_r = np.log(dr_right[mask_r])
        log_p_r = np.log(p_right[mask_r])

    if side == "left":
        log_dr, log_p = log_dr_l, log_p_l
    elif side == "right":
        log_dr, log_p = log_dr_r, log_p_r
    else:
        log_dr = np.concatenate([log_dr_l, log_dr_r])
        log_p = np.concatenate([log_p_l, log_p_r])

    if len(log_dr) < 5:
        return None, None

    # Linear fit: log(purity) = beta * log(|r - r_c|) + const
    coeffs = np.polyfit(log_dr, log_p, 1)
    beta = float(coeffs[0])
    r_squared = 1 - np.var(np.polyval(coeffs, log_dr) - log_p) / np.var(log_p)

    return beta, float(r_squared) if np.var(log_p) > 1e-20 else None


def compute_transition_sharpness(r, purity):
    """Quantify the sharpness of the purity peak.

    Metrics:
    - max_slope: maximum |d(purity)/dr|
    - FWHM: full width at half maximum of the purity curve
    - peak_prominence: peak purity minus baseline purity
    """
    dy = smooth_derivative(r, purity)
    max_slope = float(np.max(np.abs(dy)))

    peak_val = np.max(purity)
    half_max = peak_val / 2.0
    above_mask = purity >= half_max
    # Find contiguous runs above half-max and pick the widest one
    fwhm = None
    if np.any(above_mask):
        diffs = np.diff(above_mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        # Handle edge cases where the run starts at index 0 or ends at last index
        if above_mask[0]:
            starts = np.concatenate(([0], starts))
        if above_mask[-1]:
            ends = np.concatenate((ends, [len(above_mask)]))
        if len(starts) > 0 and len(starts) == len(ends):
            widths = ends - starts
            best = np.argmax(widths)
            fwhm = float(r[ends[best] - 1] - r[starts[best]])

    baseline = np.mean(purity[:10])  # first 10 points as baseline
    prominence = float(peak_val - baseline)

    return {
        "max_slope": max_slope,
        "FWHM": fwhm,
        "peak_prominence": prominence,
    }


# ============================================================
# Main analysis
# ============================================================

def run():
    """Run the full phase transition analysis."""
    t0 = time.time()
    print("=" * 64)
    print("  Phase 17: G-F Curve Phase Transition Analysis")
    print("=" * 64)

    # ---- Load yeast G-F curves ----
    gf_file = RESULTS / "gf_curves_200pts.json"
    print(f"\n[1/5] Loading yeast G-F curves from {gf_file.name} ...")
    with open(gf_file, encoding="utf-8") as f:
        gf_data = json.load(f)

    r = np.array(gf_data["r"])
    n_r = len(r)
    print(f"  r: {r[0]:.4f} to {r[-1]:.4f}, {n_r} points")

    # ---- Load topological (Betti) curves ----
    topo_file = RESULTS / "topological_analysis.json"
    print(f"\n[2/5] Loading Betti curves from {topo_file.name} ...")
    with open(topo_file, encoding="utf-8") as f:
        topo_data = json.load(f)

    r_topo = np.array(topo_data["r_vals"])
    betti_curves = topo_data["betti_curves"]
    persist_stats = topo_data["persistence_statistics"]
    topo_methods = topo_data["methods"]
    print(f"  Topological methods: {len(topo_methods)}")

    # ---- Compute derivatives for each method ----
    print(f"\n[3/5] Computing derivatives and critical points ...")
    method_results = {}

    for method in YEAST_METHODS:
        purity_key = f"{method}_purity"
        if purity_key not in gf_data:
            print(f"  SKIP {method}: no purity data")
            continue

        purity = np.array(gf_data[purity_key])

        # First derivative
        dy = smooth_derivative(r, purity)
        # Second derivative
        d2y = smooth_second_derivative(r, purity)

        # Critical points
        r_peak, peak_purity, idx_peak = find_purity_peak(r, purity)
        inflections = find_inflection_points(r, d2y)
        sharpness = compute_transition_sharpness(r, purity)

        # Critical exponent near the peak
        beta_left, r2_left = compute_critical_exponent(
            r, purity, r_peak, side="left", n_fit=25
        )
        beta_right, r2_right = compute_critical_exponent(
            r, purity, r_peak, side="right", n_fit=25
        )

        # Betti curve analysis (if available)
        betti_info = {}
        if method in betti_curves:
            b0 = np.array(betti_curves[method]["0"])
            b1 = np.array(betti_curves[method]["1"])
            r_half, idx_half = find_betti_half_merge(r_topo, b0)
            r_b1_peak, b1_max, idx_b1 = find_betti1_peak(r_topo, b1)
            betti_info = {
                "r_half_merge": r_half,
                "B1_max": b1_max,
                "r_B1_peak": r_b1_peak,
            }

        # Coincidence test: does purity inflection ~ Betti percolation?
        coincidence = None
        if betti_info.get("r_half_merge") and inflections:
            r_perc = betti_info["r_half_merge"]
            # Find closest inflection point to percolation radius
            dists = [abs(inf_r - r_perc) for inf_r, _ in inflections]
            min_dist = min(dists)
            dr = r[1] - r[0]
            coincidence = {
                "r_percolation": r_perc,
                "nearest_inflection_dist": float(min_dist),
                "coincident": bool(min_dist < 5 * dr),  # within 5 grid spacings
            }

        method_results[method] = {
            "r_peak": r_peak,
            "peak_purity": peak_purity,
            "inflection_points": [ri for ri, _ in inflections],
            "n_inflections": len(inflections),
            "critical_exponent_left": beta_left,
            "critical_exponent_left_R2": r2_left,
            "critical_exponent_right": beta_right,
            "critical_exponent_right_R2": r2_right,
            "sharpness": sharpness,
            "betti": betti_info,
            "coincidence": coincidence,
            # Store derivatives for plotting
            "_dy": dy,
            "_d2y": d2y,
            "_purity": purity,
        }

        print(f"  {method:12s}  peak r={r_peak:.3f}  purity={peak_purity:.3f}  "
              f"inflections={len(inflections)}  "
              f"beta_L={beta_left if beta_left is not None else 'N/A'}  "
              f"sharpness={sharpness['max_slope']:.2f}")

    # ---- Also add random baseline ----
    if "random_baseline_purity" in gf_data:
        rb_purity = np.array(gf_data["random_baseline_purity"])
        rb_dy = smooth_derivative(r, rb_purity)
        rb_d2y = smooth_second_derivative(r, rb_purity)
        r_peak_rb, peak_rb, _ = find_purity_peak(r, rb_purity)
        method_results["Random"] = {
            "r_peak": r_peak_rb,
            "peak_purity": peak_rb,
            "inflection_points": [],
            "n_inflections": 0,
            "sharpness": compute_transition_sharpness(r, rb_purity),
            "betti": {},
            "coincidence": None,
            "_dy": rb_dy,
            "_d2y": rb_d2y,
            "_purity": rb_purity,
        }

    # ---- Cross-method comparison ----
    print(f"\n[4/5] Cross-method analysis ...")

    # Sharpness vs GF-score correlation
    from scipy.stats import spearmanr

    # Load GF scores for correlation
    gf_scores_file = RESULTS / "gf_scores.json"
    gf_scores = {}
    if gf_scores_file.exists():
        with open(gf_scores_file, encoding="utf-8") as f:
            gs = json.load(f)
        gf_scores = gs.get("scores", gs.get("scores_paper_interval", {}))

    methods_with_gf = []
    sharpness_vals = []
    gf_vals = []
    for m in YEAST_METHODS:
        if m in method_results and m in gf_scores:
            methods_with_gf.append(m)
            sharpness_vals.append(method_results[m]["sharpness"]["max_slope"])
            gf_vals.append(gf_scores[m])

    rho_sharpness_gf = None
    p_sharpness_gf = None
    if len(methods_with_gf) >= 5:
        rho_sharpness_gf, p_sharpness_gf = spearmanr(sharpness_vals, gf_vals)
        print(f"  Sharpness-GF Spearman rho = {rho_sharpness_gf:+.3f} "
              f"(p = {p_sharpness_gf:.4f}, n = {len(methods_with_gf)})")

    # Peak radius conservation (std across methods)
    peak_radii = [method_results[m]["r_peak"] for m in YEAST_METHODS
                  if m in method_results]
    print(f"  Peak radius: mean={np.mean(peak_radii):.4f}, "
          f"std={np.std(peak_radii):.4f}, range=[{min(peak_radii):.4f}, "
          f"{max(peak_radii):.4f}]")

    # High-GF vs Low-GF sharpness comparison
    high_sharpness = [method_results[m]["sharpness"]["max_slope"]
                      for m in HIGH_GF if m in method_results]
    low_sharpness = [method_results[m]["sharpness"]["max_slope"]
                     for m in LOW_GF if m in method_results]
    from scipy.stats import mannwhitneyu
    if high_sharpness and low_sharpness:
        u_stat, u_p = mannwhitneyu(high_sharpness, low_sharpness,
                                   alternative="greater")
        print(f"  High-GF vs Low-GF sharpness (Mann-Whitney): "
              f"U={u_stat}, p={u_p:.4f}")
    else:
        u_stat, u_p = None, None

    # ---- Generate figures ----
    print(f"\n[5/5] Generating figures ...")
    plot_gf_derivatives(r, method_results)
    plot_critical_radii(r, method_results, betti_curves, r_topo)

    # ---- Save results ----
    output = {
        "description": "Phase 17: G-F Curve Phase Transition Analysis",
        "r_range": [float(r[0]), float(r[-1])],
        "n_points": n_r,
        "methods_analysed": [m for m in YEAST_METHODS if m in method_results],
        "per_method": {},
        "cross_method": {
            "sharpness_gf_rho": rho_sharpness_gf,
            "sharpness_gf_p": p_sharpness_gf,
            "n_methods": len(methods_with_gf),
            "peak_radius_mean": float(np.mean(peak_radii)),
            "peak_radius_std": float(np.std(peak_radii)),
            "high_gf_sharpness_mean": float(np.mean(high_sharpness)) if high_sharpness else None,
            "low_gf_sharpness_mean": float(np.mean(low_sharpness)) if low_sharpness else None,
            "mannwhitney_U": float(u_stat) if u_stat is not None else None,
            "mannwhitney_p": float(u_p) if u_p is not None else None,
        },
    }

    # Strip numpy arrays before saving
    for method, mdata in method_results.items():
        clean = {}
        for k, v in mdata.items():
            if k.startswith("_"):
                continue  # skip numpy arrays
            clean[k] = v
        output["per_method"][method] = clean

    out_file = RESULTS / "gf_phase_transition.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_file}")

    elapsed = time.time() - t0
    print(f"\nPhase 17 completed in {elapsed:.1f}s")
    return output


# ============================================================
# Figure: G-F Curve Derivatives (FigS21)
# ============================================================

def plot_gf_derivatives(r, method_results):
    """Three-panel figure: purity, d(purity)/dr, d^2(purity)/dr^2."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    for method in YEAST_METHODS:
        if method not in method_results:
            continue
        md = method_results[method]
        color = PALETTE.get(method, "#333333")

        # Panel A: purity curves
        axes[0].plot(r, md["_purity"], color=color, linewidth=1.8,
                     label=method)

        # Panel B: first derivative
        axes[1].plot(r, md["_dy"], color=color, linewidth=1.5,
                     label=method)

        # Panel C: second derivative
        axes[2].plot(r, md["_d2y"], color=color, linewidth=1.5,
                     label=method)

        # Mark peak
        axes[0].axvline(md["r_peak"], color=color, alpha=0.2,
                        linestyle=":", linewidth=0.8)

    # Random baseline
    if "Random" in method_results:
        axes[0].plot(r, method_results["Random"]["_purity"],
                     color=PALETTE["Random"], linewidth=2,
                     linestyle="--", label="Random baseline")

    # Zero line for derivatives
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="-")
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="-")

    # G-F integration interval shading
    for ax in axes:
        ax.axvspan(0.05, 0.422, alpha=0.05, color="blue")

    axes[0].set_ylabel("Purity", fontsize=13)
    axes[0].set_title("A. G-F Purity Curves (200-point grid)", fontsize=14,
                       fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    axes[1].set_ylabel("d(Purity) / dr", fontsize=13)
    axes[1].set_title("B. First Derivative (rate of functional change)",
                       fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)

    axes[2].set_ylabel("d$^2$(Purity) / dr$^2$", fontsize=13)
    axes[2].set_xlabel("Radius r (embedding distance)", fontsize=13)
    axes[2].set_title("C. Second Derivative (inflection = regime boundary)",
                       fontsize=14, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)

    plt.tight_layout()
    fig_path = FIGURES / "FigS21_gf_derivatives.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Figure: Critical Radii Comparison (FigS22)
# ============================================================

def plot_critical_radii(r, method_results, betti_curves, r_topo):
    """Critical radii heatmap + Betti percolation coincidence."""
    methods_sorted = sorted(
        [m for m in YEAST_METHODS if m in method_results],
        key=lambda m: method_results[m]["r_peak"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Panel A: Critical radii per method ----
    ax = axes[0]
    y_positions = list(range(len(methods_sorted)))
    for i, method in enumerate(methods_sorted):
        md = method_results[method]
        color = PALETTE.get(method, "#333333")

        # Peak
        ax.plot(md["r_peak"], i, "o", color=color, markersize=12, zorder=5)

        # Inflection points
        for inf_r in md["inflection_points"]:
            ax.plot(inf_r, i, "s", color=color, markersize=6, alpha=0.5,
                    zorder=3)

        # Betti percolation (if available)
        if md.get("betti", {}).get("r_half_merge"):
            ax.plot(md["betti"]["r_half_merge"], i, "^", color="red",
                    markersize=10, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=6)

        # B1 peak
        if md.get("betti", {}).get("r_B1_peak"):
            ax.plot(md["betti"]["r_B1_peak"], i, "D", color="purple",
                    markersize=8, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods_sorted, fontsize=11)
    ax.set_xlabel("Radius r", fontsize=13)
    ax.set_title("A. Critical Radii per Method", fontsize=14,
                  fontweight="bold")
    ax.axvspan(0.05, 0.422, alpha=0.05, color="blue",
               label="G-F interval [0.05, 0.422]")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=10, label="Purity peak"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markersize=6, label="Inflection point"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markersize=10, label="B0 percolation (n/2)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="purple",
               markersize=8, label="B1 peak (max loops)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    # ---- Panel B: Betti curves overlay (Spectral as representative) ----
    ax2 = axes[1]
    rep_method = "Spectral" if "Spectral" in betti_curves else list(betti_curves.keys())[0]
    b0 = np.array(betti_curves[rep_method]["0"])
    b1 = np.array(betti_curves[rep_method]["1"])

    ax2_twin = ax2.twinx()
    ax2.plot(r_topo, b0, color="#0072B2", linewidth=2.5, label=f"B0 ({rep_method})")
    ax2_twin.plot(r_topo, b1, color="#D55E00", linewidth=2.5,
                  label=f"B1 ({rep_method})")

    # Mark percolation
    half_n = b0[0] / 2
    ax2.axhline(half_n, color="red", linestyle="--", alpha=0.5,
                label=f"B0 = n/2 = {half_n:.0f}")
    r_perc, _ = find_betti_half_merge(r_topo, b0)
    if r_perc:
        ax2.axvline(r_perc, color="red", linestyle=":", alpha=0.5)

    # Mark B1 peak
    r_b1, b1_max, _ = find_betti1_peak(r_topo, b1)
    if r_b1:
        ax2_twin.axvline(r_b1, color="purple", linestyle=":", alpha=0.5)

    # Mark purity peak
    if rep_method in method_results:
        r_purity_peak = method_results[rep_method]["r_peak"]
        ax2.axvline(r_purity_peak, color="green", linestyle="-.",
                    alpha=0.5, label=f"Purity peak r={r_purity_peak:.3f}")

    ax2.set_xlabel("Radius r", fontsize=13)
    ax2.set_ylabel("B0 (connected components)", fontsize=13, color="#0072B2")
    ax2_twin.set_ylabel("B1 (loops)", fontsize=13, color="#D55E00")
    ax2.set_title(f"B. Topological Transitions ({rep_method})",
                   fontsize=14, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig_path = FIGURES / "FigS22_critical_radii.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()
