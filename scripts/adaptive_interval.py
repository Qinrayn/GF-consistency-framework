#!/usr/bin/env python3
"""
adaptive_interval.py
Data-driven adaptive unified interval determination algorithm.

Replaces the fixed integration interval [0.05, 0.422] used in compute_gf.py
with a principled, data-adaptive interval selected by sliding-window search
subject to stability (CV), coverage (width), and significance constraints.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from typing import Any

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEED,
    GF_R_MIN,
    GF_R_MAX,
    R_MIN,
    R_MAX,
    N_POINTS,
    ALL_CURATED_METHODS,
    get_data_dir,
    get_results_dir,
    get_embeddings_dir,
    load_curated_network,
    load_embedding,
    compute_gf_curve,
    compute_gf_score,
    compute_plateau_width,
)

# Methods to evaluate — single source of truth lives in utils.py
METHODS: list[str] = ALL_CURATED_METHODS

# Default relaxed CV threshold used when the strict pass yields no candidates
RELAXED_CV_DEFAULT: float = 0.15

# Conservative fallback for random baseline when no data is available
FALLBACK_RB_MEAN: float = 0.30
FALLBACK_RB_STD: float = 0.02


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def find_adaptive_interval(
    r_vals: np.ndarray,
    purity_vals: np.ndarray,
    random_baseline_purity: float,
    random_baseline_std: float,
    cv_threshold: float = 0.1,
    min_width: float = 0.15,
    significance_sigma: int = 2,
    relaxed_cv: float = RELAXED_CV_DEFAULT,
) -> tuple[float, float, dict[str, Any]]:
    """Find the optimal interval [r_min, r_max] for G-F Score integration.

    Given a purity curve ``purity(r)`` sampled at *r_vals*, determine the
    widest stable interval whose mean purity is significantly above the
    random baseline.

    Parameters
    ----------
    r_vals : np.ndarray
        Array of r values at which the purity curve was sampled.
    purity_vals : np.ndarray
        Corresponding purity values.
    random_baseline_purity : float
        Mean purity of the random (shuffled) baseline.
    random_baseline_std : float
        Standard deviation of the random baseline purity.
    cv_threshold : float, optional
        Maximum allowed coefficient of variation (std/mean) within the
        interval.  Default 0.1.
    min_width : float, optional
        Minimum interval width ``r_max - r_min``.  Default 0.15.
    significance_sigma : int, optional
        Number of standard deviations above the random baseline required
        for significance.  Default 2.
    relaxed_cv : float, optional
        Relaxed CV threshold used when the strict pass finds no candidates.
        Default ``RELAXED_CV_DEFAULT`` (0.15).

    Returns
    -------
    tuple of (float, float, dict)
        ``(r_min, r_max, diagnostics)`` where *diagnostics* contains
        ``cv``, ``width``, ``mean_purity``, ``significance_threshold``,
        ``n_candidates``, ``relaxed`` (bool), and ``fallback`` (bool).
    """
    r = np.asarray(r_vals, dtype=float)
    p = np.asarray(purity_vals, dtype=float)
    n = len(r)

    significance_threshold: float = (
        random_baseline_purity + significance_sigma * random_baseline_std
    )

    # ---- Pre-compute prefix sums for O(1) mean / std queries ----
    cumsum_p = np.concatenate(([0.0], np.cumsum(p)))
    cumsum_p2 = np.concatenate(([0.0], np.cumsum(p * p)))

    def _evaluate_candidates(cv_thresh: float) -> list[dict[str, Any]]:
        """Vectorised sliding-window search over all (i, j) pairs."""
        # Build upper-triangle index grids (j > i)
        idx_i, idx_j = np.triu_indices(n, k=1)

        # Width constraint
        widths = r[idx_j] - r[idx_i]
        valid = widths >= min_width

        if not np.any(valid):
            return []

        vi = idx_i[valid]
        vj = idx_j[valid]
        vw = widths[valid]
        counts = (vj - vi + 1).astype(float)

        # O(1) mean / std via prefix sums
        sums = cumsum_p[vj + 1] - cumsum_p[vi]
        sums2 = cumsum_p2[vj + 1] - cumsum_p2[vi]
        means = sums / counts
        variances = np.maximum(sums2 / counts - means ** 2, 0.0)
        stds = np.sqrt(variances)

        # CV constraint (guard against zero mean)
        cv_mask = (means >= 1e-12) & (stds / np.where(means >= 1e-12, means, 1.0) <= cv_thresh)

        # Significance constraint
        sig_mask = means > significance_threshold

        final = cv_mask & sig_mask

        candidates: list[dict[str, Any]] = [
            {
                "r_min": float(r[vi[k]]),
                "r_max": float(r[vj[k]]),
                "cv": float(stds[k] / means[k]),
                "width": float(vw[k]),
                "mean_purity": float(means[k]),
            }
            for k in np.flatnonzero(final)
        ]
        return candidates

    # --- First pass: strict CV threshold ---
    candidates = _evaluate_candidates(cv_threshold)
    relaxed = False

    # --- Second pass: relaxed CV if no candidates found ---
    if not candidates:
        candidates = _evaluate_candidates(relaxed_cv)
        relaxed = True

    # --- Select best candidate ---
    if candidates:
        # Maximise width * mean_purity (favour wide, high-purity intervals)
        best = max(candidates, key=lambda c: c["width"] * c["mean_purity"])
        r_min_out = best["r_min"]
        r_max_out = best["r_max"]
        diagnostics: dict[str, Any] = {
            "r_min": r_min_out,
            "r_max": r_max_out,
            "cv": best["cv"],
            "width": best["width"],
            "mean_purity": best["mean_purity"],
            "significance_threshold": significance_threshold,
            "n_candidates": len(candidates),
            "relaxed": relaxed,
            "fallback": False,
        }
    else:
        # Fallback: return the fixed interval and flag it
        r_min_out = GF_R_MIN
        r_max_out = GF_R_MAX
        p_sub = p[(r >= r_min_out) & (r <= r_max_out)]
        diagnostics = {
            "r_min": r_min_out,
            "r_max": r_max_out,
            "cv": float(np.std(p_sub) / np.mean(p_sub)) if len(p_sub) > 0 and np.mean(p_sub) > 0 else float("nan"),
            "width": r_max_out - r_min_out,
            "mean_purity": float(np.mean(p_sub)) if len(p_sub) > 0 else float("nan"),
            "significance_threshold": significance_threshold,
            "n_candidates": 0,
            "relaxed": True,
            "fallback": True,
        }

    return r_min_out, r_max_out, diagnostics


# ---------------------------------------------------------------------------
# Consensus with transparency
# ---------------------------------------------------------------------------

def compute_consensus_with_transparency(
    diagnostics_all: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute consensus interval with full transparency about fallbacks.

    Separates genuine (non-fallback) adaptive intervals from fallback
    intervals, computes the consensus only from genuine intervals, and
    reports the fallback ratio so callers can judge the reliability of
    the result.

    Parameters
    ----------
    diagnostics_all : dict
        ``{method_name: diagnostics_dict}`` as returned by
        :func:`find_adaptive_interval` for each method.  Each diagnostics
        dict must contain ``"r_min"``, ``"r_max"``, and ``"fallback"`` keys.

    Returns
    -------
    dict
        ``consensus_interval`` (list of two floats),
        ``genuine_methods`` (list of str),
        ``fallback_methods`` (list of str),
        ``fallback_count`` (int),
        ``total_count`` (int),
        ``fallback_ratio`` (float in [0, 1]).
    """
    genuine: dict[str, dict[str, Any]] = {}
    fallback: list[str] = []

    for method, diag in diagnostics_all.items():
        if diag.get("fallback", False):
            fallback.append(method)
        else:
            genuine[method] = diag

    total: int = len(diagnostics_all)
    fallback_count: int = len(fallback)
    fallback_ratio: float = fallback_count / total if total > 0 else 0.0

    if genuine:
        r_min_vals: list[float] = [v["r_min"] for v in genuine.values()]
        r_max_vals: list[float] = [v["r_max"] for v in genuine.values()]
        consensus: list[float] = [
            float(np.median(r_min_vals)),
            float(np.median(r_max_vals)),
        ]
    else:
        # Every method fell back — consensus degenerates to the fixed interval
        consensus = [GF_R_MIN, GF_R_MAX]

    return {
        "consensus_interval": consensus,
        "genuine_methods": list(genuine.keys()),
        "fallback_methods": fallback,
        "fallback_count": fallback_count,
        "total_count": total,
        "fallback_ratio": fallback_ratio,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_adaptive_vs_fixed(
    r_vals: np.ndarray,
    purity_vals: np.ndarray,
    fixed_interval: tuple[float, float],
    adaptive_interval: tuple[float, float],
) -> dict[str, float]:
    """Compare G-F Scores computed with fixed vs. adaptive intervals.

    Parameters
    ----------
    r_vals : np.ndarray
        r-axis sample points.
    purity_vals : np.ndarray
        Purity curve values.
    fixed_interval : tuple of (float, float)
        ``(r_min, r_max)`` for the fixed (paper) interval.
    adaptive_interval : tuple of (float, float)
        ``(r_min, r_max)`` for the adaptive interval.

    Returns
    -------
    dict
        ``fixed_score``, ``adaptive_score``, ``abs_pct_diff``.
    """
    fixed_score: float = compute_gf_score(
        r_vals, purity_vals, fixed_interval[0], fixed_interval[1],
    )
    adaptive_score: float = compute_gf_score(
        r_vals, purity_vals, adaptive_interval[0], adaptive_interval[1],
    )
    if abs(fixed_score) > 1e-12:
        abs_pct_diff: float = (
            abs(adaptive_score - fixed_score) / abs(fixed_score) * 100.0
        )
    else:
        abs_pct_diff = float("nan")
    return {
        "fixed_score": fixed_score,
        "adaptive_score": adaptive_score,
        "abs_pct_diff": abs_pct_diff,
    }


def cross_method_consistency(
    all_purities: dict[str, list[float]],
    r_vals: np.ndarray,
    random_baseline_purity: float,
    random_baseline_std: float,
    cv_threshold: float = 0.1,
    min_width: float = 0.15,
    significance_sigma: int = 2,
    relaxed_cv: float = RELAXED_CV_DEFAULT,
) -> dict[str, Any]:
    """Run adaptive interval on every method and assess consistency.

    Parameters
    ----------
    all_purities : dict
        ``{method_name: purity_list}`` for each embedding method.
    r_vals : np.ndarray
        Shared r-axis sample points.
    random_baseline_purity : float
        Mean random baseline purity.
    random_baseline_std : float
        Standard deviation of random baseline purity.
    cv_threshold : float, optional
        CV threshold.  Default 0.1.
    min_width : float, optional
        Minimum interval width.  Default 0.15.
    significance_sigma : int, optional
        Sigma above random baseline.  Default 2.
    relaxed_cv : float, optional
        Relaxed CV threshold.  Default ``RELAXED_CV_DEFAULT``.

    Returns
    -------
    dict
        ``intervals`` (per-method), ``r_min_values``, ``r_max_values``,
        ``r_min_std``, ``r_max_std``, ``consensus_interval``,
        ``transparency``.
    """
    intervals: dict[str, dict[str, Any]] = {}
    for method, purity_list in all_purities.items():
        r_min, r_max, diag = find_adaptive_interval(
            r_vals, np.array(purity_list),
            random_baseline_purity, random_baseline_std,
            cv_threshold=cv_threshold,
            min_width=min_width,
            significance_sigma=significance_sigma,
            relaxed_cv=relaxed_cv,
        )
        intervals[method] = {"r_min": r_min, "r_max": r_max, **diag}

    r_min_values: list[float] = [v["r_min"] for v in intervals.values()]
    r_max_values: list[float] = [v["r_max"] for v in intervals.values()]

    transparency = compute_consensus_with_transparency(intervals)

    return {
        "intervals": intervals,
        "r_min_values": r_min_values,
        "r_max_values": r_max_values,
        "r_min_std": float(np.std(r_min_values)) if r_min_values else float("nan"),
        "r_max_std": float(np.std(r_max_values)) if r_max_values else float("nan"),
        "consensus_interval": transparency["consensus_interval"],
        "transparency": transparency,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_gf_curves(results_dir: Path) -> dict[str, Any]:
    """Load pre-computed G-F curves from JSON (or pickle fallback)."""
    curves_file = results_dir / "gf_curves_200pts.json"
    if curves_file.exists():
        with open(curves_file) as f:
            return json.load(f)
    pkl_file = results_dir / "gf_curves_200pts.pkl"
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"Cannot find G-F curves in {results_dir}. Run compute_gf.py first."
    )


def _load_scores(results_dir: Path) -> dict[str, Any]:
    """Load pre-computed G-F scores (fixed interval) from JSON."""
    scores_file = results_dir / "gf_scores.json"
    if not scores_file.exists():
        return {}
    with open(scores_file) as f:
        return json.load(f)


def _compute_random_baseline_std(
    gf_curves: dict[str, Any],
    n_shuffles: int = 10,
) -> tuple[float, float]:
    """Extract random baseline mean and std from the curves data.

    The ``random_baseline_purity`` field in the curves JSON is a list of
    per-r purity values *averaged across shuffles*.  The variation across
    r is dominated by the shape of the curve, not by shuffle noise.  To
    estimate the shuffle-to-shuffle standard deviation we divide the
    across-r standard deviation by ``sqrt(n_shuffles)``, recovering an
    approximate per-shuffle standard error.

    Parameters
    ----------
    gf_curves : dict
        Loaded curves data containing ``random_baseline_purity``.
    n_shuffles : int, optional
        Number of shuffles used when computing the random baseline
        (must match ``compute_gf.py``).  Default 10.

    Returns
    -------
    tuple of (float, float)
        ``(mean, std)``
    """
    rb = gf_curves.get("random_baseline_purity", None)
    if rb is not None and len(rb) > 0:
        rb_arr = np.array(rb)
        mean_val: float = float(np.mean(rb_arr))
        # The stored curve is averaged over n_shuffles runs, so its
        # pointwise std underestimates the per-shuffle std by sqrt(n).
        # We use across-r std / sqrt(n_shuffles) as a conservative
        # estimate of the shuffle-to-shuffle standard error of the mean.
        std_val: float = float(np.std(rb_arr, ddof=0)) / np.sqrt(n_shuffles)
        return mean_val, std_val
    return FALLBACK_RB_MEAN, FALLBACK_RB_STD  # conservative fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-driven adaptive unified interval determination",
    )
    parser.add_argument(
        "--cv-threshold", type=float, default=0.1,
        help="CV threshold for stability constraint (default: 0.1)",
    )
    parser.add_argument(
        "--min-width", type=float, default=0.15,
        help="Minimum interval width (default: 0.15)",
    )
    parser.add_argument(
        "--significance-sigma", type=int, default=2,
        help="Number of sigma above random baseline (default: 2)",
    )
    parser.add_argument(
        "--relaxed-cv", type=float, default=RELAXED_CV_DEFAULT,
        help=f"Relaxed CV threshold for second pass (default: {RELAXED_CV_DEFAULT})",
    )
    parser.add_argument(
        "--no-consensus", action="store_true", default=False,
        help="Skip consensus interval computation (default: False)",
    )
    parser.add_argument(
        "--human", action="store_true", default=False,
        help="Also run on human validation data",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load pre-computed G-F curves and scores
    # ------------------------------------------------------------------
    print("Loading pre-computed G-F curves...")
    try:
        gf_curves = _load_gf_curves(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    r_vals = np.array(gf_curves["r"])

    scores_data = _load_scores(results_dir)
    fixed_scores = scores_data.get("scores", {})

    # Random baseline statistics
    rb_mean, rb_std = _compute_random_baseline_std(gf_curves)
    print(f"Random baseline: mean={rb_mean:.4f}, std={rb_std:.4f}")
    print(f"Significance threshold: {rb_mean + args.significance_sigma * rb_std:.4f}")

    # ------------------------------------------------------------------
    # Collect per-method purity curves
    # ------------------------------------------------------------------
    all_purities: dict[str, list[float]] = {}
    for method in METHODS:
        key = f"{method}_purity"
        if key in gf_curves:
            all_purities[method] = gf_curves[key]
        else:
            print(f"  Warning: {key} not found in curves data, skipping.")

    print(f"\nMethods available: {list(all_purities.keys())}")

    # ------------------------------------------------------------------
    # Per-method adaptive intervals
    # ------------------------------------------------------------------
    print("\n=== Adaptive Interval Determination ===")
    adaptive_intervals: dict[str, dict[str, Any]] = {}
    diagnostics_all: dict[str, dict[str, Any]] = {}

    for method, purity_list in all_purities.items():
        r_min, r_max, diag = find_adaptive_interval(
            r_vals, np.array(purity_list),
            rb_mean, rb_std,
            cv_threshold=args.cv_threshold,
            min_width=args.min_width,
            significance_sigma=args.significance_sigma,
            relaxed_cv=args.relaxed_cv,
        )
        adaptive_intervals[method] = {
            "r_min": round(r_min, 4),
            "r_max": round(r_max, 4),
            "cv": round(diag["cv"], 4),
            "width": round(diag["width"], 4),
            "mean_purity": round(diag["mean_purity"], 4),
        }
        diagnostics_all[method] = diag
        relaxed_tag = " [relaxed CV]" if diag.get("relaxed") else ""
        fallback_tag = " [FALLBACK]" if diag.get("fallback") else ""
        print(
            f"  {method:10s}  [{r_min:.4f}, {r_max:.4f}]  "
            f"width={diag['width']:.4f}  cv={diag['cv']:.4f}  "
            f"mean_p={diag['mean_purity']:.4f}"
            f"{relaxed_tag}{fallback_tag}"
        )

    # ------------------------------------------------------------------
    # Consensus interval (with transparency)
    # ------------------------------------------------------------------
    transparency = compute_consensus_with_transparency(diagnostics_all)
    consensus_interval: list[float] = [GF_R_MIN, GF_R_MAX]

    if not args.no_consensus and adaptive_intervals:
        consensus_interval = [
            round(transparency["consensus_interval"][0], 4),
            round(transparency["consensus_interval"][1], 4),
        ]

        # --- Prominent fallback-ratio banner ---
        fb_ratio: float = transparency["fallback_ratio"]
        fb_count: int = transparency["fallback_count"]
        fb_total: int = transparency["total_count"]
        print("\n" + "=" * 60)
        print(f"  TRANSPARENCY: {fb_count}/{fb_total} methods fell back to "
              f"fixed interval (fallback ratio = {fb_ratio:.2%})")
        if fb_count > 0:
            print(f"    Fallback methods : {transparency['fallback_methods']}")
        if transparency["genuine_methods"]:
            print(f"    Genuine methods : {transparency['genuine_methods']}")
        if fb_ratio > 0.5:
            print("  WARNING: majority of methods used fallback -- "
                  "consensus interval is unreliable!")
        print("=" * 60)

        r_min_vals = [v["r_min"] for v in adaptive_intervals.values()]
        r_max_vals = [v["r_max"] for v in adaptive_intervals.values()]
        print(f"\n=== Consensus Interval ===")
        print(f"  Median r_min: {np.median(r_min_vals):.4f}  "
              f"(std={np.std(r_min_vals):.4f})")
        print(f"  Median r_max: {np.median(r_max_vals):.4f}  "
              f"(std={np.std(r_max_vals):.4f})")
        print(f"  Consensus (genuine only): "
              f"[{consensus_interval[0]:.4f}, {consensus_interval[1]:.4f}]")
        print(f"  Fixed (paper):            "
              f"[{GF_R_MIN}, {GF_R_MAX}]")
        print(f"  Fallback ratio:           {fb_ratio:.2%}  "
              f"({fb_count}/{fb_total})")

    # ------------------------------------------------------------------
    # G-F Score comparison: fixed vs adaptive
    # ------------------------------------------------------------------
    print("\n=== G-F Score Comparison (Fixed vs Adaptive) ===")
    gf_scores_adaptive: dict[str, float] = {}
    gf_scores_fixed: dict[str, float] = {}
    comparison: dict[str, dict[str, float]] = {}

    for method, purity_list in all_purities.items():
        p_arr = np.array(purity_list)

        # Adaptive score (per-method adaptive interval)
        ai = adaptive_intervals[method]
        adaptive_score = compute_gf_score(r_vals, p_arr, ai["r_min"], ai["r_max"])
        gf_scores_adaptive[method] = round(adaptive_score, 4)

        # Fixed score (recompute for consistency, or use saved)
        fixed_score = compute_gf_score(r_vals, p_arr, GF_R_MIN, GF_R_MAX)
        gf_scores_fixed[method] = round(fixed_score, 4)

        abs_pct = (
            abs(adaptive_score - fixed_score) / abs(fixed_score) * 100.0
            if abs(fixed_score) > 1e-12 else float("nan")
        )
        comparison[method] = {
            "fixed": round(fixed_score, 4),
            "adaptive": round(adaptive_score, 4),
            "abs_pct_diff": round(abs_pct, 2),
        }

    # Print comparison table
    header = f"  {'Method':<12s} {'Fixed':>8s} {'Adaptive':>10s} {'|Diff|%':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    pct_diffs: list[float] = []
    for method in METHODS:
        if method not in comparison:
            continue
        c = comparison[method]
        pct_diffs.append(c["abs_pct_diff"])
        print(
            f"  {method:<12s} {c['fixed']:>8.4f} {c['adaptive']:>10.4f} "
            f"{c['abs_pct_diff']:>7.2f}%"
        )

    mean_abs_pct: float = float(np.mean(pct_diffs)) if pct_diffs else float("nan")
    max_abs_pct: float = float(np.max(pct_diffs)) if pct_diffs else float("nan")
    print(f"\n  Mean |diff|: {mean_abs_pct:.2f}%")
    print(f"  Max  |diff|: {max_abs_pct:.2f}%")

    # ------------------------------------------------------------------
    # Cross-method consistency
    # ------------------------------------------------------------------
    consistency = cross_method_consistency(
        all_purities, r_vals, rb_mean, rb_std,
        cv_threshold=args.cv_threshold,
        min_width=args.min_width,
        significance_sigma=args.significance_sigma,
        relaxed_cv=args.relaxed_cv,
    )

    # ------------------------------------------------------------------
    # Human validation (optional)
    # ------------------------------------------------------------------
    human_results: dict[str, Any] | None = None
    if args.human:
        human_pkl = results_dir / "human_gf_curves_200pts.pkl"
        if human_pkl.exists():
            print("\n=== Human Validation Data ===")
            with open(human_pkl, "rb") as f:
                human_curves = pickle.load(f)

            # Human pickle uses "r_values" (not "r") and a nested
            # curves dict: curves[method]["purity"]
            human_r = np.array(human_curves.get("r_values", human_curves.get("r", [])))
            curves_dict = human_curves.get("curves", {})

            # No random baseline stored for human data; use conservative
            # fallback estimate.
            human_rb_mean: float = FALLBACK_RB_MEAN
            human_rb_std: float = FALLBACK_RB_STD
            human_rb = human_curves.get("random_baseline_purity", None)
            if human_rb is not None and len(human_rb) > 0:
                human_rb_mean, human_rb_std = _compute_random_baseline_std(
                    human_curves
                )
            print(f"  Human random baseline: mean={human_rb_mean:.4f}, std={human_rb_std:.4f}")

            human_methods: list[str] = [m for m in METHODS if m in curves_dict]
            human_adaptive: dict[str, dict[str, Any]] = {}
            human_diagnostics: dict[str, dict[str, Any]] = {}
            for method in human_methods:
                p_list = curves_dict[method].get("purity", [])
                if not p_list:
                    continue
                r_min_h, r_max_h, diag_h = find_adaptive_interval(
                    human_r, np.array(p_list),
                    human_rb_mean, human_rb_std,
                    cv_threshold=args.cv_threshold,
                    min_width=args.min_width,
                    significance_sigma=args.significance_sigma,
                    relaxed_cv=args.relaxed_cv,
                )
                human_adaptive[method] = {
                    "r_min": round(r_min_h, 4),
                    "r_max": round(r_max_h, 4),
                    "cv": round(diag_h["cv"], 4),
                    "width": round(diag_h["width"], 4),
                    "mean_purity": round(diag_h["mean_purity"], 4),
                }
                human_diagnostics[method] = {**human_adaptive[method], **diag_h}
                print(
                    f"  {method:10s}  [{r_min_h:.4f}, {r_max_h:.4f}]  "
                    f"cv={diag_h['cv']:.4f}"
                )

            human_consensus: list[float] | None = None
            if human_adaptive:
                human_transparency = compute_consensus_with_transparency(
                    human_diagnostics
                )
                human_consensus = [
                    round(human_transparency["consensus_interval"][0], 4),
                    round(human_transparency["consensus_interval"][1], 4),
                ]
                print(
                    f"  Human consensus (genuine only): "
                    f"[{human_consensus[0]:.4f}, {human_consensus[1]:.4f}]"
                )
                h_fb = human_transparency["fallback_ratio"]
                print(
                    f"  Human fallback ratio: {h_fb:.2%}  "
                    f"({human_transparency['fallback_count']}/"
                    f"{human_transparency['total_count']})"
                )

            human_results = {
                "adaptive_intervals": human_adaptive,
                "consensus_interval": human_consensus,
            }
        else:
            print(f"\n  Warning: {human_pkl} not found, skipping human validation.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "adaptive_intervals": adaptive_intervals,
        "consensus_interval": consensus_interval,
        "gf_scores_adaptive": gf_scores_adaptive,
        "gf_scores_fixed": gf_scores_fixed,
        "comparison": comparison,
        "mean_abs_pct_diff": round(mean_abs_pct, 2),
        "max_abs_pct_diff": round(max_abs_pct, 2),
        "transparency": {
            "fallback_ratio": round(transparency["fallback_ratio"], 4),
            "fallback_count": transparency["fallback_count"],
            "total_count": transparency["total_count"],
            "genuine_methods": transparency["genuine_methods"],
            "fallback_methods": transparency["fallback_methods"],
        },
        "diagnostics": {
            m: {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in d.items()
            }
            for m, d in diagnostics_all.items()
        },
        "parameters": {
            "cv_threshold": args.cv_threshold,
            "min_width": args.min_width,
            "significance_sigma": args.significance_sigma,
            "relaxed_cv": args.relaxed_cv,
            "random_baseline_mean": round(rb_mean, 6),
            "random_baseline_std": round(rb_std, 6),
        },
        "cross_method_consistency": {
            "r_min_std": round(consistency["r_min_std"], 4),
            "r_max_std": round(consistency["r_max_std"], 4),
            "consensus_interval": consistency["consensus_interval"],
        },
    }

    if human_results is not None:
        output["human_validation"] = human_results

    output_file = results_dir / "adaptive_interval_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
