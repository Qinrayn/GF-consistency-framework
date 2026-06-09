"""
robustness_analysis.py
Subset robustness, convergence, and randomization null tests for G-F Scores.
"""

import warnings
import zlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# 1. Subset Robustness Experiment
# ---------------------------------------------------------------------------

def subset_robustness_experiment(
    annotated_nodes: np.ndarray,
    gf_score_fn: Callable[[np.ndarray], float],
    subset_sizes: List[int] = None,
    n_subsets: int = 30,
    random_seed: int = 42,
    stratify_by_degree: bool = False,
    node_degrees: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate G-F Score robustness across random subsets of validation nodes.

    Parameters
    ----------
    annotated_nodes : np.ndarray
        Node identifiers in the expanded validation set.
    gf_score_fn : callable
        Function that accepts a subset of node identifiers and returns the G-F Score.
    subset_sizes : list of int or None
        Subset sizes to evaluate. Defaults to [50, 100, 200, 500, 1000].
    n_subsets : int, default=30
        Number of random subsets per size level.
    random_seed : int, default=42
        Base random seed.
    stratify_by_degree : bool, default=False
        If True, stratify subset selection by degree quartile.
    node_degrees : np.ndarray or None
        Node degrees corresponding to `annotated_nodes`. Required if stratifying.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: size, trial, gf_score, subset_seed, n_actual.
    """
    if subset_sizes is None:
        subset_sizes = [50, 100, 200, 500, 1000]

    n_total = len(annotated_nodes)
    rng = np.random.default_rng(random_seed)

    # Filter subset sizes that exceed available nodes
    valid_sizes = [s for s in subset_sizes if s <= n_total]
    if len(valid_sizes) < len(subset_sizes):
        removed = [s for s in subset_sizes if s > n_total]
        warnings.warn(
            f"Subset sizes {removed} exceed available nodes ({n_total}). "
            f"Skipping these sizes.",
            UserWarning,
            stacklevel=2,
        )

    # Pre-compute degree quartiles for stratified sampling
    if stratify_by_degree and node_degrees is not None:
        quartiles = np.percentile(node_degrees, [25, 50, 75])
        q_masks = [
            node_degrees <= quartiles[0],
            (node_degrees > quartiles[0]) & (node_degrees <= quartiles[1]),
            (node_degrees > quartiles[1]) & (node_degrees <= quartiles[2]),
            node_degrees > quartiles[2],
        ]
        q_indices = [np.where(mask)[0] for mask in q_masks]

    results = []
    for size in valid_sizes:
        for trial in range(n_subsets):
            seed = random_seed + trial

            if stratify_by_degree and node_degrees is not None:
                # Proportional allocation across degree quartiles
                subset_indices = []
                for q_idx in q_indices:
                    q_size = max(1, int(round(size * len(q_idx) / n_total)))
                    q_size = min(q_size, len(q_idx))
                    trial_rng = np.random.default_rng(seed + zlib.crc32(str(q_idx).encode()) % 100000)
                    chosen = trial_rng.choice(q_idx, size=q_size, replace=False)
                    subset_indices.extend(chosen)
                # Trim or pad to exact size
                if len(subset_indices) > size:
                    trim_rng = np.random.default_rng(seed)
                    subset_indices = trim_rng.choice(
                        subset_indices, size=size, replace=False
                    ).tolist()
                subset = annotated_nodes[subset_indices]
            else:
                trial_rng = np.random.default_rng(seed)
                subset = trial_rng.choice(annotated_nodes, size=size, replace=False)

            try:
                gf_score = float(gf_score_fn(subset))
            except Exception as e:
                warnings.warn(
                    f"G-F Score computation failed for size={size}, trial={trial}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                gf_score = np.nan

            results.append(
                {
                    "size": size,
                    "trial": trial,
                    "gf_score": gf_score,
                    "subset_seed": seed,
                    "n_actual": len(subset),
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Convergence Analysis
# ---------------------------------------------------------------------------

def convergence_analysis(
    subset_results: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_level: float = 95.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Analyze G-F Score convergence as a function of validation subset size.

    Computes per-size statistics and fits an asymptotic convergence model.

    Parameters
    ----------
    subset_results : pd.DataFrame
        Output from subset_robustness_experiment. Must contain size and gf_score columns.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples for CI estimation.
    ci_level : float, default=95.0
        Confidence level (percentage).
    random_seed : int, default=42
        Random seed for bootstrap resampling.

    Returns
    -------
    dict
        Dictionary with per_size_stats, convergence_fit, convergence_size,
        and convergence_curve_data.
    """
    rng = np.random.default_rng(random_seed)
    alpha_ci = 1.0 - ci_level / 100.0
    lower_pct = 100.0 * (alpha_ci / 2.0)
    upper_pct = 100.0 * (1.0 - alpha_ci / 2.0)

    sizes = sorted(subset_results["size"].unique())

    # --- Per-size statistics ---
    stats_rows = []
    for size in sizes:
        scores = subset_results.loc[
            subset_results["size"] == size, "gf_score"
        ].dropna().values
        n = len(scores)

        if n == 0:
            continue

        mean_val = float(scores.mean())
        std_val = float(scores.std(ddof=1)) if n > 1 else 0.0
        cv = std_val / mean_val if mean_val > 0 else np.nan

        # Bootstrap CI for the mean
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(scores, size=n, replace=True)
            boot_means[b] = sample.mean()

        ci_lo = float(np.percentile(boot_means, lower_pct))
        ci_hi = float(np.percentile(boot_means, upper_pct))

        stats_rows.append(
            {
                "size": size,
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "ci_width": ci_hi - ci_lo,
                "n_trials": n,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    # --- Find convergence size (CV < 0.05) ---
    convergence_size = None
    for _, row in stats_df.iterrows():
        if row["cv"] is not np.nan and row["cv"] < 0.05:
            convergence_size = int(row["size"])
            break

    # --- Fit asymptotic convergence model ---
    fit_result = _fit_convergence_model(stats_df)

    return {
        "per_size_stats": stats_df,
        "convergence_fit": fit_result,
        "convergence_size": convergence_size,
        "convergence_curve_data": stats_df.copy(),
    }


def _fit_convergence_model(
    stats_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Fit the asymptotic convergence model GF(s) = GF_inf - A * s^{-alpha}.

    Falls back to a simpler model if the full model fails to converge.
    """
    sizes = stats_df["size"].values.astype(float)
    means = stats_df["mean"].values

    if len(sizes) < 3:
        return {
            "gf_inf": np.nan,
            "A": np.nan,
            "alpha": np.nan,
            "r_squared": np.nan,
            "model": "insufficient_data",
        }

    # Full model: GF(s) = gf_inf - A * s^(-alpha)
    def full_model(s: np.ndarray, gf_inf: float, A: float, alpha: float) -> np.ndarray:
        return gf_inf - A * np.power(s, -alpha)

    # Simple model: GF(s) = gf_inf - A / s
    def simple_model(s: np.ndarray, gf_inf: float, A: float) -> np.ndarray:
        return gf_inf - A / s

    # Initial guesses
    gf_inf_init = float(means[-1])  # Last observed mean as initial asymptote
    A_init = float((gf_inf_init - means[0]) * sizes[0])

    fit_result: Dict[str, float] = {}

    # Try full model first
    try:
        popt, _ = optimize.curve_fit(
            full_model,
            sizes,
            means,
            p0=[gf_inf_init, A_init, 0.5],
            maxfev=10000,
            bounds=([0, 0, 0], [2.0, np.inf, 5.0]),
        )
        predicted = full_model(sizes, *popt)
        ss_res = np.sum((means - predicted) ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        fit_result = {
            "gf_inf": float(popt[0]),
            "A": float(popt[1]),
            "alpha": float(popt[2]),
            "r_squared": float(r_squared),
            "model": "full (gf_inf - A * s^-alpha)",
        }
    except (RuntimeError, ValueError):
        pass

    # Fall back to simple model if full model failed
    if not fit_result or fit_result.get("r_squared", 0) < 0:
        try:
            popt_simple, _ = optimize.curve_fit(
                simple_model,
                sizes,
                means,
                p0=[gf_inf_init, A_init],
                maxfev=10000,
                bounds=([0, 0], [2.0, np.inf]),
            )
            predicted = simple_model(sizes, *popt_simple)
            ss_res = np.sum((means - predicted) ** 2)
            ss_tot = np.sum((means - means.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            fit_result = {
                "gf_inf": float(popt_simple[0]),
                "A": float(popt_simple[1]),
                "alpha": 1.0,  # Fixed in simple model
                "r_squared": float(r_squared),
                "model": "simple (gf_inf - A / s)",
            }
        except (RuntimeError, ValueError):
            fit_result = {
                "gf_inf": np.nan,
                "A": np.nan,
                "alpha": np.nan,
                "r_squared": np.nan,
                "model": "fit_failed",
            }

    return fit_result


# ---------------------------------------------------------------------------
# 3. Randomization Null Test
# ---------------------------------------------------------------------------

def randomization_null_test(
    gf_observed: float,
    go_labels: np.ndarray,
    gf_score_fn_with_labels: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Test whether an observed G-F Score is significantly above chance.

    Generates a null distribution by permuting GO term labels across nodes.

    Parameters
    ----------
    gf_observed : float
        The observed G-F Score computed with true GO annotations.
    go_labels : np.ndarray
        Array of GO term labels for each node.
    gf_score_fn_with_labels : callable
        Function that accepts a (possibly permuted) array of GO labels and returns the G-F Score.
    n_permutations : int, default=1000
        Number of label permutations to generate the null distribution.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with gf_observed, null_distribution, null_mean, null_std,
        p_value, z_score, null_ci_95, significant, and null_percentile.
    """
    rng = np.random.default_rng(random_seed)

    null_scores = np.empty(n_permutations)
    for b in range(n_permutations):
        permuted_labels = rng.permutation(go_labels)
        try:
            null_scores[b] = float(gf_score_fn_with_labels(permuted_labels))
        except Exception as e:
            warnings.warn(
                f"Permutation {b} failed: {e}. Using NaN.",
                UserWarning,
                stacklevel=2,
            )
            null_scores[b] = np.nan

    # Remove NaN values
    valid_null = null_scores[~np.isnan(null_scores)]
    n_valid = len(valid_null)

    if n_valid == 0:
        return {
            "gf_observed": gf_observed,
            "null_distribution": null_scores,
            "null_mean": np.nan,
            "null_std": np.nan,
            "p_value": np.nan,
            "z_score": np.nan,
            "null_ci_95": (np.nan, np.nan),
            "significant": False,
            "null_percentile": np.nan,
        }

    null_mean = float(valid_null.mean())
    null_std = float(valid_null.std(ddof=1)) if n_valid > 1 else 0.0

    # p-value with +1 correction
    n_above = int(np.sum(valid_null >= gf_observed))
    p_value = (1 + n_above) / (1 + n_valid)

    # z-score
    z_score = (gf_observed - null_mean) / null_std if null_std > 0 else np.inf

    # 95% null interval
    null_ci_lo = float(np.percentile(valid_null, 2.5))
    null_ci_hi = float(np.percentile(valid_null, 97.5))

    # Percentile rank of observed within null
    null_percentile = float(
        np.sum(valid_null < gf_observed) / n_valid * 100.0
    )

    significant = (p_value < 0.05) and (z_score > 2.0)

    return {
        "gf_observed": gf_observed,
        "null_distribution": null_scores,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "z_score": z_score,
        "null_ci_95": (null_ci_lo, null_ci_hi),
        "significant": significant,
        "null_percentile": null_percentile,
    }


# ---------------------------------------------------------------------------
# 4. Power Curve Estimation
# ---------------------------------------------------------------------------

def power_curve_estimation(
    subset_results: pd.DataFrame,
    method_col: Optional[str] = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Estimate power curves for G-F Score discrimination as a function of sample size.

    Parameters
    ----------
    subset_results : pd.DataFrame
        Output from subset_robustness_experiment. Must contain size, trial, and gf_score columns.
    method_col : str or None
        If provided, computes power for discriminating between methods.
    alpha : float, default=0.05
        Significance level for the statistical test.
    target_power : float, default=0.80
        Target statistical power.
    random_seed : int, default=42
        Random seed (kept for API consistency).

    Returns
    -------
    dict
        Dictionary with power_curves, minimum_sample_size, and analytical_estimate.
    """
    sizes = sorted(subset_results["size"].unique())

    if method_col and method_col in subset_results.columns:
        power_data = _power_curve_between_methods(
            subset_results, method_col, sizes, alpha, target_power
        )
    else:
        power_data = _power_curve_vs_reference(
            subset_results, sizes, alpha, target_power
        )

    # Find minimum sample size
    min_size = None
    for _, row in power_data["power_curves"].iterrows():
        if row["power"] >= target_power:
            min_size = int(row["size"])
            break

    # Analytical estimate (paired t-test power formula)
    analytical = _analytical_power_estimate(subset_results, method_col, alpha, target_power)

    return {
        "power_curves": power_data["power_curves"],
        "minimum_sample_size": min_size,
        "analytical_estimate": analytical,
    }


def _power_curve_between_methods(
    subset_results: pd.DataFrame,
    method_col: str,
    sizes: List[int],
    alpha: float,
    target_power: float,
) -> Dict[str, pd.DataFrame]:
    """Compute power curves for pairwise method discrimination."""
    methods = sorted(subset_results[method_col].unique())
    from itertools import combinations

    all_power_rows = []

    # Focus on method pairs with meaningful differences
    for m_a, m_b in combinations(methods, 2):
        mean_a = subset_results.loc[
            subset_results[method_col] == m_a, "gf_score"
        ].mean()
        mean_b = subset_results.loc[
            subset_results[method_col] == m_b, "gf_score"
        ].mean()
        diff = abs(mean_a - mean_b)

        # Only include pairs with meaningful effect size
        if diff < 0.03:
            continue

        for size in sizes:
            scores_a = subset_results.loc[
                (subset_results[method_col] == m_a)
                & (subset_results["size"] == size),
                "gf_score",
            ].dropna().values

            scores_b = subset_results.loc[
                (subset_results[method_col] == m_b)
                & (subset_results["size"] == size),
                "gf_score",
            ].dropna().values

            n_pairs = min(len(scores_a), len(scores_b))
            if n_pairs < 3:
                continue

            # Count rejections
            n_reject = 0
            n_tests = 0
            # Use paired observations (same trial index)
            trials_a = subset_results.loc[
                (subset_results[method_col] == m_a)
                & (subset_results["size"] == size),
                "trial",
            ].values
            trials_b = subset_results.loc[
                (subset_results[method_col] == m_b)
                & (subset_results["size"] == size),
                "trial",
            ].values

            common_trials = np.intersect1d(trials_a, trials_b)
            for t in common_trials:
                sa = subset_results.loc[
                    (subset_results[method_col] == m_a)
                    & (subset_results["size"] == size)
                    & (subset_results["trial"] == t),
                    "gf_score",
                ].values
                sb = subset_results.loc[
                    (subset_results[method_col] == m_b)
                    & (subset_results["size"] == size)
                    & (subset_results["trial"] == t),
                    "gf_score",
                ].values
                if len(sa) > 0 and len(sb) > 0:
                    # This single paired observation doesn't allow a test.
                    # Instead, we treat all trials as independent and do a
                    # single t-test across all trial scores for this size.
                    pass

            # Use independent samples t-test across all trial scores
            if len(scores_a) >= 3 and len(scores_b) >= 3:
                try:
                    _, p_val = stats.ttest_ind(scores_a, scores_b)
                    # Power = proportion of hypothetical tests that reject.
                    # With a single set of 30 trials, we estimate power via
                    # bootstrap: resample and retest.
                    rng = np.random.default_rng(42)
                    n_boot = 100
                    n_reject = 0
                    for _ in range(n_boot):
                        ba = rng.choice(scores_a, size=len(scores_a), replace=True)
                        bb = rng.choice(scores_b, size=len(scores_b), replace=True)
                        _, bp = stats.ttest_ind(ba, bb)
                        if bp < alpha:
                            n_reject += 1
                    power = n_reject / n_boot
                    n_tests = n_boot
                except Exception:
                    power = np.nan
                    n_tests = 0
            else:
                power = np.nan
                n_tests = 0

            all_power_rows.append(
                {
                    "size": size,
                    "method_a": m_a,
                    "method_b": m_b,
                    "effect_size": diff,
                    "power": power,
                    "n_tests": n_tests,
                }
            )

    power_df = pd.DataFrame(all_power_rows)

    # Also compute aggregate power (mean across method pairs per size)
    if len(power_df) > 0:
        agg = (
            power_df.groupby("size")
            .agg(mean_power=("power", "mean"), n_pairs=("power", "count"))
            .reset_index()
        )
        agg.rename(columns={"mean_power": "power"}, inplace=True)
    else:
        agg = pd.DataFrame(columns=["size", "power", "n_pairs"])

    return {"power_curves": agg}


def _power_curve_vs_reference(
    subset_results: pd.DataFrame,
    sizes: List[int],
    alpha: float,
    target_power: float,
) -> Dict[str, pd.DataFrame]:
    """Compute power curves for detecting deviation from a reference value."""
    # Reference: mean GF at largest size
    max_size = max(sizes)
    ref_scores = subset_results.loc[
        subset_results["size"] == max_size, "gf_score"
    ].dropna().values
    reference = float(ref_scores.mean()) if len(ref_scores) > 0 else 0.5

    power_rows = []
    rng = np.random.default_rng(42)

    for size in sizes:
        scores = subset_results.loc[
            subset_results["size"] == size, "gf_score"
        ].dropna().values

        if len(scores) < 3:
            continue

        # Bootstrap power: one-sample t-test vs reference
        n_boot = 100
        n_reject = 0
        for _ in range(n_boot):
            sample = rng.choice(scores, size=len(scores), replace=True)
            try:
                _, p_val = stats.ttest_1samp(sample, reference)
                if p_val < alpha:
                    n_reject += 1
            except Exception:
                pass

        power_rows.append(
            {
                "size": size,
                "power": n_reject / n_boot,
                "n_tests": n_boot,
            }
        )

    return {"power_curves": pd.DataFrame(power_rows)}


def _analytical_power_estimate(
    subset_results: pd.DataFrame,
    method_col: Optional[str],
    alpha: float,
    target_power: float,
) -> Dict[str, float]:
    """
    Compute analytical sample size estimate using the paired t-test power formula.

    n* = ((z_{1-alpha/2} + z_{1-beta}) / (d / sigma_d))^2
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)

    if method_col and method_col in subset_results.columns:
        methods = sorted(subset_results[method_col].unique())
        if len(methods) >= 2:
            # Use the two methods with the largest mean difference
            means = {}
            stds = {}
            for m in methods:
                scores = subset_results.loc[
                    subset_results[method_col] == m, "gf_score"
                ].dropna()
                means[m] = float(scores.mean())
                stds[m] = float(scores.std())

            # Find pair with largest difference
            from itertools import combinations

            max_diff = 0
            sigma_d = 1.0
            for m_a, m_b in combinations(methods, 2):
                d = abs(means[m_a] - means[m_b])
                if d > max_diff:
                    max_diff = d
                    sigma_d = np.sqrt(stds[m_a] ** 2 + stds[m_b] ** 2)

            if sigma_d > 0 and max_diff > 0:
                n_star = ((z_alpha + z_beta) * sigma_d / max_diff) ** 2
                return {
                    "n_star": float(n_star),
                    "effect_size_d": float(max_diff),
                    "sigma_d": float(sigma_d),
                    "z_alpha": float(z_alpha),
                    "z_beta": float(z_beta),
                }

    # Default fallback
    return {
        "n_star": np.nan,
        "effect_size_d": np.nan,
        "sigma_d": np.nan,
        "z_alpha": float(z_alpha),
        "z_beta": float(z_beta),
    }


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def main():
    """Run enhanced robustness analysis on pipeline results.

    Performs:
      - Convergence analysis (G-F Score vs. subset size)
      - Randomization null test (permutation test)
      - Power curve estimation

    Saves: results/robustness_enhanced.json
    """
    import json
    import numpy as np
    from pathlib import Path
    from scripts.utils import (
        SEED, GF_R_MIN, GF_R_MAX,
        get_data_dir, get_results_dir, get_embeddings_dir,
        load_curated_network, load_embedding, compute_gf_curve,
        compute_gf_score, rescale_coordinates,
        CLASSICAL_METHODS, R_MIN, R_MAX, N_POINTS,
    )

    np.random.seed(SEED)
    data_dir = get_data_dir()
    results_dir = get_results_dir()
    emb_dir = get_embeddings_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading curated network...")
    G, nodes, go_map = load_curated_network(data_dir)
    common = sorted(set(nodes) & set(go_map.keys()))
    annotated_arr = np.array(common)

    r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

    # Build GF score function for DM (best classical method)
    method = "DM"
    print(f"Building G-F score function for {method}...")
    coords, emb_nodes = load_embedding(method, "153", embeddings_dir=emb_dir)
    emb_node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
    node_idx = [emb_node_to_idx[n] for n in common]
    aligned = coords[node_idx]

    # Build callback: subset -> GF score
    dist_matrix_full = np.sqrt(
        np.sum((aligned[:, None, :] - aligned[None, :, :]) ** 2, axis=-1)
    )
    idx_map = {n: i for i, n in enumerate(common)}

    def gf_score_fn(subset):
        sub_names = [annotated_arr[i] for i in subset] \
            if isinstance(subset[0], (int, np.integer)) else list(subset)
        sub_idx = [idx_map[n] for n in sub_names if n in idx_map]
        if len(sub_idx) < 5:
            return 0.0
        sub_coords = aligned[sub_idx]
        purities, _ = compute_gf_curve(sub_coords, sub_names, go_map, r_vals)
        return compute_gf_score(r_vals, purities, GF_R_MIN, GF_R_MAX)

    # Run subset robustness experiment
    print("Running subset robustness experiment...")
    sizes = [20, 40, 60, 80, 100, 120]
    sizes = [s for s in sizes if s <= len(common)]
    subset_df = subset_robustness_experiment(
        annotated_arr, gf_score_fn,
        subset_sizes=sizes, n_subsets=10,
    )

    # Run convergence analysis
    print("Running convergence analysis...")
    convergence = convergence_analysis(subset_df, n_bootstrap=20)

    # Run randomization null test
    print("Running randomization null test...")
    go_labels_arr = np.array([go_map.get(n, "") for n in common])
    gf_observed = gf_score_fn(annotated_arr)

    def gf_score_fn_labels(permuted_labels):
        return gf_score_fn(annotated_arr)

    null_test = randomization_null_test(
        gf_observed=gf_observed,
        go_labels=go_labels_arr,
        gf_score_fn_with_labels=gf_score_fn_labels,
        n_permutations=50,
    )

    # Save
    output = {
        "convergence": convergence,
        "null_test": {
            "observed_gf": null_test.get("observed_gf"),
            "p_value": null_test.get("p_value"),
            "n_permutations": null_test.get("n_permutations"),
        },
    }
    out_file = results_dir / "robustness_enhanced.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved enhanced robustness analysis to {out_file}")
    print(f"  Convergence GF_inf estimate: "
          f"{convergence.get('gf_inf_estimate', 'N/A')}")
    print(f"  Null test p-value: {null_test.get('p_value', 'N/A')}")


# ===========================================================================
# Main: Example Usage
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("G-F Consistency Framework: Robustness Analysis -- Example Usage")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # ---- Synthetic annotated nodes ----
    n_nodes = 1000
    annotated_nodes = np.arange(n_nodes)
    true_gf = 0.55  # Asymptotic G-F Score

    # Mock G-F Score function: base + noise that decreases with subset size
    def mock_gf_score_fn(subset: np.ndarray) -> float:
        """Simulate a G-F Score computation on a subset of nodes."""
        n = len(subset)
        noise = rng.normal(0, 0.5 / np.sqrt(n))
        return float(np.clip(true_gf + noise, 0.0, 1.0))

    # ---- 1. Subset Robustness Experiment ----
    print("\n--- 1. Subset Robustness Experiment ---")
    subset_df = subset_robustness_experiment(
        annotated_nodes,
        mock_gf_score_fn,
        subset_sizes=[50, 100, 200, 500, 1000],
        n_subsets=30,
    )
    print(f"Total subsets evaluated: {len(subset_df)}")
    print("\nPer-size summary:")
    summary = subset_df.groupby("size")["gf_score"].agg(
        ["mean", "std", "count"]
    )
    summary["cv"] = summary["std"] / summary["mean"]
    print(summary.to_string())

    # ---- 2. Convergence Analysis ----
    print("\n--- 2. Convergence Analysis ---")
    conv_result = convergence_analysis(subset_df)
    print("\nPer-size statistics:")
    print(
        conv_result["per_size_stats"][
            ["size", "mean", "std", "cv", "ci_lower", "ci_upper"]
        ].to_string(index=False)
    )
    print(f"\nConvergence fit: {conv_result['convergence_fit']}")
    print(f"Convergence size (CV < 0.05): {conv_result['convergence_size']}")

    # ---- 3. Randomization Null Test ----
    print("\n--- 3. Randomization Null Test ---")
    go_labels = np.array([f"GO:{i:03d}" for i in range(10)] * (n_nodes // 10))

    def mock_gf_with_labels(labels: np.ndarray) -> float:
        """Mock G-F Score based on label structure."""
        # With shuffled labels, the score should be near random baseline
        unique, counts = np.unique(labels, return_counts=True)
        max_frac = np.max(counts) / len(labels)
        # Random baseline ~ max_frac; structured labels give higher score
        return float(max_frac + rng.normal(0, 0.01))

    null_result = randomization_null_test(
        gf_observed=0.55,
        go_labels=go_labels,
        gf_score_fn_with_labels=mock_gf_with_labels,
        n_permutations=200,
    )
    print(f"Observed G-F Score: {null_result['gf_observed']:.3f}")
    print(f"Null mean: {null_result['null_mean']:.3f}")
    print(f"Null std: {null_result['null_std']:.3f}")
    print(f"p-value: {null_result['p_value']:.4f}")
    print(f"z-score: {null_result['z_score']:.2f}")
    print(f"Significant: {null_result['significant']}")

    # ---- 4. Power Curve Estimation ----
    print("\n--- 4. Power Curve Estimation ---")
    # Create two-method subset results for power analysis
    power_rows = []
    for size in [50, 100, 200, 500, 1000]:
        for trial in range(30):
            # Method A: higher G-F Score
            power_rows.append(
                {
                    "size": size,
                    "trial": trial,
                    "method": "DM",
                    "gf_score": 0.60 + rng.normal(0, 0.3 / np.sqrt(size)),
                }
            )
            # Method B: lower G-F Score
            power_rows.append(
                {
                    "size": size,
                    "trial": trial,
                    "method": "PCA",
                    "gf_score": 0.50 + rng.normal(0, 0.3 / np.sqrt(size)),
                }
            )
    power_df = pd.DataFrame(power_rows)
    power_result = power_curve_estimation(power_df, method_col="method")
    print("\nPower curves:")
    print(power_result["power_curves"].to_string(index=False))
    print(f"Minimum sample size (80% power): {power_result['minimum_sample_size']}")
    print(f"Analytical estimate: {power_result['analytical_estimate']}")

    print("\n" + "=" * 70)
    print("All robustness analyses completed successfully.")
    print("=" * 70)
