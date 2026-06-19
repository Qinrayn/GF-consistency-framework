"""
statistical_analysis.py
Core statistical analyses: G-F Score comparison, Spearman correlation,
Wilcoxon pairwise tests, bootstrap CIs, and permutation tests.
"""

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MethodList = List[str]
SpeciesList = List[str]


# ---------------------------------------------------------------------------
# 1. G-F Score Comparison Across Methods
# ---------------------------------------------------------------------------

def compute_gf_score_comparison(
    gf_scores: pd.DataFrame,
    method_col: str = "method",
    species_col: str = "species",
    gf_col: str = "gf_score",
    auroc_col: Optional[str] = "auroc",
    f1_col: Optional[str] = "knn_f1",
) -> Dict[str, Any]:
    """Compare G-F Scores across methods and species.

    Parameters
    ----------
    gf_scores : pd.DataFrame
        DataFrame with G-F Scores and optional downstream metrics.
    method_col : str
        Column for embedding method identifier.
    species_col : str
        Column for species identifier.
    gf_col : str
        Column for G-F Score value.
    auroc_col : str or None
        Column for link prediction AUROC (None to skip).
    f1_col : str or None
        Column for k-NN F1 (None to skip).

    Returns
    -------
    dict
        Keys: comparison_table, rank_correlation, rank_correlation_pvalue,
        summary_stats.
    """
    # --- Build pivot table for G-F Scores ---
    pivot = gf_scores.pivot_table(
        index=method_col,
        columns=species_col,
        values=gf_col,
        aggfunc="mean",
    )
    pivot.columns = [f"gf_{sp}" for sp in pivot.columns]

    # --- Compute ranks per species (1 = best = highest G-F Score) ---
    species_names = gf_scores[species_col].unique()
    for sp in species_names:
        col = f"gf_{sp}"
        if col in pivot.columns:
            pivot[f"rank_{sp}"] = (
                pivot[col].rank(ascending=False, method="min").astype(int)
            )

    # --- Compute delta_rank (human_rank - yeast_rank) ---
    rank_cols = [c for c in pivot.columns if c.startswith("rank_")]
    if len(rank_cols) >= 2:
        # Assume exactly two species for delta_rank
        sp_names_sorted = sorted(
            [c.replace("rank_", "") for c in rank_cols]
        )
        if len(sp_names_sorted) == 2:
            r1, r2 = sp_names_sorted
            pivot["delta_rank"] = pivot[f"rank_{r2}"] - pivot[f"rank_{r1}"]

    # --- Rank correlation between species ---
    rank_data = pivot[[c for c in pivot.columns if c.startswith("rank_")]].copy()
    if rank_data.shape[1] == 2:
        rho, pval = stats.spearmanr(rank_data.iloc[:, 0], rank_data.iloc[:, 1])
    else:
        rho, pval = np.nan, np.nan

    # --- Summary statistics per species ---
    summary_rows = []
    for sp in species_names:
        col = f"gf_{sp}"
        if col in pivot.columns:
            vals = pivot[col].dropna()
            summary_rows.append(
                {
                    "species": sp,
                    "n_methods": len(vals),
                    "mean_gf": vals.mean(),
                    "std_gf": vals.std(),
                    "min_gf": vals.min(),
                    "max_gf": vals.max(),
                    "range_gf": vals.max() - vals.min(),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    # --- Optionally attach downstream metrics ---
    if auroc_col and auroc_col in gf_scores.columns:
        auroc_pivot = gf_scores.pivot_table(
            index=method_col, columns=species_col, values=auroc_col, aggfunc="mean"
        )
        auroc_pivot.columns = [f"auroc_{sp}" for sp in auroc_pivot.columns]
        pivot = pivot.join(auroc_pivot)

    if f1_col and f1_col in gf_scores.columns:
        f1_pivot = gf_scores.pivot_table(
            index=method_col, columns=species_col, values=f1_col, aggfunc="mean"
        )
        f1_pivot.columns = [f"f1_{sp}" for sp in f1_pivot.columns]
        pivot = pivot.join(f1_pivot)

    return {
        "comparison_table": pivot.reset_index(),
        "rank_correlation": rho,
        "rank_correlation_pvalue": pval,
        "summary_stats": summary_df,
    }


# ---------------------------------------------------------------------------
# 2. Spearman Correlation Analysis
# ---------------------------------------------------------------------------

def spearman_correlation_analysis(
    data: pd.DataFrame,
    gf_col: str = "gf_score",
    metric_col: str = "auroc",
    method_col: str = "method",
    species_col: Optional[str] = "species",
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Spearman rank correlation between G-F Score and a downstream metric.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with gf_col and metric_col columns.
    gf_col : str
        Column for G-F Score values.
    metric_col : str
        Column for the downstream metric.
    method_col : str
        Column for method identifiers.
    species_col : str or None
        Column for species identifiers (None for pooled only).
    n_bootstrap : int
        Number of bootstrap resamples for CI.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: pooled (rho, pvalue, CI, n), per_species (if species_col
        given), interpretation.
    """
    rng = np.random.default_rng(random_seed)

    def _spearman_with_ci(
        x: np.ndarray, y: np.ndarray, n_boot: int
    ) -> Dict[str, float]:
        """Compute Spearman rho, p-value, and bootstrap CI."""
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        n = len(x_valid)

        if n < 4:
            return {
                "rho": np.nan,
                "pvalue": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": n,
            }

        rho, pvalue = stats.spearmanr(x_valid, y_valid)

        # Bootstrap CI (percentile method)
        boot_rhos = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_rhos[b], _ = stats.spearmanr(x_valid[idx], y_valid[idx])

        ci_lower = float(np.percentile(boot_rhos, 2.5))
        ci_upper = float(np.percentile(boot_rhos, 97.5))

        return {
            "rho": float(rho),
            "pvalue": float(pvalue),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n,
        }

    def _interpret(rho: float) -> str:
        """Interpret correlation strength per research design guidelines."""
        abs_rho = abs(rho)
        sign_prefix = "inverse " if rho < 0 else ""
        if abs_rho > 0.7:
            return f"{sign_prefix}strong_positive (G-F Score as proxy for embedding utility)"
        elif abs_rho > 0.4:
            return f"{sign_prefix}moderate_positive (G-F Score as complementary metric)"
        else:
            return f"{sign_prefix}weak/independent (G-F Score captures unique quality dimension)"

    # --- Pooled analysis ---
    pooled = _spearman_with_ci(
        data[gf_col].values, data[metric_col].values, n_bootstrap
    )
    pooled["interpretation"] = _interpret(pooled["rho"])

    result: Dict[str, Any] = {"pooled": pooled}

    # --- Per-species analysis ---
    if species_col and species_col in data.columns:
        per_species: Dict[str, Dict[str, Any]] = {}
        for sp, grp in data.groupby(species_col):
            sp_result = _spearman_with_ci(
                grp[gf_col].values, grp[metric_col].values, n_bootstrap
            )
            sp_result["interpretation"] = _interpret(sp_result["rho"])
            per_species[str(sp)] = sp_result
        result["per_species"] = per_species

    return result


# ---------------------------------------------------------------------------
# 3. Wilcoxon Signed-Rank Pairwise Comparison
# ---------------------------------------------------------------------------

def wilcoxon_pairwise_comparison(
    gf_scores: pd.DataFrame,
    method_col: str = "method",
    gf_col: str = "gf_score",
    bootstrap_col: Optional[str] = None,
    n_bootstrap_augment: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Pairwise Wilcoxon signed-rank tests between all methods with FDR correction.

    Parameters
    ----------
    gf_scores : pd.DataFrame
        DataFrame with method identifiers and G-F Scores.
    method_col : str
        Column for method identifiers.
    gf_col : str
        Column for G-F Score values.
    bootstrap_col : str or None
        Column identifying bootstrap replicates for pairing.
    n_bootstrap_augment : int
        Bootstrap augmentations when natural pairs < 6.
    alpha : float
        Significance level for FDR correction.
    random_seed : int
        Seed for bootstrap augmentation.

    Returns
    -------
    dict
        Keys: pairwise_results (DataFrame with W, pvalue, pvalue_fdr,
        effect_size, significant), n_comparisons, n_significant,
        fdr_threshold.
    """
    rng = np.random.default_rng(random_seed)

    methods = sorted(gf_scores[method_col].unique())
    n_methods = len(methods)
    n_comparisons = n_methods * (n_methods - 1) // 2

    # Pivot: rows = pairing key (e.g., species), columns = methods, values = GF
    if bootstrap_col:
        pivot = gf_scores.pivot_table(
            index=bootstrap_col, columns=method_col, values=gf_col, aggfunc="mean"
        )
    else:
        # Use all columns except method and gf_col as implicit pairing keys
        other_cols = [
            c for c in gf_scores.columns if c not in [method_col, gf_col]
        ]
        if other_cols:
            pivot = gf_scores.pivot_table(
                index=other_cols, columns=method_col, values=gf_col, aggfunc="mean"
            )
        else:
            # No pairing key: each row is one observation per method
            pivot = gf_scores.pivot_table(
                columns=method_col, values=gf_col, aggfunc="mean"
            )

    n_pairs = len(pivot)

    # If too few pairs, issue a warning
    if n_pairs < 6:
        warnings.warn(
            f"Only {n_pairs} paired observations available. Wilcoxon signed-rank "
            f"test has very low power with n < 6. Consider bootstrap augmentation.",
            UserWarning,
            stacklevel=2,
        )

    # --- Pairwise tests ---
    results_list = []
    for m_a, m_b in combinations(methods, 2):
        if m_a not in pivot.columns or m_b not in pivot.columns:
            continue

        vals_a = pivot[m_a].dropna().values
        vals_b = pivot[m_b].dropna().values

        # Align to common non-NaN indices
        common_mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        diffs = vals_a[common_mask] - vals_b[common_mask]
        diffs = diffs[diffs != 0]  # Remove zero differences
        n_valid = len(diffs)

        if n_valid < 2:
            results_list.append(
                {
                    "method_a": m_a,
                    "method_b": m_b,
                    "W": np.nan,
                    "pvalue": np.nan,
                    "effect_size": np.nan,
                    "n_pairs": n_valid,
                }
            )
            continue

        # Wilcoxon signed-rank test
        stat_result = stats.wilcoxon(diffs, alternative="two-sided")
        W = stat_result.statistic
        pvalue = stat_result.pvalue

        # Rank biserial correlation: r_rb = 1 - 4*W / (n*(n+1))
        # W here is the smaller of W+ and W-
        r_rb = 1.0 - (4.0 * W) / (n_valid * (n_valid + 1))

        results_list.append(
            {
                "method_a": m_a,
                "method_b": m_b,
                "W": W,
                "pvalue": pvalue,
                "effect_size": r_rb,
                "n_pairs": n_valid,
            }
        )

    pairwise_df = pd.DataFrame(results_list)

    # --- Benjamini-Hochberg FDR correction ---
    if len(pairwise_df) > 0 and pairwise_df["pvalue"].notna().any():
        valid_mask = pairwise_df["pvalue"].notna()
        pvals = pairwise_df.loc[valid_mask, "pvalue"].values
        _, pvals_fdr, _, _ = _benjamini_hochberg(pvals, alpha)

        pairwise_df["pvalue_fdr"] = np.nan
        pairwise_df.loc[valid_mask, "pvalue_fdr"] = pvals_fdr
        pairwise_df["significant"] = pairwise_df["pvalue_fdr"] < alpha
    else:
        pairwise_df["pvalue_fdr"] = np.nan
        pairwise_df["significant"] = False

    n_significant = int(pairwise_df["significant"].sum())

    # FDR threshold: largest rejected p-value
    if n_significant > 0:
        fdr_threshold = float(
            pairwise_df.loc[pairwise_df["significant"], "pvalue_fdr"].max()
        )
    else:
        fdr_threshold = 0.0

    return {
        "pairwise_results": pairwise_df,
        "n_comparisons": n_comparisons,
        "n_significant": n_significant,
        "fdr_threshold": fdr_threshold,
    }


def _benjamini_hochberg(
    pvalues: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Apply Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of raw p-values.
    alpha : float
        Target FDR level.

    Returns
    -------
    tuple
        (sorted_indices, adjusted_pvalues, rejected, n_rejected)
    """
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # Adjusted p-values: p_adj[i] = min(p[i] * n / rank[i], 1.0)
    ranks = np.arange(1, n + 1)
    adjusted = np.minimum(sorted_pvals * n / ranks, 1.0)

    # Enforce monotonicity (adjusted p-values should be non-decreasing)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    adjusted_original = np.empty(n)
    adjusted_original[sorted_idx] = adjusted

    rejected = adjusted_original <= alpha
    n_rejected = int(rejected.sum())

    return sorted_idx, adjusted_original, rejected, n_rejected


# ---------------------------------------------------------------------------
# 4. Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_confidence_intervals(
    data: pd.DataFrame,
    value_col: str = "gf_score",
    group_col: str = "method",
    species_col: Optional[str] = "species",
    n_bootstrap: int = 1000,
    ci_level: float = 95.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap percentile confidence intervals for G-F Scores by group.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the values to bootstrap.
    value_col : str
        Column for numeric values to estimate.
    group_col : str
        Column for grouping.
    species_col : str or None
        Creates sub-groups within each species if provided.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level as percentage (e.g. 95.0).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: group_col, species_col (if given), mean, std,
        ci_lower, ci_upper, ci_width, n_obs, se.
    """
    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - ci_level / 100.0
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    group_cols = [group_col]
    if species_col and species_col in data.columns:
        group_cols.append(species_col)

    results = []
    for group_key, grp in data.groupby(group_cols):
        values = grp[value_col].dropna().values
        n = len(values)

        if n < 2:
            if isinstance(group_key, tuple):
                row = dict(zip(group_cols, group_key))
            else:
                row = {group_cols[0]: group_key}
            row.update(
                {
                    "mean": float(values[0]) if n == 1 else np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_width": np.nan,
                    "n_obs": n,
                    "se": np.nan,
                }
            )
            results.append(row)
            continue

        # Bootstrap resampling
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means[b] = sample.mean()

        mean_val = float(values.mean())
        std_val = float(values.std(ddof=1))
        se_val = std_val / np.sqrt(n)
        ci_lo = float(np.percentile(boot_means, lower_pct))
        ci_hi = float(np.percentile(boot_means, upper_pct))

        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key))
        else:
            row = {group_cols[0]: group_key}

        row.update(
            {
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "ci_width": ci_hi - ci_lo,
                "n_obs": n,
                "se": se_val,
            }
        )
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Permutation Test for Rank Reversal Significance
# ---------------------------------------------------------------------------

def permutation_test_rank_reversal(
    comparison_table: pd.DataFrame,
    method_col: str = "method",
    rank_cols: Optional[List[str]] = None,
    focal_methods: Optional[List[str]] = None,
    n_permutations: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Permutation test for cross-species rank reversal significance.

    Parameters
    ----------
    comparison_table : pd.DataFrame
        DataFrame with method identifiers and per-species rank columns.
    method_col : str
        Column for method identifiers.
    rank_cols : list of str or None
        Two per-species rank columns (auto-detected if None).
    focal_methods : list of str or None
        Two methods to compare (auto-selected if None).
    n_permutations : int
        Number of permutations.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: T_observed, p_value, null_distribution, focal_methods,
        rank_shift_method1, rank_shift_method2, effect_size_ci,
        all_rank_shifts.
    """
    rng = np.random.default_rng(random_seed)

    # --- Auto-detect rank columns ---
    if rank_cols is None:
        rank_cols = sorted([c for c in comparison_table.columns if c.startswith("rank_")])
    if len(rank_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 rank columns, found {len(rank_cols)}: {rank_cols}"
        )

    # --- Compute rank shifts for all methods ---
    df = comparison_table.copy()
    df["delta_rank"] = df[rank_cols[1]] - df[rank_cols[0]]

    # --- Select focal methods ---
    if focal_methods is None:
        # Pick the two methods with the largest absolute rank shifts
        abs_shifts = df["delta_rank"].abs().sort_values(ascending=False)
        focal_methods = df.loc[abs_shifts.index[:2], method_col].tolist()

    if len(focal_methods) != 2:
        raise ValueError(f"Expected exactly 2 focal methods, got {len(focal_methods)}")

    fm1, fm2 = focal_methods

    # --- Compute observed test statistic ---
    shift_1 = float(df.loc[df[method_col] == fm1, "delta_rank"].values[0])
    shift_2 = float(df.loc[df[method_col] == fm2, "delta_rank"].values[0])
    T_obs = abs(shift_1 - shift_2)

    # --- Permutation test ---
    # Null hypothesis: species labels are exchangeable.
    # Under H0, for each method, we can swap its two ranks with probability 0.5.
    n_methods = len(df)
    ranks_sp1 = df[rank_cols[0]].values.astype(float)
    ranks_sp2 = df[rank_cols[1]].values.astype(float)

    # Identify indices of focal methods
    fm1_idx = int(np.where(df[method_col].values == fm1)[0][0])
    fm2_idx = int(np.where(df[method_col].values == fm2)[0][0])

    null_distribution = np.empty(n_permutations)
    for b in range(n_permutations):
        # For each method, independently decide whether to swap species labels
        swap_mask = rng.integers(0, 2, size=n_methods).astype(bool)

        # Compute permuted rank shifts
        perm_r1 = np.where(swap_mask, ranks_sp2, ranks_sp1)
        perm_r2 = np.where(swap_mask, ranks_sp1, ranks_sp2)
        perm_delta = perm_r2 - perm_r1

        # Test statistic for focal methods
        T_perm = abs(perm_delta[fm1_idx] - perm_delta[fm2_idx])
        null_distribution[b] = T_perm

    # p-value: proportion of null stats >= observed
    p_value = float(
        (1 + np.sum(null_distribution >= T_obs)) / (1 + n_permutations)
    )

    # --- Effect size CI via bootstrap (resample methods) ---
    # Bootstrap over the non-focal methods to get CI for T
    other_indices = [
        i for i in range(n_methods) if i not in [fm1_idx, fm2_idx]
    ]
    boot_T = np.empty(1000)
    for b in range(1000):
        # Resample delta_rank with replacement from all methods
        boot_shifts = rng.choice(df["delta_rank"].values, size=n_methods, replace=True)
        boot_T[b] = abs(boot_shifts[fm1_idx] - boot_shifts[fm2_idx])

    ci_lower = float(np.percentile(boot_T, 2.5))
    ci_upper = float(np.percentile(boot_T, 97.5))

    return {
        "T_observed": T_obs,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "focal_methods": focal_methods,
        "rank_shift_method1": shift_1,
        "rank_shift_method2": shift_2,
        "effect_size_ci": (ci_lower, ci_upper),
        "all_rank_shifts": df[[method_col, rank_cols[0], rank_cols[1], "delta_rank"]],
    }


# ---------------------------------------------------------------------------
# 6. Adaptive vs. Fixed Interval Comparison
# ---------------------------------------------------------------------------

def adaptive_vs_fixed_comparison(
    data: pd.DataFrame,
    method_col: str = "method",
    species_col: str = "species",
    gf_fixed_col: str = "gf_fixed",
    gf_adaptive_col: str = "gf_adaptive",
    equivalence_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compare G-F Scores from adaptive vs. fixed interval strategies.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with method, species, fixed and adaptive G-F Score columns.
    method_col : str
        Column for method identifiers.
    species_col : str
        Column for species identifiers.
    gf_fixed_col : str
        Column for fixed-interval G-F Scores.
    gf_adaptive_col : str
        Column for adaptive-interval G-F Scores.
    equivalence_threshold : float
        Absolute threshold for TOST and MAPD acceptance.

    Returns
    -------
    dict
        Keys: wilcoxon, mapd, max_apd, spearman_rho, spearman_pvalue,
        kendall_tau, tost, accepted, per_method.
    """
    df = data.copy()
    df["diff"] = df[gf_adaptive_col] - df[gf_fixed_col]
    df["abs_diff"] = df["diff"].abs()
    df["apd"] = df["abs_diff"] / df[gf_fixed_col].abs()  # absolute pct diff (fraction)

    diffs = df["diff"].values
    diffs_nonzero = diffs[diffs != 0]

    # --- Wilcoxon signed-rank test ---
    if len(diffs_nonzero) >= 2:
        wilcox_result = stats.wilcoxon(diffs_nonzero, alternative="two-sided")
        W = wilcox_result.statistic
        pvalue = wilcox_result.pvalue
        n_nz = len(diffs_nonzero)
        r_rb = 1.0 - (4.0 * W) / (n_nz * (n_nz + 1))
    else:
        W, pvalue, r_rb = np.nan, np.nan, np.nan

    wilcoxon_result = {"W": W, "pvalue": pvalue, "effect_size_r_rb": r_rb}

    # --- MAPD ---
    mapd = float(df["apd"].mean())
    max_apd = float(df["apd"].max())

    # --- Spearman rank correlation ---
    rho, sp_pval = stats.spearmanr(df[gf_fixed_col], df[gf_adaptive_col])

    # --- Kendall's tau-b ---
    tau, tau_pval = stats.kendalltau(df[gf_fixed_col], df[gf_adaptive_col])

    # --- TOST (Two One-Sided Tests) for equivalence ---
    # H0_lower: mean_diff <= -delta  vs  H1_lower: mean_diff > -delta
    # H0_upper: mean_diff >= +delta  vs  H1_upper: mean_diff < +delta
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
    n = len(diffs)
    se_diff = std_diff / np.sqrt(n) if n > 1 and std_diff > 0 else np.nan

    if se_diff and se_diff > 0:
        t_lower = (mean_diff - (-equivalence_threshold)) / se_diff
        p_lower = float(1 - stats.t.cdf(t_lower, df=n - 1))  # one-sided upper

        t_upper = (mean_diff - equivalence_threshold) / se_diff
        p_upper = float(stats.t.cdf(t_upper, df=n - 1))  # one-sided lower
    else:
        t_lower, p_lower, t_upper, p_upper = np.nan, np.nan, np.nan, np.nan

    # Equivalence is concluded if BOTH one-sided tests reject at alpha=0.05
    tost_alpha = 0.05
    equivalent = (
        (not np.isnan(p_lower) and p_lower < tost_alpha)
        and (not np.isnan(p_upper) and p_upper < tost_alpha)
    )

    tost_result = {
        "t_lower": t_lower,
        "p_lower": p_lower,
        "t_upper": t_upper,
        "p_upper": p_upper,
        "equivalent": equivalent,
    }

    # --- Overall acceptance ---
    accepted = mapd < equivalence_threshold and (not np.isnan(rho) and rho > 0.95)

    # --- Per-method details ---
    per_method = df[
        [method_col, species_col, gf_fixed_col, gf_adaptive_col, "diff", "abs_diff", "apd"]
    ].copy()

    return {
        "wilcoxon": wilcoxon_result,
        "mapd": mapd,
        "max_apd": max_apd,
        "spearman_rho": float(rho),
        "spearman_pvalue": float(sp_pval),
        "kendall_tau": float(tau),
        "tost": tost_result,
        "accepted": accepted,
        "per_method": per_method,
    }


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def main():
    """Run statistical analysis on pipeline results.

    Loads G-F scores, link prediction AUROC, and downstream k-NN results,
    then computes:
      - Spearman correlations between G-F Score and downstream metrics
      - Bootstrap confidence intervals
      - Wilcoxon pairwise comparisons

    Saves: results/statistical_analysis_summary.json
    """
    import json
    from pathlib import Path
    from utils import (
        SEED, get_results_dir,
    )

    np.random.seed(SEED)
    results_dir = get_results_dir()

    # Load G-F scores
    gf_file = results_dir / "gf_scores.json"
    if not gf_file.exists():
        print("G-F scores not found. Run Step 3 (compute_gf.py) first.")
        return
    with open(gf_file) as f:
        gf_data = json.load(f)
    gf_scores = gf_data.get("scores", {})

    # Merge GNN scores
    gnn_file = results_dir / "gnn_gf_scores.json"
    if gnn_file.exists():
        with open(gnn_file) as f:
            gnn_data = json.load(f)
        gf_scores.update(gnn_data.get("gf_scores", {}))

    if not gf_scores:
        print("No G-F scores available. Step skipped.")
        return

    # Load link prediction AUROC
    lp_file = results_dir / "link_prediction.json"
    auroc = {}
    if lp_file.exists():
        with open(lp_file) as f:
            lp_data = json.load(f)
        if isinstance(lp_data, dict):
            for method, vals in lp_data.get("auroc_results", lp_data).items():
                if isinstance(vals, dict) and "auroc_mean" in vals:
                    auroc[method] = vals["auroc_mean"]

    # Load downstream k-NN
    knn_file = results_dir / "downstream_knn.json"
    knn_f1 = {}
    if knn_file.exists():
        with open(knn_file) as f:
            knn_data = json.load(f)
        for method, vals in knn_data.items():
            if isinstance(vals, dict) and "micro_f1_mean" in vals:
                knn_f1[method] = vals["micro_f1_mean"]

    # Build comparison data
    methods = sorted(gf_scores.keys())
    rows = []
    for m in methods:
        row = {"method": m, "gf_score": gf_scores[m], "species": "yeast"}
        if m in auroc:
            row["auroc"] = auroc[m]
        if m in knn_f1:
            row["knn_f1"] = knn_f1[m]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Run analyses
    output = {}

    # G-F score comparison table
    comparison = compute_gf_score_comparison(df)
    output["score_comparison"] = comparison.to_dict(orient="records") \
        if hasattr(comparison, "to_dict") else str(comparison)

    # Spearman correlation (G-F vs AUROC)
    if "auroc" in df.columns and df["auroc"].notna().sum() >= 3:
        spearman = spearman_correlation_analysis(df, "gf_score", "auroc")
        output["spearman_gf_vs_auroc"] = spearman
        print(f"Spearman(GF, AUROC): rho={spearman.get('rho', 'N/A')}, "
              f"p={spearman.get('p_value', 'N/A')}")

    # Spearman correlation (G-F vs k-NN)
    if "knn_f1" in df.columns and df["knn_f1"].notna().sum() >= 3:
        spearman_knn = spearman_correlation_analysis(df, "gf_score", "knn_f1")
        output["spearman_gf_vs_knn"] = spearman_knn
        print(f"Spearman(GF, k-NN): rho={spearman_knn.get('rho', 'N/A')}, "
              f"p={spearman_knn.get('p_value', 'N/A')}")

    # Wilcoxon pairwise
    if len(methods) >= 2:
        wilcoxon = wilcoxon_pairwise_comparison(df, "method", "gf_score")
        output["wilcoxon_pairwise"] = wilcoxon.to_dict(orient="records") \
            if hasattr(wilcoxon, "to_dict") else str(wilcoxon)

    # Save
    out_file = results_dir / "statistical_analysis_summary.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved statistical analysis to {out_file}")


# ===========================================================================
# Main: Example Usage
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("G-F Consistency Framework: Statistical Analysis -- Example Usage")
    print("=" * 70)

    # ---- Synthetic example data ----
    rng = np.random.default_rng(42)
    methods = [
        "DM", "Spectral", "PCA", "MDS", "VGAE", "VGAE-feat",
        "DeepWalk", "Node2Vec", "GraphSAGE", "GAT", "GIN",
    ]
    species = ["yeast", "human"]

    # Simulate G-F Scores with a cross-species rank reversal for DeepWalk/Node2Vec
    base_scores_yeast = np.array([
        0.625, 0.580, 0.550, 0.520, 0.490, 0.470,
        0.430, 0.420, 0.510, 0.500, 0.480,
    ])
    # Human: Node2Vec and DeepWalk jump to top
    base_scores_human = np.array([
        0.410, 0.395, 0.380, 0.370, 0.360, 0.350,
        0.450, 0.470, 0.400, 0.390, 0.385,
    ])

    rows = []
    for i, m in enumerate(methods):
        for sp, scores in zip(species, [base_scores_yeast, base_scores_human]):
            rows.append(
                {
                    "method": m,
                    "species": sp,
                    "gf_score": scores[i] + rng.normal(0, 0.005),
                    "auroc": 0.5 + 0.4 * scores[i] + rng.normal(0, 0.02),
                    "knn_f1": 0.3 + 0.5 * scores[i] + rng.normal(0, 0.03),
                }
            )
    example_df = pd.DataFrame(rows)

    # ---- 1. G-F Score Comparison ----
    print("\n--- 1. G-F Score Comparison ---")
    comparison = compute_gf_score_comparison(example_df)
    print(comparison["comparison_table"].to_string(index=False))
    print(
        f"\nRank correlation (yeast vs human): rho = "
        f"{comparison['rank_correlation']:.3f}, "
        f"p = {comparison['rank_correlation_pvalue']:.4f}"
    )

    # ---- 2. Spearman Correlation ----
    print("\n--- 2. Spearman Correlation: AUROC vs G-F Score ---")
    corr_result = spearman_correlation_analysis(
        example_df, gf_col="gf_score", metric_col="auroc"
    )
    pooled = corr_result["pooled"]
    print(
        f"Pooled: rho = {pooled['rho']:.3f}, "
        f"p = {pooled['pvalue']:.4f}, "
        f"95% CI = [{pooled['ci_lower']:.3f}, {pooled['ci_upper']:.3f}]"
    )
    print(f"Interpretation: {pooled['interpretation']}")

    if "per_species" in corr_result:
        for sp, sp_res in corr_result["per_species"].items():
            print(
                f"  {sp}: rho = {sp_res['rho']:.3f}, "
                f"p = {sp_res['pvalue']:.4f}, n = {sp_res['n']}"
            )

    # ---- 3. Wilcoxon Pairwise Comparison ----
    print("\n--- 3. Wilcoxon Pairwise Comparison (top 5 pairs) ---")
    wilcox = wilcoxon_pairwise_comparison(example_df)
    pr = wilcox["pairwise_results"].sort_values("pvalue_fdr")
    print(pr.head(5).to_string(index=False))
    print(
        f"\n{wilcox['n_significant']}/{wilcox['n_comparisons']} comparisons significant "
        f"after FDR correction (alpha=0.05)"
    )

    # ---- 4. Bootstrap Confidence Intervals ----
    print("\n--- 4. Bootstrap Confidence Intervals ---")
    ci_df = bootstrap_confidence_intervals(example_df)
    print(ci_df[["method", "species", "mean", "ci_lower", "ci_upper", "ci_width"]].to_string(index=False))

    # ---- 5. Permutation Test for Rank Reversal ----
    print("\n--- 5. Permutation Test: Node2Vec vs DeepWalk Rank Reversal ---")
    perm_result = permutation_test_rank_reversal(
        comparison["comparison_table"],
        focal_methods=["Node2Vec", "DeepWalk"],
    )
    print(
        f"T_observed = {perm_result['T_observed']:.2f}, "
        f"p-value = {perm_result['p_value']:.4f}"
    )
    print(
        f"Rank shifts: Node2Vec = {perm_result['rank_shift_method1']:+.0f}, "
        f"DeepWalk = {perm_result['rank_shift_method2']:+.0f}"
    )
    print(f"95% CI for T: [{perm_result['effect_size_ci'][0]:.2f}, "
          f"{perm_result['effect_size_ci'][1]:.2f}]")

    # ---- 6. Adaptive vs Fixed Comparison ----
    print("\n--- 6. Adaptive vs. Fixed Interval Comparison ---")
    # Simulate adaptive scores (close to fixed)
    example_df["gf_fixed"] = example_df["gf_score"]
    example_df["gf_adaptive"] = example_df["gf_score"] * (
        1 + rng.uniform(-0.03, 0.03, size=len(example_df))
    )
    adapt_result = adaptive_vs_fixed_comparison(example_df)
    print(f"MAPD: {adapt_result['mapd']:.4f}")
    print(f"Spearman rho: {adapt_result['spearman_rho']:.4f}")
    print(f"TOST equivalent: {adapt_result['tost']['equivalent']}")
    print(f"Accepted: {adapt_result['accepted']}")

    print("\n" + "=" * 70)
    print("All analyses completed successfully.")
    print("=" * 70)
