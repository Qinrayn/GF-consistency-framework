# Phase 8: Cross-Network Validation & Bootstrap Confidence Intervals

## 8.1 Overview

Phases 1-7 were all developed and validated on the curated 153-node yeast
PPI subnetwork. While human PPI validation exists for the core pipeline
(Steps 13, 33, 37-38) and for spectral alignment (Phase 5A), the deeper
predictive models (two-factor, three-factor) and TDA bridge features have
never been tested out-of-sample. Phase 8 addresses two limitations:

(A) **Cross-network transfer**: Tests the two-factor (spectral alignment +
effective rank) and three-factor (+ H1 persistence) models on the human
PPI network (15,882 nodes, 2,000 subsampled).

(B) **Statistical robustness**: Computes bootstrap 95% confidence intervals
for all Phase 7 single-factor and partial correlations via 10,000 resamples
of the 11-method yeast feature matrix.

## 8.2 Human Two-Factor Validation (11 Methods)

Using SVD-based effective rank (computed directly from human embeddings)
and spectral alignment (from Phase 5A), the two-factor model was tested
on all 11 methods:

    Predictor                      rho       p
    Spectral alignment            0.200    0.555
    Effective rank                0.418    0.201
    Effective dim (Phase 5A)      0.082    0.811
    Two-factor (spec+eff_rank)    0.543    0.085
    Two-factor (spec+eff_dim)     0.146    0.667

Key observations:

(1) The SVD-based effective rank (rho=0.418) outperforms the Phase 5A
PCA-based effective dimensionality (rho=0.082). This suggests that the
SVD participation ratio captures geometric structure that the PCA
eigenvalue participation ratio misses on the larger human network.

(2) The two-factor model with effective rank achieves rho=0.543 (p=0.085),
a marginal trend that substantially improves over Phase 5A's rho=0.218
(p=0.519). While not significant at alpha=0.05, the improvement is
notable: the geometric predictors capture a meaningful signal that was
previously masked by the inferior effective dimensionality measure.

(3) Individual factor correlations are weak on human: spectral alignment
(rho=0.200) and effective rank (rho=0.418) are both below their yeast
counterparts (0.609 and 0.873 respectively). This confirms Phase 5A's
conclusion that the geometric-functional correspondence is network-specific.

## 8.3 Human Three-Factor Validation (6 Methods)

For the 6 methods with available human TDA data (DM, MDS, Spectral,
DeepWalk, Node2Vec, VGAE):

    Predictor                      rho       p
    Spectral alignment            0.657    0.156
    Effective rank                0.600    0.208
    H1 max persistence           -0.086    0.872
    Two-factor (spec+eff_rank)    0.880    0.021  *
    Three-factor (+h1_max)        0.600    0.208

Key findings:

(1) The two-factor model IS significant on the 6-method subset
(rho=0.880, p=0.021), providing the first evidence that the spectral +
effective rank combination can transfer to human PPI when the method
sample is restricted.

(2) H1 max persistence shows essentially zero correlation with human
G-F Score (rho=-0.086). This contrasts sharply with yeast (rho=0.764).
The likely explanation is that the human TDA features were computed with
different parameters (r range [0.05, 0.3] vs yeast's [0.05, 0.55],
different subsampling), making the persistence values not directly
comparable.

(3) Adding H1 persistence to the two-factor model degrades performance
(rho=0.600 vs 0.880), confirming that the human H1 feature is uninformative
in this context. The three-factor model does NOT transfer to human.

## 8.4 Yeast vs Human Comparison

    Factor/Model              Yeast rho    Human rho    Transfer?
    Spectral alignment        0.609        0.200        No
    Effective rank            0.873        0.418        Partial
    Two-factor (11 methods)   0.809        0.543        Marginal
    Two-factor (6 methods)    0.809        0.880        Yes *
    H1 max persistence        0.764       -0.086        No
    Three-factor              0.909        0.600        No

The two-factor model shows partial transfer when using SVD-based effective
rank, and significant transfer on the 6-method subset. The TDA component
does not transfer, likely due to methodological differences in the human
TDA computation pipeline.

## 8.5 Bootstrap Confidence Intervals: Single-Factor Correlations

10,000 bootstrap resamples of the yeast 11-method feature matrix:

    Feature                       rho      95% CI            Significant?
    topo_gf_score                0.973    [+0.832, +1.000]   Yes ***
    effective_rank               0.873    [+0.484, +1.000]   Yes ***
    h1_max_persistence           0.764    [+0.279, +0.972]   Yes **
    h1_topological_complexity    0.636    [+0.009, +0.971]   Yes *
    spectral_alignment           0.609    [-0.093, +0.924]   Borderline
    topo_consistency            -0.382    [-0.850, +0.333]   No

Key findings:

(1) Four features have bootstrap-robust significant correlations (CI
excludes 0): topo_gf_score, effective_rank, h1_max_persistence, and
h1_topological_complexity. These are the most reliable Phase 7 findings.

(2) spectral_alignment (rho=0.609) is borderline — the 95% CI just barely
includes 0 at -0.093. This is consistent with the marginal p=0.047 from
the original analysis; the correlation is real but fragile at n=11.

(3) topo_consistency (rho=-0.382) is not robust — the CI spans [-0.850,
+0.333], confirming the original p=0.247.

## 8.6 Bootstrap Confidence Intervals: Partial Correlations

Partial correlations controlling for spectral alignment + effective rank:

    Feature                       partial_rho   95% CI            Significant?
    topo_gf_score                  +0.809     [-0.196, +1.000]    No
    h1_max_persistence             +0.318     [-0.608, +0.935]    No
    h1_topological_complexity      -0.282     [-0.907, +0.886]    No
    topo_consistency               -0.182     [-0.856, +0.634]    No

None of the partial correlations are robust under bootstrap resampling.
This is a critical honesty check on the Phase 7 claims:

(1) The Phase 7 claim that "topo_gf_score adds independent signal
(partial rho=0.845, p=0.001)" is NOT robust at n=11. While the point
estimate is high, the bootstrap CI spans [-0.196, +1.000], meaning
the true partial correlation could plausibly be zero.

(2) The fundamental issue is statistical power. Partial correlations
require estimating 4 variables' joint distribution, which is very
sensitive to small samples. With n=11, even a strong partial rho of
0.8 can have a CI that includes 0.

(3) This does NOT mean the Phase 7 conclusions are wrong — it means
they are not yet confirmed at the level of statistical certainty
required for publication. The single-factor correlations (topo_gf_score
rho=0.973, effective_rank rho=0.873, h1_max_persistence rho=0.764)
are robust, providing strong evidence that these features capture
real signal. The partial correlation analysis needs validation on a
larger sample (multiple networks, or subsampled networks from the same
species).

## 8.7 Revised Phase 7 Conclusions

Based on the bootstrap analysis, the Phase 7 conclusions should be
revised as follows:

(1) **Confirmed (robust):** topo_gf_score, effective_rank, and
h1_max_persistence are strong single-factor predictors of G-F Score
(bootstrap 95% CI excludes 0 for all three).

(2) **Suggestive (not yet robust):** The claim that TDA features add
INDEPENDENT signal beyond spectral alignment + effective rank is
supported by the point estimates but not by bootstrap CIs. This claim
requires validation on a larger dataset.

(3) **Practical recommendation:** For embedding quality prediction,
use the robust single-factor predictors (topo_gf_score or effective_rank)
rather than the multi-factor partial-correlation-based claims.

## 8.8 Figures

- **Fig46**: Human cross-network validation (4 panels)
  - A: Human spectral alignment vs G-F Score (rho=0.200, n=11)
  - B: Human effective rank vs G-F Score (rho=0.418, n=11)
  - C: Yeast vs human factor correlation comparison (bar chart)
  - D: Yeast vs human model comparison for 6-method TDA subset

- **Fig47**: Bootstrap confidence intervals (3 panels)
  - A: Single-factor Spearman CIs (green=significant, gray=ns)
  - B: Partial correlation CIs (all gray=ns after bootstrap)
  - C: Bootstrap summary table with significance indicators

## 8.9 Limitations

(1) Human TDA data was computed with different parameters (older pipeline)
and only for 6 of 11 methods. A complete human TDA analysis with current
parameters would strengthen the three-factor validation.

(2) n=11 is fundamentally insufficient for robust partial correlations.
Future work should test on multiple networks (different species,
tissue-specific, or subsampled) to increase effective sample size.

(3) The bootstrap analysis uses Spearman rank-based residualization for
partial correlations, which may have different statistical properties
than the Pearson-based approach used in Phase 7.
