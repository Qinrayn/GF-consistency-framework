# Phase 8C: Leave-One-Out Sensitivity Analysis — Supplement Report

## 8C.1 Motivation

Phase 8B found that H1 max persistence has near-zero correlation with
human G-F Score (rho=0.073, p=0.832). The verification audit identified
Spectral as a catastrophic outlier: its H1 max persistence on human
(0.0012) is 80x lower than its yeast counterpart (0.096), while all
other methods show at most 6x cross-species change. This analysis tests
whether the null TDA result is driven by this single point.

## 8C.2 Leave-One-Out Results

    Predictor          Full rho    Excl Spectral   Delta    Excl GAT      Delta
    Spectral align.    +0.200      +0.321          +0.121   +0.600        +0.400
    Effective rank     +0.418      +0.224          -0.194   +0.224        -0.194
    H1 max persist.    +0.073      +0.430          +0.357   -0.115        -0.188
    Two-factor         +0.543      +0.351          -0.192   +0.622        +0.079
    Three-factor       +0.282      +0.382          +0.100   +0.103        -0.179

Key findings:

(1) H1 MAX PERSISTENCE: Excluding Spectral causes the most dramatic
change of any LOO test — rho jumps from +0.073 to +0.430 (delta =
+0.357). This is a 6x increase. The "TDA has no signal on human"
conclusion is primarily driven by one anomalous data point.

(2) SPECTRAL ALIGNMENT: Excluding GAT causes rho to jump from +0.200 to
+0.600 (delta = +0.400). GAT's collapsed embedding (eff_rank=1.002)
gives it a disproportionate influence on spectral alignment correlation.

(3) EFFECTIVE RANK: Spectral supports the effective rank correlation —
excluding it drops rho from +0.418 to +0.224. This is expected: Spectral
has both max eff_rank (2.0) and max GF (0.402), acting as an anchor.

(4) TWO-FACTOR MODEL: Also anchored by Spectral (rho drops to +0.351
when excluded), but the model remains the most robust across all LOO
tests (range: 0.351 to 0.652).

(5) THREE-FACTOR MODEL: VGAE is unexpectedly influential — excluding it
drops rho to +0.067. The three-factor model is the least stable.

## 8C.3 Spectral's Anomalous Topology: Root Cause

Spectral has 420 H1 features on human (normal count), but their max
persistence is 0.0012 (44-100x lower than other methods with similar
loop counts). The topology EXISTS but is trivially short-lived.

Mechanism: Spectral embeddings are constructed from Laplacian
eigenvectors, which become increasingly smooth as graph size grows.
On the 2000-node human subsample (from 15,882 nodes), the eigenvector
coordinates produce an extremely smooth 2D manifold. After rescaling
to std=0.3, loops form and merge at very small filtration radii,
yielding near-zero persistence lifetimes.

On yeast (n=153), the same mechanism produces Spectral H1_max = 0.096
(high), because the 153-point manifold has enough geometric complexity
to sustain persistent loops. The 80x drop from yeast to human is
unique to Spectral — no other method shows a comparable change.

## 8C.4 Cross-Species Spectral Asymmetry

    Species    Spectral H1_max   Spectral GF    Spectral eff_rank
    Yeast      0.096             0.163          2.000
    Human      0.001             0.402          2.000

In yeast, Spectral has HIGH H1 persistence AND high G-F Score, making
it a key point supporting the positive H1-vs-GF correlation (rho=0.764).
In human, Spectral has NEAR-ZERO H1 persistence AND high G-F Score,
creating an anti-correlation anchor that pulls rho toward zero.

This asymmetry means:
- Yeast LOO (excl Spectral): H1 rho would DECREASE (Spectral supports)
- Human LOO (excl Spectral): H1 rho INCREASES to +0.430 (Spectral opposes)

The same method has opposite topological behavior in the two species,
not because of a computational error, but because Laplacian eigenvector
smoothness scales with graph size.

## 8C.5 Revised Interpretation

Phase 8B's conclusion that "TDA loop signal does not transfer to human"
should be revised:

(1) There IS a moderate positive H1-vs-GF signal on human (rho=+0.430
after excluding Spectral), but it is masked by Spectral's anomalous
topology at the human scale.

(2) The three-factor model does not improve over two-factor even after
excluding Spectral (rho=0.382 vs 0.351), suggesting that H1 persistence
adds marginal value on human regardless.

(3) The two-factor model (spectral alignment + effective rank) remains
the most robust predictor combination, with the widest LOO stability
range.

(4) For future cross-species TDA comparisons, Laplacian-based methods
(Spectral) should be flagged as potential outliers due to scale-dependent
eigenvector smoothness. The analysis should report both full and
Spectral-excluded results.

## 8C.6 Recommendations for the Paper

(1) Report the LOO analysis as a robustness check in Supplementary
Materials. Show that excluding Spectral reveals latent H1 signal
(rho=+0.430), but that the two-factor model remains the recommended
cross-species predictor.

(2) Add a caveat: "The near-zero H1-vs-GF correlation on human (rho=
0.073) is primarily driven by Spectral's anomalously low topological
persistence (0.0012 vs 0.096 on yeast), a consequence of Laplacian
eigenvector smoothness scaling with graph size. Excluding Spectral
reveals a moderate positive signal (rho=+0.430), but the three-factor
model does not consistently outperform the two-factor model."

(3) Consider using persistence-normalized features (e.g., H1 persistence
/ max_pairwise_distance) to correct for scale-dependent effects.

## 8C.7 Figures

- **Fig49**: LOO sensitivity (3 panels)
  - A: Delta-rho bar chart (LOO impact per method per predictor)
  - B: H1 max persistence LOO horizontal bars (Spectral in orange)
  - C: Scatter with Spectral outlier highlighted, dual trend lines
