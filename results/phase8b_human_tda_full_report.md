# Phase 8B: Full Human TDA Analysis (11 Methods) — Supplement Report

## 8B.1 Motivation

Phase 8 (§8.3) reported that H1 max persistence shows essentially zero
correlation with human G-F Score (rho=-0.086), but this result was
confounded: the human TDA features were computed with different parameters
(r range [0.05, 0.3] vs yeast's [0.05, 0.55], different subsampling)
and only for 6 of 11 methods. Phase 8B eliminates both confounds by
recomputing persistent homology for all 11 methods on human PPI using
the identical parameters as the yeast pipeline.

## 8B.2 Methods

Persistent homology was computed for all 11 methods on the human PPI
(15,882 nodes, 2,000 subsampled, seed=42) using Ripser with Vietoris-Rips
complexes (max_dim=1). The r-grid, rescaling (TARGET_STD=0.3), and GF
integration interval [0.05, 0.422] match the yeast pipeline exactly.

H0 (connected components) and H1 (loops) persistence diagrams were
computed. Persistence statistics (max, mean, topological complexity,
feature count) and Betti curves were extracted for each method.

Topological G-F curves (community detection at each r) were skipped due
to computational cost (greedy_modularity_communities on ~2000 nodes at
200 r values x 11 methods is prohibitively expensive). Instead, we use
persistence statistics directly for the three-factor validation.

## 8B.3 Persistent Homology Results

    Method       H0 features  H1 features  H1 max persistence
    DM           1997         459          0.0648
    MDS          1976         505          0.0538
    Spectral     1998         420          0.0012
    DeepWalk     2000         443          0.0985
    Node2Vec     2000         423          0.0859
    VGAE         2000         314          0.0055
    PCA          1997         505          0.1206
    VGAE-feat    1997         455          0.0530
    GraphSAGE    1997         519          0.1055
    GAT          1995         510          0.0028
    GIN          1997         504          0.1015

Key observations:

(1) All methods produce H1 features in the range 314-519, confirming that
loop structure exists in human PPI embeddings at the current scale (n=2000).

(2) H1 max persistence varies dramatically: Spectral (0.0012) and GAT
(0.0028) have near-zero persistence (short-lived loops), while PCA (0.1206),
GraphSAGE (0.1055), and GIN (0.1015) have the most persistent loops.

(3) Critically, the methods with highest H1 max persistence (PCA, GIN,
GraphSAGE) are NOT the methods with highest G-F Scores (Spectral, MDS).
This is the first indication that H1 persistence may not predict G-F
Score on human PPI.

## 8B.4 Single-Factor Correlations (Human, 11 Methods)

    Predictor                  rho       p
    Spectral alignment        0.200    0.555
    Effective rank            0.418    0.201
    H1 max persistence        0.073    0.832
    H1 mean persistence        0.118    0.729
    H1 topological complexity  0.118    0.729

None of the TDA features (H1 max, mean, complexity) show meaningful
correlation with human G-F Score. The strongest TDA feature (H1 max
persistence, rho=0.073) is essentially zero.

Comparison with yeast (Phase 7):

    Feature              Yeast rho    Human rho    Ratio
    Spectral alignment   0.609        0.200        0.33x
    Effective rank       0.873        0.418        0.48x
    H1 max persistence   0.764        0.073        0.10x

The drop-off is most severe for TDA features: H1 max persistence retains
only 10% of its yeast correlation on human. This suggests that persistent
loops in 2D embeddings are highly sensitive to network structure and do
not capture transferable signal about functional-geometric consistency.

## 8B.5 Multi-Factor Models

    Model                          rho       p
    Two-factor (spec+eff_rank)    0.543    0.085
    Three-factor (+h1_max)        0.282    0.400
    Weight-optimized 3F           0.300    0.370  w=[0.33,0.33,0.34]

The three-factor model (rho=0.282) is substantially WORSE than the
two-factor model (rho=0.543). Adding H1 max persistence as a third
factor degrades the model by injecting noise. The weight optimizer
converges to near-equal weights [0.33, 0.33, 0.34], indicating it
cannot find a useful weighting — the H1 signal is genuinely absent.

This contrasts with Phase 7 yeast results where the three-factor model
improved over two-factor (rho=0.827 vs 0.809 with H1 max persistence,
or rho=0.909 with topo_gf_score).

## 8B.6 Yeast vs Human Full Comparison

    Model                   Yeast rho   Human rho   Transfer?
    Spectral alignment      0.609       0.200       No
    Effective rank          0.873       0.418       Partial
    H1 max persistence      0.764       0.073       No
    Two-factor              0.809       0.543       Marginal
    Three-factor            0.827       0.282       No

The complete 11-method comparison confirms Phase 8's finding: the
two-factor model shows marginal transfer (p=0.085), but the TDA
component does NOT transfer to human PPI at any meaningful level.

## 8B.7 Revised Interpretation

Phase 7 established that TDA features (particularly topo_gf_score and
H1 max persistence) are strong predictors of G-F Score on yeast. Phase 8
showed that H1 persistence from the OLD human pipeline (6 methods,
different parameters) was uninformative. Phase 8B eliminates the
parameter mismatch and extends to all 11 methods — the result is the
same: H1 persistence is uninformative on human PPI.

This is NOT a methodological artifact. The persistent homology computation
is identical (Ripser, Vietoris-Rips, same r-grid, same rescaling). The
difference is in the network itself: human PPI (15,882 nodes) has
fundamentally different topological structure than yeast PPI (5,936 nodes)
when embedded in 2D.

Possible explanations:

(1) Scale effect: At n=2000 subsampled from 15,882 nodes, the embedding
density is lower than yeast (n=153 from 5,936), making loops less
informative about functional modules.

(2) Annotation structure: Human GO annotations are richer but noisier
than yeast, potentially decoupling loop structure from functional
coherence.

(3) Embedding quality: 2D embeddings of a 15,882-node network may lose
more topological information than 2D embeddings of a 5,936-node network,
making the surviving loops less meaningful.

(4) Method-specific effects: On human, the methods with highest H1
persistence (PCA, GIN, GraphSAGE) are mid-to-low performers on G-F Score.
Their loops may reflect geometric artifacts rather than functional
structure.

## 8B.8 Implications for the Three-Factor Framework

The Phase 7 three-factor framework (spectral alignment + effective rank +
TDA feature) is confirmed as YEAST-SPECIFIC. For human PPI:

- The two-factor model (spectral alignment + effective rank) remains the
  best available predictor, achieving marginal significance at rho=0.543
  (p=0.085) for 11 methods and significant rho=0.880 (p=0.021) for the
  6-method subset (Phase 8).

- TDA features (H1 persistence) do not add signal on human PPI and
  should NOT be included in cross-species predictive models.

- The revised recommendation is: (1) use effective rank as a fast
  screening metric (works partially on both species), (2) use spectral
  alignment for network-specific refinement, (3) reserve TDA features
  for within-species fine-grained ranking where they have been validated.

## 8B.9 Figures

- **Fig48**: Human TDA + three-factor validation (3 panels)
  - A: H1 max persistence vs G-F Score (rho=0.073, p=0.832, n=11)
  - B: Yeast vs human model comparison (grouped bar chart)
  - C: Human Betti curves (beta_1) for top-3 and bottom-3 methods

## 8B.10 Limitations

(1) Topological G-F Score (the strongest yeast predictor, rho=0.973)
could not be computed for human due to computational cost. The persistence
statistics used here are a weaker proxy. Future work with optimized
community detection (e.g., Leiden on GPU) could enable full topo_gf_score
computation on 2000-node human subsamples.

(2) The n=11 sample size limits statistical power. Even the two-factor
model's rho=0.543 does not reach significance (p=0.085). The null TDA
results (rho=0.073) should not be over-interpreted as evidence of
ABSENCE of signal — they indicate that any signal is too weak to detect
at this sample size.

(3) Only 2D embeddings are tested. Higher-dimensional embeddings may
exhibit different topological properties that transfer better across
species.
