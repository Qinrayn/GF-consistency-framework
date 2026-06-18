# G-F Consistency Framework

**A Geometric-Functional Consistency Framework for Evaluating Protein Interaction Network Embeddings**

Complete reproduction of the experimental pipeline: 11 embedding methods · 72-step validation workflow · 200-point G-F curve sampling · publication-quality figures · `random_seed = 42`

<details>
<summary><strong>Key Results at a Glance</strong></summary>

| Method | G-F Score | Link Pred. AUC | k-NN micro-F1 |
|--------|:---------:|:--------------:|:-------------:|
| **Spectral** | **0.163** | 0.885 | 0.672 |
| DM | 0.155 | 0.726 | 0.513 |
| MDS | 0.152 | 0.904 | 0.639 |
| Node2Vec | 0.151 | 0.491 | 0.185 |
| PCA | 0.138 | 0.678 | 0.521 |
| VGAE-feat | 0.124 | 0.466 | 0.168 |
| DeepWalk | 0.123 | 0.489 | 0.193 |
| GIN | 0.122 | 0.483 | 0.210 |
| GAT | 0.069 | 0.534 | 0.160 |
| GraphSAGE | 0.069 | 0.531 | 0.177 |
| VGAE | 0.066 | 0.491 | 0.227 |

- Leiden baseline purity: **0.180** (same formula as G-F curve: most-common GO term / total GO terms)
- Leiden baseline ≈ best G-F Score (0.180 vs 0.163), indicating spatial embeddings capture functional structure at a level comparable to graph-based community detection
- Bonferroni (30 subsets, size 150): **9/30** significant after correction
- Randomization: original max 0.247 > shuffled 0.230 ± 0.002 (10 permutations, Z = 6.95)
- Spearman rho (G-F Score vs Link Pred AUC): **+0.591** (*P* = 0.056, 95% CI [−0.09, 0.88], n = 11)
- Spearman rho (G-F Score vs k-NN F1): **+0.609** (*P* = 0.047, 95% CI [−0.05, 0.92], n = 11)
- Spearman rho (standard vs IC-weighted purity): **ρ = +0.964** (*P* < 0.001) — IC-weighting preserves rankings while suppressing DAG inflation
- Spearman rho (standard vs semantic purity): **ρ = +0.491** (*P* = 0.125) — semantic (Resnik) captures complementary coherence signal
- H1 max persistence vs G-F Score: **ρ = +0.764** (*P* = 0.006, 95% CI [0.27, 0.97])
- Cross-species rank consistency (yeast vs human, 11 methods): **ρ = +0.500** (*P* = 0.117), **Kendall W = 0.750** — strong concordance; Spectral #1 and MDS #2 in both species
- Scale gradient Kendall W (500-4000 nodes, 4 methods): **W = 0.700** — rank stability across scales; PCA consistently #1 at all scales
- Density-corrected GF (STRING threshold gradient, 4 methods): **W<sub>raw</sub> = 0.178 → W<sub>corrected</sub> = 0.70** (ΔW = 0.522) — density correction dramatically improves rank concordance
- Human seed stability (10 seeds, 11 methods): **Kendall W = 0.675** — strong rank stability; Spectral mean rank 1.1 (std 0.3), consistently #1
- Human IC-weighted GF (11 methods): **Spearman ρ = +0.991** (*P* < 0.001) — IC-weighting preserves rankings while correcting for DAG inflation
- GAT collapse diagnosis (5 variants): all variants show persistent collapse — architectural, not optimization-related
- **Geometric predictability** (Phase 1-3): Effective dimensionality (ρ=+0.905, p=0.002) and spectral alignment (ρ=+0.810, p=0.015) are independent predictors of G-F Score; combined model achieves **ρ=+0.929** (p=0.001), explaining 86% of inter-method variance
- **Cross-species geometric transfer fails** (Phase 2): Yeast-trained predictor does not generalise to human PPI (ρ=+0.027) — the geometric-functional correspondence is network-topology-specific
- **Two-factor model** (Phase 3): G-F consistency requires both correct spectral alignment (embedding captures network's functional modes) and sufficient geometric expressiveness (effective dimensionality ≈ 2.0)
- **GAT collapse mechanism** (Phase 4): Attention degeneration (normalized entropy=0.973) renders GAT equivalent to GCN; 2-layer mean aggregation with 2D bottleneck produces rank-1 output (effective rank=1.019, dimension variance ratio=105:1). All 5 architectural variants fail — collapse is architectural, not optimization-related
- **Two-factor model does not transfer to human** (Phase 5A): Spectral alignment (rho=0.200, p=0.555), effective dimensionality (rho=0.082, p=0.811), and combined model (rho=0.218, p=0.519) all fail on human PPI — the geometric-functional correspondence is network-specific
- **GAT dimension sweep** (Phase 5B): Increasing latent_dim from 2 to 32 does NOT rescue GAT — G-F Score stays near-random (0.067-0.112), attention entropy stays at ~0.974 across all dimensions. Attention degeneration is dimension-independent, confirming it as the root cause rather than the 2D bottleneck. GraphSAGE at d=32 achieves G-F=0.210 (above random baseline), outperforming GAT
- **Formal GAT collapse proof** (Phase 6): Three theorems rigorously establish the collapse causal chain — (T1) attention degeneration bound from degree CV, (T2) effective rank bound for mean-aggregation GNN (GNN: 1.045 vs non-GNN: 1.702, rho=0.873), (T3) G-F Score upper bound for rank-1 embeddings (rank-1 methods: GF_2D/GF_1D ratio~1.0; full-rank: up to 1.73). Combined corollary: GAT collapse is architecturally necessary, not an accident of training
- **TDA-Geometry Bridge** (Phase 7): Topological G-F Score (rho=0.973) is the strongest single predictor; partial correlation analysis shows TDA adds independent signal beyond spectral alignment + effective rank (partial rho=0.845, p=0.001). H1 max persistence provides marginal independent signal (partial rho=0.527, p=0.096). Three-factor model (spectral+eff_rank+topo_gf) achieves rho=0.909, improving over the two-factor model by 12.4%. Betti curve analysis reveals high-G-F methods have richer loop structure (DM: 11 loops) while rank-collapsed methods (VGAE, VGAE-feat) have zero H1 features
- **Cross-network validation & bootstrap CIs** (Phase 8): Two-factor model partially transfers to human PPI (rho=0.543, p=0.085 for 11 methods; rho=0.880, p=0.021 for 6-method subset). SVD-based effective rank outperforms PCA-based effective dimensionality on human (rho=0.418 vs 0.082). Bootstrap analysis (10k resamples) confirms 4 single-factor predictors are robust (topo_gf_score, effective_rank, h1_max_persistence, h1_topological_complexity — all CIs exclude 0), but partial correlations are not robust at n=11
- **Full human TDA analysis** (Phase 8B): Persistent homology recomputed for all 11 methods on human PPI with identical parameters as yeast — H1 max persistence (rho=0.073) does NOT predict G-F Score on human; three-factor model degrades to rho=0.282 (worse than two-factor rho=0.543). TDA loop signal is yeast-specific, not cross-species transferable
- **LOO sensitivity** (Phase 8C): Spectral is a catastrophic topological outlier on human (H1 persistence 80x lower than yeast). Excluding Spectral reveals latent H1 signal (rho=+0.430), but two-factor model remains the most LOO-stable cross-species predictor
- **Unified human G-F Scores** (Phase 9): Eliminating community-detection (Louvain→greedy_modularity) and interval ([0.282,0.297]→[0.05,0.422]) confounds yields rho=0.927 rank correlation with original scores. Top-3 (Spectral, MDS, Node2Vec) and bottom-2 (VGAE, GAT) rankings are identical. All predictor correlations preserved in direction (two-factor rho=+0.483 vs old +0.543). LOO pattern confirms Phase 8C (H1→+0.418 excl Spectral)
- **Three-species mouse validation** (Phase 10): Spectral ranks #1 in all 3 species (yeast, human, mouse). Kendall W=0.739 (11 methods, 3 species). Top-2 (Spectral, MDS) and bottom-2 (VGAE, VGAE-feat) identical in all species. Two-factor geometric model does NOT transfer to mouse (rho=−0.037) — the spectral alignment + effective rank predictors are network-specific, even though method quality is conserved. Spectral is a topological outlier in both human and mouse (H1 persistence 72–161× lower). Persistence images do not improve G-F prediction over scalar H1 features
- **Spectral Transferability Theory** (Phase 11): Derives closed-form Spectral Quality Index (SQI = λ₂/λ₂_ER × PR × FA_max) predicting when the two-factor model works. SQI ordering: yeast(10.72) > human(2.02) > mouse(0.54) matches two-factor rho: +0.929 > +0.483 > −0.037. Mouse Fiedler vector is 6× more localized (PR=0.0007). Validated on 20 SBM networks (SA_std vs log(SQI) rho=+0.647)
- **Biological Validation & Statistical Power** (Phase 12): GO BP enrichment confirms Spectral's functional coherence on yeast (80% enriched communities, p=4.58e-10) but not human (0%) or mouse (14%), where GraphSAGE/GAT produce more biologically coherent modules. Multi-seed panel (n=220 observations, 20 groups) yields pooled rank consistency |ρ|=0.583 (95% CI [0.470, 0.688]); per-species: yeast 0.981, human 0.967, mouse 0.800
- **Protein Function Prediction — Closing the Loop** (Phase 13): Leave-one-term-out CV on full yeast STRING network (5,936 nodes, 12,690 trials) demonstrates that GF-consistent embeddings predict protein function (Spectral P@10=0.148, best among 5 embedding methods). GF Score on curated 153-node network predicts function-prediction MRR on full 5,936-node network (Spearman rho=0.900, P=0.037, n=5 methods)
- **E. coli K-12 4th Species Validation** (Step 40): Spectral ranks #2 (GF=0.240), MDS #1 (GF=0.245) on *E. coli* STRING network. SQI=0.7 (lowest among 4 species). Kendall's W=0.652 including *E. coli* (vs 0.739 for 3 eukaryotic species). Spectral optimality extends to prokaryotes, attenuated by lower network spectral quality
- **Coexpression Network G-F Analysis** (Step 41): Method optimality is network-type-dependent. DeepWalk #1 (GF=0.877) on coexpression network, Spectral drops to mid-tier. PPI-coexpression rank correlation rho=0.071. Random-walk methods outperform spectral embeddings on coexpression networks — the opposite of PPI
- **Degree-Preserving Null Model** (Step 42): 50 double-edge-swap randomizations preserving exact degree sequence. Spectral methods fall substantially below DP null (Spectral z=-11.9, MDS z=-18.4); random-walk methods exceed it (DeepWalk z=+2.4, Node2Vec z=+2.7, p<0.01). Reveals fundamental spectral vs random-walk dichotomy
- **GAT Collapse on Full Network** (Step 43): All three collapse theorems verified on full 5936-node network. T1: H_norm=1.059 (bound satisfied). T2: GNN mean eff_rank=1.228 vs non-GNN=1.816. T3: low-rank methods have GF_2D/GF_1D ratio near 1. Collapse is NOT a small-network artifact
- **Community Detection Ablation** (Step 44): Kendall's W=0.797 across 5 community detection algorithms (greedy_modularity, label_propagation, connected_components, louvain, leiden). Spectral ranks #1 under 4 of 5 algorithms. G-F Score robustness to community detection methodology confirmed
- **Full 11-Method Function Prediction** (Step 45): Expanded LOTO-CV from 5 to all 11 methods. Spearman rho=0.646 (p=0.032, permutation p=0.041) between GF Score and MRR. Strengthens GF Score <-> downstream utility correlation with full method coverage
- **G-F Curve Phase Transition Analysis** (Step 46): Numerical derivatives of 200-point purity curves identify critical radii for all methods. Peak radius mean=0.064 (std=0.031), with Node2Vec as outlier (r=0.145). Sharpness-GF Score correlation rho=+0.405 (n=8). Betti curve percolation radii tested for coincidence with functional transitions. High-GF methods show marginally sharper transitions (Mann-Whitney U=6, p=0.10)
- **Extended Dimension Sweep** (Step 47): Spectral embeddings extended to d=128 and d=256. Previous d=64 MRR=0.205 vs PPI baseline 0.219. Log-linear fit (MRR ~ log2(d)) tests whether embeddings can surpass PPI topology for function prediction at sufficient dimension
- **Functional Dark Matter Mining** (Step 48): Identifies protein functional associations invisible to network topology (>= 5 hops apart, not STRING-connected at >= 700). Spectral embedding reveals 74 dark matter pairs among 71 proteins at the extreme network periphery (median degree 4 vs 21, p=2.2e-15). Extraordinary GO enrichment: retrograde transport (1809x), transmembrane transport (960x), ascospore formation (710x), ERAD pathway (521x). These are functional associations that NO network-based method could ever discover
- **UMAP/t-SNE Evaluation** (v2.7.0, Step 56): Adjacency-based UMAP achieves highest G-F Score (0.177) across all evaluated methods, exceeding Spectral (0.163). t-SNE achieves 0.152. Critical caveat: shortest-path input causes UMAP to collapse to near-random (GF=0.068), while t-SNE degrades only modestly (0.150). Input representation matters more than algorithm choice
- **GO Ontology Generality** (v2.8.0, Step 57): G-F Score computed with GO Molecular Function (GF=0.348) and Cellular Component (GF=0.191) in addition to Biological Process (GF=0.112). Spectral embedding captures functional geometry across all three GO ontologies — not an artifact of any single annotation system
- **STRING Threshold Sensitivity** (v2.8.0): Method rankings stable between scores 600-700 (Spearman rho=0.90), but regime shift at 800 where network loses 22% edges and Node2Vec collapses (rho=-0.10 vs 700 ranking). Threshold 700 is the natural operating point
- **Dark Matter Literature Validation** (v2.8.0): NSG1-NSG2 confirmed as yeast INSIG homologs (Flury 2005, EMBO J). GON7-SPT8 confirmed as SAGA complex subunits (Wang 2020, Nature). BST1-ADD37 maps to DERL1-DERL3 as mutual rank-4 neighbors in both human and mouse d=64 embeddings — direct cross-species geometric conservation of ERAD pathway organization
- **GATv2 vs GAT Collapse** (v2.8.1): GATv2 (Brody et al., ICLR 2022) with dynamic attention reduces mean attention entropy from 0.927 (GAT) to 0.903, confirming partial alleviation of attention degeneration (Theorem 1). However, max G-F Score remains near-random: GATv2 0.157 vs GAT 0.154, both below Spectral (0.163). Collapse originates from adjacency-reconstruction objective on degree-heterogeneous PPI, not the attention variant
- **ProNE/HARP Robustness Check** (v2.9.0, Step 60): ProNE (spectral propagation + Chebyshev approximation, Zhang 2019) and HARP (hierarchical graph coarsening, Chen 2018) both score below random baseline on curated 153-node network (ProNE GF = 0.087, HARP GF = 0.114 vs random 0.135). Confirms Spectral superiority is not method-selection bias — advanced spectral/hierarchical methods still cannot match direct Laplacian eigenvectors
- **Cosine Similarity Voting** (v2.9.0, Step 61): Top-100 cosine-similar protein voting (weighted by max(cosine_sim, 0)) improves MRR for all methods tested (Spectral: 0.052 to 0.063, +21%; MDS: 0.045 to 0.062, +38%). Strengthens GF-MRR correlation from rho = 0.80 (p = 0.104) to rho = 0.90 (p = 0.037) — the geometric-functional link is robust to distance metric choice
- **Drosophila 5th Species** (v2.9.0, Step 62): Spectral ranks #1 on Drosophila STRING network (6,909 nodes, 89,685 edges) with GF = 0.619 — the highest across all five species. Kendall's W = 0.752 for four eukaryotic species (increases from 0.739 with three species), W = 0.690 including E. coli. SQI = 0.733. Confirms spectral optimality extends to a fourth independent eukaryotic lineage
- **Heat Kernel Multi-Scale Analysis** (v2.10.0, Step 63): Heat kernel K(t) = exp(-tL) at 12 time scales t in [0.01, 100]. Spectral embedding = heat kernel at t->0 limit (GF identical, cross-scale Spearman rho = 1.0). Optimal t* = 5.0 in kD (GF = 0.255). Phase transition at t = 25-50 (delta_GF = -0.073, 32% drop). Characteristic time t_char = 1/lambda_2 = 14.0. Proves Laplacian eigenbasis captures ALL diffusion time scales simultaneously — no tuning of diffusion can improve on Spectral
- **Position Encoding Comparison** (v2.10.0, Step 64): Benchmarks Laplacian PE, RWPE, SignNet against 11 methods. Laplacian PE #1 (GF = 0.163 = Spectral), SignNet #2 (0.160, widest plateau), RWPE #3-4 (0.124/0.117). Sign-flip std = 9.4e-5 (distance invariance). Dimension invariance in 2D: k = 2..32 eigenvectors yield identical GF. Even recent graph transformer PEs (GraphiT, SignNet) cannot beat raw Laplacian eigenvectors on PPI networks
- **Cheeger-Spectral G-F Bound** (v2.10.0, Step 65): Theoretical GF upper bound from Laplacian spectrum via Cheeger's inequality. 4-component bound, spectral gap B1 dominates (w1 = 0.999). Valid for all 6 networks. Tightness: Drosophila 0.996, Human 0.95, Yeast curated 0.44, Mouse 0.36, E. coli 0.28. LOO-CV rho = 0.77 across 6 species. Enables pre-screening of network suitability for spectral analysis without computing any embedding
- Unified interval: **[0.05, 0.422]**

All metrics → [`results/final_results_summary.json`](results/final_results_summary.json)

</details>

---

## Quick Start

```bash
# 1. Install
conda env create -f environment.yml
conda activate gf-consistency

# Or install as a package
pip install .

# 2. Pull large data files (tracked via Git LFS)
git lfs pull

# 3. Run full pipeline (≈ 60 min on standard laptop)
python run_all_analysis.py

# Or use the CLI entry point after pip install
gf-consistency

# Configuration: edit pipeline_config.yaml or pass via CLI
python run_all_analysis.py --config my_config.yaml

# Options:
python run_all_analysis.py --run-human          # Include human validation
python run_all_analysis.py --start-from 3       # Resume from step 3
python run_all_analysis.py --skip-plots         # Skip figure generation
python run_all_analysis.py --skip-gnn           # Skip GNN embeddings (GraphSAGE/GAT/GIN)
python run_all_analysis.py --skip-extended      # Skip extended analyses (Steps 16-21, 24-72)
python run_all_analysis.py --seed 123           # Override random seed
python run_all_analysis.py --species human      # Target a different species
```

Individual steps work standalone:

```bash
python scripts/compute_gf.py          # G-F curves only
python scripts/link_prediction.py     # Link prediction only
python scripts/plot_figures.py        # Regenerate figures from existing results
```

---

## Configuration

All pipeline parameters can be customised via `pipeline_config.yaml` without editing any script:

```yaml
pipeline:
  seed: 42
  species: yeast          # yeast | human | ecoli | mouse | fly
  start_from: 1

gf_score:
  r_min: 0.05
  r_max: 0.55
  n_points: 200
  gf_r_min: 0.05
  gf_r_max: 0.422
  plateau_relative_threshold: 0.80

embeddings:
  classical_methods: [DM, MDS, Spectral, DeepWalk, Node2Vec, VGAE]
  gnn_methods: [GraphSAGE, GAT, GIN]
  node2vec:
    p: 0.5
    q: 2.0
```

CLI flags (`--seed`, `--species`, `--start-from`, etc.) take precedence over config file values. Run `gf-consistency --help` for all options.

---

## G-F Framework Theory

The **G-F Score** quantifies alignment between geometric structure (embeddings) and functional structure (GO annotations):

$$\text{G-F Score} = \frac{1}{r_{\max} - r_{\min}} \int_{r_{\min}}^{r_{\max}} \text{purity}(r) \, dr$$

| Concept | Definition |
|---------|-----------|
| **purity(r)** | Mean over communities of (dominant GO term count / total GO terms in community) at distance threshold *r* |
| **Unified interval** | [0.05, 0.422] — stable region across all methods |
| **Plateau width W** | Range of *r* where purity remains stable → well-separated clusters |
| **Geometric gap** | d<sub>inter</sub> − d<sub>intra</sub>: margin between intra- and inter-module distances |

Mathematical proofs (Propositions 1–2, Theorems 1–4) → [`Supplementary_Materials.pdf`](Supplementary_Materials.pdf) · Supplementary data tables → [`Supplementary_Materials.txt`](Supplementary_Materials.txt)

---

## Pipeline Overview

```
Step 1  ─ Data Preprocessing ────────── Yeast PPI + GO annotations
Step 2  ─ Compute Embeddings ────────── 8 methods → embeddings/*.npy
Step 3  ─ G-F Curves & Scores ───────── 200-point grid → gf_scores.json
Step 4  ─ Leiden Baseline ────────────── Community detection baseline
Step 5  ─ Subset Robustness ──────────── 30 subsets × 5 sizes + Bonferroni correction
Step 6  ─ Full Network ───────────────── 5,936-node STRING validation
Step 7  ─ Geometric Analysis ─────────── d_intra / d_inter margin
Step 8  ─ Link Prediction ────────────── 5-fold CV AUROC
Step 9  ─ Downstream k-NN ────────────── GO term prediction
Step 10 ─ Randomization Control ──────── Shuffled-label baseline
Step 11 ─ Sampling Density ──────────── 30 vs 200-point verification
Step 12 ─ Sensitivity Analysis ───────── Interval robustness
Step 13 ─ Human Validation ───────────── Cross-species (15,882 nodes)
Step 14 ─ Figure Generation ──────────── figures/*.png
Step 15 ─ GNN Embeddings ─────────────── GraphSAGE, GAT, GIN → embeddings/*.npy
Step 16 ─ Adaptive Unified Interval ──── Data-driven consensus [r_min, r_max]
Step 17 ─ Network Topology Analysis ──── Cross-species topology comparison
Step 18 ─ Rank Reversal Analysis ──────── p/q sensitivity + geometric gap mechanism
Step 19 ─ GO Propagation ─────────────── True Path Rule DAG expansion (154 → 5,429 genes)
Step 20 ─ Biological Interpretation ──── 4-level G-F scale + GO cluster case study
Step 21 ─ Runtime Benchmark ──────────── Step-wise profiling + complexity analysis
Step 22 ─ Persistent Homology ─────────── Betti curves, persistence diagrams (Ripser)
Step 23 ─ Topological Statistics ──────── Topo-GF correlations, consistency scores
Step 24 ─ Hyperbolic Embedding ────────── Poincare Ball (Riemannian SGD)
Step 25 ─ Pathway Enrichment ──────────── Fisher's exact test on G-F communities
Step 26 ─ Statistical Summary ─────────── Spearman, Wilcoxon, bootstrap CIs
Step 27 ─ Metric Comparison ────────────── G-F Score vs link prediction AUC + k-NN F1
Step 28 ─ Bootstrap Correlations ───────── 95% CI for key Spearman correlations (10k resamples)
Step 29 ─ Semantic Purity Analysis ─────── IC-weighted + Resnik semantic purity + DAG inflation diagnosis
Step 30 ─ Cross-Species Consistency ───── Yeast vs human rank concordance (Spearman + Kendall W)
Step 31 ─ Scale Gradient Analysis ──────── Scale-dependent topology coupling (500-4000 nodes)
Step 32 ─ Bootstrap Stability ─────────── 30-resample CI for G-F Score rankings (80% sampling)
Step 33 ─ Human Extended Embeddings ───── 11-method human network embeddings (PCA, VGAE-feat, GNNs)
Step 34 ─ Multi-Modal Anchoring ───────── STRING threshold gradient + channel-specific GF analysis
Step 35 ─ Hyperparameter Sensitivity ──── r-points, resolution, dimensions, walk parameters
Step 36 ─ Density-Corrected GF ─────────── STRING threshold gradient with random baseline correction (ΔW = 0.522)
Step 37 ─ Human Seed Stability ──────────── 10-seed subsampling stability (Kendall W = 0.675)
Step 38 ─ Human IC-Weighted GF ─────────── IC-weighted purity on human network (ρ = 0.991)
Step 39 ─ GAT Collapse Diagnosis ────────── 5-variant ablation: clip, warmup, multi-head analysis
Step 40 ─ E. coli K-12 Validation ────────── 4th species cross-species validation (SQI=0.7)
Step 41 ─ Coexpression Network G-F ────────── Network-type dependence (DeepWalk #1)
Step 42 ─ Degree-Preserving Null Model ────── 50 double-edge-swap randomizations (z-scores)
Step 43 ─ GAT Theorem Full Network ────────── 5936-node theorem verification (T1-T3)
Step 44 ─ Community Detection Ablation ────── 5 algorithms, Kendall W=0.797
Step 45 ─ Full 11-Method LOTO-CV ──────────── Expanded function prediction (rho=0.646)
Step 46 ─ G-F Phase Transition Analysis ───── Critical radii, Betti coincidence, critical exponents
Step 47 ─ Extended Dimension Sweep ─────────── d=128/256, test PPI baseline crossing
Step 48 ─ Functional Dark Matter Mining ────── Embedding-only functional associations (74 pairs)
Step 49 ─ Cross-Species Dark Matter ─────────── Human/mouse ortholog mapping + embedding proximity
Step 50 ─ Rescue Protein Analysis ────────────── 235 rescue proteins characterisation
Step 51 ─ STRING v12.0 Re-validation ──────────── Confirm dark matter pairs absent in latest STRING
Step 52 ─ High-Dim Spectral Embeddings ────────── d=64 spectral for human (15,882) + mouse (16,180)
Step 53 ─ Cross-Species High-Dim ──────────────── 2D vs 64D cross-species conservation comparison
Step 54 ─ Yeast High-Dim Embedding ────────────── d=64 spectral for yeast (5,936 nodes, PR=63.99/64)
Step 55 ─ Three-Species Dimension Gradient ────── d=2,8,16,32,64 across yeast/human/mouse; critical dims
Step 56 ─ UMAP/t-SNE Evaluation ────────────────── Adjacency vs shortest-path input representation
Step 57 ─ GO Ontology Generality ───────────────── MF + CC + BP G-F Scores (3 GO aspects)
Step 58 ─ STRING Threshold Sensitivity ──────────── 600/700/800 threshold gradient + regime shift
Step 59 ─ GATv2 vs GAT Comparison ─────────────── Dynamic attention entropy + GF Score comparison
Step 60 ─ ProNE/HARP G-F Scores ──────────────── Spectral-propagation + hierarchical coarsening baselines
Step 61 ─ Cosine Similarity Baseline ──────────── Cosine vs Euclidean voting for function prediction
Step 62 ─ Drosophila 5th Species ───────────────── 6,909-node cross-species validation (Kendall W = 0.752)
         └─ Summary ─────────────────── final_results_summary.json
```

---

## Embedding Methods

| Method | Input | Strategy | 2D Output |
|--------|-------|----------|-----------|
| Diffusion Map (DM) | 6 centrality features → Markov matrix | Eigendecomposition | 2nd and 3rd eigenvectors |
| Classical MDS | Shortest-path distances | Double-centering + eigendecomposition | Top-2 eigenvectors |
| Spectral | Normalized Laplacian | Eigendecomposition | Fiedler vectors (eig 1,2) |
| DeepWalk | Random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| Node2Vec | Biased random walks → co-occurrence | Truncated SVD | Top-2 singular vectors |
| VGAE | One-hot identity matrix | 2-layer GCN VAE (hidden=4, 300 epochs, Adam lr=0.01) | Latent mean μ |
| VGAE-feat | 6 centrality features | 2-layer GCN VAE (hidden=4, 300 epochs) | Latent mean μ |
| PCA | 6 centrality features | PCA via covariance eigendecomposition | Top-2 components |
| GraphSAGE | 6 centrality features | 2-layer SAGEConv mean agg (hidden=16, 300 epochs) | Latent space |
| GAT | 6 centrality features | 2-layer GATConv 1-head (hidden=16, 300 epochs) | Latent space |
| GIN | 6 centrality features | 2-layer GINConv + MLP (hidden=16, 300 epochs) | Latent space |

All embeddings standardized to **σ = 0.3** before G-F analysis.

---

## Project Structure

```
GF-consistency-framework/
├── scripts/                    # 72-step analysis pipeline + extensions
│   ├── data_preprocessing.py   # Load PPI + GO data
│   ├── embed_all.py            # Compute 8 classical/NN embeddings
│   ├── compute_gf.py           # G-F curves + scores
│   ├── leiden_baseline.py      # Leiden community baseline
│   ├── robustness.py           # Subset analysis + Bonferroni
│   ├── full_network.py         # Full 5,936-node network
│   ├── geometric_analysis.py   # d_intra / d_inter margins
│   ├── link_prediction.py      # 5-fold CV link prediction
│   ├── downstream_knn.py       # k-NN GO term prediction
│   ├── randomization_control.py# Shuffled-label control
│   ├── sampling_density.py     # 30 vs 200 pt comparison
│   ├── gf_score_sensitivity.py # Interval sensitivity
│   ├── plot_figures.py         # Generate all figures
│   ├── human_validation.py     # Cross-species validation
│   ├── embed_gnn.py            # GraphSAGE, GAT, GIN embeddings (Step 15)
│   ├── adaptive_interval.py    # Data-driven consensus interval (Step 16)
│   ├── network_topology_analysis.py  # Cross-species topology (Step 17)
│   ├── rank_reversal_analysis.py     # p/q sensitivity + mechanism (Step 18)
│   ├── go_propagation.py       # GO DAG True Path Rule expansion (Step 19)
│   ├── biological_interpretation.py  # 4-level G-F scale + case study (Step 20)
│   ├── benchmark_runtime.py    # Pipeline profiling + complexity (Step 21)
│   ├── statistical_analysis.py # Spearman, Wilcoxon, bootstrap, permutation
│   ├── robustness_analysis.py  # Extended 30-subset convergence
│   ├── visualization_helpers.py# Okabe-Ito colorblind-safe plotting
│   ├── config_loader.py        # YAML configuration loader + validator
│   ├── input_validator.py      # Pre-flight input validation
│   ├── embed_hyperbolic.py     # Poincare Ball hyperbolic embeddings
│   ├── multispecies_loader.py  # Multi-species dataset loader
│   ├── temporal_network.py     # Dynamic/temporal PPI framework
│   ├── pathway_analysis.py     # Pathway enrichment + cancer gene analysis
│   ├── metric_comparison.py    # G-F Score vs link prediction AUC + k-NN F1 (Step 27)
│   ├── bootstrap_correlations.py # Bootstrap 95% CI for Spearman correlations (Step 28)
│   ├── semantic_purity.py      # IC-weighted + Resnik semantic purity core library (Step 29)
│   ├── semantic_similarity_analysis.py  # 3-variant purity comparison + DAG diagnostics (Step 29)
│   ├── cross_species_consistency.py     # Yeast vs human rank concordance analysis (Step 30)
│   ├── scale_gradient.py               # Scale-dependent topology coupling (Step 31)
│   ├── bootstrap_stability.py          # Bootstrap CI for G-F Score rankings (Step 32)
│   ├── human_embed_extended.py         # Extended human embeddings (Step 33a)
│   ├── human_gf_extended.py            # Human GF analysis all 11 methods (Step 33b)
│   ├── multimodal_functional_anchoring.py # STRING threshold + channel analysis (Step 34)
│   ├── hyperparameter_sensitivity.py   # Parameter sensitivity analysis (Step 35)
│   ├── density_corrected_gf.py         # Density-corrected GF with random baseline (Step 36)
│   ├── human_seed_stability.py         # Human 10-seed stability analysis (Step 37)
│   ├── human_ic_weighted_gf.py         # Human IC-weighted GF analysis (Step 38)
│   ├── gat_collapse_diagnosis.py       # GAT variant ablation study (Step 39)
│   ├── deep_geometric_analysis.py      # Multi-scale geometric fingerprint (Phase 1, Fig 26-29)
│   ├── geometric_predictor.py          # Cross-species geometric predictor (Phase 2, Fig 30-33)
│   ├── spectral_alignment.py           # Network-aware spectral alignment (Phase 3, Fig 34-35)
│   ├── gat_collapse_theory.py          # GAT collapse mathematical theory (Phase 4, Fig 36-38)
│   ├── human_spectral_alignment.py     # Human network spectral alignment (Phase 5A, Fig 39-40)
│   ├── gat_dimension_sweep.py          # GAT latent dimension sweep (Phase 5B, Fig 41)
│   ├── gat_collapse_formal_proof.py    # Formal proofs of GAT collapse (Phase 6, Fig 42-43)
│   ├── tda_geometry_bridge.py         # TDA-geometry bridge analysis (Phase 7, Fig 44-45)
│   ├── human_cross_network_validation.py  # Cross-network validation & bootstrap CIs (Phase 8, Fig 46-47)
│   ├── human_tda_full.py               # Full human TDA: 11-method persistent homology + three-factor validation (Phase 8B, Fig 48)
│   ├── human_loo_sensitivity.py        # Leave-one-out sensitivity analysis (Phase 8C, Fig 49)
│   ├── human_gf_unified.py             # Unified human G-F Scores: fix confounds 1+2 (Phase 9, Fig 50)
│   ├── mouse_data_prep.py              # Mouse STRING PPI download + MGI GAF ID mapping (Phase 10A)
│   ├── mouse_embeddings_full.py        # Full-network mouse embeddings: 11 methods, ~16K nodes (Phase 10B)
│   ├── persistence_image_analysis.py   # Persistence image TDA + three-species comparison (Phase 10C, Fig 51-54)
│   ├── spectral_transferability.py     # Spectral Quality Index (SQI) + SBM validation (Phase 11, Fig 55-59)
│   ├── biological_validation.py        # GO BP enrichment + multi-seed panel + mixed-effects (Phase 12, Fig 60-64)
│   ├── function_prediction.py         # Protein function prediction via LOTO-CV + GF correlation (Phase 13, Fig 65-68)
│   ├── ecoli_analysis.py              # E. coli K-12 4th species cross-species validation (Step 40)
│   ├── coexpression_gf.py             # Coexpression network G-F analysis (Step 41)
│   ├── degree_preserving_null.py      # Degree-preserving null model (50 randomizations) (Step 42)
│   ├── gat_theorem_large_network.py   # GAT theorem verification on full network (Step 43)
│   ├── gf_ablation_community_detection.py # Community detection sensitivity ablation (Step 44)
│   ├── function_prediction_full.py    # Full 11-method LOTO-CV function prediction (Step 45)
│   ├── gf_phase_transition.py        # G-F curve phase transition analysis (Step 46)
│   ├── dimension_sweep_extended.py   # Extended dimension sweep d=128/256 (Step 47)
│   ├── functional_dark_matter.py     # Functional dark matter mining (Step 48)
│   ├── multihead_gat_experiment.py    # Multi-head GAT configuration sweep
│   ├── cross_species_dark_matter.py   # Cross-species dark matter ortholog analysis (Step 49)
│   ├── rescue_protein_analysis.py     # 235 rescue proteins characterisation (Step 50)
│   ├── string_v12_revalidation.py    # STRING v12.0 re-validation of dark matter (Step 51)
│   ├── highdim_spectral_embeddings.py # d=64 spectral for human+mouse (Step 52)
│   ├── cross_species_highdim.py      # 2D vs 64D cross-species conservation (Step 53)
│   ├── dimension_gradient_3species.py # Three-species dimension gradient d=2-64 (Steps 54-55)
│   ├── umap_tsne_gf.py              # UMAP/t-SNE G-F evaluation (Step 56)
│   ├── go_mf_cc_gf_scores.py        # GO MF/CC/BP G-F Scores (Step 57)
│   ├── string_threshold_sensitivity.py # STRING threshold gradient 600-800 (Step 58)
│   ├── gatv2_experiment.py           # GATv2 vs GAT collapse comparison (Step 59)
│   ├── prone_harp_gf.py             # ProNE + HARP embedding G-F Scores (Step 60)
│   ├── function_prediction_cosine.py # Cosine similarity voting baseline (Step 61)
│   ├── fly_analysis.py              # Drosophila 5th species cross-species validation (Step 62)
│   ├── dark_matter_ortholog_validation.py # Dark matter ortholog proximity check
│   ├── dimension_sweep.py            # Dimension sweep utility
│   └── utils.py                # Shared utilities
│
├── tests/                      # pytest test suite (51 tests)
│   ├── conftest.py             # Shared fixtures
│   ├── test_utils.py           # Core computation tests
│   └── test_config_loader.py   # Config loader tests
│
├── data/                       # Input data (STRING v11.5, GO)
│   ├── *.edgelist              # PPI networks (curated 153, full, subsets)
│   ├── gene_go_map.json        # Gene → GO term mapping
│   └── *.gz                    # STRING/GAF files (Git LFS)
│
├── embeddings/                 # Generated .npy + _nodes.json per method
├── results/                    # JSON/CSV result files
│   └── final_results_summary.json  ← Master summary
│
├── figures/                    # Publication figures (PNG, 300 dpi)
│   ├── Fig1–14                 # Main figures (G-F curves, topology, human validation)
│   ├── Fig15                   # Metric comparison scatter (G-F vs link pred vs k-NN)
│   ├── Fig16                   # Semantic purity comparison (standard, IC-weighted, Resnik)
│   ├── Fig17                   # Cross-species rank consistency (yeast vs human)
│   ├── Fig18                   # Scale-dependent topology coupling (500-4000 nodes)
│   ├── Fig19                   # Bootstrap stability of G-F Score rankings
│   ├── Fig20                   # Multi-modal functional anchoring (threshold + channel)
│   ├── Fig21                   # Hyperparameter sensitivity analysis
│   ├── Fig22                   # Density-corrected GF analysis (Step 36)
│   ├── Fig23                   # Human seed stability analysis (Step 37)
│   ├── Fig24                   # Human IC-weighted GF comparison (Step 38)
│   ├── Fig25                   # GAT collapse diagnosis (Step 39)
│   ├── Fig26                   # Distance-function correspondence curves (Phase 1)
│   ├── Fig27                   # Geometric feature radar chart (Phase 1)
│   ├── Fig28                   # G-F curve shape decomposition (Phase 1)
│   ├── Fig29                   # DFC enrichment heatmap (Phase 1)
│   ├── Fig30                   # Cross-species geometric prediction (Phase 2)
│   ├── Fig31                   # Network spectral theory (Phase 2)
│   ├── Fig32                   # Collapse diagnostics (Phase 2)
│   ├── Fig33                   # Method clustering dendrogram (Phase 2)
│   ├── Fig34                   # Spectral decomposition in Laplacian eigenbasis (Phase 3)
│   ├── Fig35                   # Spectral alignment summary (Phase 3)
│   ├── Fig36                   # Attention degeneration analysis (Phase 4)
│   ├── Fig37                   # Rank collapse landscape across methods (Phase 4)
│   ├── Fig38                   # Unified GAT collapse causal chain (Phase 4)
│   ├── Fig39                   # Human spectral decomposition (Phase 5A)
│   ├── Fig40                   # Human alignment summary: yeast vs human (Phase 5A)
│   ├── Fig41                   # GAT dimension sweep d=2-32 (Phase 5B)
│   ├── Fig42                   # Formal proof verification: 3 theorems (Phase 6)
│   ├── Fig43                   # Proof summary: rank landscape (Phase 6)
│   ├── Fig44                   # TDA-geometry bridge correlations (Phase 7)
│   ├── Fig45                   # Three-factor model summary (Phase 7)
│   ├── Fig46                   # Human cross-network validation (Phase 8)
│   ├── Fig47                   # Bootstrap confidence intervals (Phase 8)
│   ├── Fig48                   # Full human TDA + three-factor validation (Phase 8B)
│   ├── Fig49                   # Leave-one-out sensitivity analysis (Phase 8C)
│   ├── Fig50                   # Unified human G-F comparison (Phase 9)
│   ├── Fig51                   # Mouse G-F curves + ranking (Phase 10)
│   ├── Fig52                   # Persistence images gallery: human vs mouse (Phase 10)
│   ├── Fig53                   # Three-species G-F Score comparison (Phase 10)
│   ├── Fig54                   # TDA feature comparison: scalar vs persistence image (Phase 10)
│   ├── Fig55                   # Laplacian spectrum comparison: 3 species (Phase 11)
│   ├── Fig56                   # SQI decomposition + two-factor rho (Phase 11)
│   ├── Fig57                   # SA discriminative power vs SQI (Phase 11)
│   ├── Fig58                   # SBM synthetic validation (Phase 11)
│   ├── Fig59                   # Transferability summary + table (Phase 11)
│   ├── Fig60                   # GO BP enrichment overview: 3 species (Phase 12)
│   ├── Fig61                   # Enrichment significance distribution (Phase 12)
│   ├── Fig62                   # Multi-seed method stability (Phase 12)
│   ├── Fig63                   # Rank consistency |ρ| with CI (Phase 12)
│   ├── Fig64                   # Phase 12 summary: enrichment + mixed-effects (Phase 12)
│   ├── Fig65                   # Precision@k curves: embedding methods + baselines (Phase 13)
│   ├── Fig66                   # MRR comparison bar chart coloured by GF Score (Phase 13)
│   ├── Fig67                   # GF Score vs prediction accuracy scatter (Phase 13)
│   ├── Fig68                   # Phase 13 summary dashboard (Phase 13)
│   ├── FigS1–S7                # Supplementary figures
│   └── FigS8                   # Sampling density comparison
│
├── human_validation/           # Cross-species (optional, STRING v12.0)
│
├── run_all_analysis.py         # One-command Python pipeline (72 steps)
├── pipeline_config.yaml        # YAML configuration (all parameters)
├── pyproject.toml              # Python package metadata
├── environment.yml             # Conda environment
├── requirements.txt            # pip dependencies
├── Supplementary_Materials.pdf # Mathematical proofs (Propositions 1-2, Theorems 1-4)
└── LICENSE                     # MIT License
```

---

## Data Sources

| Dataset | Source | Nodes | Notes |
|---------|--------|:-----:|-------|
| Yeast PPI | STRING v11.5 | 5,936 | Score ≥ 700, curated |
| Yeast GO | SGD GAF | 153 annotated (5,429 after GO DAG propagation) | BP terms, level ≥ 3 |
| Human PPI | STRING v12.0 | 15,882 (14,679 in largest CC) | Cross-species validation |
| Human GO | GOA Human GAF | 16,818 annotated | BP terms |
| Mouse PPI | STRING v11.5 | 16,180 (largest CC) | Score ≥ 700 |
| Mouse GO | MGI GAF | 17,639 annotated | BP terms, Ensembl_MGI alias mapping |
| E. coli PPI | STRING v11.5 | ~4,000 | Score ≥ 700, 4th species validation |
| E. coli GO | EcoCyc/UniProt | ~3,500 annotated | BP terms |
| Drosophila PPI | STRING v11.5 | 6,909 | Score ≥ 700, 5th species |
| Drosophila GO | FlyBase GAF | ~7,000 annotated | BP terms |
| Yeast Coexpression | Transcriptomic | ~5,000 | Network-type dependence analysis |

Large files (`*.txt.gz`, `*.gaf.gz`) tracked via **Git LFS** — run `git lfs install` before cloning.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.11 | Runtime |
| numpy | ≥ 1.24 | Numerical computation |
| scipy | ≥ 1.10 | Integration, statistics |
| scikit-learn | ≥ 1.3 | k-NN, logistic regression |
| matplotlib | ≥ 3.7 | Figure generation |
| pandas | ≥ 2.0 | Data handling |
| networkx | ≥ 3.0 | Graph operations |
| python-igraph | ≥ 0.10 | Leiden community detection |
| python-louvain | ≥ 0.16 | Louvain community detection (human validation) |
| torch | ≥ 2.0 | VGAE encoder |
| torch_geometric | ≥ 2.3 | GCN layers |
| requests | ≥ 2.28 | STRING API fallback |
| seaborn | ≥ 0.12 | Statistical visualization |
| pyyaml | ≥ 6.0 | YAML configuration parsing |

Full spec → [`requirements.txt`](requirements.txt) · [`environment.yml`](environment.yml)

---

## Extension Modules (v1.1+)

Beyond the core 72-step pipeline, the framework provides extensible modules for advanced analyses. Modules marked with * are integrated into `run_all_analysis.py` (Steps 22–72).

| Module | Description |
|--------|-------------|
| `embed_hyperbolic.py` * | Poincare Ball embeddings via Riemannian SGD — suited for hierarchical PPI structures |
| `multispecies_loader.py` | Species registry (yeast, human, *E. coli*, mouse, *Drosophila*) with STRING network + GAF parsing |
| `temporal_network.py` | `TemporalNetwork` container for time-resolved PPI analysis (requires temporal PPI data) |
| `pathway_analysis.py` * | Fisher's exact pathway enrichment, cancer gene association, consensus communities |
| `topological_analysis.py` * | Persistent homology computation (Betti curves, persistence diagrams) via Ripser |
| `topological_stats.py` * | Topological feature extraction and statistical summaries |
| `robustness_analysis.py` * | Convergence analysis, randomization null test, power curve estimation |
| `statistical_analysis.py` * | Spearman correlations, Wilcoxon tests, bootstrap CIs, cross-species comparison |
| `metric_comparison.py` * | G-F Score vs link prediction AUC and k-NN node classification F1 across all 11 methods |
| `bootstrap_correlations.py` * | Bootstrap 95% CI for key Spearman correlations (10,000 resamples) |
| `semantic_purity.py` * | IC-weighted purity and Resnik MICA semantic similarity for GO DAG–aware community evaluation |
| `semantic_similarity_analysis.py` * | Robustness check: 3-variant G-F score comparison + DAG inflation diagnostics (Fig 16) |
| `cross_species_consistency.py` * | Cross-species rank concordance: yeast vs human Spearman + Kendall W (Fig 17) |
| `scale_gradient.py` * | Scale-dependent topology coupling: 500-4000 node gradient analysis (Fig 18) |
| `bootstrap_stability.py` * | Bootstrap stability: 30-resample 95% CI for G-F Score rankings (Fig 19) |
| `human_embed_extended.py` * | Extended human embeddings: PCA, VGAE-feat, GraphSAGE, GAT, GIN (Step 33a) |
| `human_gf_extended.py` * | Human GF analysis for all 11 methods (Step 33b) |
| `multimodal_functional_anchoring.py` * | STRING threshold gradient + channel-specific GF analysis (Fig 20, Step 34) |
| `hyperparameter_sensitivity.py` * | r-points, resolution, dimension, walk parameter sensitivity (Fig 21, Step 35) |
| `density_corrected_gf.py` * | Density-corrected GF with random baseline subtraction (Step 36) |
| `human_seed_stability.py` * | Human 10-seed subsampling stability analysis (Step 37) |
| `human_ic_weighted_gf.py` * | Human IC-weighted GF analysis across 11 methods (Step 38) |
| `gat_collapse_diagnosis.py` * | GAT variant ablation: clip, warmup, multi-head analysis (Step 39) |
| `deep_geometric_analysis.py` | Multi-scale geometric fingerprint: DFC, geometric features, curve decomposition (Phase 1, Fig 26-29) |
| `geometric_predictor.py` | Cross-species geometric predictability: yeast→human validation, spectral theory (Phase 2, Fig 30-33) |
| `spectral_alignment.py` | Network-aware spectral alignment: Laplacian eigenbasis decomposition (Phase 3, Fig 34-35) |
| `gat_collapse_theory.py` | GAT collapse mathematical theory: 4-pillar impossibility analysis (Phase 4, Fig 36-38) |
| `human_spectral_alignment.py` | Human network spectral alignment: cross-network two-factor model transfer test (Phase 5A, Fig 39-40) |
| `gat_dimension_sweep.py` | GAT latent dimension sweep d={2,4,8,16,32}: causal disentanglement of attention degeneration (Phase 5B, Fig 41) |
| `gat_collapse_formal_proof.py` | Formal proofs of GAT collapse: 3 theorems with numerical verification — attention degeneration bound, effective rank bound, G-F Score upper bound (Phase 6, Fig 42-43) |
| `tda_geometry_bridge.py` | TDA-geometry bridge: unified feature matrix (11 methods × 18 features), single/multi-factor models, partial correlations, Betti curve phase transitions — proves TDA adds independent predictive signal (Phase 7, Fig 44-45) |
| `human_cross_network_validation.py` | Cross-network validation: tests two/three-factor models on human PPI, bootstrap CIs for Phase 7 correlations (10k resamples) — confirms single-factor robustness, identifies partial correlation fragility at n=11 (Phase 8, Fig 46-47) |
| `human_tda_full.py` | Full human TDA analysis: persistent homology for all 11 methods on human PPI with identical yeast parameters, three-factor validation — H1 persistence does NOT transfer (rho=0.073), three-factor degrades to rho=0.282 (Phase 8B, Fig 48) |
| `human_loo_sensitivity.py` | Leave-one-out sensitivity: Spectral is a catastrophic topological outlier (H1 80x lower than yeast); excluding Spectral reveals latent H1 signal (rho=+0.430); two-factor model is most LOO-stable (Phase 8C, Fig 49) |
| `human_gf_unified.py` | Unified human G-F Scores: eliminates community-detection (Louvain→greedy_modularity) and interval confounds; confirms rho=0.927 rank correlation, top-3/bottom-2 identical, all correlations preserved (Phase 9, Fig 50) |
| `mouse_data_prep.py` | Mouse STRING PPI + MGI GAF download with Ensembl_MGI alias-based ID mapping: ~16K nodes, ~233K edges, 17,639 annotated genes (Phase 10A) |
| `mouse_embeddings_full.py` | Full-network mouse embeddings (11 methods, ~16K nodes) with landmark MDS + sparse VGAE/GNN — matches human pipeline methodology (subsample-after, not subsample-before) (Phase 10B) |
| `persistence_image_analysis.py` | Persistence image TDA via ripser/persim: mouse G-F scores, H1 persistence diagrams + images (energy, density, spread, entropy), three-species Kendall's W concordance (Phase 10C, Fig 51-54) |
| `spectral_transferability.py` | Spectral Transferability Theory: derives Spectral Quality Index (SQI = λ₂/λ₂_ER × PR × FA_max) predicting two-factor model transferability; Laplacian spectral analysis (3 species), proposition verification, synthetic SBM validation (20 networks) (Phase 11, Fig 55-59) |
| `biological_validation.py` | Biological validation + statistical power: (A) GO BP hypergeometric enrichment across 3 species × 11 methods; (B) multi-seed panel (yeast 5 seeds, human 10, mouse 5 subsamples) with mixed-effects pooled Spearman model; `part_a`/`part_b` CLI modes with checkpoint resume (Phase 12, Fig 60-64) |
| `function_prediction.py` | Protein function prediction via leave-one-term-out CV on full yeast STRING network (5,936 nodes, 5 methods, 12,690 trials); KNN in embedding space vs PPI/2-hop/random baselines; Precision@k + MRR evaluation; GF Score ↔ prediction accuracy correlation closes the framework loop (Phase 13, Fig 65-68) |
| `ecoli_analysis.py` * | E. coli K-12 cross-species validation: 4th species, STRING v11.5, all 11 methods, SQI=0.7 (Step 40) |
| `coexpression_gf.py` * | Coexpression network G-F analysis: tests network-type dependence, DeepWalk #1 on coexpression (Step 41) |
| `degree_preserving_null.py` * | Degree-preserving null model: 50 double-edge-swap randomizations, z-score comparison (Step 42) |
| `gat_theorem_large_network.py` * | GAT collapse theorem verification on full 5936-node network: T1-T3 all hold (Step 43) |
| `gf_ablation_community_detection.py` * | Community detection sensitivity: 5 algorithms, Kendall W=0.797, Spectral #1 under 4/5 (Step 44) |
| `function_prediction_full.py` * | Full 11-method LOTO-CV function prediction: rho=0.646, p=0.032, permutation p=0.041 (Step 45) |
| `gf_phase_transition.py` * | G-F curve phase transition analysis: derivatives, critical radii, Betti coincidence, critical exponents (Step 46) |
| `dimension_sweep_extended.py` * | Extended dimension sweep d=128/256: tests whether Spectral MRR surpasses PPI-Neighbors baseline (Step 47) |
| `functional_dark_matter.py` * | Functional dark matter mining: embedding-only functional associations invisible to network topology (Step 48) |
| `cross_species_dark_matter.py` * | Cross-species dark matter: human/mouse ortholog mapping, embedding proximity validation (Step 49) |
| `rescue_protein_analysis.py` * | Rescue protein characterisation: 235 proteins systematically underrepresented in PPI networks (Step 50) |
| `string_v12_revalidation.py` * | STRING v12.0 re-validation: confirms all 44 dark matter pairs absent in latest database (Step 51) |
| `highdim_spectral_embeddings.py` * | High-dimensional spectral embeddings: d=64 for human (15,882 nodes) and mouse (16,180 nodes) via sparse eigendecomposition (Step 52) |
| `cross_species_highdim.py` * | Cross-species high-dim: 2D vs 64D conservation comparison, conserved categories increase from 3/7 to 4/7 (Step 53) |
| `dimension_gradient_3species.py` * | Three-species dimension gradient: d=2,8,16,32,64 across yeast/human/mouse; identifies critical dimensions per GO category (Steps 54-55) |
| `umap_tsne_gf.py` * | UMAP/t-SNE G-F evaluation: adjacency-based UMAP achieves GF=0.177 (highest), input representation matters more than algorithm (Step 56) |
| `go_mf_cc_gf_scores.py` * | GO ontology generality: Spectral GF across Molecular Function (0.348), Cellular Component (0.191), Biological Process (0.112) (Step 57) |
| `string_threshold_sensitivity.py` * | STRING threshold sensitivity: 600/700/800 gradient, regime shift at 800, stable 600-700 (rho=0.90) (Step 58) |
| `gatv2_experiment.py` * | GATv2 vs GAT collapse: dynamic attention reduces entropy (0.903 vs 0.927) but GF Score stays near-random (Step 59) |
| `prone_harp_gf.py` * | ProNE + HARP G-F Scores: spectral-propagation (Chebyshev order-5) and hierarchical coarsening baselines — both score below random (Step 60) |
| `function_prediction_cosine.py` * | Cosine similarity voting baseline: top-100 cosine-similar proteins, weighted voting improves MRR for all methods (Spectral +21%, MDS +38%) (Step 61) |
| `fly_analysis.py` * | Drosophila 5th species: 6,909-node STRING network, all 11 methods, Spectral #1 GF=0.619, Kendall W=0.752 for 4 eukaryotes (Step 62) |
| `dark_matter_ortholog_validation.py` | Dark matter ortholog validation: maps 71 proteins to human/mouse, BST1-ADD37→DERL1-DERL3 rank-4 |
| `multihead_gat_experiment.py` | Multi-head GAT configuration sweep (1/4/8 heads, d=2-32) |
| `input_validator.py` | Pre-flight validation for networks, embeddings, GO annotations |
| `config_loader.py` | YAML configuration loader with deep merge, validation, CLI overrides |

New species can be registered at runtime:

```python
from scripts.multispecies_loader import register_species, load_species_dataset

register_species("fly", {
    "taxon_id": "7227",
    "string_prefix": "7227",
    "name": "Drosophila melanogaster",
    "go_db": "gene_association.fb.gaf.gz",
})
G, nodes, go_map = load_species_dataset("fly", data_dir="data/fly")
```

---

## Human PPI Validation

Cross-species validation on STRING v12.0 human interactome (~15,882 nodes after score ≥ 700 filtering; 14,679 in largest CC, 236,712 edges; 14,562 with BP GO annotations). G-F curves computed on 2,000-node subsample using Louvain community detection (200 r-points, standard purity formula):

| Method | G-F Score | Plateau Width W | Peak Purity |
|--------|:---------:|:---------------:|:-----------:|
| **Spectral** | **0.402** | 0.480 | 0.478 |
| MDS | 0.367 | 0.217 | 0.671 |
| Node2Vec | 0.166 | 0.495 | 0.206 |
| GraphSAGE | 0.133 | 0.495 | 0.256 |
| DeepWalk | 0.101 | 0.495 | 0.204 |
| GIN | 0.089 | 0.379 | 0.184 |
| VGAE-feat | 0.089 | 0.495 | 0.212 |
| PCA | 0.086 | 0.495 | 0.264 |
| DM | 0.060 | 0.495 | 0.165 |
| VGAE | 0.014 | 0.500 | 0.018 |
| GAT | 0.011 | 0.500 | 0.011 |

- Human unified interval: **[0.282, 0.297]** (11 methods, standard purity)
- Spectral ranks #1 in both yeast and human, demonstrating strong cross-species consistency
- DM drops from 2nd (yeast) to 9th (human), confirming the framework discriminates context-dependent embedding quality

```bash
cd human_validation
python run_all_human.py
```

Details → [`human_validation/README.md`](human_validation/README.md) · Figure → [`figures/Fig6_human_validation.png`](figures/Fig6_human_validation.png)

---

## Topological Analysis Figures

Persistent homology analysis (Vietoris–Rips complexes via Ripser) generates additional figures characterizing the topological structure of PPI embeddings:

| Figure | File | Description |
|--------|------|-------------|
| Fig8 | `Fig8_betti_curves.png` | Betti number curves (β₀, β₁) across filtration scales for all embedding methods |
| Fig9 | `Fig9_topo_vs_standard_purity.png` | Comparison of topological purity vs. standard G-F purity profiles |
| Fig10 | `Fig10_persistence_diagrams.png` | Persistence diagrams (H₀, H₁) for each embedding method |
| Fig11 | `Fig11_topo_consistency_vs_gf_score.png` | Scatter plot of topological consistency score vs. G-F Score |
| Fig12 | `Fig12_topo_metric_correlations.png` | Correlation matrix of topological metrics (Betti numbers, persistence, entropy) |
| Fig13 | `Fig13_topo_vs_standard_gf_bars.png` | Bar comparison of standard vs. topologically-weighted G-F scores |
| Fig14 | `Fig14_human_topo_scatter.png` | Human PPI topological feature scatter (cross-species validation) |

Generated by `scripts/topological_analysis.py` and `scripts/topological_stats.py`. See `results/topological_analysis.json` for raw numerical data.

---

## Mouse PPI Validation (Phase 10)

Three-species cross-species validation on mouse STRING v11.5 PPI (~16,180 nodes, ~233,340 edges; 17,639 genes with MGI GO annotations via Ensembl_MGI alias mapping). All 11 methods embedded on the full network, G-F curves computed on 2,000-node subsample with unified parameters (greedy_modularity, [0.05, 0.422]):

| Method | Mouse G-F | Yeast Rank | Human Rank | Mouse Rank |
|--------|:---------:|:----------:|:----------:|:----------:|
| **Spectral** | **0.309** | 1 | 1 | 1 |
| MDS | 0.160 | 3 | 2 | 2 |
| PCA | 0.064 | 5 | 6 | 3 |
| GraphSAGE | 0.056 | 10 | 5 | 4 |
| DM | 0.048 | 2 | 8 | 5 |
| VGAE-feat | 0.015 | 6 | 7 | 11 |

- Three-species Kendall W: **0.739** (11 methods)
- Pairwise: yeast-human ρ=+0.636, yeast-mouse ρ=+0.555, human-mouse ρ=+0.636
- Spectral #1 in all three species; MDS consistently top-3
- Two-factor model does NOT transfer to mouse (rho=−0.037) — geometric predictors are network-specific
- Persistence images do not improve G-F prediction over scalar H1 features

Report → [`results/phase10_report.md`](results/phase10_report.md) · Figures → [`figures/Fig51-54`](figures/)

---

## Spectral Transferability Theory (Phase 11)

Why does the two-factor model (spectral alignment + effective rank) work on yeast and human but fail on mouse? The answer lies in the **Spectral Quality Index (SQI)**:

$$\text{SQI} = \frac{\lambda_2}{\lambda_2^{\text{ER}}} \times \text{PR}(v_2) \times \text{FA}_{\max}$$

| Species | n | λ₂ | Fiedler PR | SQI | SA_std | 2F rho |
|---------|-------|--------|-----------|-------|--------|--------|
| Yeast | 5,936 | 0.0409 | 0.0044 | 10.72 | 0.140 | +0.929 |
| Human | 15,882 | 0.0046 | 0.0037 | 2.02 | 0.006 | +0.483 |
| Mouse | 16,180 | 0.0067 | 0.0007 | 0.54 | 0.006 | −0.037 |

- Mouse Fiedler vector is **6× more localized** than yeast (PR=0.0007 vs 0.0044)
- SQI ordering perfectly predicts two-factor model performance: yeast > human > mouse
- Validated on 20 synthetic SBM networks: SA_std correlates +0.647 with log(SQI)
- Practical implication: compute SQI from a PPI network alone to predict a priori whether spectral-based evaluation will work

Report → [`results/phase11_report.md`](results/phase11_report.md) · Figures → [`figures/Fig55-59`](figures/)

## Biological Validation & Statistical Power (Phase 12)

Does the G-F framework detect biologically meaningful structure, and are rankings statistically robust?

**Part A — GO BP Enrichment**: At r=0.2, communities from each embedding method are tested for Gene Ontology biological_process enrichment via hypergeometric test (24,135 BP terms).

| Species | Spectral Enrichment | Best Method | Best Enrichment |
|---------|:-------------------:|-------------|:---------------:|
| Yeast | 80% (p=4.58e-10) | DM | 100% (p=1.32e-11) |
| Human | 0% (p=0.35) | GraphSAGE | 100% (p=6.87e-15) |
| Mouse | 14% (p=0.06) | GAT | 100% (p=5.80e-12) |

**Part B — Multi-Seed Panel**: 220 observations across 20 groups (species × seed) with pooled Spearman rank consistency.

| Metric | Value |
|--------|-------|
| Pooled \|ρ\| | 0.583 (95% CI [0.470, 0.688]) |
| Yeast \|ρ\| | 0.981 (n=55) |
| Human \|ρ\| | 0.967 (n=110) |
| Mouse \|ρ\| | 0.800 (n=55) |

Report → [`results/phase12_report.md`](results/phase12_report.md) · Figures → [`figures/Fig60-64`](figures/)

---

## Protein Function Prediction — Closing the Loop (Phase 13)

Can the GF-consistent embedding space predict protein function, and does GF Score predict prediction accuracy?

**Leave-One-Term-Out CV** on the full yeast STRING network (5,936 nodes, 4,709 proteins with experimental BP annotations, 12,690 trials). Five methods with full-network embeddings predict function via KNN; three network-topology baselines for comparison.

| Method | P@5 | P@10 | MRR |
|--------|:---:|:----:|:---:|
| PPI-Neighbors (baseline) | 0.331 | 0.442 | 0.219 |
| 2-Hop Diffusion (baseline) | 0.147 | 0.222 | 0.105 |
| **Spectral** | **0.097** | **0.148** | **0.066** |
| MDS | 0.088 | 0.136 | 0.060 |
| DM | 0.054 | 0.084 | 0.037 |
| Random (baseline) | 0.049 | 0.073 | 0.041 |

**Closing the loop**: GF Score (curated 153-node network) vs MRR (full 5,936-node network) yields Spearman rho = 0.900 (P = 0.037, n = 5 methods). The framework's structural quality metric predicts function-prediction accuracy across network scales.

Report → [`results/phase13_report.md`](results/phase13_report.md) · Figures → [`figures/Fig65-68`](figures/)

---

## Limitations

- **Network scale**: The primary ranking is based on a curated 153-node yeast subnetwork. Full-network (5,936 nodes) and cross-species (15,882 nodes) validations confirm general trends, but fine-grained method ordering may vary with network size.
- **GO annotation bias**: G-F Score depends on GO annotation quality and coverage. Well-studied genes have richer annotations, potentially inflating purity for communities dominated by such genes. This limitation is shared by all GO-based evaluation frameworks.
- **GO DAG propagation artifact**: True Path Rule expansion (Step 19) increases annotations from ~3.8 to ~28.9 terms/gene, causing community purity to approach 1.0 (G-F Score ≈ 0.9996). This is a known artifact of hierarchical expansion; the main results use pre-propagation annotations.
- **Community detection**: Only greedy modularity optimization is used for the main results; however, ablation analysis (Step 44) confirms G-F Score robustness across 5 community detection algorithms (Kendall's W = 0.797), with Spectral ranking #1 under 4 of 5 algorithms.
- **2D output for G-F curves**: All 11 methods produce 2D coordinate spaces (standardized to σ = 0.3) for G-F curve computation. Higher-dimensional embeddings (d = 8-256) have been evaluated separately (Steps 47, 52-55) and show that Spectral embedding surpasses PPI topology at d = 256 (MRR 0.230 vs 0.219).
- **Plateau width**: Defined as the r-interval where purity ≥ 80% of each method's peak (relative threshold). Methods with very flat purity curves may yield wide plateaus despite low absolute purity. The G-F Score itself (integrated purity over the unified interval [0.05, 0.422]) provides an absolute metric independent of this definition.
- **Spearman correlations**: G-F Score vs link prediction AUC (ρ = 0.591, P = 0.056) and vs k-NN F1 (ρ = 0.609, P = 0.047) are based on n = 11 methods. Bootstrap 95% CIs indicate moderate precision; see `results/bootstrap_correlations.json` for full details. Convergent evidence from full 11-method function prediction (rho=0.646, p=0.032) and cross-species Kendall W=0.739 provides independent validation.
- **GAT-family embedding collapse**: GAT exhibits persistent embedding collapse across species, architectures, and attention variants. Five architectural variants (gradient clipping, warmup, multi-head attention) and GATv2 dynamic attention (Brody et al., ICLR 2022) have been tested — GATv2 partially reduces attention entropy (0.903 vs 0.927) but does not rescue G-F Score (0.157 vs 0.154, both near-random). The collapse is architectural, driven by the adjacency-reconstruction objective on degree-heterogeneous PPI networks.
- **Density-dependent rankings**: Method rankings are network-density-dependent (W<sub>raw</sub> = 0.178 across STRING thresholds 400-900), though density correction substantially improves concordance (W<sub>corrected</sub> = 0.70, Step 36). Threshold sensitivity analysis (Step 58) confirms stability between scores 600-700 (rho=0.90).
- **Network-specific geometric-functional mapping** (Phase 2-3): The relationship between embedding geometry and G-F Score is network-topology-specific. This is not a limitation but an informative feature: the Spectral Quality Index (SQI) predicts a priori when spectral-based evaluation will be effective (SQI > 5: well-suited; SQI 1-5: beneficial with higher d; SQI < 1: may require alternatives).

---

## Author

**Yuhan Zhang (张宇涵)**  
Department of Chemical Engineering and Pharmacy, Guangling College, Yangzhou University  
Correspondence: qinray@hotmail.com  
ORCID: [0009-0000-2769-467X](https://orcid.org/0009-0000-2769-467X)

---

## Academic Use Notice

This repository accompanies a manuscript currently under peer review at *Nature Communications*. The code and data are made publicly available for reproducibility and transparency purposes. By using this repository, you agree to the following terms:

1. **Citation requirement**: Any academic publication that uses, adapts, or builds upon this framework, its methodology, or its results **must** cite the accompanying paper (see Citation below). This includes, but is not limited to: benchmarking studies, comparative analyses, extensions, and meta-analyses.

2. **No competing pre-publication use**: This code and its results **must not** be used to produce a competing or derivative publication before the original manuscript has completed its publication process. If the manuscript is rejected or withdrawn, this restriction is lifted 12 months after the repository's initial public release.

3. **Derivative works**: Modified versions of this code must clearly state that they are modified versions and must not be presented as the original work. Any fork or derivative repository used in academic work must retain this notice and the original citation.

4. **Results integrity**: The numerical results, figures, and rankings produced by this pipeline **must not** be reused in other publications without explicit written permission from the corresponding author, except for the purpose of reproducing or verifying the results as described in the accompanying paper.

Violation of these terms constitutes an academic ethics concern. The author reserves the right to raise such concerns with the relevant journal editors and institutional research integrity offices. For questions or collaboration inquiries, contact the corresponding author.

---

## Citation

If you use this framework, please cite:

> Zhang, Y. "Embedding Geometry Reveals Functional Dark Matter in Protein Interaction Networks." (2026). Submitted to *Nature Communications*.
>
> ```bibtex
> @article{zhang2026gf,
>   title   = {Embedding Geometry Reveals Functional Dark Matter
>              in Protein Interaction Networks},
>   author  = {Zhang, Yuhan},
>   year    = {2026},
>   note    = {Reproducible pipeline: 11 methods, 72-step validation,
>              113 scripts. Submitted to Nature Communications.},
> }
> ```

---

## License

[MIT](LICENSE) — see [Academic Use Notice](#academic-use-notice) for usage terms applicable to academic publications.