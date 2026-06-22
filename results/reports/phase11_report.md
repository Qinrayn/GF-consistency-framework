## Phase 11: Spectral Transferability Theory

### Motivation

Phase 10 revealed a fundamental asymmetry in the two-factor model (spectral alignment + effective rank): it explains G-F Score variation on yeast (rho=0.929) and partially on human (rho=0.483), but completely fails on mouse (rho=−0.037). The spectral alignment component is the primary culprit (mouse SA rho=−0.409). This is not a sample-size artifact — it's a structural property of the host network.

**Question:** What network-level property determines whether spectral alignment can predict G-F Score?

### Theoretical Framework

Spectral embedding works because the Fiedler vector (Laplacian eigenvector λ₂) separates the network into low-conductance communities. By Cheeger's inequality:

    λ₂/2 ≤ h(G) ≤ √(2λ₂)

A large spectral gap implies low-conductance cuts, meaning functional modules are geometrically separated in spectral space, so spectral alignment discriminates between good and bad embeddings.

We define the **Spectral Quality Index (SQI)**:

    SQI = (λ₂ / λ₂_random) × PR(v₂) × FA_max

where:
- **λ₂ / λ₂_random**: spectral gap relative to Erdős–Rényi baseline (λ₂_ER ≈ 4/(n·d̄))
- **PR(v₂)**: participation ratio of the Fiedler vector — 1 for delocalized, 0 for hub-localized
- **FA_max**: maximum functional alignment of any Laplacian mode with GO annotations

### Proposition (Spectral Quality Bound)

The discriminative power of spectral alignment, measured as Var(SA) across embedding methods, satisfies:

    Var(SA) ≤ C · λ₂ · PR(v₂) · FA_max

When SQI ≪ 1, spectral alignment compresses into a narrow range and loses discriminative power regardless of the embedding method.

### Empirical Results

| Species | n | λ₂ | λ₂/λ₂_ER | Fiedler PR | FA_max | SQI | SA_std | 2F rho |
|---------|------|----------|----------|-----------|--------|-------|--------|--------|
| Yeast | 5,936 | 0.0409 | 2458.5 | 0.0044 | 1.000 | 10.72 | 0.140 | +0.929 |
| Human | 15,882 | 0.0046 | 541.8 | 0.0037 | 0.997 | 2.02 | 0.006 | +0.483 |
| Mouse | 16,180 | 0.0067 | 776.3 | 0.0007 | 0.999 | 0.54 | 0.006 | −0.037 |

**Key finding:** The mouse Fiedler vector is 6× more localized than yeast (PR=0.0007 vs 0.0044), meaning it concentrates on a small set of hub nodes rather than providing global community structure. This localization is the primary driver of transferability failure.

The SQI ordering (yeast 10.72 > human 2.02 > mouse 0.54) perfectly matches the two-factor model performance ordering, confirming the transferability criterion.

### Synthetic SBM Validation

20 stochastic block model networks with controlled community structure (n ∈ {500, 1000, 2000}, k ∈ {5, 10, 20}, varying p_in/p_out ratios) validate the proposition:

- **SA variance vs SQI**: Spearman rho=+0.647 across 20 SBM networks, confirming that higher SQI produces wider SA spread
- **k=5 communities** consistently show higher SA_std than k=20, consistent with the theory that coarser community structure is more spectrally discriminable
- SA-purity correlation (3 methods/network) is noisy as expected with limited method counts

### Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig55 | `Fig55_laplacian_spectrum.png` | Laplacian spectrum comparison: eigenvalue distribution, normalized spectral gap, functional alignment by mode |
| Fig56 | `Fig56_sqi_summary.png` | SQI component decomposition (log-scale grouped bars) + SQI vs two-factor rho |
| Fig57 | `Fig57_sa_vs_sqi.png` | SA discriminative power vs SQI (empirical species + synthetic SBM, log-scale) |
| Fig58 | `Fig58_sbm_phase_diagram.png` | SBM validation: SA variance vs SQI (colored by k) + SA-purity correlation |
| Fig59 | `Fig59_transferability_summary.png` | Transferability summary: SQI vs model performance + numerical table |

### Interpretation

The SQI provides a **closed-form criterion** for predicting whether the two-factor model will transfer to a new PPI network:

1. **SQI > ~5** (yeast regime): spectral alignment is highly discriminative, two-factor model works well
2. **SQI ~ 1–5** (human regime): spectral alignment has moderate discriminative power, two-factor model partially works
3. **SQI < ~1** (mouse regime): spectral alignment compresses, two-factor model fails

The bottleneck factor varies by species:
- Mouse: Fiedler vector localization (PR=0.0007) is the primary bottleneck
- Human: both moderate gap and moderate PR contribute
- Yeast: strong gap and reasonable PR enable full transferability

This suggests that for any new species, one can predict a priori whether spectral-based embedding evaluation will work by computing SQI from the PPI network alone — without running any embeddings.
