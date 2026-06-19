# Verification Log (June 2026)

Day-by-day record of pre-submission verification. Each entry documents which scripts or results were audited and what issues were found.

---

## 2026-06-16

**Scope:** Code quality pass across all scripts.

- Fixed Windows GBK encoding crashes: `print()` statements containing Unicode (→, ρ, ±) replaced with ASCII equivalents
- Replaced hardcoded absolute paths with `Path(__file__).parent` in 6 scripts
- Added `encoding='utf-8'` to all `open()` calls writing non-ASCII content
- Unified `random_seed` propagation: `np.random.seed()`, `random.seed()`, `torch.manual_seed()` all set from `pipeline_config.yaml`
- Fixed build-backend in `pyproject.toml`

## 2026-06-17

**Scope:** v2.8.1 — GATv2 experiment review.

- `gatv2_experiment.py`: verified attention entropy values (0.903 vs 0.927) match output JSON
- Confirmed GATv2 GF Score (0.157) vs GAT (0.154) — dynamic attention does not rescue collapse
- Added academic use protection notice to README

## 2026-06-18

**Scope:** v2.9.0 through v2.11.0 — new experiment modules + full pipeline run.

- `fly_analysis.py` (Step 60): verified Drosophila network stats (6,909 nodes, 89,685 edges), Spectral GF=0.619
- `prone_harp_gf.py` (Step 61): confirmed ProNE GF=0.087, HARP GF=0.114 both below random baseline (0.135)
- `function_prediction_cosine.py` (Step 62): cosine voting improves MRR for all 5 methods, GF-MRR rho=0.90
- `heat_kernel_multiscale.py` (Step 63): verified Spectral = t→0 limit (rho=1.0 across all time scales)
- `position_encoding_comparison.py` (Step 64): Laplacian PE GF=0.163 matches Spectral; sign-flip std=9.4e-5 confirmed
- `cheeger_gf_bound.py` (Step 65): 4-component bound valid for all 6 networks; tightness values cross-checked
- `function_prediction_atlas.py` (Step 67): Spectral d=256 exceeds PPI in 9/9 ontology-species cases
- `uncharacterized_prediction.py` (Step 68): 511 predictions for 285 proteins verified
- `dark_matter_pan_species.py` (Step 72): fixed STRING v12.0 header crash (try/except for malformed header lines)
- `cross_species_atlas.py` (Step 69): fixed invalid f-string format bug in output logging
- Rebuilt `cross_species_atlas.json` — Human BP results were all zeros (LOTO-CV bug)
- Full pipeline run: all 72 steps complete, all figures regenerated

## 2026-06-19

**Scope:** Final engineering audit — documentation consistency, tag/release alignment, module table completeness.

Audit 1 — Step count and method count:
- Found 5 remaining "65-step" references in README → all corrected to "72-step"
- Found "11 methods" in README header → corrected to "18 methods" (7 additional: UMAP, UMAP-adj, t-SNE, t-SNE-sp, ProNE, HARP, GATv2)
- `scripts/__init__.py` docstring: "65-step" → "72-step"
- `scripts/benchmark_runtime.py` docstring: "14-step" → "72-step"

Audit 2 — Project structure completeness:
- README directory tree missing Steps 63-72 → added all 10 scripts with descriptions
- Steps 60-62 ordering wrong in 3 of 4 README sections (Key Results, Step list, module table had prone=60, cosine=61, fly=62) → corrected to fly=60, prone=61, cosine=62 per `run_all_analysis.py` authoritative ordering
- `run_all_analysis.py` docstring: Steps 69-70 descriptions were swapped → corrected

Audit 3 — Orphan script review:
- `cross_species_atlas_optimized.py` (591 lines): zero references anywhere, no output artifacts → removed
- `gf_mrr_bridge.py`: regression analysis over `function_prediction_atlas.json` data, unique interaction model (R²=0.282) but fully reconstructable from existing JSON → removed (data preserved in `results/gf_mrr_bridge.json`)
- `human_atlas.py`: fully duplicated by `cross_species_atlas.py` (identical MRR values in output) → removed

Audit 4 — Module name / filename mismatches:
- `scripts/__init__.py`: `dimension_sweep_512_1024` → `dimension_sweep_512` (actual filename)
- `scripts/__init__.py`: `uncharacterized_protein_mining` → `uncharacterized_prediction` (actual filename)
- `scripts/__init__.py`: `atlas_extension` → `atlas_extension_512` (actual filename)

Audit 5 — Script count:
- "113 scripts" in BibTeX was pre-deletion count → corrected to 110 (verified: `find scripts/ -name "*.py" | wc -l` = 110)

Audit 6 — Tag/release alignment:
- Deleted orphan tags v1.1.1 and v1.2.0 (had tags but no releases)
- Final state: 3 tags = 3 releases (v1.0.0, v1.1.0, v2.11.0-paper-submission)

Audit 7 — Pipeline Overview:
- Step 50 was "Rescue Protein Analysis" but `run_all_analysis.py` defines Step 50 as `generate_missing_figures.py` → corrected
- `rescue_protein_analysis.py` is a supplementary module, not a pipeline step → removed step number from module table
- Module table missing `generate_missing_figures.py` (Step 50) → added
- Module table missing Steps 63-72 (10 entries) → added all

Audit 8 — Source code patterns:
- Verified all 100+ `json.dump` calls use protective patterns: explicit `float()`/`int()` casts, `default=str` fallback, or custom `_json_default` handler — no unprotected numpy types
- Verified zero Unicode characters in `print()` statements — all statistical symbols only in matplotlib labels
- Verified no stale imports referencing deleted files
- Verified all 48 `from scripts.` imports in `run_all_analysis.py` target existing modules
