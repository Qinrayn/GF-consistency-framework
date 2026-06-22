# human_validation/ (DEPRECATED)

This directory contains early human PPI validation scripts from the initial
cross-species analysis. It has been **superseded** by more complete scripts
in `scripts/`:

| Legacy Script | Replaced By |
|---------------|-------------|
| `human_gf_all.py` | `scripts/human_gf_extended.py` (11 methods, greedy_modularity) |
| `human_embed_all.py` | `scripts/embed_all.py` + `scripts/embed_gnn.py` |
| `human_topological_analysis.py` | `scripts/human_tda_full.py` |
| `run_all_human.py` | `run_all_analysis.py` (Steps 37-38) |

The compressed data files (`9606.protein.links.v12.0.txt.gz`, etc.) are
retained as they may be needed for STRING v12.0 human network access.

**Do not use these scripts for new analysis.** They use the older Louvain
community detection algorithm and outdated GF Score intervals.
