"""
human_validation/ — Cross-species validation on human PPI network.

This package validates the G-F consistency framework on the STRING v12.0
human interactome (~15,882 nodes after score filtering; 14,679 in largest CC)
using all six embedding methods.

Modules
-------
run_human_validation    Unified runner for the full human pipeline
human_embed_all         Generate 6 embeddings for human network
human_gf_all            Compute G-F curves, scores, and plateau widths
plot_human_results      Generate comparative figures
"""
