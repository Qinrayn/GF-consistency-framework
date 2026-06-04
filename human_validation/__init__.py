"""
human_validation/ — Cross-species validation on human PPI network.

This package validates the G-F consistency framework on the STRING v12.0
human interactome (14,679 nodes) using all six embedding methods.

Modules
-------
run_human_validation    Unified runner for the full human pipeline
human_embed_all         Generate 6 embeddings for human network
human_gf_all            Compute G-F curves, scores, and plateau widths
plot_human_results      Generate comparative figures

Legacy scripts (deprecated, kept for reference):
    12_human_ppi_validation.py
    12d_human_n2v_cleaned_scan.py
    12e_human_dm_quick.py
"""
