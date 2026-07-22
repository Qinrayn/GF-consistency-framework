#!/usr/bin/env bash
# =========================================================================
# Server Run Script — G-F Consistency Framework Extensions
# =========================================================================
# Usage:
#   chmod +x server_run.sh
#   ./server_run.sh                    # Full run (all 3 extensions)
#   ./server_run.sh --quick            # Quick test (SQI quick + GFAE small)
#   ./server_run.sh --gpu              # GFAE on full 5936-node network (GPU)
#
# Output:
#   results/sqi_validation_extended.json     # Direction C
#   results/functional_aware_embedding_metadata.json  # Direction A
#   results/dark_matter_external_validation.json      # Direction B
# =========================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  G-F Consistency Framework — Server Extension Run"
echo "  $(date)"
echo "============================================================"

# ---- Parse args ----
QUICK=false
GPU=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=true; shift ;;
        --gpu)   GPU=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- 1. Direction C: SQI Systematic Validation ----
echo ""
echo "============================================================"
echo "  [1/3] SQI Systematic Validation (Direction C)"
echo "============================================================"
if [ "$QUICK" = true ]; then
    python scripts/sqi_validation_extended.py --quick
else
    python scripts/sqi_validation_extended.py
fi
echo "  ✓ SQI validation complete → results/sqi_validation_extended.json"

# ---- 2. Direction A: Functional-Aware Embedding (GFAE) ----
echo ""
echo "============================================================"
echo "  [2/3] Functional-Aware Embedding (Direction A)"
echo "============================================================"
if [ "$GPU" = true ]; then
    # Full network, GPU training
    python scripts/functional_aware_embedding.py --network full --epochs 500
    echo "  ✓ GFAE (full, GPU) → embeddings/GFAE_full.npy"
else
    # Curated 153-node network (CPU-friendly)
    python scripts/functional_aware_embedding.py --network curated --epochs 300
    echo "  ✓ GFAE (curated) → embeddings/GFAE_153.npy"
fi

# Evaluate GFAE GF Score
python -c "
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, 'scripts')
from utils import load_curated_network, load_embedding, compute_gf_curve, compute_gf_score, GF_R_MIN, GF_R_MAX, R_MIN, R_MAX, N_POINTS

data_dir = Path('data')
emb_dir = Path('embeddings')
G, nodes, go_map = load_curated_network(data_dir)
r_vals = np.linspace(R_MIN, R_MAX, N_POINTS)

tag = 'full' if '$GPU' == 'true' else '153'
coords, emb_nodes = load_embedding('GFAE', tag, embeddings_dir=emb_dir)
node_to_idx = {n: i for i, n in enumerate(emb_nodes)}
common = sorted(set(node_to_idx) & set(nodes) & set(go_map))
idx = [node_to_idx[n] for n in common]
aligned = coords[idx]
pur, _ = compute_gf_curve(aligned, common, go_map, r_vals)
score = compute_gf_score(r_vals, pur, GF_R_MIN, GF_R_MAX)
print(f'  GFAE ({tag}) GF Score: {score:.4f}')
" 2>&1
echo "  ✓ GFAE GF Score evaluated"

# ---- 3. Direction B: Dark Matter External Validation ----
echo ""
echo "============================================================"
echo "  [3/3] Dark Matter External Validation (Direction B)"
echo "============================================================"
python scripts/dark_matter_external_validation.py
echo "  ✓ Dark matter validation → results/dark_matter_external_validation.json"

# ---- Summary ----
echo ""
echo "============================================================"
echo "  ALL EXTENSIONS COMPLETE"
echo "============================================================"
echo "  Results:"
echo "    results/sqi_validation_extended.json"
echo "    results/functional_aware_embedding_metadata.json"
echo "    results/dark_matter_external_validation.json"
echo "    embeddings/GFAE_*.npy"
echo ""
echo "  Download these files back to your local machine."
echo "  Run \"python scripts/compute_gf.py\" to compare GFAE against baselines."
echo "============================================================"