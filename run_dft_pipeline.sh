#!/bin/bash
# Runs DFT in parallel with ongoing MD on the same instance.
# Watches the trajectory, extracts clusters after equilibration,
# and feeds them to PySCF on idle CPUs while GPU does MD.
#
# Usage: nohup bash run_dft_pipeline.sh > dft_pipeline.log 2>&1 &
#
# Prerequisites:
#   - run_phase1.py is running (or finished) and phase1/trajectory.traj exists
#   - go/no-go has passed (phase1/go_nogo_report.txt exists)
#   - PySCF is installed: pip install pyscf pyscf-dftd3
set -eo pipefail

TRAJ_FILE="phase1/trajectory.traj"
GO_NOGO_FILE="phase1/go_nogo_report.txt"
CLUSTERS_DIR="clusters"
DFT_DIR="dft_results"
DFT_HYBRID_DIR="dft_hybrid"     # supplementary revPBE0-D3 spot-check outputs
N_SURFACE=120
N_INTERFACE=90
N_BULK=90
N_HYBRID_SPOTCHECK=30           # surface clusters re-run at hybrid level
EXTRACT_INTERVAL=600            # check for new frames every 10 min
CUTOFF=6.0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ── Wait for go/no-go ───────────────────────────────────────────
log "DFT pipeline waiting for go/no-go report..."
while [ ! -f "$GO_NOGO_FILE" ]; do
    sleep 300
    log "  Still waiting for $GO_NOGO_FILE"
done

log "Go/no-go report found:"
cat "$GO_NOGO_FILE"
log ""

# Check for fatal issues
if grep -q "VERDICT: NO-GO" "$GO_NOGO_FILE" 2>/dev/null; then
    log "NO-GO detected. Not starting DFT. Exiting."
    exit 1
fi

# ── Install PySCF if needed ─────────────────────────────────────
python -c "import pyscf" 2>/dev/null || {
    log "Installing PySCF..."
    pip install pyscf pyscf-dftd3 > /dev/null 2>&1
    log "PySCF installed"
}

# ── Wait for enough production frames ───────────────────────────
# Need at least N_TOTAL distinct production frames for statistical diversity.
# At 0.03 ns/day = ~300 frames/day (TRAJ_INT=200, 0.1ps/frame), ~1 day wait.
N_TOTAL=$((N_SURFACE + N_INTERFACE + N_BULK))
MIN_PROD_FRAMES=$N_TOTAL

log "Waiting for at least $MIN_PROD_FRAMES production frames in trajectory..."
while true; do
    N_TRAJ_FRAMES=$(python -c "from ase.io.trajectory import Trajectory; print(len(Trajectory('$TRAJ_FILE')))" 2>/dev/null || echo "0")
    # Auto-detect equilibration frames from go/no-go report
    EQUIL_NS=$(grep -oP 'Equilibration time:\s+\K[\d.]+' "$GO_NOGO_FILE" 2>/dev/null || echo "0.5")
    EQUIL_FRAMES=$(python -c "print(int(float('$EQUIL_NS') * 1e3 / 0.1) + 100)")
    PROD_FRAMES=$((N_TRAJ_FRAMES - EQUIL_FRAMES))
    if [ "$PROD_FRAMES" -ge "$MIN_PROD_FRAMES" ] 2>/dev/null; then
        log "  $PROD_FRAMES production frames available (need $MIN_PROD_FRAMES). Proceeding."
        break
    fi
    log "  $PROD_FRAMES/$MIN_PROD_FRAMES production frames. Waiting 1h..."
    sleep 3600
done

# ── Extract clusters from trajectory (stratified) ───────────────
log "Extracting $N_TOTAL clusters (stratified: $N_SURFACE surface / $N_INTERFACE interface / $N_BULK bulk)..."
python extract_clusters.py \
    --n-surface "$N_SURFACE" \
    --n-interface "$N_INTERFACE" \
    --n-bulk "$N_BULK" \
    --cutoff "$CUTOFF" 2>&1

if [ ! -d "$CLUSTERS_DIR" ] || [ -z "$(ls $CLUSTERS_DIR/cluster_*.xyz 2>/dev/null)" ]; then
    log "ERROR: No clusters extracted. Check trajectory."
    exit 1
fi

N_EXTRACTED=$(ls "$CLUSTERS_DIR"/cluster_*.xyz 2>/dev/null | wc -l)
log "Extracted $N_EXTRACTED clusters"

# ── Run primary DFT pass on CPUs (revPBE-D3) ────────────────────
# Use 2 parallel workers with 4 cores each (leaves 1 core free for MD)
export OMP_NUM_THREADS=4
N_WORKERS=2

log "Starting DFT primary pass: $N_EXTRACTED clusters, $N_WORKERS parallel workers, $OMP_NUM_THREADS threads each"
log "Functional: revPBE-D3 / def2-SVP (adequate for MACE fine-tuning forces)"

python run_dft.py \
    --clusters-dir "$CLUSTERS_DIR" \
    --output-dir "$DFT_DIR" \
    --workers "$N_WORKERS" 2>&1

N_DONE=$(ls "$DFT_DIR"/cluster_*.json 2>/dev/null | wc -l)
N_OK=$(grep -l '"status": "ok"' "$DFT_DIR"/cluster_*.json 2>/dev/null | wc -l)
log "Primary DFT complete: $N_OK/$N_DONE converged"

# ── Supplementary hybrid spot-check on surface clusters ─────────
# revPBE0-D3 is ~5× slower than revPBE-D3. Limited to N_HYBRID_SPOTCHECK
# surface clusters to bound cost while quantifying the functional error
# on the physics that matters for the paper.
log ""
log "Starting hybrid spot-check: $N_HYBRID_SPOTCHECK surface clusters at revPBE0-D3/def2-TZVP"
python run_dft.py \
    --clusters-dir "$CLUSTERS_DIR" \
    --output-dir "$DFT_HYBRID_DIR" \
    --functional revpbe0 \
    --basis def2-tzvp \
    --filter '*_surface.xyz' \
    --end "$N_HYBRID_SPOTCHECK" \
    --workers "$N_WORKERS" 2>&1

N_HYBRID_DONE=$(ls "$DFT_HYBRID_DIR"/cluster_*.json 2>/dev/null | wc -l)
N_HYBRID_OK=$(grep -l '"status": "ok"' "$DFT_HYBRID_DIR"/cluster_*.json 2>/dev/null | wc -l)
log "Hybrid spot-check complete: $N_HYBRID_OK/$N_HYBRID_DONE converged"

log ""
log "Results in $DFT_DIR/ (primary) and $DFT_HYBRID_DIR/ (hybrid spot-check)"
log "Next: python validate_polarmace_vs_dft.py --dft-dir $DFT_DIR"
log "      python finetune_mace.py --dft-dir $DFT_DIR"
