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
N_CLUSTERS=300
EXTRACT_INTERVAL=600  # check for new frames every 10 min
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

# ── Extract clusters from trajectory ────────────────────────────
log "Extracting $N_CLUSTERS clusters from production frames..."
python extract_clusters.py --n-clusters "$N_CLUSTERS" --cutoff "$CUTOFF" 2>&1

if [ ! -d "$CLUSTERS_DIR" ] || [ -z "$(ls $CLUSTERS_DIR/cluster_*.xyz 2>/dev/null)" ]; then
    log "ERROR: No clusters extracted. Check trajectory."
    exit 1
fi

N_EXTRACTED=$(ls "$CLUSTERS_DIR"/cluster_*.xyz 2>/dev/null | wc -l)
log "Extracted $N_EXTRACTED clusters"

# ── Run DFT on CPUs ─────────────────────────────────────────────
# Use 2 parallel workers with 4 cores each (leaves 1 core free for MD)
export OMP_NUM_THREADS=4
N_WORKERS=2

log "Starting DFT: $N_EXTRACTED clusters, $N_WORKERS parallel workers, $OMP_NUM_THREADS threads each"
log "Functional: revPBE-D3 / def2-TZVP"

python run_dft.py \
    --clusters-dir "$CLUSTERS_DIR" \
    --output-dir "$DFT_DIR" \
    --workers "$N_WORKERS" 2>&1

N_DONE=$(ls "$DFT_DIR"/cluster_*.json 2>/dev/null | wc -l)
N_OK=$(grep -l '"status": "ok"' "$DFT_DIR"/cluster_*.json 2>/dev/null | wc -l)

log "DFT complete: $N_OK/$N_DONE converged"
log ""
log "Results in $DFT_DIR/"
log "Next: python finetune_mace.py --dft-dir $DFT_DIR"
