#!/bin/bash
# Phase 1 orchestrator — runs full pipeline end-to-end.
# Check progress anytime: cat phase1_status.txt
#
# Usage:
#   bash run_phase1_full.sh          # full run from scratch
#   bash run_phase1_full.sh --resume # resume interrupted MD
#
# Configure via env vars (defaults shown):
#   DROPLET_DIAMETER_NM=5.0  MACE_MODEL=medium  bash run_phase1_full.sh
set -eo pipefail

export DROPLET_DIAMETER_NM="${DROPLET_DIAMETER_NM:-5.0}"
export MACE_MODEL="${MACE_MODEL:-medium}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESUME="${1:-}"
STATUS_FILE="phase1_status.txt"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$STATUS_FILE"
}

log "============================================================"
log "Phase 1: Water Microdroplet E-field — Full Pipeline"
log "============================================================"
log "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'local')"
log "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'none')"

# ── Phase 1a: Build droplet ─────────────────────────────────────
if [ "$RESUME" != "--resume" ] && [ ! -f droplet_initial.xyz ]; then
    log "Phase 1a: Building 8nm water droplet..."
    python build_droplet.py 2>&1 | tee -a "$STATUS_FILE"

    if [ ! -f droplet_initial.xyz ]; then
        log "FAIL: droplet_initial.xyz not created. Exiting."
        exit 1
    fi
    log "Phase 1a: DONE"
else
    log "Phase 1a: Skipped (droplet_initial.xyz exists)"
fi

# ── Phase 1b+1c: 20 ns MD ───────────────────────────────────────
log "Phase 1b+1c: Starting 20 ns NVT MD..."
log "  Go/no-go report will appear in phase1/go_nogo_report.txt after ~1 ns"
log "  Check progress: cat phase1_status.txt"

if [ "$RESUME" == "--resume" ]; then
    python run_phase1.py --resume 2>&1 | tee -a "$STATUS_FILE"
else
    python run_phase1.py 2>&1 | tee -a "$STATUS_FILE"
fi

log "Phase 1b+1c: DONE (20 ns MD complete)"

# Copy go/no-go report into status file
if [ -f phase1/go_nogo_report.txt ]; then
    echo "" >> "$STATUS_FILE"
    echo "=== GO/NO-GO REPORT ===" >> "$STATUS_FILE"
    cat phase1/go_nogo_report.txt >> "$STATUS_FILE"
    echo "" >> "$STATUS_FILE"
fi

# ── Phase 1d: E-field analysis ──────────────────────────────────
log "Phase 1d: Running E-field analysis (PolarMACE + fixed charges)..."
python analyze_efield.py --trajectory phase1/trajectory.traj --n-frames 100 2>&1 | tee -a "$STATUS_FILE"
log "Phase 1d: DONE"

# ── Sanity check ────────────────────────────────────────────────
log "=== RESULTS SUMMARY ==="
python -c "
import numpy as np
data = np.load('analysis/efield_analysis.npz', allow_pickle=True)
r = data['r_angstrom']
r90 = float(data['r90_angstrom'])

surface = (r > r90 - 5) & (r < r90 + 5)
bulk = r < r90 - 10

has_polar = 'efield_polar_mean_mvcm' in data

if has_polar:
    sp = data['efield_polar_mean_mvcm'][surface].mean()
    bp = data['efield_polar_mean_mvcm'][bulk].mean()
    enhancement = sp - bp
    print(f'PolarMACE surface enhancement: {enhancement:.1f} MV/cm (ref: 16 MV/cm)')
    if 8 <= enhancement <= 32:
        print('IN RANGE: Right ballpark vs C-GeM. Publishable result.')
    elif enhancement > 0:
        print('POSITIVE but out of 8-32 range. Proceed to Phase 2.')
    else:
        print('NO ENHANCEMENT. Check charge model / trajectory.')

sf = data['efield_fixed_mean_mvcm'][surface].mean()
bf = data['efield_fixed_mean_mvcm'][bulk].mean()
print(f'Fixed-charge enhancement: {sf - bf:.1f} MV/cm')

orient = data['orient_cos_mean']
so = orient[surface].mean()
print(f'Surface <cos theta>: {so:+.3f}')
" 2>&1 | tee -a "$STATUS_FILE"

# ── Comparison figure ───────────────────────────────────────────
log "Generating comparison figures..."
python compare_to_cgem.py 2>&1 | tee -a "$STATUS_FILE"

# ── Final status ────────────────────────────────────────────────
log "============================================================"
log "Phase 1 COMPLETE"
log "============================================================"
log "Output files:"
log "  phase1_status.txt              — this log"
log "  phase1/go_nogo_report.txt      — go/no-go checkpoint"
log "  analysis/efield_analysis.npz   — numerical data"
log "  analysis/efield_analysis.png   — 4-panel analysis"
log "  analysis/comparison_cgem.pdf   — publication figure"
log ""
log "Next: python extract_clusters.py --n-clusters 300"
