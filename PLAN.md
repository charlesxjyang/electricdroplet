# PLAN.md — Context and Instructions for Claude Code

## What this project is

We're computing the radial electric field profile at the air-water interface
of a 5nm water nanodroplet (~2,100 water molecules, ~6,300 atoms) using
MACE-MP-0 machine learning molecular dynamics. This is the first MLIP-based
calculation of this quantity — the only prior work is Head-Gordon's 2022
Nat. Commun. paper using ReaxFF/C-GeM, which found ~9 MV/cm surface
electric field (size-independent across R40-R80).

The result will be posted on ChemRxiv and submitted to JPC Letters (4-page
limit, no APC for subscription access).

## Hardware

**Single instance**: AWS g6e.2xlarge on-demand
- 1× NVIDIA L40S GPU (46 GB VRAM) — runs MACE MD + PolarMACE analysis
- 8 vCPU, 64 GB RAM — runs DFT on idle CPUs during production MD
- $1.11/hr on-demand
- Instance ID: i-015a994614ddbb34f
- IP: 34.201.54.227

**Local**: MacBook Pro M5 24GB — SSH remote control only. No local compute.

**P-instance quota request pending** — if approved (96 vCPU for p4de.24xlarge),
we could rerun at 8nm with MACE large on A100 80GB. Spot price ~$14/hr.

## Model choice

**MACE-MP-0 medium** for MD. Configurable via `MACE_MODEL` env var.

Why medium, not large:
- MACE large OOMs on every GPU ≤48 GB for >6K atoms (neighbor list
  tensor products dominate, not model weights)
- Medium fits on L40S 46 GB for 5nm droplet (33.8 GB peak, 12 GB headroom)
- DFT fine-tuning matters more than model size for accuracy
- If P-instance quota is approved, can rerun with `MACE_MODEL=large
  DROPLET_DIAMETER_NM=8.0` on p4de.24xlarge (A100 80GB)

## Workflow

**Everything runs on ONE g6e.2xlarge instance.** GPU does MD, idle CPUs
do DFT in parallel during production. No multi-instance coordination needed.

```
Phase 1a: Build 5nm droplet + FIRE energy minimization + validate
          Duration: ~5 minutes
          Go/No-Go: max force < 10 eV/A after minimization          ✅ PASSED

Phase 1b: 0.5 ns equilibration → go/no-go checkpoint
          Duration: ~17 days (~1.5s/step, 1M steps)
          Cost: ~$450
          Go/No-Go: spread < 1.3, T ±20K, density ±20%, surface <cos θ> > 0

Phase 1c: 2 ns production MD (matching Hao et al. 2022 protocol)
          Duration: ~69 days
          Cost: ~$1,840
          DFT runs on CPUs in parallel (see below)

Phase 1d: PolarMACE E-field analysis
          Duration: ~17 minutes (single GPU)
          Go/No-Go: surface E-field enhancement > 0 MV/cm
          >>> FIRST PUBLISHABLE RESULT <<<

Phase 2:  DFT (runs on CPUs DURING Phase 1c — zero extra cost/time)
          300 clusters × revPBE-D3/def2-TZVP via PySCF
          2 parallel workers × 4 threads on 8 vCPU
          Duration: ~5 days (finishes long before Phase 1c)

Phase 3:  Fine-tune MACE medium on DFT data → 2.5 ns production MD
          Same instance, same GPU
          Duration: ~86 days
          Post-process with PolarMACE + fixed charges
          Compare off-the-shelf vs fine-tuned
```

Total wall time: ~3 months. Total cost: ~$2,300 at $1.11/hr.

## Key design decisions

### Why 5nm, not 8nm?
- 8nm (27K atoms) OOMs on ALL GPUs ≤48 GB with ALL MACE model sizes
- Single A100 80GB instances don't exist on AWS — p4de.24xlarge has 8× A100
  at $14/hr spot (~$3,400 for 2.5 ns), way over budget
- 5nm (6.3K atoms) fits on L40S with headroom
- Hao et al. show E-field is size-independent at R≥40 A; 5nm (R=25 A) is
  a novel prediction in an untested curvature regime

### Why 2.5 ns, not 20 ns?
- Hao et al. published in Nature Communications with 0.5 ns equil + 2 ns production
- Matching their protocol gives direct comparison
- 20 ns would take 1+ year on L40S with zero scientific justification

### Why co-located DFT on CPUs?
- GPU MD uses 100% GPU but only 12.5% CPU (1 core for neighbor lists)
- 7 idle cores + 58 GB idle RAM can run PySCF in parallel
- Phase 2 becomes free — no extra instances, no extra cost
- run_dft_pipeline.sh automates: waits for go/no-go → extract → DFT

### Why FIRE minimization before MD?
- Grid-placed waters have overlapping atoms (max force 5,161 eV/A)
- 210 FIRE steps relax to max force < 1 eV/A
- Without this, MD blows up immediately

## Configuration

All scripts read from environment variables with sensible defaults:

```bash
export DROPLET_DIAMETER_NM=5.0      # 5nm droplet (default)
export MACE_MODEL=medium             # MACE-MP-0 medium (default)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For A100 80GB runs (if P quota approved):
```bash
export DROPLET_DIAMETER_NM=8.0
export MACE_MODEL=large
```

## Current status

- Phase 1a: ✅ PASSED (5nm, 2109 waters, 6327 atoms, max force 1.0 eV/A)
- Phase 1b: 🔄 RUNNING on g6e.2xlarge (34.201.54.227)
- Go/no-go expected: ~April 28, 2026
- P-instance quota: PENDING (96 vCPU on-demand + spot)

## Key references

- Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022) — ReaxFF/C-GeM E-field
- Heindel, LaCour & Head-Gordon, Nat. Commun. 15, 3670 (2024) — charge + redox
- LaCour et al., JACS 147, 6299 (2025) — interfaces + charge review
- Batatia et al., JCP 163, 184110 (2025) — MACE-MP-0 foundation model paper
- Xia et al., PNAS 122, e2519491122 (2025) — curvature-dependent reactivity
