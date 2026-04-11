# PLAN.md — Context and Instructions for Claude Code

## What this project is

We're computing the radial electric field profile at the air-water interface
of an 8nm water nanodroplet (~9,000 water molecules, ~27,000 atoms) using
MACE-MP-0 machine learning molecular dynamics. This is the first MLIP-based
calculation of this quantity — the only prior work is Head-Gordon's 2022
Nat. Commun. paper using ReaxFF/C-GeM, which found ~16 MV/cm surface
electric field enhancement.

The result will be posted on ChemRxiv and submitted to JPC Letters (4-page
limit, no APC for subscription access).

## Hardware

- **Compute**: AWS g5.2xlarge spot instance (1× NVIDIA A10G 24GB, 8 vCPU, 32GB RAM, ~$0.40/hr spot)
- **DFT**: 4× AWS c6i.8xlarge spot instances (32 vCPU, 64GB RAM each, ~$0.45/hr spot)
- **Local**: MacBook Pro M5 24GB — used ONLY as remote control via SSH. No local compute.
- **Budget**: <$500 total AWS spend

## Model choice

**MACE-MP-0 large** for all phases. Not medium, not small.

Reasons:
- Large has more capacity to represent polarization effects at curved interfaces
- Fine-tuning a larger model on the same DFT data produces a strictly better result
- 24GB A10G VRAM fits the large model + 27K atoms (tight but OK)
- The accuracy difference between medium and large matters less than the DFT
  fine-tuning, but since we're paying for AWS anyway, there's no reason to
  use a smaller model

If large OOMs on the A10G, step up to g5.4xlarge (same GPU, more system RAM)
or p4d.24xlarge (A100 80GB, but much more expensive — only if needed).

## Workflow

Everything runs on AWS. The MacBook runs `run-everything.sh` which SSHes
into instances and executes commands remotely.

```
Phase 1: MACE large off-the-shelf → 20ns NVT MD → structural validation
         Instance: g5.2xlarge spot
         Duration: ~7-10 days
         Cost: ~$100
         Checks: density, orientational order, droplet integrity
         Checks: fixed-charge E-field sign (positive or negative?)
         Checks: PolarMACE E-field on sampled frames
         Output: off-the-shelf MACE + PolarMACE E-field profile
                 (publishable on ChemRxiv as-is)

Phase 2: Extract 300 interfacial clusters → DFT single points (revPBE-D3/TZV2P)
         Instance: 4× c6i.8xlarge spot
         Duration: ~1-2 days
         Cost: ~$70
         Also: extract Bader charges from DFT for comparison

Phase 3: Fine-tune MACE large on DFT data → 50ns production MD → analysis
         Instance: g5.2xlarge spot (same as Phase 1)
         Duration: ~15-20 days
         Cost: ~$200
         Post-process with PolarMACE charges (primary result)
         Post-process with fixed charges (for comparison)
         Compare off-the-shelf vs fine-tuned structural profiles
         Compare all charge models to C-GeM reference
         Output: refined E-field profile + comparison figures
```

Total wall time: ~4 weeks. Total cost: ~$370-500.

## Changes needed to the repo

The repo was originally written for MACE medium with a hybrid Mac/AWS
workflow. The following changes bring it to MACE large, AWS-only:

### 1. All Python scripts: model='medium' → model='large'

Files to change:
- `05-run-phase1.py`: the `get_calc()` or calculator setup function
- `03-gpu-userdata.sh`: the smoke test at the end

In every place where `mace_mp()` is called, use:
```python
mace_mp(model='large', dispersion=False, default_dtype='float32', device='cuda')
```

### 2. Fine-tuning script: foundation_model="large"

In `08-finetune-mace.sh`, the `mace_run_train` command should use:
```
--foundation_model="large"
```

Also update `--num_channels=256` (large uses 256, medium uses 128).

### 3. Device is always 'cuda', never 'mps'

Since nothing runs on the MacBook, remove any `device='mps'` references.
All compute uses `device='cuda'` on the A10G.

### 4. run-everything.sh is the entry point

This single script handles the entire lifecycle from the MacBook:
- `./run-everything.sh setup` — S3, keys, security group, IAM
- `./run-everything.sh phase1` — launch g5.2xlarge, install MACE, build droplet, start MD
- `./run-everything.sh check` — SSH in and read go/no-go report
- `./run-everything.sh phase2` — extract clusters, launch CPUs, run DFT
- `./run-everything.sh phase3` — fine-tune, start production MD
- `./run-everything.sh results` — download analysis + figures
- `./run-everything.sh teardown` — terminate everything

### 5. Go/no-go checkpoint

After 1ns of equilibration (~2-3 days), the Phase 1 script automatically
checks:
- Droplet integrity (spread ratio < 1.3 = spherical, > 1.5 = fragmented)
- Temperature (within ±20K of 300K)
- Energy per water (-0.3 to -0.6 eV typical)
- Interior density (~33.4 molecules/nm³ for bulk water)
- Orientational order: surface <cos θ> > 0 (OH points outward = correct)
- Bulk <cos θ> ≈ 0 (isotropic = correct)
- Simulation throughput (ns/day)

Prints a clear GO / NO-GO verdict with suggested fixes if problems.

### 6. Checkpointing for spot instance resilience

All MD scripts checkpoint every 2000 steps (1 ps). The g5.2xlarge spot
instance is set to `InstanceInterruptionBehavior: stop` (not terminate),
so the EBS volume persists across interruptions. On restart, the scripts
accept `--resume` to continue from the last checkpoint.

### 7. Two publishable results

The paper includes both:
1. Off-the-shelf MACE large E-field profile (from Phase 1)
2. DFT-fine-tuned MACE large E-field profile (from Phase 3)

If they agree → validates that the foundation model is already good enough
for this interface. If they diverge → interesting finding about when
fine-tuning matters. Either way it's a result.

### 8. Electric field calculation — PolarMACE

The analysis script (`analyze_efield.py`) computes E-fields using two
charge models on each trajectory frame:

1. **PolarMACE charges** (MACE-POLAR-1): ML-predicted per-atom charges
   that vary with local environment. Surface vs bulk water molecules get
   different charges, capturing the polarization physics.
   Requires: `mace-torch>=0.3.16` (develop branch) + `graph_longrange`
   (from `pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git`)

2. **Fixed SPC/E charges** (O: -0.8476e, H: +0.4238e): for comparison.
   Classical fixed-charge models get the sign of the surface potential
   wrong (negative vs positive) — this is the key finding of Hao et al.

Additionally computes:
- Radial density profile ρ(r) — structural validation
- Orientational order <cos θ>(r) — comparison with C-GeM

The comparison script (`compare_to_cgem.py`) generates publication figures
against digitized Hao et al. 2022 reference data.

## Key references

- Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022) — ReaxFF/C-GeM E-field
- Heindel, LaCour & Head-Gordon, Nat. Commun. 15, 3670 (2024) — charge + redox
- LaCour et al., JACS 147, 6299 (2025) — interfaces + charge review
- Batatia et al., JCP 163, 184110 (2025) — MACE-MP-0 foundation model paper
- Xia et al., PNAS 122, e2519491122 (2025) — curvature-dependent reactivity

## What Claude Code should do

When asked to update the repo:

1. Run `bash update-to-large.sh` if it exists, OR manually sed all
   `model='medium'` → `model='large'` and `model="medium"` → `model="large"`
   across .py and .sh files

2. Verify `run-everything.sh` exists and is executable

3. Verify no references to `device='mps'` remain in any AWS-targeted scripts

4. Verify `05-run-phase1.py` has the go/no-go checkpoint logic that fires
   after EQUIL_STEPS

5. Verify `08-finetune-mace.sh` uses `--foundation_model="large"`

That's it. The repo should then be ready to clone onto an EC2 instance
or run entirely via `run-everything.sh` from the MacBook.
