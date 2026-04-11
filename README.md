# Water Microdroplet Electric Field Simulation

Ab initio-quality MD simulation of an 8nm water microdroplet using MACE-MP-0, studying electric fields at the droplet surface and interior.

## Infrastructure

| Phase | Instance | Hardware | Purpose |
|-------|----------|----------|---------|
| 1 (MD) | g5.2xlarge | A10G 24GB VRAM, 32GB RAM | 20ns preliminary MD |
| 2 (DFT) | 4x c6i.8xlarge | 32 vCPU, 64GB RAM each | DFT single-points on clusters |
| 3 (MD) | g5.2xlarge | A10G 24GB VRAM, 32GB RAM | Fine-tune + 50ns production MD |

## Prerequisites

- An S3 bucket for data transfer between instances
- AWS CLI configured (`aws configure`) on all instances
- Edit `s3_config.py` to set your bucket name

## Quick Start

### Phase 1 — GPU instance (g5.2xlarge)

```bash
git clone <this-repo> && cd electricdroplet
bash setup_gpu.sh
conda activate mace

# Edit S3 bucket name
vi s3_config.py   # set BUCKET = "s3://your-bucket/electricdroplet"

# Build droplet + run 20ns MD
python build_droplet.py
python run_phase1.py              # ~1-2 weeks on A10G
python run_phase1.py --resume     # if interrupted

# Extract clusters -> auto-uploads to S3
python extract_clusters.py --n-clusters 300
```

### Phase 2 — DFT instances (4x c6i.8xlarge)

```bash
git clone <this-repo> && cd electricdroplet
bash setup_dft.sh
conda activate dft
vi s3_config.py   # same bucket name

# Each instance runs a shard (auto-downloads clusters from S3)
python run_dft.py --start 0 --end 75      # instance 1
python run_dft.py --start 75 --end 150    # instance 2
python run_dft.py --start 150 --end 225   # instance 3
python run_dft.py --start 225 --end 300   # instance 4
# Results auto-upload to S3 when done
```

### Phase 3 — Back on GPU instance

```bash
# Fine-tune (auto-downloads DFT results from S3)
python finetune_mace.py --dft-dir dft_results/

# 50ns production MD with fine-tuned model
python run_phase3.py --model mace_droplet.model
python run_phase3.py --model mace_droplet.model --resume

# Analysis
python analyze_efield.py --trajectory phase3/trajectory.traj
```

## Data Flow

```
GPU (g5.2xlarge)                S3 Bucket                   CPU (4x c6i.8xlarge)
────────────────                ─────────                   ────────────────────
build_droplet.py
run_phase1.py
extract_clusters.py
  clusters/ ──────────────>  /clusters/  ──────────────>  clusters/
                                                          run_dft.py
                             /dft_results/ <────────────  dft_results/
finetune_mace.py <────────
  mace_droplet.model ────>  /models/
run_phase3.py
analyze_efield.py
```

## Pipeline Overview

1. **build_droplet.py** — Generates 8nm spherical water droplet (~9000 waters, ~27K atoms) and validates with MACE
2. **run_phase1.py** — 20ns NVT Langevin MD at 300K with MACE-MP-0 large. Includes automatic go/no-go checkpoint after 1ns equilibration
3. **extract_clusters.py** — Samples frames from production trajectory, extracts local water clusters for DFT
4. **run_dft.py** — PySCF DFT single-points (PBE/def2-SVP) on clusters. Parallelized across cores, shardable across instances
5. **finetune_mace.py** — Fine-tunes MACE-MP-0 large on DFT data for improved accuracy on the droplet system
6. **run_phase3.py** — 50ns production MD with the fine-tuned model
7. **analyze_efield.py** — Computes radial electric field profiles using point-charge model, generates plots

## Go/No-Go Checkpoint

Phase 1 automatically runs a diagnostic after 1ns equilibration, checking:
- Droplet integrity (is it still spherical or fragmenting?)
- Temperature stability
- Energy per water molecule
- Density vs bulk water
- Performance (ns/day) and ETA

Results are printed and saved to `phase1/go_nogo_report.txt`.

## Checkpointing

Both MD phases save checkpoints every 1ps. Resume any interrupted run with `--resume`. Trajectory files are append-mode safe.
