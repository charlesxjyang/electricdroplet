# Water Microdroplet Electric Field Simulation

Ab initio-quality MD simulation of a water nanodroplet using MACE-MP-0, computing the radial electric field at the air-water interface via PolarMACE charge analysis.

## Architecture

Everything runs on **one g6e.2xlarge instance** (L40S 46GB GPU, 8 vCPU, 64GB RAM):
- GPU runs MACE MD + PolarMACE analysis
- Idle CPUs run DFT in parallel during production MD

## Quick Start

```bash
# On g6e.2xlarge
git clone https://github.com/charlesxjyang/electricdroplet.git && cd electricdroplet
bash setup_gpu.sh
conda activate mace

# Phase 1: Build + 2.5 ns MD (all-in-one)
bash run_phase1_full.sh
# Check progress: cat phase1_status.txt
# Go/no-go report: cat phase1/go_nogo_report.txt

# Phase 2: DFT on idle CPUs (start after go/no-go passes)
pip install pyscf pyscf-dftd3
nohup bash run_dft_pipeline.sh > dft_pipeline.log 2>&1 &

# Phase 3: Fine-tune + production MD
python finetune_mace.py --dft-dir dft_results/
python run_phase3.py --model mace_droplet.model

# Analysis
python analyze_efield.py --trajectory phase3/trajectory.traj
python compare_to_cgem.py
```

## Configuration

```bash
# Default: 5nm droplet, MACE medium (fits L40S 46GB)
export DROPLET_DIAMETER_NM=5.0
export MACE_MODEL=medium

# For A100 80GB (p4de.24xlarge): 8nm droplet, MACE large
export DROPLET_DIAMETER_NM=8.0
export MACE_MODEL=large
```

## Pipeline

| Step | What | Hardware | Duration | Go/No-Go |
|------|------|----------|----------|----------|
| 1a | Build droplet + FIRE minimize | GPU | 5 min | Max force < 10 eV/A |
| 1b | 0.5 ns equilibration | GPU | ~17 days | Spread, temp, density, orientation |
| 1c | 2 ns production MD | GPU | ~69 days | — |
| 1d | PolarMACE E-field analysis | GPU | ~17 min | E-field enhancement > 0 MV/cm |
| 2 | DFT on 300 clusters | CPU (parallel with 1c) | ~5 days | >90% SCF converged |
| 3 | Fine-tune + 2.5 ns MD + analysis | GPU | ~86 days | Compare to Phase 1 |

## Scripts

1. **build_droplet.py** — Build spherical water droplet + FIRE energy minimization + MACE validation
2. **run_phase1.py** — NVT Langevin MD at 300K with automatic go/no-go after equilibration
3. **run_phase1_full.sh** — Orchestrates Phase 1: build → MD → PolarMACE analysis → comparison figures
4. **run_dft_pipeline.sh** — Watches for go/no-go, then runs cluster extraction + DFT on idle CPUs
5. **extract_clusters.py** — Samples frames from trajectory, extracts local water clusters
6. **run_dft.py** — PySCF DFT single-points (revPBE-D3/def2-TZVP)
7. **finetune_mace.py** — Fine-tunes MACE on DFT data
8. **run_phase3.py** — Production MD with fine-tuned model
9. **analyze_efield.py** — Dual E-field analysis (PolarMACE + fixed charges), density, orientational order
10. **compare_to_cgem.py** — Publication figures vs Hao et al. 2022

## Go/No-Go Checkpoint

After 0.5 ns equilibration, `run_phase1.py` checks:
- Droplet integrity (spread ratio < 1.3)
- Temperature (±20K of 300K)
- Density (±20% of 33.4 mol/nm³)
- Orientational order (surface `<cos θ>` > 0 = OH outward)
- Surface potential sign from fixed charges
- Performance (ns/day) and ETA

Report saved to `phase1/go_nogo_report.txt`.

## Reference Data

`reference_data/hao2022_digitized.py` contains benchmark values from Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022). Hard numbers from the paper text are exact; digitized figure values are approximate and should be refined with [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) before submission.
