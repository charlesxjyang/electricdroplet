#!/bin/bash
# Setup for g5.2xlarge (NVIDIA A10G, 24GB VRAM, 32GB RAM)
# Run: bash setup_gpu.sh
set -euo pipefail

echo "=== GPU instance setup (g5.2xlarge) ==="

# Install miniforge if needed
if ! command -v conda &>/dev/null; then
    echo "Installing miniforge..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
    conda init bash
    rm /tmp/miniforge.sh
else
    eval "$(conda shell.bash hook)"
fi

# Create environment
conda create -n mace python=3.11 -y
conda activate mace

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MACE (develop branch for PolarMACE support) + dependencies
pip install git+https://github.com/ACEsuit/mace.git@develop
pip install ase numpy scipy matplotlib h5py

# PolarMACE long-range electrostatics dependency
pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git

# Verify CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Verify MACE + PolarMACE
python -c "
from mace.calculators import mace_mp, mace_polar
print('MACE-MP-0 large: OK')
print('PolarMACE: OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate mace"
echo "Then run:      python build_droplet.py"
