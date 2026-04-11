#!/bin/bash
# Setup for c6i.8xlarge (32 vCPU, 64GB RAM) — DFT with PySCF
# Run: bash setup_dft.sh
set -euo pipefail

echo "=== DFT instance setup (c6i.8xlarge) ==="

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

conda create -n dft python=3.11 -y
conda activate dft

pip install pyscf numpy scipy h5py ase torch

# PySCF extras: libxc for functionals, dftd3 for dispersion
pip install pyscf[all] pyscf-dftd3

python -c "
import pyscf
print(f'PySCF {pyscf.__version__}')
import multiprocessing
print(f'CPUs available: {multiprocessing.cpu_count()}')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate dft"
echo "Then run:      python run_dft.py --clusters-dir clusters/"
