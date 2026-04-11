"""Build 8nm water droplet and run initial validation with MACE-MP-0 large."""
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

DIAMETER_NM = 8.0
OUTPUT = Path("droplet_initial.xyz")

def build_water_droplet(diameter_nm):
    radius_A = diameter_nm * 10.0 / 2.0
    oh_bond = 0.9572
    angle_rad = np.radians(104.52)
    density = 0.0334  # molecules per A^3
    spacing = (1.0 / density) ** (1/3)
    n_side = int(2 * radius_A / spacing) + 2

    positions = []
    symbols = []
    rng = np.random.default_rng(42)

    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                cx = (ix - n_side/2) * spacing
                cy = (iy - n_side/2) * spacing
                cz = (iz - n_side/2) * spacing

                if np.sqrt(cx**2 + cy**2 + cz**2) > radius_A:
                    continue

                phi = rng.uniform(0, 2*np.pi)
                cos_t = rng.uniform(-1, 1)
                sin_t = np.sqrt(1 - cos_t**2)

                positions.append([cx, cy, cz])
                symbols.append('O')

                h1x = cx + oh_bond * (np.sin(angle_rad/2) * np.cos(phi) * sin_t)
                h1y = cy + oh_bond * (np.sin(angle_rad/2) * np.sin(phi) * sin_t)
                h1z = cz + oh_bond * np.cos(angle_rad/2) * cos_t
                positions.append([h1x, h1y, h1z])
                symbols.append('H')

                h2x = cx - oh_bond * (np.sin(angle_rad/2) * np.cos(phi) * sin_t)
                h2y = cy - oh_bond * (np.sin(angle_rad/2) * np.sin(phi) * sin_t)
                h2z = cz + oh_bond * np.cos(angle_rad/2) * cos_t
                positions.append([h2x, h2y, h2z])
                symbols.append('H')

    positions = np.array(positions)
    positions -= positions.mean(axis=0)
    n_water = len(symbols) // 3

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[2*radius_A + 30]*3,
        pbc=False
    )
    return atoms, n_water


print("Building 8nm water droplet...")
atoms, n_water = build_water_droplet(DIAMETER_NM)
write(str(OUTPUT), atoms)

print(f"  Water molecules: {n_water}")
print(f"  Total atoms:     {len(atoms)}")
print(f"  Saved to:        {OUTPUT}")
print()

print("Validating with MACE-MP-0 large...")
import torch
from mace.calculators import mace_mp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

calc = mace_mp(model='large', dispersion=False,
               default_dtype='float32', device=device)
atoms.calc = calc

e = atoms.get_potential_energy()
f = atoms.get_forces()
f_norms = np.linalg.norm(f, axis=1)

print(f"  Energy:          {e:.2f} eV ({e/n_water:.4f} eV/water)")
print(f"  Max force:       {np.max(f_norms):.4f} eV/A")
print(f"  Mean force:      {np.mean(f_norms):.4f} eV/A")
print()

e_per_water = e / n_water
if -0.8 < e_per_water < -0.2:
    print("  Energy per water looks reasonable. Safe to proceed.")
else:
    print(f"  Energy per water ({e_per_water:.3f} eV) seems off.")
    print("  Try a short energy minimization before MD.")
