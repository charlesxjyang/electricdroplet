"""
Lightweight intermediate E-field check during production MD.

Uses fixed SPC/E charges (CPU-only, no GPU needed) to compute the radial
E-field profile from the latest production frames. Gives a directional
signal on whether the surface E-field enhancement exists before the full
PolarMACE analysis in Phase 1d.

Appends results to analysis/intermediate_efield.log so we can track
convergence over time. Takes ~30-60 seconds per run.

Usage: python intermediate_efield_check.py [--n-frames 50]
"""
import argparse
import numpy as np
import time
import re
from pathlib import Path
from ase.io.trajectory import Trajectory

# Fixed SPC/E charges
Q_O = -0.8476  # e
Q_H = +0.4238  # e

# Physics constants
E_CHARGE_C = 1.602176634e-19
ANGSTROM_TO_M = 1e-10
K_COULOMB = 8.9875517873681764e9
VM_TO_MVCM = 1e-8

TRAJ_FILE = Path("phase1/trajectory.traj")
OUTPUT_DIR = Path("analysis")
LOG_FILE = OUTPUT_DIR / "intermediate_efield.log"
NBINS = 40
N_PROBES = 30  # fewer than full analysis (50) — speed over precision


def _fibonacci_sphere(n):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    y = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.clip(1 - y * y, 0.0, None))
    theta = phi * i
    return np.stack([r_xy * np.cos(theta), y, r_xy * np.sin(theta)], axis=-1)


def compute_efield(atoms, com, bin_edges):
    """Fixed-charge Coulomb E-field at radial shells."""
    symbols = atoms.get_chemical_symbols()
    charges_e = np.array([Q_O if s == 'O' else Q_H for s in symbols])
    positions_m = atoms.positions * ANGSTROM_TO_M
    com_m = com * ANGSTROM_TO_M
    charges_C = charges_e * E_CHARGE_C

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    field_mag = np.zeros(len(bin_centers))
    unit_probes = _fibonacci_sphere(N_PROBES)

    for bi, r_A in enumerate(bin_centers):
        if r_A < 0.5:
            continue
        r_m = r_A * ANGSTROM_TO_M
        probes_m = unit_probes * r_m + com_m

        shell_fields = np.zeros(N_PROBES)
        for pi, p in enumerate(probes_m):
            dr = p - positions_m
            r = np.linalg.norm(dr, axis=1)
            # Exclude atoms within 1 Å of probe — point-charge Coulomb
            # diverges at short range where the electron density is spread out
            mask = r > 1.0 * ANGSTROM_TO_M
            if mask.sum() == 0:
                continue
            e_vec = K_COULOMB * charges_C[mask, None] * dr[mask] / (r[mask, None] ** 3)
            shell_fields[pi] = np.linalg.norm(e_vec.sum(axis=0))

        field_mag[bi] = shell_fields.mean()

    return bin_centers, field_mag * VM_TO_MVCM


def get_equil_frames():
    """Read equilibration frame count from go/no-go report."""
    gng = Path("phase1/go_nogo_report.txt")
    if gng.exists():
        m = re.search(r'Equilibration time:\s+([\d.]+)\s+ns', gng.read_text())
        if m:
            return int(float(m.group(1)) * 1e3 / 0.1) + 100
    return 2500


def main(n_frames=50):
    OUTPUT_DIR.mkdir(exist_ok=True)

    traj = Trajectory(str(TRAJ_FILE))
    n_total = len(traj)
    equil_frames = get_equil_frames()
    prod_frames = list(range(equil_frames, n_total))

    if len(prod_frames) < 10:
        print(f"Only {len(prod_frames)} production frames — too few. Skipping.")
        return

    # Sample evenly across available production
    n_sample = min(n_frames, len(prod_frames))
    indices = np.linspace(0, len(prod_frames) - 1, n_sample, dtype=int)
    frame_list = [prod_frames[i] for i in indices]

    # Get geometry from last frame
    atoms_last = traj[frame_list[-1]]
    symbols = atoms_last.get_chemical_symbols()
    o_idx = [j for j, s in enumerate(symbols) if s == 'O']
    o_pos = atoms_last.positions[o_idx]
    com = o_pos.mean(axis=0)
    r90 = float(np.percentile(np.linalg.norm(o_pos - com, axis=1), 90))
    bin_edges = np.linspace(0, r90 + 10, NBINS + 1)

    print(f"Intermediate E-field check")
    print(f"  Trajectory: {n_total} frames, {len(prod_frames)} production")
    print(f"  Sampling {n_sample} frames, r90={r90:.1f} Å")

    t0 = time.time()
    all_fields = []
    for k, fi in enumerate(frame_list):
        atoms = traj[fi]
        o_pos_frame = atoms.positions[o_idx]
        com_frame = o_pos_frame.mean(axis=0)
        _, field = compute_efield(atoms, com_frame, bin_edges)
        all_fields.append(field)
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{n_sample} frames...")

    traj.close()
    elapsed = time.time() - t0

    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    field_mean = np.mean(all_fields, axis=0)
    field_std = np.std(all_fields, axis=0)

    surface_mask = (r > r90 - 5) & (r < r90 + 5)
    bulk_mask = r < r90 - 10

    surface_field = float(field_mean[surface_mask].mean()) if surface_mask.any() else 0
    bulk_field = float(field_mean[bulk_mask].mean()) if bulk_mask.any() else 0
    enhancement = surface_field - bulk_field

    # Estimate sim time covered
    sim_time_ps = len(prod_frames) * 0.1  # 0.1 ps per frame

    print(f"\n  Results ({elapsed:.0f}s, {n_sample} frames over {sim_time_ps:.0f} ps production):")
    print(f"    Bulk E-field:    {bulk_field:.2f} MV/cm")
    print(f"    Surface E-field: {surface_field:.2f} MV/cm")
    print(f"    Enhancement:     {enhancement:+.2f} MV/cm  (ref: ~9 MV/cm from Hao et al.)")
    print(f"    Surface std:     {field_std[surface_mask].mean():.2f} MV/cm")

    if enhancement > 0:
        print(f"    Signal: POSITIVE — surface field exceeds bulk ✓")
    else:
        print(f"    Signal: NEGATIVE — no enhancement yet (may need more production data)")

    # Append to running log
    import datetime
    entry = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"prod_frames={len(prod_frames):>6d} ({sim_time_ps:>7.1f} ps) | "
             f"sampled={n_sample:>3d} | "
             f"bulk={bulk_field:>6.2f} | surface={surface_field:>6.2f} | "
             f"enhancement={enhancement:>+6.2f} MV/cm\n")

    with open(LOG_FILE, 'a') as f:
        f.write(entry)
    print(f"\n  Appended to {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=50,
                        help="Number of production frames to sample")
    args = parser.parse_args()
    main(n_frames=args.n_frames)
