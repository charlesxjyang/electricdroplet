"""
Analyze electric field profiles from MD trajectory.

Two charge models computed on each frame:
  1. PolarMACE — ML-predicted environment-dependent per-atom charges
  2. Fixed SPC/E charges (O: -0.8476e, H: +0.4238e) — for comparison

Also computes:
  - Radial density profile rho(r)
  - Orientational order parameter <cos theta>(r)

Usage:
  python analyze_efield.py --trajectory phase3/trajectory.traj
  python analyze_efield.py --trajectory phase1/trajectory.traj --n-frames 50
  python analyze_efield.py --trajectory phase1/trajectory.traj --skip-polar  # fixed charges only
"""
import argparse
import numpy as np
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io.trajectory import Trajectory

# Physical constants
E_CHARGE_C = 1.602176634e-19
ANGSTROM_TO_M = 1e-10
K_COULOMB = 8.9875517873681764e9  # N m^2 / C^2

# Fixed SPC/E charges
Q_O_FIXED = -0.8476  # e
Q_H_FIXED = +0.4238  # e

OUTPUT_DIR = Path("analysis")
NBINS = 50


def get_polar_calc(device='cuda'):
    """Load PolarMACE calculator."""
    from mace.calculators import mace_polar
    return mace_polar(model='polar-1-s', device=device, default_dtype='float32')


def get_charges_polarmace(atoms, calc):
    """Run PolarMACE on a frame and return per-atom charges in units of e.

    Verifies charge conservation globally (droplet neutrality) and per-water
    (atoms assumed to be in O, H, H order per molecule from build_droplet.py).
    Small numerical drift accumulated over thousands of atoms can produce a
    spurious net charge that dominates the radial E-field, so we fail loud
    rather than let it through silently.
    """
    atoms_copy = atoms.copy()
    atoms_copy.info['charge'] = 0
    atoms_copy.info['spin'] = 1
    atoms_copy.info['external_field'] = [0.0, 0.0, 0.0]
    atoms_copy.calc = calc
    atoms_copy.get_potential_energy()
    charges = calc.results['charges'].copy()

    # Conservation checks
    total = float(charges.sum())
    if abs(total) > 0.01:
        raise ValueError(
            f"PolarMACE total charge violates neutrality: {total:.4f} e "
            f"(threshold 0.01). A 0.01 e excess over {len(atoms)} atoms "
            "changes the macroscopic droplet field measurably."
        )

    if len(atoms) % 3 == 0:
        per_water = charges.reshape(-1, 3).sum(axis=1)
        worst = float(np.abs(per_water).max())
        if worst > 0.01:
            raise ValueError(
                f"PolarMACE per-water charge conservation violated: "
                f"worst |sum| = {worst:.4f} e on some molecule "
                "(expected 0 for neutral waters; check O,H,H ordering)."
            )

    return charges


def _polar_worker(args):
    """Worker for multi-GPU PolarMACE. Each process owns one GPU."""
    gpu_id, frame_indices, traj_path, bin_edges = args
    device = f'cuda:{gpu_id}'
    calc = get_polar_calc(device=device)
    traj = Trajectory(str(traj_path))

    results = []
    for fi in frame_indices:
        atoms = traj[fi]
        symbols = atoms.get_chemical_symbols()
        o_idx = [j for j, s in enumerate(symbols) if s == 'O']
        com = atoms.positions[o_idx].mean(axis=0)

        charges = get_charges_polarmace(atoms, calc)
        _, efield = compute_efield_radial(atoms, charges, com, bin_edges)

        o_charges = charges[[j for j, s in enumerate(symbols) if s == 'O']]
        h_charges = charges[[j for j, s in enumerate(symbols) if s == 'H']]
        results.append((efield, o_charges, h_charges))

    traj.close()
    return results


def get_charges_fixed(atoms):
    """Return fixed SPC/E charges for each atom."""
    return np.array([Q_O_FIXED if s == 'O' else Q_H_FIXED
                     for s in atoms.get_chemical_symbols()])


def _fibonacci_sphere(n):
    """Quasi-uniform unit vectors on a sphere via golden-angle spiral.

    Deterministic, no RNG dependence. Convergence is ~n^{-3/2} for smooth
    integrands — much faster than random sampling's n^{-1/2}. At n=50, an
    order-l=5 spherical harmonic is integrated to ~1e-4 precision, which
    is ~20× more than random with the same n.
    """
    phi_golden = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    y = 1.0 - 2.0 * i / max(n - 1, 1)
    r_xy = np.sqrt(np.clip(1 - y * y, 0.0, None))
    theta = phi_golden * i
    return np.stack([r_xy * np.cos(theta), y, r_xy * np.sin(theta)], axis=-1)


def compute_efield_radial(atoms, charges_e, com, bin_edges, n_probes=50):
    """
    Compute radially-averaged E-field magnitude from point charges.

    For each radial shell, place Fibonacci-spiral probe points on a sphere
    and compute the Coulomb E-field from all atomic charges. Returns field
    in V/m.
    """
    positions_m = atoms.positions * ANGSTROM_TO_M
    com_m = com * ANGSTROM_TO_M
    charges_C = charges_e * E_CHARGE_C

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    field_mag = np.zeros(len(bin_centers))

    unit_probes = _fibonacci_sphere(n_probes)  # shape (n_probes, 3)

    for bi, r_A in enumerate(bin_centers):
        if r_A < 0.5:
            continue
        r_m = r_A * ANGSTROM_TO_M
        probes_m = unit_probes * r_m + com_m

        shell_fields = np.zeros(n_probes)
        for pi, p in enumerate(probes_m):
            dr = p - positions_m
            r = np.linalg.norm(dr, axis=1)
            r = np.maximum(r, 1e-12)
            e_vec = K_COULOMB * charges_C[:, None] * dr / (r[:, None] ** 3)
            shell_fields[pi] = np.linalg.norm(e_vec.sum(axis=0))

        field_mag[bi] = shell_fields.mean()

    return bin_centers, field_mag


def compute_density_profile(atoms, com, bin_edges):
    """Compute radial number density of oxygen atoms (molecules/nm^3)."""
    symbols = atoms.get_chemical_symbols()
    o_pos = atoms.positions[[i for i, s in enumerate(symbols) if s == 'O']]
    radii = np.linalg.norm(o_pos - com, axis=1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts, _ = np.histogram(radii, bins=bin_edges)

    # Shell volumes in nm^3
    r_lo = bin_edges[:-1] * 0.1  # A -> nm
    r_hi = bin_edges[1:] * 0.1
    shell_vol = 4/3 * np.pi * (r_hi**3 - r_lo**3)
    shell_vol = np.maximum(shell_vol, 1e-30)

    density = counts / shell_vol  # molecules/nm^3
    return bin_centers, density


def compute_orientational_order_profile(atoms, com, bin_edges):
    """
    Compute <cos theta>(r) where theta is the angle between OH bond vector
    and the radial outward direction from COM.
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cos_sums = np.zeros(len(bin_centers))
    cos_counts = np.zeros(len(bin_centers))

    for i, s in enumerate(symbols):
        if s != 'O':
            continue
        o_pos = positions[i]
        r_vec = o_pos - com
        r_dist = np.linalg.norm(r_vec)
        if r_dist < 1e-6:
            continue
        r_hat = r_vec / r_dist

        bi = np.searchsorted(bin_edges, r_dist) - 1
        if bi < 0 or bi >= len(bin_centers):
            continue

        for h_idx in [i + 1, i + 2]:
            if h_idx >= len(symbols) or symbols[h_idx] != 'H':
                continue
            oh_vec = positions[h_idx] - o_pos
            oh_len = np.linalg.norm(oh_vec)
            if oh_len < 1e-6:
                continue
            cos_theta = np.dot(oh_vec / oh_len, r_hat)
            cos_sums[bi] += cos_theta
            cos_counts[bi] += 1

    mask = cos_counts > 0
    cos_avg = np.zeros(len(bin_centers))
    cos_avg[mask] = cos_sums[mask] / cos_counts[mask]
    return bin_centers, cos_avg


def main(traj_path, n_frames=100, skip_polar=False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    traj_path = Path(traj_path)

    print(f"Loading trajectory: {traj_path}")
    traj = Trajectory(str(traj_path))
    n_total = len(traj)
    print(f"  Total frames: {n_total}")

    frame_indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
    print(f"  Analyzing {len(frame_indices)} frames")

    # Determine droplet geometry from last frame
    atoms_last = traj[-1]
    symbols = atoms_last.get_chemical_symbols()
    o_idx = [j for j, s in enumerate(symbols) if s == 'O']
    o_pos = atoms_last.positions[o_idx]
    com = o_pos.mean(axis=0)
    r90 = np.percentile(np.linalg.norm(o_pos - com, axis=1), 90)
    r_max = r90 + 10.0

    bin_edges = np.linspace(0, r_max, NBINS + 1)

    # Detect GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Accumulate results
    all_efield_polar = []
    all_efield_fixed = []
    all_density = []
    all_orient = []
    all_polar_charges_o = []
    all_polar_charges_h = []

    # --- Fixed charges + density + orientational order (CPU, sequential) ---
    t0 = time.time()
    for idx, fi in enumerate(frame_indices):
        atoms = traj[fi]
        symbols = atoms.get_chemical_symbols()
        o_idx_frame = [j for j, s in enumerate(symbols) if s == 'O']
        com_frame = atoms.positions[o_idx_frame].mean(axis=0)

        charges_fixed = get_charges_fixed(atoms)
        r, efield_fixed = compute_efield_radial(atoms, charges_fixed, com_frame, bin_edges)
        all_efield_fixed.append(efield_fixed)

        _, density = compute_density_profile(atoms, com_frame, bin_edges)
        all_density.append(density)

        _, orient = compute_orientational_order_profile(atoms, com_frame, bin_edges)
        all_orient.append(orient)

        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"    Fixed charges: [{idx+1}/{len(frame_indices)}] {(idx+1)/elapsed:.1f} frames/s")

    traj.close()
    print(f"  Fixed charges + structure done in {time.time()-t0:.0f}s")

    # --- PolarMACE charges (GPU, multi-GPU if available) ---
    if not skip_polar and n_gpus > 0:
        from concurrent.futures import ProcessPoolExecutor
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        use_gpus = min(n_gpus, len(frame_indices))
        print(f"  Running PolarMACE on {use_gpus} GPU(s)...")

        # Split frames across GPUs
        chunks = [[] for _ in range(use_gpus)]
        for i, fi in enumerate(frame_indices):
            chunks[i % use_gpus].append(fi)

        worker_args = [(gpu_id, chunk, traj_path, bin_edges)
                       for gpu_id, chunk in enumerate(chunks) if chunk]

        t1 = time.time()
        if use_gpus == 1:
            # Single GPU — run in-process to avoid spawn overhead
            all_results = [_polar_worker(worker_args[0])]
        else:
            with ProcessPoolExecutor(max_workers=use_gpus) as pool:
                all_results = list(pool.map(_polar_worker, worker_args))

        # Collect results in frame order
        flat_results = []
        for gpu_results in all_results:
            flat_results.extend(gpu_results)

        for efield, o_ch, h_ch in flat_results:
            all_efield_polar.append(efield)
            all_polar_charges_o.extend(o_ch)
            all_polar_charges_h.extend(h_ch)

        print(f"  PolarMACE done in {time.time()-t1:.0f}s ({use_gpus} GPUs)")
    elif not skip_polar:
        print(f"  No GPU available, skipping PolarMACE")

    # Aggregate
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    efield_fixed_mean = np.mean(all_efield_fixed, axis=0)
    efield_fixed_std = np.std(all_efield_fixed, axis=0)
    density_mean = np.mean(all_density, axis=0)
    orient_mean = np.mean(all_orient, axis=0)

    # Convert E-field from V/m to MV/cm (1 MV/cm = 1e8 V/m)
    VM_TO_MVCM = 1e-8
    efield_fixed_mean_mvcm = efield_fixed_mean * VM_TO_MVCM
    efield_fixed_std_mvcm = efield_fixed_std * VM_TO_MVCM

    results = {
        'r_angstrom': r,
        'r90_angstrom': r90,
        'efield_fixed_mean_mvcm': efield_fixed_mean_mvcm,
        'efield_fixed_std_mvcm': efield_fixed_std_mvcm,
        'density_mean_per_nm3': density_mean,
        'orient_cos_mean': orient_mean,
    }

    if all_efield_polar:
        efield_polar_mean = np.mean(all_efield_polar, axis=0)
        efield_polar_std = np.std(all_efield_polar, axis=0)
        efield_polar_mean_mvcm = efield_polar_mean * VM_TO_MVCM
        efield_polar_std_mvcm = efield_polar_std * VM_TO_MVCM
        results['efield_polar_mean_mvcm'] = efield_polar_mean_mvcm
        results['efield_polar_std_mvcm'] = efield_polar_std_mvcm
        results['polar_charges_o_mean'] = np.mean(all_polar_charges_o)
        results['polar_charges_o_std'] = np.std(all_polar_charges_o)
        results['polar_charges_h_mean'] = np.mean(all_polar_charges_h)
        results['polar_charges_h_std'] = np.std(all_polar_charges_h)

    np.savez(OUTPUT_DIR / "efield_analysis.npz", **results)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: E-field profiles
    ax = axes[0, 0]
    ax.plot(r, efield_fixed_mean_mvcm, 'b-', linewidth=2, label='Fixed SPC/E charges')
    ax.fill_between(r, efield_fixed_mean_mvcm - efield_fixed_std_mvcm,
                    efield_fixed_mean_mvcm + efield_fixed_std_mvcm, alpha=0.2, color='blue')
    if all_efield_polar:
        ax.plot(r, efield_polar_mean_mvcm, 'r-', linewidth=2, label='PolarMACE charges')
        ax.fill_between(r, efield_polar_mean_mvcm - efield_polar_std_mvcm,
                        efield_polar_mean_mvcm + efield_polar_std_mvcm, alpha=0.2, color='red')
    ax.axvline(r90, color='gray', linestyle='--', alpha=0.5, label=f'Surface (~{r90:.0f} A)')
    ax.axhline(16.0, color='green', linestyle=':', alpha=0.7, label='C-GeM ref: 16 MV/cm')
    ax.set_xlabel('Distance from center (A)')
    ax.set_ylabel('E-field magnitude (MV/cm)')
    ax.set_title('Radial Electric Field Profile')
    ax.legend(fontsize=9)
    ax.set_xlim(0, r_max)
    ax.grid(True, alpha=0.3)

    # Plot 2: Density profile
    ax = axes[0, 1]
    ax.plot(r, density_mean, 'k-', linewidth=2)
    ax.axhline(33.4, color='blue', linestyle=':', alpha=0.5, label='Bulk water (33.4/nm3)')
    ax.axvline(r90, color='gray', linestyle='--', alpha=0.5, label=f'Surface')
    ax.set_xlabel('Distance from center (A)')
    ax.set_ylabel('Density (molecules/nm3)')
    ax.set_title('Radial Density Profile')
    ax.legend(fontsize=9)
    ax.set_xlim(0, r_max)
    ax.grid(True, alpha=0.3)

    # Plot 3: Orientational order parameter
    ax = axes[1, 0]
    ax.plot(r, orient_mean, 'k-', linewidth=2)
    ax.axhline(0.0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(r90, color='gray', linestyle='--', alpha=0.5, label=f'Surface')
    ax.set_xlabel('Distance from center (A)')
    ax.set_ylabel('<cos theta>')
    ax.set_title('Orientational Order Parameter')
    ax.legend(fontsize=9)
    ax.set_xlim(0, r_max)
    ax.grid(True, alpha=0.3)

    # Plot 4: Charge distribution (PolarMACE) or summary stats
    ax = axes[1, 1]
    if all_polar_charges_o:
        ax.hist(all_polar_charges_o, bins=50, alpha=0.7, color='red', label='O charges', density=True)
        ax.hist(all_polar_charges_h, bins=50, alpha=0.7, color='blue', label='H charges', density=True)
        ax.axvline(Q_O_FIXED, color='red', linestyle=':', label=f'SPC/E O ({Q_O_FIXED})')
        ax.axvline(Q_H_FIXED, color='blue', linestyle=':', label=f'SPC/E H ({Q_H_FIXED})')
        ax.set_xlabel('Charge (e)')
        ax.set_ylabel('Probability density')
        ax.set_title('PolarMACE Charge Distribution')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'PolarMACE disabled\n(--skip-polar)',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title('PolarMACE Charges')
    ax.grid(True, alpha=0.3)

    fig.suptitle('8nm Water Microdroplet — Electric Field Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efield_analysis.png", dpi=150, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "efield_analysis.pdf", bbox_inches='tight')
    plt.close()

    # --- Print summary ---
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  efield_analysis.npz")
    print(f"  efield_analysis.png")
    print(f"  efield_analysis.pdf")

    surface_mask = (r > r90 - 5) & (r < r90 + 5)
    bulk_mask = r < r90 - 10

    print(f"\nSummary:")
    print(f"  Droplet radius (90th pct): {r90:.1f} A")

    sf = efield_fixed_mean_mvcm[surface_mask].mean() if surface_mask.any() else 0
    bf = efield_fixed_mean_mvcm[bulk_mask].mean() if bulk_mask.any() else 0
    print(f"  Fixed-charge E-field:")
    print(f"    Bulk:    {bf:.2f} MV/cm")
    print(f"    Surface: {sf:.2f} MV/cm")
    print(f"    Enhancement: {sf - bf:.2f} MV/cm (ref: 16 MV/cm)")

    if all_efield_polar:
        sp = efield_polar_mean_mvcm[surface_mask].mean() if surface_mask.any() else 0
        bp = efield_polar_mean_mvcm[bulk_mask].mean() if bulk_mask.any() else 0
        print(f"  PolarMACE E-field:")
        print(f"    Bulk:    {bp:.2f} MV/cm")
        print(f"    Surface: {sp:.2f} MV/cm")
        print(f"    Enhancement: {sp - bp:.2f} MV/cm (ref: 16 MV/cm)")
        print(f"  PolarMACE charges:")
        print(f"    O: {results['polar_charges_o_mean']:.4f} +/- {results['polar_charges_o_std']:.4f} e")
        print(f"    H: {results['polar_charges_h_mean']:.4f} +/- {results['polar_charges_h_std']:.4f} e")

    so = orient_mean[surface_mask].mean() if surface_mask.any() else 0
    bo = orient_mean[bulk_mask].mean() if bulk_mask.any() else 0
    print(f"  Orientational order:")
    print(f"    Bulk <cos theta>:    {bo:+.3f}")
    print(f"    Surface <cos theta>: {so:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="phase3/trajectory.traj")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--skip-polar", action="store_true",
                        help="Skip PolarMACE, use fixed charges only")
    args = parser.parse_args()
    main(args.trajectory, args.n_frames, args.skip_polar)
