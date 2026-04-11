"""
Analyze electric field profiles from Phase 3 production trajectory.

Computes:
  - Radial electric field profile E(r) from oxygen-hydrogen charge distribution
  - Time-averaged and instantaneous field maps
  - Surface vs bulk field comparison

Usage: python analyze_efield.py [--trajectory phase3/trajectory.traj]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io.trajectory import Trajectory

# Partial charges (SPC/E-like, reasonable for field estimation)
Q_O = -0.8476  # e
Q_H = +0.4238  # e
E_CHARGE = 1.602176634e-19       # C
BOHR_TO_M = 5.29177210903e-11
ANGSTROM_TO_M = 1e-10
K_COULOMB = 8.9875517873681764e9  # N m^2 / C^2
V_PER_M_TO_V_PER_NM = 1e-9

OUTPUT_DIR = Path("analysis")


def compute_efield_at_points(atoms, probe_points):
    """
    Compute electric field at probe points from point-charge model.
    Returns field vectors in V/nm at each probe point.
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions * ANGSTROM_TO_M  # to meters
    probe_m = probe_points * ANGSTROM_TO_M

    charges = np.array([Q_O if s == 'O' else Q_H for s in symbols])
    charges_C = charges * E_CHARGE

    fields = np.zeros((len(probe_m), 3))
    for i, p in enumerate(probe_m):
        dr = p - positions  # (N, 3)
        r = np.linalg.norm(dr, axis=1)
        r = np.maximum(r, 1e-12)  # avoid division by zero
        # E = k * q * r_hat / r^2
        e_contrib = K_COULOMB * charges_C[:, None] * dr / (r[:, None] ** 3)
        fields[i] = e_contrib.sum(axis=0)

    # Convert V/m to V/nm
    return fields * V_PER_M_TO_V_PER_NM


def radial_field_profile(atoms, n_bins=50, r_max=None):
    """Compute radially-averaged electric field magnitude as function of distance from COM."""
    symbols = atoms.get_chemical_symbols()
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    o_pos = atoms.positions[o_indices]
    com = o_pos.mean(axis=0)

    if r_max is None:
        r_max = np.max(np.linalg.norm(o_pos - com, axis=1)) + 2.0

    # Probe points along radial shells
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    field_magnitudes = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    # Sample probe points on each shell
    n_probes_per_shell = 50
    rng = np.random.default_rng(0)

    for bi in range(n_bins):
        r = bin_centers[bi]
        if r < 0.5:
            continue

        # Random points on sphere of radius r
        phi = rng.uniform(0, 2 * np.pi, n_probes_per_shell)
        cos_theta = rng.uniform(-1, 1, n_probes_per_shell)
        sin_theta = np.sqrt(1 - cos_theta**2)

        probes = np.column_stack([
            r * sin_theta * np.cos(phi),
            r * sin_theta * np.sin(phi),
            r * cos_theta,
        ]) + com

        fields = compute_efield_at_points(atoms, probes)
        field_mags = np.linalg.norm(fields, axis=1)

        field_magnitudes[bi] = field_mags.mean()
        counts[bi] = n_probes_per_shell

    return bin_centers, field_magnitudes


def main(traj_path, n_frames=100):
    OUTPUT_DIR.mkdir(exist_ok=True)
    traj_path = Path(traj_path)

    print(f"Loading trajectory: {traj_path}")
    traj = Trajectory(str(traj_path))
    n_total = len(traj)
    print(f"  Total frames: {n_total}")

    # Sample evenly across trajectory
    frame_indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
    print(f"  Analyzing {len(frame_indices)} frames...")

    all_profiles = []
    for i, fi in enumerate(frame_indices):
        atoms = traj[fi]
        r, E_r = radial_field_profile(atoms)
        all_profiles.append(E_r)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(frame_indices)}")

    traj.close()

    profiles = np.array(all_profiles)
    mean_profile = profiles.mean(axis=0)
    std_profile = profiles.std(axis=0)

    # Save data
    np.savez(
        OUTPUT_DIR / "efield_radial.npz",
        r_angstrom=r,
        mean_field_v_per_nm=mean_profile,
        std_field_v_per_nm=std_profile,
        all_profiles=profiles,
    )

    # Get droplet radius for annotation
    atoms_last = traj[-1] if hasattr(traj, '__getitem__') else atoms
    symbols = atoms.get_chemical_symbols()
    o_idx = [j for j, s in enumerate(symbols) if s == 'O']
    o_pos = atoms.positions[o_idx]
    com = o_pos.mean(axis=0)
    r90 = np.percentile(np.linalg.norm(o_pos - com, axis=1), 90)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, mean_profile, 'b-', linewidth=2, label='Mean E-field')
    ax.fill_between(r, mean_profile - std_profile, mean_profile + std_profile,
                    alpha=0.3, color='blue', label='1 std')
    ax.axvline(r90, color='red', linestyle='--', alpha=0.7, label=f'Droplet surface (~{r90:.0f} A)')
    ax.set_xlabel('Distance from center (A)', fontsize=14)
    ax.set_ylabel('Electric field magnitude (V/nm)', fontsize=14)
    ax.set_title('Radial Electric Field Profile — 8nm Water Microdroplet', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim(0, r.max())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efield_radial.png", dpi=150)
    plt.close()

    # Surface field stats
    surface_mask = (r > r90 - 5) & (r < r90 + 5)
    bulk_mask = r < r90 - 10
    surface_field = mean_profile[surface_mask].mean() if surface_mask.any() else 0
    bulk_field = mean_profile[bulk_mask].mean() if bulk_mask.any() else 0

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  efield_radial.npz  — raw data")
    print(f"  efield_radial.png  — plot")
    print(f"\nSummary:")
    print(f"  Droplet radius (90th pct): {r90:.1f} A")
    print(f"  Bulk field (<{r90-10:.0f} A):       {bulk_field:.2f} V/nm")
    print(f"  Surface field (~{r90:.0f} A):     {surface_field:.2f} V/nm")
    print(f"  Surface/Bulk ratio:         {surface_field/bulk_field:.1f}x" if bulk_field > 0 else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="phase3/trajectory.traj")
    parser.add_argument("--n-frames", type=int, default=100)
    args = parser.parse_args()
    main(args.trajectory, args.n_frames)
