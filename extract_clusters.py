"""
Extract water clusters from Phase 1 trajectory for DFT single-point calculations.

Uses STRATIFIED sampling by radial position: the center oxygen of each cluster
is drawn from one of three concentric regions defined relative to the droplet's
90th-percentile radius r90:

  - surface:   r > r90 - 4 Å       (outermost ~4 Å shell)
  - interface: r90 - 8 < r ≤ r90-4 (next 4 Å inward)
  - bulk:      r ≤ r90 - 8         (deeper core)

Uniform random O selection would give a Boltzmann-weighted sample — fine for
homogeneous systems but under-represents surface environments in a finite
droplet where the paper's physics claim lives. Stratifying to 40/30/30
(surface/interface/bulk) ensures the MLIP fine-tuning set has the surface
coverage needed to defend the surface E-field enhancement result.

Usage:
  python extract_clusters.py                                   # 120/90/90 default
  python extract_clusters.py --n-surface 150 --n-interface 75 --n-bulk 75
  python extract_clusters.py --cutoff 7.0                      # bigger clusters
"""
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase import Atoms

TRAJ_FILE = Path("phase1/trajectory.traj")
EQUIL_FRAMES = 10_000  # first 10K frames = 1ns equilibration, skip these
OUTPUT_DIR = Path("clusters")

# Stratum boundaries in Å, as inward offsets from r90
SURFACE_THICKNESS = 4.0    # surface: r > r90 - 4
INTERFACE_THICKNESS = 8.0  # interface: r90 - 8 < r ≤ r90 - 4


def extract_cluster(atoms, center_o_idx, cutoff_A):
    """Extract all waters with O within cutoff of center oxygen."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']

    center_pos = positions[center_o_idx]
    dists = np.linalg.norm(positions[o_indices] - center_pos, axis=1)
    nearby_o = [o_indices[j] for j, d in enumerate(dists) if d < cutoff_A]

    # For each O, grab the two H that follow it (O, H, H ordering from build).
    # Fail loud if the invariant breaks — a silent reorder would poison the
    # DFT training set with broken molecule geometries.
    cluster_indices = []
    for o in nearby_o:
        assert o + 2 < len(symbols) and symbols[o+1] == 'H' and symbols[o+2] == 'H', (
            f"O,H,H ordering broken at atom {o}: "
            f"{symbols[o:o+3] if o+3 <= len(symbols) else symbols[o:]}"
        )
        cluster_indices.extend([o, o+1, o+2])

    cluster = Atoms(
        symbols=[symbols[i] for i in cluster_indices],
        positions=positions[cluster_indices],
    )
    # Center at origin
    cluster.positions -= cluster.positions[
        [i for i, s in enumerate(cluster.get_chemical_symbols()) if s == 'O']
    ].mean(axis=0)

    return cluster


def _classify_by_radius(radii, r90):
    """Return label array ('surface' | 'interface' | 'bulk') for each radius."""
    labels = np.full(len(radii), 'bulk', dtype=object)
    labels[radii > r90 - INTERFACE_THICKNESS] = 'interface'
    labels[radii > r90 - SURFACE_THICKNESS] = 'surface'
    return labels


def _droplet_geometry(atoms):
    """Return (com, r90) for the current frame's oxygen positions."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    o_positions = positions[[i for i, s in enumerate(symbols) if s == 'O']]
    com = o_positions.mean(axis=0)
    radii = np.linalg.norm(o_positions - com, axis=1)
    r90 = float(np.percentile(radii, 90))
    return com, r90


def main(n_surface, n_interface, n_bulk, cutoff):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading trajectory from {TRAJ_FILE}...")
    traj = Trajectory(str(TRAJ_FILE))
    n_frames = len(traj)
    print(f"  Total frames: {n_frames}")
    print(f"  Skipping first {EQUIL_FRAMES} (equilibration)")

    prod_frames = list(range(EQUIL_FRAMES, n_frames))
    if not prod_frames:
        raise SystemExit("No production frames available after equilibration skip.")

    # Reference r90 from a late-production frame (stable geometry)
    ref_frame = prod_frames[len(prod_frames) // 2]
    _, ref_r90 = _droplet_geometry(traj[ref_frame])
    print(f"  Reference r90 (from frame {ref_frame}): {ref_r90:.2f} Å")
    print(f"  Stratum boundaries:")
    print(f"    surface:   r > {ref_r90 - SURFACE_THICKNESS:.2f} Å")
    print(f"    interface: {ref_r90 - INTERFACE_THICKNESS:.2f} < r ≤ {ref_r90 - SURFACE_THICKNESS:.2f} Å")
    print(f"    bulk:      r ≤ {ref_r90 - INTERFACE_THICKNESS:.2f} Å")

    targets = {'surface': n_surface, 'interface': n_interface, 'bulk': n_bulk}
    counts = {k: 0 for k in targets}
    total_target = sum(targets.values())
    print(f"  Target: {targets} (total {total_target})")

    rng = np.random.default_rng(123)
    manifest = []
    cluster_idx = 0

    # Safety limit on attempts: if we can't fill a stratum in ~20× target
    # attempts, something is wrong with geometry or equilibration.
    max_attempts = 20 * total_target
    attempts = 0

    while sum(counts.values()) < total_target and attempts < max_attempts:
        attempts += 1

        # Which strata still need samples?
        needed = [s for s, t in targets.items() if counts[s] < t]
        if not needed:
            break
        # Prefer least-filled stratum to avoid early saturation of one bucket
        stratum = min(needed, key=lambda s: counts[s])

        # Try to find a frame with at least one eligible O in this stratum
        fi = int(rng.choice(prod_frames))
        atoms = traj[fi]
        symbols = atoms.get_chemical_symbols()
        positions = atoms.positions
        o_indices = np.array([i for i, s in enumerate(symbols) if s == 'O'])
        o_pos = positions[o_indices]
        com = o_pos.mean(axis=0)
        radii = np.linalg.norm(o_pos - com, axis=1)
        labels = _classify_by_radius(radii, ref_r90)

        eligible = o_indices[labels == stratum]
        if len(eligible) == 0:
            continue  # this frame has no O in the target stratum — retry

        center_o = int(rng.choice(eligible))
        cluster = extract_cluster(atoms, center_o, cutoff)
        n_waters = sum(1 for s in cluster.get_chemical_symbols() if s == 'O')
        r_center = float(np.linalg.norm(positions[center_o] - com))

        fname = f"cluster_{cluster_idx:04d}_{stratum}.xyz"
        write(str(OUTPUT_DIR / fname), cluster)
        manifest.append({
            'filename': fname, 'frame': fi, 'stratum': stratum,
            'r_center_A': r_center, 'n_waters': n_waters,
            'n_atoms': len(cluster),
        })
        counts[stratum] += 1
        cluster_idx += 1

        if cluster_idx % 25 == 0:
            print(f"    {cluster_idx}/{total_target}  "
                  f"(S={counts['surface']}, I={counts['interface']}, B={counts['bulk']})")

    traj.close()

    if sum(counts.values()) < total_target:
        print(f"\n  WARNING: filled only {sum(counts.values())}/{total_target} "
              f"after {attempts} attempts. Counts: {counts}")

    # Write manifest
    with open(OUTPUT_DIR / "manifest.csv", 'w') as f:
        f.write("filename,frame,stratum,r_center_A,n_waters,n_atoms\n")
        for m in manifest:
            f.write(f"{m['filename']},{m['frame']},{m['stratum']},"
                    f"{m['r_center_A']:.3f},{m['n_waters']},{m['n_atoms']}\n")

    sizes = [m['n_waters'] for m in manifest]
    print(f"\nDone! {len(manifest)} clusters saved to {OUTPUT_DIR}/")
    print(f"  Counts by stratum: {counts}")
    print(f"  Waters per cluster: {np.min(sizes)}-{np.max(sizes)} (mean {np.mean(sizes):.0f})")
    print(f"  Manifest: {OUTPUT_DIR}/manifest.csv")

    # Upload to S3 if available
    try:
        from s3_config import CLUSTERS_S3, sync_up
        print(f"\nUploading clusters to S3...")
        sync_up(OUTPUT_DIR, CLUSTERS_S3)
    except Exception:
        print(f"\nS3 upload skipped (running locally)")
    print(f"\nNext: python run_dft.py --clusters-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-surface", type=int, default=120,
                        help="Clusters centered in the surface shell (r > r90-4 Å)")
    parser.add_argument("--n-interface", type=int, default=90,
                        help="Clusters centered in the interface shell (r90-8 < r ≤ r90-4 Å)")
    parser.add_argument("--n-bulk", type=int, default=90,
                        help="Clusters centered in the bulk core (r ≤ r90-8 Å)")
    parser.add_argument("--cutoff", type=float, default=6.0,
                        help="Cutoff radius in Angstrom for cluster extraction")
    args = parser.parse_args()
    main(n_surface=args.n_surface, n_interface=args.n_interface,
         n_bulk=args.n_bulk, cutoff=args.cutoff)
