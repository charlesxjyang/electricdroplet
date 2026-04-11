"""
Extract water clusters from Phase 1 trajectory for DFT single-point calculations.

Samples frames across the production portion of the trajectory, then for each
frame selects a random oxygen and extracts all waters within a cutoff radius.

Usage: python extract_clusters.py [--n-clusters 300] [--cutoff 6.0]
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


def extract_cluster(atoms, center_o_idx, cutoff_A):
    """Extract all waters with O within cutoff of center oxygen."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']

    center_pos = positions[center_o_idx]
    dists = np.linalg.norm(positions[o_indices] - center_pos, axis=1)
    nearby_o = [o_indices[j] for j, d in enumerate(dists) if d < cutoff_A]

    # For each O, grab the two H that follow it (O, H, H ordering from build)
    cluster_indices = []
    for o in nearby_o:
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


def main(n_clusters=300, cutoff=6.0):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading trajectory from {TRAJ_FILE}...")
    traj = Trajectory(str(TRAJ_FILE))
    n_frames = len(traj)
    print(f"  Total frames: {n_frames}")
    print(f"  Skipping first {EQUIL_FRAMES} (equilibration)")

    prod_frames = list(range(EQUIL_FRAMES, n_frames))
    if len(prod_frames) < n_clusters:
        print(f"  Warning: only {len(prod_frames)} production frames, reducing cluster count")
        n_clusters = len(prod_frames)

    rng = np.random.default_rng(123)
    frame_indices = sorted(rng.choice(prod_frames, size=n_clusters, replace=False))

    print(f"  Extracting {n_clusters} clusters (cutoff={cutoff} A)...")

    manifest = []
    for i, fi in enumerate(frame_indices):
        atoms = traj[fi]
        symbols = atoms.get_chemical_symbols()
        o_indices = [j for j, s in enumerate(symbols) if s == 'O']

        center_o = rng.choice(o_indices)
        cluster = extract_cluster(atoms, center_o, cutoff)
        n_waters = sum(1 for s in cluster.get_chemical_symbols() if s == 'O')

        fname = f"cluster_{i:04d}.xyz"
        write(str(OUTPUT_DIR / fname), cluster)
        manifest.append(f"{fname},{fi},{n_waters},{len(cluster)}")

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_clusters} done")

    # Write manifest
    with open(OUTPUT_DIR / "manifest.csv", 'w') as f:
        f.write("filename,frame,n_waters,n_atoms\n")
        for line in manifest:
            f.write(line + "\n")

    traj.close()

    sizes = [int(m.split(',')[2]) for m in manifest]
    print(f"\nDone! {n_clusters} clusters saved to {OUTPUT_DIR}/")
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
    parser.add_argument("--n-clusters", type=int, default=300)
    parser.add_argument("--cutoff", type=float, default=6.0,
                        help="Cutoff radius in Angstrom for cluster extraction")
    args = parser.parse_args()
    main(n_clusters=args.n_clusters, cutoff=args.cutoff)
