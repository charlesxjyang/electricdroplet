"""
Phase 2: DFT single-point calculations on extracted clusters using PySCF.
Run on c6i.8xlarge instances (32 vCPU, 64GB RAM).

Computes energy and forces at revPBE-D3 / def2-TZVP level for MACE fine-tuning.

Usage:
  python run_dft.py --clusters-dir clusters/
  python run_dft.py --clusters-dir clusters/ --start 0 --end 75   # shard across instances
"""
import argparse
import numpy as np
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ase.io import read, write

# DFT settings
FUNCTIONAL = "pbe"       # PySCF functional name
BASIS = "def2-svp"       # def2-SVP for speed; upgrade to def2-TZVP if budget allows
MAX_WORKERS = 8           # parallel DFT jobs per instance (each uses ~4 cores)


def run_single_dft(cluster_path, output_dir):
    """Run DFT on one cluster, return results dict."""
    import pyscf
    from pyscf import gto, dft

    fname = Path(cluster_path).name
    out_file = output_dir / fname.replace(".xyz", ".json")

    if out_file.exists():
        return {"file": fname, "status": "skipped"}

    atoms = read(str(cluster_path))
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    # Build PySCF mol
    atom_str = "\n".join(
        f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        for s, p in zip(symbols, positions)
    )

    try:
        mol = gto.M(atom=atom_str, basis=BASIS, unit="Angstrom", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = FUNCTIONAL
        mf.max_cycle = 200
        mf.conv_tol = 1e-7

        t0 = time.time()
        energy = mf.kernel()
        elapsed = time.time() - t0

        if not mf.converged:
            return {"file": fname, "status": "not_converged", "time": elapsed}

        # Analytical nuclear gradients -> forces
        g = mf.nuc_grad_method().kernel()
        forces = -g  # gradient -> force

        # Convert to eV and eV/A
        ha_to_ev = 27.211386245988
        bohr_to_ang = 0.529177249

        result = {
            "file": fname,
            "status": "ok",
            "energy_ha": float(energy),
            "energy_ev": float(energy * ha_to_ev),
            "forces_ev_per_ang": (forces * ha_to_ev / bohr_to_ang).tolist(),
            "symbols": symbols,
            "positions_ang": positions.tolist(),
            "n_atoms": len(atoms),
            "functional": FUNCTIONAL,
            "basis": BASIS,
            "time_s": elapsed,
        }

        with open(out_file, 'w') as f:
            json.dump(result, f)

        return {"file": fname, "status": "ok", "time": elapsed,
                "energy_ev": result["energy_ev"]}

    except Exception as e:
        return {"file": fname, "status": "error", "error": str(e)}


def main(clusters_dir, output_dir, start=None, end=None):
    clusters_dir = Path(clusters_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Pull clusters from S3 if local dir is empty
    if not any(clusters_dir.glob("cluster_*.xyz")):
        clusters_dir.mkdir(exist_ok=True)
        from s3_config import CLUSTERS_S3, sync_down
        print("No local clusters found, downloading from S3...")
        sync_down(CLUSTERS_S3, clusters_dir)
        print()

    cluster_files = sorted(clusters_dir.glob("cluster_*.xyz"))
    if start is not None or end is not None:
        s = start or 0
        e = end or len(cluster_files)
        cluster_files = cluster_files[s:e]

    print(f"DFT single-points: {len(cluster_files)} clusters")
    print(f"  Functional: {FUNCTIONAL} / {BASIS}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Output: {output_dir}/")
    print()

    results = []
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single_dft, str(cf), output_dir): cf
            for cf in cluster_files
        }
        for future in futures:
            r = future.result()
            results.append(r)
            done += 1
            if r["status"] == "ok":
                elapsed = time.time() - t0
                rate = done / elapsed * 3600
                remaining = (len(cluster_files) - done) / (done / elapsed)
                print(f"  [{done}/{len(cluster_files)}] {r['file']} "
                      f"E={r['energy_ev']:.4f} eV  {r['time']:.0f}s  "
                      f"({rate:.0f}/hr, ETA {remaining/3600:.1f}h)")
            else:
                print(f"  [{done}/{len(cluster_files)}] {r['file']} — {r['status']}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(cluster_files)} converged")
    print(f"Total time: {(time.time()-t0)/3600:.1f} hours")

    # Upload results to S3
    from s3_config import DFT_RESULTS_S3, sync_up
    print(f"\nUploading DFT results to S3...")
    sync_up(output_dir, DFT_RESULTS_S3)
    print(f"\nNext (on GPU instance): python finetune_mace.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-dir", required=True)
    parser.add_argument("--output-dir", default="dft_results")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    MAX_WORKERS = args.workers
    main(args.clusters_dir, Path(args.output_dir), args.start, args.end)
