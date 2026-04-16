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

# DFT settings — must match Hao et al. validation level
FUNCTIONAL = "revpbe"    # revPBE functional (PySCF name)
BASIS = "def2-tzvp"      # TZV2P-quality basis
USE_D3 = True            # DFT-D3(BJ) dispersion correction
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
        mf.grids.level = 4  # fine integration grid
        mf.max_cycle = 200
        mf.conv_tol = 1e-7

        # D3(BJ) dispersion correction
        if USE_D3:
            from pyscf import dftd3
            mf = dftd3.dftd3(mf)

        t0 = time.time()
        energy = mf.kernel()
        elapsed = time.time() - t0

        if not mf.converged:
            return {"file": fname, "status": "not_converged", "time": elapsed}

        # Sanity check: HOMO-LUMO gap. Sub-2 eV on a closed-shell water cluster
        # is diagnostic of pathological SCF (fractional occupations, near-
        # metallic state) — flag so fine-tuning doesn't silently learn bad
        # forces.
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        homo = float(mo_energy[mo_occ > 0].max())
        lumo = float(mo_energy[mo_occ == 0].min())
        gap_ev = (lumo - homo) * 27.211386245988

        # Analytical nuclear gradients -> forces
        g = mf.nuc_grad_method().kernel()
        forces = -g  # gradient -> force

        # Post-SCF quantities for PolarMACE cross-validation (Phase 1d):
        #   - Mulliken charges: per-atom, for direct comparison to PolarMACE
        #   - Dipole moment: integrated test (PolarMACE was trained on dipoles)
        mulliken_charges = mf.mulliken_pop(verbose=0)[1]  # returns (pop, chg)
        dipole_debye = mf.dip_moment(unit='Debye', verbose=0)

        # Convert to eV and eV/A
        ha_to_ev = 27.211386245988
        bohr_to_ang = 0.529177249

        result = {
            "file": fname,
            "status": "ok",
            "energy_ha": float(energy),
            "energy_ev": float(energy * ha_to_ev),
            "forces_ev_per_ang": (forces * ha_to_ev / bohr_to_ang).tolist(),
            "mulliken_charges_e": [float(q) for q in mulliken_charges],
            "dipole_debye": [float(d) for d in dipole_debye],
            "homo_lumo_gap_ev": gap_ev,
            "symbols": symbols,
            "positions_ang": positions.tolist(),
            "n_atoms": len(atoms),
            "functional": FUNCTIONAL,
            "basis": BASIS,
            "time_s": elapsed,
        }

        if gap_ev < 2.0:
            result["status"] = "low_gap"
            result["warning"] = f"HOMO-LUMO gap {gap_ev:.2f} eV below 2 eV threshold"

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

    # Upload results to S3 if available
    try:
        from s3_config import DFT_RESULTS_S3, sync_up
        print(f"\nUploading DFT results to S3...")
        sync_up(output_dir, DFT_RESULTS_S3)
    except Exception:
        print(f"\nS3 upload skipped (running locally)")
    print(f"\nNext: python finetune_mace.py --dft-dir {output_dir}")


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
