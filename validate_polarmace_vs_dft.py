"""
Cross-validate PolarMACE charges and dipoles against DFT on Phase 2 clusters.

PolarMACE was trained on small-molecule dipole moments (SPICE / ωB97X-D).
To defend its use on nanodroplet interfaces, we compare its predictions on
the Phase 2 cluster geometries to DFT-derived reference values at the same
level used for fine-tuning.

What we compare:
  - Per-atom Mulliken charges (direct atom-by-atom parity)
  - Cluster dipole moment magnitude (the quantity PolarMACE was trained on)

Usage:
  python validate_polarmace_vs_dft.py --dft-dir dft_results/
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ase import Atoms


E_CHARGE_DEBYE = 1.0 / 0.2081943  # e·Å → Debye


def dipole_from_charges(symbols, positions_ang, charges_e):
    """Dipole vector in Debye for a set of point charges on a cluster.

    Uses the centre-of-geometry as the origin. For a neutral cluster this
    is invariant; if the model produces a small net charge (which would
    itself be a red flag), we subtract the monopole contribution.
    """
    positions = np.asarray(positions_ang)
    q = np.asarray(charges_e)
    net = q.sum()
    if abs(net) > 1e-3:
        # Remove monopole by origin choice (COM of the cluster)
        origin = positions.mean(axis=0)
        d_vec = ((q - net / len(q))[:, None] * (positions - origin)).sum(axis=0)
    else:
        d_vec = (q[:, None] * positions).sum(axis=0)
    return d_vec * E_CHARGE_DEBYE


def main(dft_dir):
    dft_dir = Path(dft_dir)
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)

    # Load PolarMACE once
    from mace.calculators import mace_polar
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc = mace_polar(model='polar-1-s', device=device, default_dtype='float32')

    files = sorted(dft_dir.glob("cluster_*.json"))
    print(f"Loaded {len(files)} DFT cluster results from {dft_dir}")

    all_dft_q = []
    all_pm_q = []
    elem_per_charge = []
    dft_dipoles = []
    pm_dipoles = []
    failures = 0

    for path in files:
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "ok":
            continue
        if "mulliken_charges_e" not in d:
            print(f"  skipping {path.name}: no charges saved (rerun DFT)")
            continue

        symbols = d["symbols"]
        positions = np.array(d["positions_ang"])
        q_dft = np.array(d["mulliken_charges_e"])
        dip_dft = np.array(d["dipole_debye"])

        # PolarMACE on the same geometry
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info['charge'] = 0
        atoms.info['spin'] = 1
        atoms.info['external_field'] = [0.0, 0.0, 0.0]
        atoms.calc = calc
        try:
            atoms.get_potential_energy()
            q_pm = calc.results['charges'].copy()
        except Exception as e:
            print(f"  {path.name}: PolarMACE failed — {e}")
            failures += 1
            continue

        dip_pm = dipole_from_charges(symbols, positions, q_pm)

        all_dft_q.extend(q_dft.tolist())
        all_pm_q.extend(q_pm.tolist())
        elem_per_charge.extend(symbols)
        dft_dipoles.append(np.linalg.norm(dip_dft))
        pm_dipoles.append(np.linalg.norm(dip_pm))

    if not all_dft_q:
        print("No usable clusters found. Did you run updated run_dft.py?")
        return

    all_dft_q = np.array(all_dft_q)
    all_pm_q = np.array(all_pm_q)
    dft_dipoles = np.array(dft_dipoles)
    pm_dipoles = np.array(pm_dipoles)
    elem = np.array(elem_per_charge)
    is_O = elem == 'O'
    is_H = elem == 'H'

    # Metrics
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    rmse_O = rmse(all_dft_q[is_O], all_pm_q[is_O])
    rmse_H = rmse(all_dft_q[is_H], all_pm_q[is_H])
    rmse_dip = rmse(dft_dipoles, pm_dipoles)
    r2_dip = r2(dft_dipoles, pm_dipoles)

    print(f"\nCharge RMSE: O={rmse_O:.4f} e   H={rmse_H:.4f} e")
    print(f"Dipole magnitude RMSE: {rmse_dip:.3f} Debye   R² = {r2_dip:.3f}")
    print(f"Clusters compared: {len(dft_dipoles)}   "
          f"Failures: {failures}")

    # Parity plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Charge parity
    ax = axes[0]
    ax.scatter(all_dft_q[is_O], all_pm_q[is_O], s=8, alpha=0.4, color='red', label=f'O (RMSE={rmse_O:.3f} e)')
    ax.scatter(all_dft_q[is_H], all_pm_q[is_H], s=8, alpha=0.4, color='blue', label=f'H (RMSE={rmse_H:.3f} e)')
    lo = min(all_dft_q.min(), all_pm_q.min())
    hi = max(all_dft_q.max(), all_pm_q.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel('DFT Mulliken charge (e)')
    ax.set_ylabel('PolarMACE charge (e)')
    ax.set_title('Per-atom charge parity')
    ax.legend()
    ax.grid(alpha=0.3)

    # Dipole parity
    ax = axes[1]
    ax.scatter(dft_dipoles, pm_dipoles, s=20, alpha=0.5)
    lo = min(dft_dipoles.min(), pm_dipoles.min())
    hi = max(dft_dipoles.max(), pm_dipoles.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel('DFT dipole |μ| (Debye)')
    ax.set_ylabel('PolarMACE dipole |μ| (Debye)')
    ax.set_title(f'Cluster dipole parity (R²={r2_dip:.3f})')
    ax.grid(alpha=0.3)

    fig.suptitle('PolarMACE vs DFT on Phase 2 clusters', fontsize=13)
    fig.tight_layout()
    out_png = out_dir / "polarmace_vs_dft.png"
    out_pdf = out_dir / "polarmace_vs_dft.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}, {out_pdf}")

    np.savez(out_dir / "polarmace_vs_dft.npz",
             dft_charges=all_dft_q, pm_charges=all_pm_q, elements=elem,
             dft_dipoles=dft_dipoles, pm_dipoles=pm_dipoles,
             rmse_O=rmse_O, rmse_H=rmse_H, rmse_dipole=rmse_dip, r2_dipole=r2_dip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-dir", default="dft_results")
    args = parser.parse_args()
    main(args.dft_dir)
