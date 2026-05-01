"""
E-field analysis using Hao et al.'s definition:
  E projected onto OH bonds, excluding parent molecule.

Runs on existing trajectory with SPC/E charges (fast, CPU only).
Also supports PolarMACE charges if provided.

Usage:
  python analyze_efield_hao.py --trajectory phase1/trajectory.traj --n-frames 20
  python analyze_efield_hao.py --polar-charges analysis/polar_charges.npz
"""
import argparse
import numpy as np
import time
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io.trajectory import Trajectory
from compute_efield_hao import compute_efield_oh_fast

Q_O_SPCE = -0.8476
Q_H_SPCE = +0.4238
OUTPUT_DIR = Path("analysis")


def main(traj_path, n_frames=20, polar_charges_file=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    traj = Trajectory(str(traj_path))
    n_total = len(traj)

    # Determine production frames
    equil_frames = 0
    gng = Path("phase1/go_nogo_report.txt")
    if gng.exists():
        m = re.search(r"Equilibration time:\s+([\d.]+)\s+ns", gng.read_text())
        if m:
            equil_frames = int(float(m.group(1)) * 1e3 / 0.1) + 100

    prod_frames = list(range(equil_frames, n_total))
    if not prod_frames:
        prod_frames = list(range(n_total))

    n_sample = min(n_frames, len(prod_frames))
    indices = np.linspace(0, len(prod_frames) - 1, n_sample, dtype=int)
    frame_list = [prod_frames[i] for i in indices]

    # Geometry reference
    atoms_ref = traj[frame_list[-1]]
    sym_ref = atoms_ref.get_chemical_symbols()
    o_idx = [j for j, s in enumerate(sym_ref) if s == "O"]
    o_pos = atoms_ref.positions[o_idx]
    com = o_pos.mean(axis=0)
    r90 = float(np.percentile(np.linalg.norm(o_pos - com, axis=1), 90))
    bin_edges = np.linspace(0, r90 + 10, 41)

    # Load PolarMACE charges if available
    polar_charges_per_frame = None
    if polar_charges_file and Path(polar_charges_file).exists():
        pdata = np.load(polar_charges_file, allow_pickle=True)
        polar_charges_per_frame = pdata.get("charges_per_frame")
        print("Loaded PolarMACE charges for {} frames".format(
            len(polar_charges_per_frame) if polar_charges_per_frame is not None else 0))

    print("E-field analysis (Hao et al. definition: E dot OH, excluding parent molecule)")
    print("  Trajectory: {} frames, {} production".format(n_total, len(prod_frames)))
    print("  Sampling {} frames".format(n_sample))
    print("  r90 = {:.1f} A".format(r90))

    all_ef_fixed = []
    all_ef_polar = []
    t0 = time.time()

    for k, fi in enumerate(frame_list):
        atoms = traj[fi]
        sym = atoms.get_chemical_symbols()
        o_pos_f = atoms.positions[[i for i, s in enumerate(sym) if s == "O"]]
        com_f = o_pos_f.mean(axis=0)

        # SPC/E fixed charges
        q_fixed = np.array([Q_O_SPCE if s == "O" else Q_H_SPCE for s in sym])
        r_bins, ef_f, _, _ = compute_efield_oh_fast(atoms, q_fixed, com_f, bin_edges)
        all_ef_fixed.append(ef_f)

        # PolarMACE charges (if available)
        if polar_charges_per_frame is not None and k < len(polar_charges_per_frame):
            q_polar = polar_charges_per_frame[k]
            _, ef_p, _, _ = compute_efield_oh_fast(atoms, q_polar, com_f, bin_edges)
            all_ef_polar.append(ef_p)

        if (k + 1) % 5 == 0 or k == 0:
            print("  [{}/{}] ({:.0f}s)".format(k + 1, n_sample, time.time() - t0))

    traj.close()

    surface = (r_bins > r90 - 5) & (r_bins < r90 + 5)
    bulk = r_bins < r90 - 10

    ef_f_mean = np.mean(all_ef_fixed, axis=0)
    sf = float(ef_f_mean[surface].mean()) if surface.any() else 0
    bf = float(ef_f_mean[bulk].mean()) if bulk.any() else 0
    enhs_f = [float(np.mean(ef[surface]) - np.mean(ef[bulk])) for ef in all_ef_fixed]

    print("\n" + "=" * 65)
    print("RESULTS: E dot OH (Hao et al. definition), {} frames".format(n_sample))
    print("=" * 65)
    print("SPC/E fixed:   bulk={:+.2f}  surface={:+.2f}  enh={:+.2f} +/- {:.2f} MV/cm".format(
        bf, sf, sf - bf, np.std(enhs_f) / np.sqrt(len(enhs_f))))

    if all_ef_polar:
        ef_p_mean = np.mean(all_ef_polar, axis=0)
        sp = float(ef_p_mean[surface].mean()) if surface.any() else 0
        bp = float(ef_p_mean[bulk].mean()) if bulk.any() else 0
        enhs_p = [float(np.mean(ef[surface]) - np.mean(ef[bulk])) for ef in all_ef_polar]
        print("PolarMACE:     bulk={:+.2f}  surface={:+.2f}  enh={:+.2f} +/- {:.2f} MV/cm".format(
            bp, sp, sp - bp, np.std(enhs_p) / np.sqrt(len(enhs_p))))

    print("Hao et al:     enh=+16 MV/cm (C-GeM, E dot OH definition)")
    print("               surface field ~9 MV/cm larger than Hao's interior")

    # Save results
    results = {
        "method": "E_dot_OH_hao_definition",
        "n_frames": n_sample,
        "r90_A": r90,
        "fixed_bulk": bf, "fixed_surface": sf,
        "fixed_enhancement": sf - bf,
        "fixed_err": float(np.std(enhs_f) / np.sqrt(len(enhs_f))),
    }
    if all_ef_polar:
        results["polar_bulk"] = bp
        results["polar_surface"] = sp
        results["polar_enhancement"] = sp - bp
        results["polar_err"] = float(np.std(enhs_p) / np.sqrt(len(enhs_p)))

    with open(OUTPUT_DIR / "efield_hao_definition.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    ax.plot(r_bins, ef_f_mean, 'steelblue', lw=1.5, label='SPC/E')
    if all_ef_polar:
        ax.plot(r_bins, ef_p_mean, 'crimson', lw=1.5, label='PolarMACE')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.3)
    ax.axvline(r90, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('$r_O$ (\\AA)')
    ax.set_ylabel('$E \\cdot \\hat{OH}$ (MV cm$^{-1}$)')
    ax.set_xlim(0, r90 + 12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efield_hao_definition.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "efield_hao_definition.pdf", bbox_inches='tight')
    plt.close()
    print("\nSaved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="phase1/trajectory.traj")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--polar-charges", default=None,
                        help="Path to .npz with PolarMACE charges per frame")
    args = parser.parse_args()
    main(args.trajectory, args.n_frames, args.polar_charges)
