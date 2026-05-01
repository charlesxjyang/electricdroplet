"""
Compute E-field using DFT-derived radially-binned charges instead of
PolarMACE or fixed SPC/E. This tests whether the surface E-field sign
is a property of DFT itself or an artifact of PolarMACE's charge model.

Reads the 289 DFT cluster results (with Mulliken charges), bins charges
by radial stratum (surface/interface/bulk), then applies stratum-dependent
charges to the full MD trajectory frames.

Usage: python dft_charge_efield.py [--n-frames 20]
"""
import argparse
import json
import numpy as np
import time
from pathlib import Path
from ase.io.trajectory import Trajectory
from analyze_efield import compute_efield_radial, _fibonacci_sphere

# Fixed SPC/E for comparison
Q_O_SPCE = -0.8476
Q_H_SPCE = +0.4238

TRAJ_FILE = Path("phase1/trajectory.traj")
DFT_DIR = Path("dft_results")
MANIFEST = Path("clusters/manifest.csv")


def load_dft_charges_by_stratum():
    """Load DFT Mulliken charges, return mean O/H charge per stratum."""
    import csv

    # Read manifest to get stratum for each cluster
    cluster_strata = {}
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"].replace(".xyz", "")
            cluster_strata[name] = row["stratum"]

    # Collect charges by stratum
    charges_by_stratum = {
        "surface": {"O": [], "H": []},
        "interface": {"O": [], "H": []},
        "bulk": {"O": [], "H": []},
    }

    for jf in sorted(DFT_DIR.glob("cluster_*.json")):
        with open(jf) as f:
            d = json.load(f)
        if d.get("status") not in ("ok", "low_gap"):
            continue
        if "mulliken_charges_e" not in d:
            continue

        name = jf.stem
        stratum = cluster_strata.get(name)
        if stratum not in charges_by_stratum:
            continue

        symbols = d["symbols"]
        charges = d["mulliken_charges_e"]
        for s, q in zip(symbols, charges):
            if s == "O":
                charges_by_stratum[stratum]["O"].append(q)
            elif s == "H":
                charges_by_stratum[stratum]["H"].append(q)

    # Compute means
    result = {}
    for stratum in ["surface", "interface", "bulk"]:
        o_charges = charges_by_stratum[stratum]["O"]
        h_charges = charges_by_stratum[stratum]["H"]
        if o_charges and h_charges:
            result[stratum] = {
                "O": float(np.mean(o_charges)),
                "H": float(np.mean(h_charges)),
                "O_std": float(np.std(o_charges)),
                "H_std": float(np.std(h_charges)),
                "n_clusters": len(o_charges),
            }
    return result


def assign_dft_charges(atoms, com, r90, dft_charges):
    """Assign DFT stratum-dependent charges to each atom."""
    sym = atoms.get_chemical_symbols()
    pos = atoms.positions
    radii = np.linalg.norm(pos - com, axis=1)

    charges = np.zeros(len(atoms))
    for i, s in enumerate(sym):
        r = radii[i]
        if s == "O":
            # Classify by radial position
            if r > r90 - 4:
                stratum = "surface"
            elif r > r90 - 8:
                stratum = "interface"
            else:
                stratum = "bulk"
        else:
            # H atoms: use same stratum as their parent O (i-1 or i-2)
            parent_r = radii[i - 1] if sym[i - 1] == "O" else radii[i - 2]
            if parent_r > r90 - 4:
                stratum = "surface"
            elif parent_r > r90 - 8:
                stratum = "interface"
            else:
                stratum = "bulk"

        elem = "O" if s == "O" else "H"
        if stratum in dft_charges:
            charges[i] = dft_charges[stratum][elem]
        else:
            charges[i] = Q_O_SPCE if s == "O" else Q_H_SPCE

    return charges


def main(n_frames=20):
    print("Loading DFT charges by stratum...")
    dft_charges = load_dft_charges_by_stratum()
    for stratum, vals in dft_charges.items():
        print("  {:12s}: O={:+.4f} +/- {:.4f}  H={:+.4f} +/- {:.4f}  (n={})".format(
            stratum, vals["O"], vals["O_std"], vals["H"], vals["H_std"], vals["n_clusters"]))

    print("\nComparison:")
    print("  SPC/E fixed:  O={:+.4f}  H={:+.4f}".format(Q_O_SPCE, Q_H_SPCE))
    print("  PolarMACE:    O~-0.46   H~+0.23 (from prior analysis)")

    traj = Trajectory(str(TRAJ_FILE))
    n = len(traj)

    # Get equilibration offset from go/no-go
    equil_frames = 0
    gng = Path("phase1/go_nogo_report.txt")
    if gng.exists():
        import re
        m = re.search(r"Equilibration time:\s+([\d.]+)\s+ns", gng.read_text())
        if m:
            equil_frames = int(float(m.group(1)) * 1e3 / 0.1) + 100

    prod_frames = list(range(equil_frames, n))
    if len(prod_frames) < n_frames:
        n_frames = len(prod_frames)
    indices = np.linspace(0, len(prod_frames) - 1, n_frames, dtype=int)
    frame_list = [prod_frames[i] for i in indices]

    # Geometry reference
    atoms_ref = traj[frame_list[-1]]
    sym_ref = atoms_ref.get_chemical_symbols()
    o_idx = [j for j, s in enumerate(sym_ref) if s == "O"]
    o_pos = atoms_ref.positions[o_idx]
    r90 = float(np.percentile(np.linalg.norm(o_pos - o_pos.mean(axis=0), axis=1), 90))
    bin_edges = np.linspace(0, r90 + 10, 41)

    print("\nComputing E-field from multiple charge models on {} frames...".format(n_frames))

    all_ef_fixed = []
    all_ef_dft = []
    extra_results = []
    traj_cache = {}
    t0 = time.time()

    for k, fi in enumerate(frame_list):
        atoms = traj[fi]
        pos = atoms.positions
        sym = atoms.get_chemical_symbols()
        o_pos_f = pos[[i for i, s in enumerate(sym) if s == "O"]]
        com = o_pos_f.mean(axis=0)

        # Fixed SPC/E
        q_fixed = np.array([Q_O_SPCE if s == "O" else Q_H_SPCE for s in sym])
        _, ef_f = compute_efield_radial(atoms, q_fixed, com, bin_edges)
        all_ef_fixed.append(ef_f)

        # DFT radial charges
        q_dft = assign_dft_charges(atoms, com, r90, dft_charges)
        _, ef_d = compute_efield_radial(atoms, q_dft, com, bin_edges)
        all_ef_dft.append(ef_d)

        if (k + 1) % 5 == 0:
            print("  [{}/{}] ({:.0f}s)".format(k + 1, n_frames, time.time() - t0))

    traj.close()

    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    VM = 1e-8
    surface = (r > r90 - 5) & (r < r90 + 5)
    bulk = r < r90 - 10

    ef_f_mean = np.mean(all_ef_fixed, axis=0) * VM
    ef_d_mean = np.mean(all_ef_dft, axis=0) * VM

    sf = float(ef_f_mean[surface].mean())
    bf = float(ef_f_mean[bulk].mean())
    sd = float(ef_d_mean[surface].mean())
    bd = float(ef_d_mean[bulk].mean())

    enhs_f = [(np.mean(ef[surface] * VM) - np.mean(ef[bulk] * VM)) for ef in all_ef_fixed]
    enhs_d = [(np.mean(ef[surface] * VM) - np.mean(ef[bulk] * VM)) for ef in all_ef_dft]

    # Also try Lowdin/Becke charges if recompute_charges has been run
    recharge_dir = Path("dft_charges")
    for method in ["lowdin", "becke"]:
        charge_files = sorted(recharge_dir.glob("*_charges.json")) if recharge_dir.exists() else []
        if not charge_files:
            continue

        # Build stratum-averaged charges for this method
        method_charges = {"surface": {"O": [], "H": []},
                          "interface": {"O": [], "H": []},
                          "bulk": {"O": [], "H": []}}
        import csv
        strata = {}
        if MANIFEST.exists():
            with open(MANIFEST) as f:
                for row in csv.DictReader(f):
                    strata[row["filename"].replace(".xyz", "")] = row["stratum"]

        for jf in charge_files:
            with open(jf) as f:
                d = json.load(f)
            name = jf.stem.replace("_charges", "")
            stratum = strata.get(name)
            if stratum not in method_charges or "{}_charges".format(method) not in d:
                continue
            for s, q in zip(d["symbols"], d["{}_charges".format(method)]):
                if s in ("O", "H"):
                    method_charges[stratum][s].append(q)

        if all(method_charges[s]["O"] for s in method_charges):
            method_dft = {s: {"O": float(np.mean(v["O"])), "H": float(np.mean(v["H"]))}
                          for s, v in method_charges.items() if v["O"]}

            all_ef_method = []
            for fi in frame_list:
                atoms = traj_cache[fi] if fi in traj_cache else traj[fi]
                pos = atoms.positions
                sym = atoms.get_chemical_symbols()
                o_pos_f = pos[[i for i, s in enumerate(sym) if s == "O"]]
                com = o_pos_f.mean(axis=0)
                q_m = assign_dft_charges(atoms, com, r90, method_dft)
                _, ef_m = compute_efield_radial(atoms, q_m, com, bin_edges)
                all_ef_method.append(ef_m)

            ef_m_mean = np.mean(all_ef_method, axis=0) * VM
            sm = float(ef_m_mean[surface].mean())
            bm = float(ef_m_mean[bulk].mean())
            enhs_m = [(np.mean(ef[surface] * VM) - np.mean(ef[bulk] * VM)) for ef in all_ef_method]
            extra_results.append((method.capitalize(), bm, sm, sm - bm,
                                  np.std(enhs_m) / np.sqrt(len(enhs_m))))

    print("\n" + "=" * 60)
    print("RESULTS: Multi-method E-field comparison ({} frames)".format(n_frames))
    print("=" * 60)
    print("Fixed SPC/E:     bulk={:+.2f}  surface={:+.2f}  enh={:+.2f} +/- {:.2f} MV/cm".format(
        bf, sf, sf - bf, np.std(enhs_f) / np.sqrt(len(enhs_f))))
    print("DFT Mulliken:    bulk={:+.2f}  surface={:+.2f}  enh={:+.2f} +/- {:.2f} MV/cm".format(
        bd, sd, sd - bd, np.std(enhs_d) / np.sqrt(len(enhs_d))))
    for name, b, s, enh, err in extra_results:
        print("DFT {:12s} bulk={:+.2f}  surface={:+.2f}  enh={:+.2f} +/- {:.2f} MV/cm".format(
            name + ":", b, s, enh, err))
    print("PolarMACE:       enh=-9.96 +/- 0.96 MV/cm (prior full-droplet run)")
    print("Hao et al C-GeM: enh=+9 MV/cm")
    print("")

    all_enh = [sf - bf, sd - bd] + [x[3] for x in extra_results]
    if all(e < 0 for e in all_enh):
        print("ALL charge methods give NEGATIVE enhancement → sign is robust")
        print("across DFT charge partitions. Not a PolarMACE artifact.")
    elif all(e > 0 for e in all_enh):
        print("ALL charge methods give POSITIVE enhancement → agrees with C-GeM.")
    else:
        print("MIXED signs across methods → sign depends on charge partition.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=20)
    args = parser.parse_args()
    main(n_frames=args.n_frames)
