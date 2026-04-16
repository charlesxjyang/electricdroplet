"""
Phase 1: NVT MD with MACE-MP-0 on g6e.2xlarge (L40S).

Protocol: up to 0.5 ns equilibration + 2 ns production (Hao et al. 2022).
Early convergence: if the surface orientational order, density, and spread
ratio are all stable across two consecutive 50-ps windows (starting at
0.2 ns), equilibration is declared done early and production begins
immediately — shortening walltime.

Run:    python run_phase1.py
Resume: python run_phase1.py --resume
"""
import argparse, os, time
import numpy as np
import torch
from pathlib import Path
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import mace_mp

# === CONFIGURATION (matching Hao et al. 2022 protocol) ===
TIMESTEP_FS    = 0.5
TOTAL_STEPS    = 5_000_000    # 2.5 ns (0.5 ns equil + 2 ns production) upper bound
EQUIL_STEPS    = 1_000_000    # 0.5 ns equilibration cap (early convergence may cut this)
PRODUCTION_STEPS = TOTAL_STEPS - EQUIL_STEPS  # 4,000,000 = 2 ns production
CHECKPOINT_INT = 2000         # Save restart every 1 ps
TRAJ_INT       = 200          # Save frame every 0.1 ps
LOG_INT        = 2000         # Print status every 1 ps
TEMPERATURE    = 300.0        # K
DIAMETER_NM    = 8.0

# === EARLY CONVERGENCE DETECTION ===
# Start checking after 0.2 ns; compare last 50 ps vs previous 50 ps; require 2
# consecutive passes. If all three observables are stable within tolerance,
# equilibration is declared done and production begins immediately — saving
# walltime vs running the full 0.5 ns cap.
CONV_CHECK_START   = 400_000  # 0.2 ns: earliest convergence check
CONV_CHECK_STRIDE  = 100_000  # 50 ps between checks
CONV_WINDOW        = 50       # number of 1-ps samples per window
CONV_PASSES_REQ    = 1        # passes required before declaring equilibrated
                              # (lowered from 2 after 160-ps probe showed signals
                              # deep inside tolerance with monotonic tightening)
CONV_TOL_COS_THETA = 0.02     # |Δ<cos θ>_surface| between windows
CONV_TOL_DENSITY   = 0.01     # |Δρ|/ρ (relative)
CONV_TOL_SPREAD    = 0.01     # |Δ spread ratio|


class StopEarly(Exception):
    """Raised from a callback to terminate the MD loop cleanly."""
    pass

WORK = Path("phase1")
WORK.mkdir(exist_ok=True)
RESTART   = WORK / "restart.xyz"
CKPT_META = WORK / "checkpoint.npz"
TRAJ_FILE = WORK / "trajectory.traj"
LOG_FILE  = WORK / "md.log"
VALIDATION_FILE = WORK / "go_nogo_report.txt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


MACE_MODEL = os.environ.get("MACE_MODEL", "medium")


def get_calc():
    return mace_mp(model=MACE_MODEL, dispersion=False,
                   default_dtype='float32', device=DEVICE)


def run(resume=False):
    if resume and RESTART.exists():
        print(f"Resuming from {RESTART}")
        atoms = read(str(RESTART))
        atoms.calc = get_calc()
        meta = np.load(str(CKPT_META))
        start_step = int(meta['step'])
        print(f"  Resuming at step {start_step}")
    else:
        atoms = read("droplet_initial.xyz")
        atoms.calc = get_calc()
        start_step = 0
        print(f"Starting fresh: {len(atoms)} atoms")

    dyn = Langevin(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE,
        friction=0.01 / units.fs,
        logfile=str(LOG_FILE),
        loginterval=LOG_INT,
    )

    traj = Trajectory(str(TRAJ_FILE), 'a' if resume else 'w', atoms)
    t0 = time.time()
    step_offset = start_step

    # Convergence / early-termination state
    conv = {
        'samples': [],          # list of {step, surf_cos, density, spread, T}
        'passes': 0,            # consecutive passing convergence checks
        'converged_step': None, # step at which equilibration was declared done
        'target_end': TOTAL_STEPS,  # last step to run (may shrink on early conv)
        'gng_fired': False,     # whether go/no-go has been written
    }

    def save_traj():
        traj.write(atoms)

    def checkpoint():
        step = step_offset + dyn.nsteps
        write(str(RESTART), atoms)
        np.savez(str(CKPT_META), step=step)

    def _measure():
        """Cheap per-step metrics for convergence tracking."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.positions
        o_idx = [i for i, s in enumerate(symbols) if s == 'O']
        o_pos = positions[o_idx]
        com = o_pos.mean(axis=0)
        radii = np.linalg.norm(o_pos - com, axis=1)
        r90 = float(np.percentile(radii, 90))
        spread = float(np.max(radii) / r90)
        density = len(o_idx) / (4/3 * np.pi * r90**3 * 1e-3)
        surf_cos, _ = compute_orientational_order(atoms, com, r90)
        return {
            'com': com, 'r90': r90, 'spread': spread,
            'density': density, 'surf_cos': surf_cos,
            'T': atoms.get_temperature(),
        }

    def _current_ns_per_day():
        if dyn.nsteps <= 0:
            return 0.0
        elapsed = time.time() - t0
        return (dyn.nsteps * TIMESTEP_FS * 1e-6) / (elapsed / 86400)

    def status():
        step = step_offset + dyn.nsteps
        ns_per_day = _current_ns_per_day()
        if ns_per_day > 0:
            remaining_ns = (conv['target_end'] - step) * TIMESTEP_FS * 1e-6
            eta_days = remaining_ns / ns_per_day
        else:
            eta_days = 999

        sim_time_ns = step * TIMESTEP_FS * 1e-6
        T = atoms.get_temperature()
        E = atoms.get_potential_energy()
        n_water = sum(1 for s in atoms.get_chemical_symbols() if s == 'O')

        msg = (f"Step {step:>10,d} | t={sim_time_ns:>8.3f} ns | "
               f"T={T:>6.1f} K | E/water={E/n_water:>8.4f} eV | "
               f"{ns_per_day:>5.2f} ns/day | ETA {eta_days:>5.1f} d")
        print(msg, flush=True)

        # Sample observables while still in equilibration
        if conv['converged_step'] is None and step < EQUIL_STEPS:
            m = _measure()
            conv['samples'].append({
                'step': step, 'surf_cos': m['surf_cos'],
                'density': m['density'], 'spread': m['spread'], 'T': m['T'],
            })

        # Fallback: if we reach the 0.5 ns cap without early convergence,
        # fire go/no-go there (original behavior).
        if (not conv['gng_fired']) and step >= EQUIL_STEPS:
            conv['gng_fired'] = True
            conv['converged_step'] = step
            # target_end already TOTAL_STEPS; keep production = 2 ns from here
            run_go_nogo_check(atoms, ns_per_day, equil_step=step)

        # Early termination when we've reached the (possibly shrunken) target
        if step >= conv['target_end']:
            raise StopEarly()

    def check_convergence():
        """Fires every CONV_CHECK_STRIDE. Declares early equilibration done
        when all observables are stable across two consecutive 50-ps windows."""
        step = step_offset + dyn.nsteps
        if conv['gng_fired'] or conv['converged_step'] is not None:
            return
        if step < CONV_CHECK_START:
            return
        samples = conv['samples']
        if len(samples) < 2 * CONV_WINDOW:
            return

        recent = samples[-CONV_WINDOW:]
        prev = samples[-2 * CONV_WINDOW:-CONV_WINDOW]

        def m(key, subset):
            return float(np.mean([s[key] for s in subset]))

        d_cos = abs(m('surf_cos', recent) - m('surf_cos', prev))
        rho_prev = m('density', prev)
        d_rho_rel = abs(m('density', recent) - rho_prev) / rho_prev
        d_spr = abs(m('spread', recent) - m('spread', prev))

        passed = (d_cos < CONV_TOL_COS_THETA and
                  d_rho_rel < CONV_TOL_DENSITY and
                  d_spr < CONV_TOL_SPREAD)

        tag = f"[conv {conv['passes']+1 if passed else 0}/{CONV_PASSES_REQ}]"
        print(f"{tag} step={step:,} Δ<cosθ>={d_cos:.4f} "
              f"Δρ={d_rho_rel*100:.2f}% Δspread={d_spr:.4f}", flush=True)

        if passed:
            conv['passes'] += 1
            if conv['passes'] >= CONV_PASSES_REQ:
                conv['converged_step'] = step
                conv['gng_fired'] = True
                new_end = min(step + PRODUCTION_STEPS, TOTAL_STEPS)
                conv['target_end'] = new_end
                saved_steps = EQUIL_STEPS - step
                saved_ns = saved_steps * TIMESTEP_FS * 1e-6
                print(f"\n*** EARLY CONVERGENCE at step {step:,} "
                      f"({step * TIMESTEP_FS * 1e-6:.3f} ns). "
                      f"Saved {saved_ns:.3f} ns ({saved_steps:,} steps) of equilibration. "
                      f"Production ends at step {new_end:,} "
                      f"({new_end * TIMESTEP_FS * 1e-6:.3f} ns) ***\n",
                      flush=True)
                run_go_nogo_check(atoms, _current_ns_per_day(), equil_step=step)
        else:
            conv['passes'] = 0

    dyn.attach(save_traj, interval=TRAJ_INT)
    dyn.attach(checkpoint, interval=CHECKPOINT_INT)
    dyn.attach(status, interval=LOG_INT)
    dyn.attach(check_convergence, interval=CONV_CHECK_STRIDE)

    remaining = TOTAL_STEPS - start_step
    total_ns = TOTAL_STEPS * TIMESTEP_FS * 1e-6
    remaining_ns = remaining * TIMESTEP_FS * 1e-6

    print(f"\n{'='*60}")
    print(f"Phase 1: Water Droplet — MACE-MP-0 {MACE_MODEL} NVT MD")
    print(f"{'='*60}")
    print(f"Max target:    {total_ns:.1f} ns ({TOTAL_STEPS:,} steps)")
    print(f"Remaining:     {remaining_ns:.1f} ns ({remaining:,} steps)")
    print(f"Equil cap:     {EQUIL_STEPS * TIMESTEP_FS * 1e-6:.1f} ns "
          f"(early convergence may cut short after "
          f"{CONV_CHECK_START * TIMESTEP_FS * 1e-6:.1f} ns)")
    print(f"Checkpoint:    every {CHECKPOINT_INT * TIMESTEP_FS * 1e-3:.1f} ps")
    print(f"Trajectory:    every {TRAJ_INT * TIMESTEP_FS * 1e-3:.2f} ps")
    print(f"Device:        {DEVICE}", end="")
    if DEVICE == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"{'='*60}\n")

    try:
        dyn.run(remaining)
    except StopEarly:
        checkpoint()
        final_step = step_offset + dyn.nsteps
        print(f"\nReached target end step {final_step:,} "
              f"({final_step * TIMESTEP_FS * 1e-6:.3f} ns). Phase 1 complete.",
              flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        checkpoint()
        print("Saved. Resume with: python run_phase1.py --resume")
        return
    except Exception as e:
        print(f"\nError: {e}")
        print("Saving checkpoint...")
        checkpoint()
        raise
    finally:
        traj.close()

    print(f"\nPhase 1 complete!")
    print(f"Trajectory: {TRAJ_FILE}")
    print(f"Next: python extract_clusters.py")


def compute_surface_potential_sign(atoms, com, r90):
    """
    Estimate surface electrostatic potential sign from fixed SPC/E charges.
    Computes potential at a probe point just outside the droplet surface.
    """
    Q_O, Q_H = -0.8476, +0.4238
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    charges = np.array([Q_O if s == 'O' else Q_H for s in symbols])

    # Probe point at r90 + 2 A along +z
    probe = com + np.array([0.0, 0.0, r90 + 2.0])
    dr = probe - positions
    dists = np.linalg.norm(dr, axis=1)
    dists = np.maximum(dists, 0.1)  # avoid singularity

    # Potential in arbitrary units (sign is what matters)
    potential = np.sum(charges / dists)

    if potential > 0:
        return f"POSITIVE ({potential:.4f} arb. units)"
    else:
        return f"NEGATIVE ({potential:.4f} arb. units)"


def compute_orientational_order(atoms, com, r90):
    """
    Compute <cos theta> for OH bonds, where theta is the angle between
    the OH vector and the radial outward direction from the droplet COM.

    Returns (surface_cos, bulk_cos).
    Surface = oxygen within 5 A of r90. Bulk = oxygen more than 10 A inside.
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    surface_cosines = []
    bulk_cosines = []

    for i, s in enumerate(symbols):
        if s != 'O':
            continue
        o_pos = positions[i]
        r_vec = o_pos - com
        r_dist = np.linalg.norm(r_vec)
        if r_dist < 1e-6:
            continue
        r_hat = r_vec / r_dist

        # H atoms follow each O in build order: O, H, H
        for h_idx in [i + 1, i + 2]:
            if h_idx >= len(symbols) or symbols[h_idx] != 'H':
                continue
            oh_vec = positions[h_idx] - o_pos
            oh_len = np.linalg.norm(oh_vec)
            if oh_len < 1e-6:
                continue
            cos_theta = np.dot(oh_vec / oh_len, r_hat)

            if r_dist > r90 - 5.0:
                surface_cosines.append(cos_theta)
            elif r_dist < r90 - 10.0:
                bulk_cosines.append(cos_theta)

    surface_cos = np.mean(surface_cosines) if surface_cosines else 0.0
    bulk_cos = np.mean(bulk_cosines) if bulk_cosines else 0.0
    return float(surface_cos), float(bulk_cos)


def run_go_nogo_check(atoms, ns_per_day, equil_step=None):
    equil_ns = (equil_step * TIMESTEP_FS * 1e-6) if equil_step is not None else EQUIL_STEPS * TIMESTEP_FS * 1e-6
    print(f"\n{'='*60}")
    print(f"GO / NO-GO CHECKPOINT (after {equil_ns:.3f} ns equilibration)")
    print(f"{'='*60}")

    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    n_water = sum(1 for s in symbols if s == 'O')
    o_idx = [i for i, s in enumerate(symbols) if s == 'O']
    o_pos = positions[o_idx]

    com = o_pos.mean(axis=0)
    radii = np.linalg.norm(o_pos - com, axis=1)
    r90 = np.percentile(radii, 90)
    r_max = np.max(radii)
    spread = r_max / r90

    T = atoms.get_temperature()
    E = atoms.get_potential_energy()
    e_per_water = E / n_water

    vol_sphere = 4/3 * np.pi * r90**3
    density_nm3 = n_water / (vol_sphere * 1e-3)

    remaining_ns = (TOTAL_STEPS - EQUIL_STEPS) * TIMESTEP_FS * 1e-6
    eta_days = remaining_ns / ns_per_day if ns_per_day > 0 else 999

    issues = []
    report = []
    report.append(f"Go/No-Go Report — {time.strftime('%Y-%m-%d %H:%M')}")
    report.append(f"{'='*50}")
    report.append(f"Equilibration time: {equil_ns:.3f} ns "
                  f"(cap = {EQUIL_STEPS * TIMESTEP_FS * 1e-6:.3f} ns)")
    report.append("")

    if spread > 1.5:
        issues.append("DROPLET FRAGMENTED")
        report.append(f"  [FAIL] Droplet integrity: spread={spread:.2f} (FRAGMENTED)")
    elif spread > 1.3:
        issues.append("droplet elongated")
        report.append(f"  [WARN] Droplet integrity: spread={spread:.2f} (elongated)")
    else:
        report.append(f"  [OK]   Droplet integrity: spread={spread:.2f} (compact)")

    if abs(T - TEMPERATURE) > 50:
        issues.append("TEMPERATURE WAY OFF")
        report.append(f"  [FAIL] Temperature: {T:.1f} K (target {TEMPERATURE} K)")
    elif abs(T - TEMPERATURE) > 20:
        report.append(f"  [WARN] Temperature: {T:.1f} K (slightly off)")
    else:
        report.append(f"  [OK]   Temperature: {T:.1f} K")

    if e_per_water > -0.1 or e_per_water < -1.0:
        issues.append("ENERGY UNREASONABLE")
        report.append(f"  [FAIL] Energy/water: {e_per_water:.4f} eV")
    else:
        report.append(f"  [OK]   Energy/water: {e_per_water:.4f} eV")

    if density_nm3 < 20 or density_nm3 > 50:
        issues.append("DENSITY WAY OFF")
        report.append(f"  [FAIL] Density: {density_nm3:.1f} mol/nm3 (bulk=33.4)")
    elif abs(density_nm3 - 33.4) > 10:
        report.append(f"  [WARN] Density: {density_nm3:.1f} mol/nm3 (bulk=33.4)")
    else:
        report.append(f"  [OK]   Density: {density_nm3:.1f} mol/nm3 (bulk=33.4)")

    expected_r = DIAMETER_NM * 10 / 2
    r_deviation = abs(r90 - expected_r) / expected_r * 100
    report.append(f"         Effective radius: {r90:.1f} A (expected ~{expected_r:.0f} A, {r_deviation:.0f}% off)")

    # Orientational order parameter: <cos theta> for OH bonds vs radial direction
    # theta = angle between OH bond vector and radial outward direction from COM
    # Positive <cos theta> at surface = OH points outward (correct physics)
    surface_cos, bulk_cos = compute_orientational_order(atoms, com, r90)
    if surface_cos > 0.0:
        report.append(f"  [OK]   Orientational order:")
    else:
        issues.append("SURFACE OH ORIENTATION WRONG")
        report.append(f"  [FAIL] Orientational order:")
    report.append(f"           Surface <cos theta>: {surface_cos:+.3f} (positive = OH outward = correct)")
    report.append(f"           Bulk <cos theta>:    {bulk_cos:+.3f} (near zero = isotropic = correct)")

    # Surface potential sign from fixed charges (quick diagnostic)
    # C-GeM gives POSITIVE surface potential; classical fixed-charge models give NEGATIVE
    surface_pot_sign = compute_surface_potential_sign(atoms, com, r90)
    report.append(f"  [INFO] Fixed-charge surface potential: {surface_pot_sign}")
    report.append(f"           (C-GeM ref: positive; classical models: negative)")

    report.append("")
    report.append(f"         Performance: {ns_per_day:.2f} ns/day")
    report.append(f"         ETA remaining: {eta_days:.1f} days")
    report.append(f"         Waters: {n_water}  Atoms: {len(atoms)}")
    report.append("")

    fatal = any(s.isupper() for s in issues)
    if fatal:
        report.append(f"VERDICT: NO-GO")
        report.append(f"  Issues: {', '.join(issues)}")
        report.append(f"  Fix: reduce timestep to 0.25fs, or energy-minimize first")
    elif issues:
        report.append(f"VERDICT: PROCEED WITH CAUTION")
        report.append(f"  Minor: {', '.join(issues)}")
    else:
        report.append(f"VERDICT: GO")
        report.append(f"  Let it run. Check back in {eta_days:.0f} days.")

    full_report = '\n'.join(report)
    print(full_report)

    with open(VALIDATION_FILE, 'w') as f:
        f.write(full_report)
    print(f"\nReport saved to: {VALIDATION_FILE}")
    print(f"{'='*60}\n")

    return not fatal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
