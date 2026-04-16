"""
One-off: populate run_phase1.py's convergence-sample buffer from the existing
trajectory. Needed after an MD restart where the in-process samples list was
lost — recomputes the observables from trajectory.traj and writes them into
checkpoint.npz so the next --resume picks up with a full rolling buffer.

Usage:
  python prefill_conv_buffer.py [--stride 10]

Note: step labels are approximate — the 1-ps stride is correct but the
absolute step count across multiple resume points is not tracked by ASE's
trajectory format. This is fine for the convergence check, which only cares
about sample ordering and window-to-window comparison.
"""
import argparse
import numpy as np
from pathlib import Path
from ase.io.trajectory import Trajectory

TRAJ_FILE = Path("phase1/trajectory.traj")
CKPT_FILE = Path("phase1/checkpoint.npz")
TRAJ_INT = 200      # trajectory frame every 0.1 ps
LOG_INT = 2000      # one sample per 1 ps → take every 10th frame
TOTAL_STEPS = 5_000_000


def compute_metrics(atoms):
    sym = atoms.get_chemical_symbols()
    pos = atoms.positions
    o_idx = [i for i, s in enumerate(sym) if s == 'O']
    opos = pos[o_idx]
    com = opos.mean(axis=0)
    radii = np.linalg.norm(opos - com, axis=1)
    r90 = float(np.percentile(radii, 90))
    spread = float(radii.max() / r90)
    density = len(o_idx) / (4/3 * np.pi * r90**3 * 1e-3)

    surf_cos_vals = []
    for i, s in enumerate(sym):
        if s != 'O':
            continue
        rv = pos[i] - com
        rd = np.linalg.norm(rv)
        if rd < 1e-6:
            continue
        rhat = rv / rd
        for h in [i + 1, i + 2]:
            if h < len(sym) and sym[h] == 'H':
                ov = pos[h] - pos[i]
                ol = np.linalg.norm(ov)
                if ol > 1e-6 and rd > r90 - 5.0:
                    surf_cos_vals.append(float(np.dot(ov / ol, rhat)))
    surf_cos = float(np.mean(surf_cos_vals)) if surf_cos_vals else 0.0

    try:
        T = float(atoms.get_temperature())
    except Exception:
        T = 300.0  # fallback if velocities not in the frame

    return {
        'surf_cos': surf_cos,
        'density': density,
        'spread': spread,
        'T': T,
        'r90': r90,
    }


def main(stride):
    traj = Trajectory(str(TRAJ_FILE))
    n_frames = len(traj)
    print(f"Trajectory: {n_frames} frames ({n_frames * 0.1:.1f} ps)")

    # Read checkpoint
    ckpt = dict(np.load(CKPT_FILE, allow_pickle=True))
    current_step = int(ckpt['step'])
    print(f"Checkpoint step: {current_step:,}")

    samples = []
    frame_indices = list(range(0, n_frames, stride))
    print(f"Computing observables for {len(frame_indices)} frames "
          f"(stride {stride} = {stride * 0.1:.1f} ps between samples)...")

    for k, fi in enumerate(frame_indices):
        atoms = traj[fi]
        m = compute_metrics(atoms)
        # Approximate step label from frame index — see module docstring
        step = fi * TRAJ_INT
        samples.append({
            'step': step,
            'surf_cos': m['surf_cos'],
            'density': m['density'],
            'spread': m['spread'],
            'T': m['T'],
        })
        if (k + 1) % 25 == 0 or k == len(frame_indices) - 1:
            print(f"  {k+1}/{len(frame_indices)}  "
                  f"step≈{step:,}  surf_cos={m['surf_cos']:+.4f}  "
                  f"ρ={m['density']:.2f}  spread={m['spread']:.4f}")
    traj.close()

    print(f"\nWriting {len(samples)} samples to checkpoint...")
    np.savez(
        str(CKPT_FILE),
        step=current_step,
        conv_samples=np.array(samples, dtype=object),
        conv_passes=0,
        conv_converged_step=np.array(None, dtype=object),
        conv_target_end=TOTAL_STEPS,
        conv_gng_fired=False,
    )
    print(f"Done. Next --resume will restore {len(samples)} samples into conv buffer.")
    print("First automatic convergence check will fire at step 400,000 as originally planned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10,
                        help="Take every Nth trajectory frame (10 = 1 ps between samples, default)")
    args = parser.parse_args()
    main(args.stride)
