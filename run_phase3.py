"""
Phase 3: 50ns production NVT MD with fine-tuned MACE model.
Run on g5.2xlarge.

Usage:
  python run_phase3.py --model mace_droplet.model
  python run_phase3.py --model mace_droplet.model --resume
"""
import argparse, time
import numpy as np
import torch
from pathlib import Path
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import MACECalculator

TIMESTEP_FS    = 0.5
TOTAL_STEPS    = 5_000_000    # 2.5 ns (0.5 ns equil + 2 ns production)
CHECKPOINT_INT = 2000
TRAJ_INT       = 200          # 0.1 ps
LOG_INT        = 2000
TEMPERATURE    = 300.0

WORK = Path("phase3")
WORK.mkdir(exist_ok=True)
RESTART   = WORK / "restart.xyz"
CKPT_META = WORK / "checkpoint.npz"
TRAJ_FILE = WORK / "trajectory.traj"
LOG_FILE  = WORK / "md.log"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_calc(model_path):
    return MACECalculator(
        model_paths=[model_path],
        device=DEVICE,
        default_dtype='float32',
    )


def run(model_path, resume=False):
    if resume and RESTART.exists():
        print(f"Resuming from {RESTART}")
        atoms = read(str(RESTART))
        atoms.calc = get_calc(model_path)
        meta = np.load(str(CKPT_META))
        start_step = int(meta['step'])
        print(f"  Resuming at step {start_step}")
    else:
        # Start from end of Phase 1
        atoms = read(str(Path("phase1") / "restart.xyz"))
        atoms.calc = get_calc(model_path)
        start_step = 0
        print(f"Starting Phase 3: {len(atoms)} atoms")

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

    def save_traj():
        traj.write(atoms)

    def checkpoint():
        step = step_offset + dyn.nsteps
        write(str(RESTART), atoms)
        np.savez(str(CKPT_META), step=step)

    def status():
        step = step_offset + dyn.nsteps
        elapsed = time.time() - t0
        if dyn.nsteps > 0:
            ns_per_day = (dyn.nsteps * TIMESTEP_FS * 1e-6) / (elapsed / 86400)
            remaining_ns = (TOTAL_STEPS - step) * TIMESTEP_FS * 1e-6
            eta_days = remaining_ns / ns_per_day if ns_per_day > 0 else 999
        else:
            ns_per_day, eta_days = 0, 999

        sim_time_ns = step * TIMESTEP_FS * 1e-6
        T = atoms.get_temperature()
        E = atoms.get_potential_energy()
        n_water = sum(1 for s in atoms.get_chemical_symbols() if s == 'O')

        msg = (f"Step {step:>10,d} | t={sim_time_ns:>8.3f} ns | "
               f"T={T:>6.1f} K | E/water={E/n_water:>8.4f} eV | "
               f"{ns_per_day:>5.2f} ns/day | ETA {eta_days:>5.1f} d")
        print(msg, flush=True)

    dyn.attach(save_traj, interval=TRAJ_INT)
    dyn.attach(checkpoint, interval=CHECKPOINT_INT)
    dyn.attach(status, interval=LOG_INT)

    remaining = TOTAL_STEPS - start_step
    total_ns = TOTAL_STEPS * TIMESTEP_FS * 1e-6
    remaining_ns = remaining * TIMESTEP_FS * 1e-6

    print(f"\n{'='*60}")
    print(f"Phase 3: Production MD — Fine-tuned MACE")
    print(f"{'='*60}")
    print(f"Model:         {model_path}")
    print(f"Total target:  {total_ns:.1f} ns ({TOTAL_STEPS:,} steps)")
    print(f"Remaining:     {remaining_ns:.1f} ns ({remaining:,} steps)")
    print(f"Trajectory:    every {TRAJ_INT * TIMESTEP_FS * 1e-3:.2f} ps")
    print(f"Device:        {DEVICE}", end="")
    if DEVICE == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"{'='*60}\n")

    try:
        dyn.run(remaining)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        checkpoint()
        print("Saved. Resume with: python run_phase3.py --model {model_path} --resume")
        return
    except Exception as e:
        print(f"\nError: {e}")
        print("Saving checkpoint...")
        checkpoint()
        raise
    finally:
        traj.close()

    print(f"\nPhase 3 complete!")
    print(f"Trajectory: {TRAJ_FILE}")
    print(f"Next: python analyze_efield.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned MACE model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.model, resume=args.resume)
