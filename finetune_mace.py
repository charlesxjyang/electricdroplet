"""
Fine-tune MACE-MP-0 on DFT cluster data.
Run on g5.2xlarge (GPU) after DFT results are collected.

Usage: python finetune_mace.py --dft-dir dft_results/
"""
import argparse
import json
import numpy as np
import subprocess
import sys
from pathlib import Path
from ase import Atoms
from ase.io import write


def collect_training_data(dft_dir, output_xyz):
    """Convert DFT JSON results to extended XYZ for MACE training."""
    dft_dir = Path(dft_dir)
    results = sorted(dft_dir.glob("cluster_*.json"))

    print(f"Collecting training data from {len(results)} DFT results...")

    atoms_list = []
    skipped = 0
    for jf in results:
        with open(jf) as f:
            data = json.load(f)

        if data["status"] != "ok":
            skipped += 1
            continue

        atoms = Atoms(
            symbols=data["symbols"],
            positions=np.array(data["positions_ang"]),
        )
        atoms.info["REF_energy"] = data["energy_ev"]
        atoms.arrays["REF_forces"] = np.array(data["forces_ev_per_ang"])
        atoms.info["config_type"] = "dft_cluster"
        atoms_list.append(atoms)

    print(f"  Valid: {len(atoms_list)}, Skipped: {skipped}")

    # Shuffle and split 90/10
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(atoms_list))
    n_train = int(0.9 * len(atoms_list))

    train_atoms = [atoms_list[i] for i in indices[:n_train]]
    test_atoms = [atoms_list[i] for i in indices[n_train:]]

    train_file = output_xyz.replace(".xyz", "_train.xyz")
    test_file = output_xyz.replace(".xyz", "_test.xyz")

    write(train_file, train_atoms, format="extxyz")
    write(test_file, test_atoms, format="extxyz")

    print(f"  Train: {len(train_atoms)} -> {train_file}")
    print(f"  Test:  {len(test_atoms)} -> {test_file}")
    return train_file, test_file


def finetune(train_file, test_file, output_model="mace_droplet.model"):
    """Run MACE fine-tuning via CLI."""
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", "mace_droplet",
        "--foundation_model", "large",
        "--train_file", train_file,
        "--valid_file", test_file,
        "--energy_key", "REF_energy",
        "--forces_key", "REF_forces",
        "--model", "MACE",
        "--num_interactions", "2",
        "--max_num_epochs", "200",
        "--lr", "0.0001",
        "--batch_size", "4",
        "--patience", "20",
        "--default_dtype", "float32",
        "--device", "cuda",
        "--seed", "42",
        "--loss", "weighted",
        "--energy_weight", "1.0",
        "--forces_weight", "100.0",
        "--output_dir", "finetune_output",
    ]

    print(f"\nStarting MACE fine-tuning...")
    print(f"  Foundation model: MACE-MP-0 large")
    print(f"  Train: {train_file}")
    print(f"  Test:  {test_file}")
    print(f"  Output: finetune_output/")
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=True)

    # Copy best model
    best = Path("finetune_output") / "mace_droplet.model"
    if best.exists():
        import shutil
        shutil.copy(best, output_model)
        print(f"\nFine-tuned model saved to: {output_model}")
    else:
        print(f"\nWarning: expected model at {best} not found.")
        print("Check finetune_output/ for the trained model.")

    print(f"Next: python run_phase3.py --model {output_model}")


def main(dft_dir, output_model="mace_droplet.model"):
    dft_path = Path(dft_dir)

    # Pull DFT results from S3 if local dir is empty
    if not any(dft_path.glob("cluster_*.json")):
        dft_path.mkdir(exist_ok=True)
        from s3_config import DFT_RESULTS_S3, sync_down
        print("No local DFT results found, downloading from S3...")
        sync_down(DFT_RESULTS_S3, dft_path)
        print()

    train_file, test_file = collect_training_data(
        dft_dir, "training_data.xyz"
    )
    finetune(train_file, test_file, output_model)

    # Upload fine-tuned model to S3
    from s3_config import MODELS_S3, upload_file
    print(f"\nUploading model to S3...")
    upload_file(output_model, f"{MODELS_S3}/{output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-dir", required=True)
    parser.add_argument("--output-model", default="mace_droplet.model")
    args = parser.parse_args()
    main(args.dft_dir, args.output_model)
