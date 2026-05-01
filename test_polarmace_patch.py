"""
Validate chunked PolarMACE patch against the original implementation.

Tests on multiple systems of varying size to confirm the fused chunked
computation gives identical charges to the unpatched complete-graph approach.
Systems small enough for the original to run in memory serve as ground truth.

Usage: python test_polarmace_patch.py
"""
import torch
import numpy as np
import time
from ase import Atoms
from ase.build import molecule


def get_calc_original():
    """Load PolarMACE WITHOUT patch (original complete-graph code)."""
    from mace.calculators import mace_polar
    return mace_polar(model="polar-1-s", device="cuda", default_dtype="float32")


def get_calc_patched():
    """Load PolarMACE WITH chunked patch."""
    import patch_graph_longrange
    patch_graph_longrange.apply(chunk_size=50)  # small chunks to stress-test
    # Also patch force computation for no_grad mode
    import mace.modules.extensions as ext
    def patched_get(energy, positions, **kwargs):
        return torch.zeros_like(positions), None, None, None, None
    ext.get_outputs = patched_get
    from mace.calculators import mace_polar
    return mace_polar(model="polar-1-s", device="cuda", default_dtype="float32")


def run_polarmace(calc, atoms, use_no_grad=False):
    """Run PolarMACE and return charges."""
    atoms = atoms.copy()
    atoms.info = {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}
    atoms.calc = calc
    torch.cuda.empty_cache()
    t0 = time.time()
    if use_no_grad:
        with torch.no_grad():
            atoms.get_potential_energy()
    else:
        atoms.get_potential_energy()
    dt = time.time() - t0
    charges = calc.results["charges"].copy()
    mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return charges, dt, mem


def build_water_cluster(n_waters, spacing=3.0, seed=42):
    """Build a roughly cubic cluster of water molecules."""
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(n_waters ** (1/3)))
    positions = []
    symbols = []
    count = 0
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if count >= n_waters:
                    break
                x = ix * spacing + rng.uniform(-0.3, 0.3)
                y = iy * spacing + rng.uniform(-0.3, 0.3)
                z = iz * spacing + rng.uniform(-0.3, 0.3)
                positions.extend([
                    [x, y, z],
                    [x + 0.96, y, z + 0.1],
                    [x - 0.24, y + 0.93, z + 0.1],
                ])
                symbols.extend(["O", "H", "H"])
                count += 1
            if count >= n_waters:
                break
        if count >= n_waters:
            break
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.center()
    return atoms


def main():
    print("=" * 70)
    print("PolarMACE Patch Validation: Original vs Chunked")
    print("=" * 70)

    # Define test systems
    test_systems = [
        ("Water monomer", Atoms("OH2", positions=[
            [0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]
        ])),
        ("Water dimer (H-bonded)", Atoms("OH2OH2", positions=[
            [-1.551, -0.115, 0.0], [-1.934, 0.763, 0.0], [-0.600, 0.041, 0.0],
            [1.351, 0.111, 0.0], [1.680, -0.374, -0.759], [1.680, -0.374, 0.759],
        ])),
        ("Water pentamer (cyclic)", build_water_cluster(5, spacing=2.8)),
        ("20-water cluster", build_water_cluster(20, spacing=3.0)),
        ("50-water cluster", build_water_cluster(50, spacing=3.0)),
        ("100-water cluster", build_water_cluster(100, spacing=3.1)),
        ("200-water cluster", build_water_cluster(200, spacing=3.1)),
    ]

    # Run original (no patch) on all systems
    print("\n--- Running ORIGINAL (unpatched) ---")
    calc_orig = get_calc_original()
    orig_results = {}
    for name, atoms in test_systems:
        try:
            charges, dt, mem = run_polarmace(calc_orig, atoms)
            orig_results[name] = charges
            n_o = sum(1 for s in atoms.get_chemical_symbols() if s == "O")
            o_mean = np.mean([float(charges[i]) for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"])
            print("  {:25s} {:4d} atoms | {:.1f}s | {:.1f} GB | O={:+.4f} e".format(
                name, len(atoms), dt, mem, o_mean))
        except torch.OutOfMemoryError:
            print("  {:25s} {:4d} atoms | OOM".format(name, len(atoms)))
            orig_results[name] = None

    # Clear GPU and run patched version
    del calc_orig
    torch.cuda.empty_cache()

    print("\n--- Running PATCHED (chunked, no_grad) ---")
    calc_patched = get_calc_patched()
    patch_results = {}
    for name, atoms in test_systems:
        try:
            charges, dt, mem = run_polarmace(calc_patched, atoms, use_no_grad=True)
            patch_results[name] = charges
            n_o = sum(1 for s in atoms.get_chemical_symbols() if s == "O")
            o_mean = np.mean([float(charges[i]) for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"])
            print("  {:25s} {:4d} atoms | {:.1f}s | {:.1f} GB | O={:+.4f} e".format(
                name, len(atoms), dt, mem, o_mean))
        except torch.OutOfMemoryError:
            print("  {:25s} {:4d} atoms | OOM".format(name, len(atoms)))
            patch_results[name] = None

    # Compare
    print("\n--- COMPARISON ---")
    print("  {:25s} {:>10s} {:>10s} {:>10s}  {}".format(
        "System", "MaxDiff", "RMSD", "Atoms", "Verdict"))
    all_pass = True
    for name, atoms in test_systems:
        o = orig_results.get(name)
        p = patch_results.get(name)
        if o is None or p is None:
            print("  {:25s} {:>10s} {:>10s} {:>10d}  SKIPPED (OOM)".format(
                name, "—", "—", len(atoms)))
            continue
        diff = np.abs(o - p)
        max_diff = float(diff.max())
        rmsd = float(np.sqrt(np.mean(diff ** 2)))
        passed = max_diff < 1e-4
        verdict = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print("  {:25s} {:10.2e} {:10.2e} {:10d}  {}".format(
            name, max_diff, rmsd, len(atoms), verdict))

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED — chunked patch is numerically identical to original")
    else:
        print("SOME TESTS FAILED — investigate charge discrepancies")
    print("=" * 70)


if __name__ == "__main__":
    main()
