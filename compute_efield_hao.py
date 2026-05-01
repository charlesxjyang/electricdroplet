"""
Compute the electric field using Hao et al.'s definition:
E-field at each H atom, projected onto the OH bond direction,
excluding the parent water molecule.

This is what the "16 MV/cm enhancement" and "+9 MV/cm" numbers
from Hao, Leven & Head-Gordon (Nat. Commun. 2022) actually measure.

Usage:
    from compute_efield_hao import compute_efield_oh
    r_bins, efield = compute_efield_oh(atoms, charges, com, bin_edges)
"""
import numpy as np

# Constants
E_CHARGE_C = 1.602176634e-19
ANGSTROM_TO_M = 1e-10
K_COULOMB = 8.9875517873681764e9
VM_TO_MVCM = 1e-8


def compute_efield_oh(atoms, charges_e, com, bin_edges):
    """Compute E·OH at each H atom, excluding the parent water molecule.

    For each H atom:
      1. Identify parent O (atoms are in O,H,H order from build_droplet.py)
      2. Compute Coulomb field at H from ALL other atoms except the 3 atoms
         of the parent water molecule (parent O + sibling H + self)
      3. Project onto the OH bond unit vector (H_pos - O_pos, normalized)
      4. Bin by radial position of the parent O

    Returns:
      bin_centers (array): radial bin centers in Angstrom
      efield_mvcm (array): mean E·OH in MV/cm per radial bin
      efield_std_mvcm (array): std of E·OH per bin
      counts (array): number of OH bonds per bin
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions  # Angstrom
    n_atoms = len(atoms)

    # Convert to SI for Coulomb calculation
    positions_m = positions * ANGSTROM_TO_M
    charges_C = charges_e * E_CHARGE_C

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # Collect E·OH values per bin
    bin_values = [[] for _ in range(n_bins)]

    for i, s in enumerate(symbols):
        if s != 'H':
            continue

        # Identify parent O: atoms are in O,H,H order
        if i >= 1 and symbols[i - 1] == 'O':
            parent_o = i - 1
            sibling_h = i + 1 if (i + 1 < n_atoms and symbols[i + 1] == 'H') else None
        elif i >= 2 and symbols[i - 2] == 'O':
            parent_o = i - 2
            sibling_h = i - 1  # the other H
        else:
            continue  # can't identify parent

        # Exclude parent molecule (3 atoms: parent_o, self, sibling_h)
        exclude = {i, parent_o}
        if sibling_h is not None:
            exclude.add(sibling_h)

        # OH bond unit vector
        oh_vec = positions[i] - positions[parent_o]
        oh_len = np.linalg.norm(oh_vec)
        if oh_len < 1e-6:
            continue
        oh_hat = oh_vec / oh_len

        # Coulomb field at H from all other atoms (excluding parent molecule)
        E_total = np.zeros(3)
        h_pos_m = positions_m[i]
        for j in range(n_atoms):
            if j in exclude:
                continue
            dr = h_pos_m - positions_m[j]  # field point - source
            dist = np.linalg.norm(dr)
            if dist < 0.5 * ANGSTROM_TO_M:  # numerical safety (shouldn't happen)
                continue
            E_total += K_COULOMB * charges_C[j] * dr / (dist ** 3)

        # Project onto OH direction
        e_oh = np.dot(E_total, oh_hat)  # V/m, positive = along OH (outward for free OH)

        # Bin by radial position of parent O
        r_o = np.linalg.norm(positions[parent_o] - com)
        bi = np.searchsorted(bin_edges, r_o) - 1
        if 0 <= bi < n_bins:
            bin_values[bi].append(e_oh)

    # Aggregate
    efield_mean = np.zeros(n_bins)
    efield_std = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for bi in range(n_bins):
        if bin_values[bi]:
            efield_mean[bi] = np.mean(bin_values[bi])
            efield_std[bi] = np.std(bin_values[bi])
            counts[bi] = len(bin_values[bi])

    return bin_centers, efield_mean * VM_TO_MVCM, efield_std * VM_TO_MVCM, counts


def compute_efield_oh_fast(atoms, charges_e, com, bin_edges):
    """Vectorized version for speed on large systems.

    Same physics as compute_efield_oh but uses numpy broadcasting
    instead of Python loops over atom pairs. ~10x faster for 6000+ atoms.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.positions
    positions_m = positions * ANGSTROM_TO_M
    charges_C = charges_e * E_CHARGE_C
    n_atoms = len(atoms)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)
    bin_values = [[] for _ in range(n_bins)]

    # Identify all H atoms and their parent O
    h_indices = np.where(symbols == 'H')[0]

    for h_idx in h_indices:
        # Parent O identification (O,H,H ordering)
        if h_idx >= 1 and symbols[h_idx - 1] == 'O':
            parent_o = h_idx - 1
            sibling_h = h_idx + 1 if (h_idx + 1 < n_atoms and symbols[h_idx + 1] == 'H') else -1
        elif h_idx >= 2 and symbols[h_idx - 2] == 'O':
            parent_o = h_idx - 2
            sibling_h = h_idx - 1
        else:
            continue

        # OH bond direction
        oh_vec = positions[h_idx] - positions[parent_o]
        oh_len = np.linalg.norm(oh_vec)
        if oh_len < 1e-6:
            continue
        oh_hat = oh_vec / oh_len

        # Mask: exclude parent molecule
        mask = np.ones(n_atoms, dtype=bool)
        mask[h_idx] = False
        mask[parent_o] = False
        if sibling_h >= 0:
            mask[sibling_h] = False

        # Coulomb field at H from all non-excluded atoms
        dr = positions_m[h_idx] - positions_m[mask]  # (N-3, 3)
        dist = np.linalg.norm(dr, axis=1)            # (N-3,)
        dist_safe = np.maximum(dist, 0.5 * ANGSTROM_TO_M)

        E_vec = K_COULOMB * (charges_C[mask, None] * dr) / (dist_safe[:, None] ** 3)
        E_total = E_vec.sum(axis=0)  # (3,)

        e_oh = np.dot(E_total, oh_hat)

        r_o = np.linalg.norm(positions[parent_o] - com)
        bi = np.searchsorted(bin_edges, r_o) - 1
        if 0 <= bi < n_bins:
            bin_values[bi].append(e_oh)

    efield_mean = np.zeros(n_bins)
    efield_std = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for bi in range(n_bins):
        if bin_values[bi]:
            efield_mean[bi] = np.mean(bin_values[bi])
            efield_std[bi] = np.std(bin_values[bi])
            counts[bi] = len(bin_values[bi])

    return bin_centers, efield_mean * VM_TO_MVCM, efield_std * VM_TO_MVCM, counts
