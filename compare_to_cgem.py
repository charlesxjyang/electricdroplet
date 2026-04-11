"""
Generate publication-quality comparison figure:
  Our MACE/PolarMACE E-field vs C-GeM reference (Hao et al. 2022).

Reads output from analyze_efield.py and reference data.
Produces PNG (ChemRxiv) and PDF (JPC Letters).

Usage: python compare_to_cgem.py [--analysis-dir analysis/]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from reference_data.hao2022_digitized import (
    SURFACE_FIELD_ENHANCEMENT_MV_PER_CM,
    EFIELD_INTERIOR_MEAN_MV_PER_CM,
    EFIELD_SURFACE_MEAN_MV_PER_CM,
    FIG2B_RADIAL_POTENTIAL_CGEM,
    FIG2B_RADIAL_POTENTIAL_SPCFW,
    DROPLET_RADIUS_A,
)


def main(analysis_dir="analysis"):
    analysis_dir = Path(analysis_dir)
    data = np.load(analysis_dir / "efield_analysis.npz", allow_pickle=True)

    r = data['r_angstrom']
    r90 = float(data['r90_angstrom'])
    efield_fixed = data['efield_fixed_mean_mvcm']

    has_polar = 'efield_polar_mean_mvcm' in data
    if has_polar:
        efield_polar = data['efield_polar_mean_mvcm']

    # Convert r to distance from surface (positive = outside, negative = inside)
    r_from_surface = r - r90

    # --- Figure 1: E-field comparison (main result) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: E-field vs radial distance
    ax = axes[0]
    ax.plot(r, efield_fixed, 'b-', linewidth=1.5, label='MACE + fixed charges (SPC/E)')
    if has_polar:
        ax.plot(r, efield_polar, 'r-', linewidth=2, label='MACE + PolarMACE charges')

    # Reference annotation
    ax.axhline(SURFACE_FIELD_ENHANCEMENT_MV_PER_CM, color='green', linestyle=':',
               alpha=0.7, linewidth=1.5)
    ax.annotate(f'C-GeM surface enhancement\n({SURFACE_FIELD_ENHANCEMENT_MV_PER_CM:.0f} MV/cm)',
                xy=(r90 - 8, SURFACE_FIELD_ENHANCEMENT_MV_PER_CM),
                xytext=(r90 - 20, SURFACE_FIELD_ENHANCEMENT_MV_PER_CM + 8),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    ax.axvline(r90, color='gray', linestyle='--', alpha=0.5)
    ax.text(r90 + 0.5, ax.get_ylim()[1] * 0.9, 'surface', fontsize=9,
            color='gray', rotation=90, va='top')

    ax.set_xlabel('Distance from center (A)', fontsize=12)
    ax.set_ylabel('|E| (MV/cm)', fontsize=12)
    ax.set_title('(a) Radial electric field profile', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, r90 + 10)
    ax.grid(True, alpha=0.2)

    # Panel B: Surface potential comparison
    ax = axes[1]
    cgem_r = np.array([p[0] for p in FIG2B_RADIAL_POTENTIAL_CGEM])
    cgem_v = np.array([p[1] for p in FIG2B_RADIAL_POTENTIAL_CGEM])
    spc_r = np.array([p[0] for p in FIG2B_RADIAL_POTENTIAL_SPCFW])
    spc_v = np.array([p[1] for p in FIG2B_RADIAL_POTENTIAL_SPCFW])

    ax.plot(cgem_r, cgem_v, 'g-', linewidth=2, label='C-GeM (Hao et al. 2022)')
    ax.plot(spc_r, spc_v, 'b--', linewidth=1.5, label='SPC/Fw fixed charges (ref)')

    ax.axvline(DROPLET_RADIUS_A, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Distance from center (A)', fontsize=12)
    ax.set_ylabel('Electrostatic potential (eV)', fontsize=12)
    ax.set_title('(b) Reference: radial potential (Hao et al.)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 45)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Electric Field at the Water Microdroplet Interface\n'
                 '8nm droplet, MACE-MP-0 large, 300 K NVT',
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(analysis_dir / "comparison_cgem.png", dpi=300, bbox_inches='tight')
    fig.savefig(analysis_dir / "comparison_cgem.pdf", bbox_inches='tight')
    plt.close()

    # --- Print comparison summary ---
    surface_mask = (r > r90 - 5) & (r < r90 + 5)
    bulk_mask = r < r90 - 10

    print("Comparison with Hao et al. 2022 (C-GeM):")
    print(f"  Reference surface E-field enhancement: {SURFACE_FIELD_ENHANCEMENT_MV_PER_CM:.0f} MV/cm")
    print()

    sf = efield_fixed[surface_mask].mean() if surface_mask.any() else 0
    bf = efield_fixed[bulk_mask].mean() if bulk_mask.any() else 0
    print(f"  Fixed charges (SPC/E):")
    print(f"    Surface: {sf:.2f} MV/cm | Bulk: {bf:.2f} MV/cm | Enhancement: {sf-bf:.2f} MV/cm")

    if has_polar:
        sp = efield_polar[surface_mask].mean() if surface_mask.any() else 0
        bp = efield_polar[bulk_mask].mean() if bulk_mask.any() else 0
        print(f"  PolarMACE charges:")
        print(f"    Surface: {sp:.2f} MV/cm | Bulk: {bp:.2f} MV/cm | Enhancement: {sp-bp:.2f} MV/cm")

    print()
    print(f"  Figures saved to:")
    print(f"    {analysis_dir}/comparison_cgem.png (ChemRxiv)")
    print(f"    {analysis_dir}/comparison_cgem.pdf (JPC Letters)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()
    main(args.analysis_dir)
