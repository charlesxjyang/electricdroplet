"""
Generate all 4 publication figures for JPC Letters.

Reads pre-computed analysis results from analysis/ directory.
Outputs PNG (300 dpi, for ChemRxiv) and PDF (vector, for journal).

Usage: python paper/generate_figures.py [--analysis-dir analysis/]

Prerequisites: run analyze_efield.py and validate_polarmace_vs_dft.py first.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# JPC Letters column width
COL_WIDTH = 3.25  # inches
FULL_WIDTH = 7.0  # inches

# Consistent style
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


def fig1_structure(data, out_dir):
    """Figure 1: Droplet density and orientational order profiles."""
    r = data['r_angstrom']
    r90 = float(data['r90_angstrom'])
    density = data['density_mean_per_nm3']
    orient = data['orient_cos_mean']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    # (a) Density
    ax1.plot(r, density, 'k-', linewidth=1.5)
    ax1.axhline(33.4, color='steelblue', ls=':', lw=0.8, label='Bulk (33.4 nm$^{-3}$)')
    ax1.axvline(r90, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.set_xlabel('$r$ (\\AA)')
    ax1.set_ylabel('$\\rho$ (molecules nm$^{-3}$)')
    ax1.set_xlim(0, r90 + 12)
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Orientational order
    ax2.plot(r, orient, 'k-', linewidth=1.5)
    ax2.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax2.axvline(r90, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('$r$ (\\AA)')
    ax2.set_ylabel('$\\langle\\cos\\theta\\rangle$')
    ax2.set_xlim(0, r90 + 12)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(out_dir / 'fig1_structure.png', bbox_inches='tight')
    fig.savefig(out_dir / 'fig1_structure.pdf', bbox_inches='tight')
    plt.close()
    print('  Fig 1: structure profiles')


def fig2_efield(data, out_dir):
    """Figure 2: E-field profiles — the main result."""
    r = data['r_angstrom']
    r90 = float(data['r90_angstrom'])
    ef_fixed = data['efield_fixed_mean_mvcm']
    ef_fixed_std = data['efield_fixed_std_mvcm']

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8))

    ax.plot(r, ef_fixed, 'steelblue', lw=1.5, label='Fixed SPC/E')
    ax.fill_between(r, ef_fixed - ef_fixed_std, ef_fixed + ef_fixed_std,
                     color='steelblue', alpha=0.15)

    if 'efield_polar_mean_mvcm' in data:
        ef_polar = data['efield_polar_mean_mvcm']
        ef_polar_std = data['efield_polar_std_mvcm']
        ax.plot(r, ef_polar, 'crimson', lw=1.5, label='PolarMACE')
        ax.fill_between(r, ef_polar - ef_polar_std, ef_polar + ef_polar_std,
                         color='crimson', alpha=0.15)

    # Hao et al. reference
    ax.axhline(9.0, color='forestgreen', ls=':', lw=1.0, alpha=0.7,
               label='Hao et al. C-GeM (+9)')
    ax.axhline(0, color='gray', ls='-', lw=0.4, alpha=0.3)
    ax.axvline(r90, color='gray', ls='--', lw=0.8, alpha=0.5,
               label='$r_{90}$ = %.0f \\AA' % r90)

    ax.set_xlabel('$r$ (\\AA)')
    ax.set_ylabel('$E \\cdot \\hat{r}$ (MV cm$^{-1}$)')
    ax.set_xlim(0, r90 + 12)
    ax.legend(frameon=False, loc='lower left')

    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_efield.png', bbox_inches='tight')
    fig.savefig(out_dir / 'fig2_efield.pdf', bbox_inches='tight')
    plt.close()
    print('  Fig 2: E-field profiles')


def fig3_validation(analysis_dir, out_dir):
    """Figure 3: PolarMACE vs DFT validation."""
    val_file = analysis_dir / 'polarmace_vs_dft.npz'
    if not val_file.exists():
        print('  Fig 3: SKIPPED (run validate_polarmace_vs_dft.py first)')
        return

    v = np.load(val_file, allow_pickle=True)
    dft_dip = v['dft_dipoles']
    pm_dip = v['pm_dipoles']
    r2 = float(v['r2_dipole'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    # (a) Dipole parity
    ax1.scatter(dft_dip, pm_dip, s=12, alpha=0.5, c='steelblue', edgecolors='none')
    lo = min(dft_dip.min(), pm_dip.min())
    hi = max(dft_dip.max(), pm_dip.max())
    ax1.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax1.set_xlabel('DFT $|\\mu|$ (Debye)')
    ax1.set_ylabel('PolarMACE $|\\mu|$ (Debye)')
    ax1.text(0.05, 0.92, '$R^2$ = %.3f\n%d clusters' % (r2, len(dft_dip)),
             transform=ax1.transAxes, fontsize=7, va='top')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

    # (b) Monomer/dimer calibration
    models = ['PolarMACE', 'SPC/E', 'Experiment']
    mono = [1.582, 2.388, 1.855]
    dimer = [2.360, None, 2.643]  # no SPC/E dimer
    x = np.arange(2)
    w = 0.25
    colors = ['crimson', 'steelblue', 'gray']

    for i, (m, c) in enumerate(zip(models, colors)):
        vals = [mono[i], dimer[i] if dimer[i] else 0]
        bars = ax2.bar(x + (i - 1) * w, vals, w, label=m, color=c, alpha=0.8)
        if dimer[i] is None:
            bars[1].set_alpha(0)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Monomer', 'Dimer'])
    ax2.set_ylabel('$|\\mu|$ (Debye)')
    ax2.legend(frameon=False, fontsize=6)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(out_dir / 'fig3_validation.png', bbox_inches='tight')
    fig.savefig(out_dir / 'fig3_validation.pdf', bbox_inches='tight')
    plt.close()
    print('  Fig 3: PolarMACE validation')


def fig4_charges(data, out_dir):
    """Figure 4: PolarMACE charge distribution by radial position."""
    if 'polar_charges_o_mean' not in data:
        print('  Fig 4: SKIPPED (no PolarMACE charge data)')
        return

    # This figure needs per-frame radial charge data.
    # For now, generate from the summary statistics.
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    o_mean = float(data['polar_charges_o_mean'])
    o_std = float(data['polar_charges_o_std'])
    h_mean = float(data['polar_charges_h_mean'])
    h_std = float(data['polar_charges_h_std'])

    # Bar chart: PolarMACE vs SPC/E
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [o_mean, h_mean], w, yerr=[o_std, h_std],
           label='PolarMACE', color='crimson', alpha=0.8, capsize=3)
    ax.bar(x + w/2, [-0.8476, 0.4238], w,
           label='SPC/E (fixed)', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Oxygen', 'Hydrogen'])
    ax.set_ylabel('Charge ($e$)')
    ax.axhline(0, color='gray', lw=0.4)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / 'fig4_charges.png', bbox_inches='tight')
    fig.savefig(out_dir / 'fig4_charges.pdf', bbox_inches='tight')
    plt.close()
    print('  Fig 4: charge comparison')


def main(analysis_dir='analysis'):
    analysis_dir = Path(analysis_dir)
    out_dir = Path('paper/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    efield_file = analysis_dir / 'efield_analysis.npz'
    if not efield_file.exists():
        print('ERROR: run analyze_efield.py first to generate', efield_file)
        return

    data = dict(np.load(efield_file, allow_pickle=True))
    print('Generating JPC Letters figures...')

    fig1_structure(data, out_dir)
    fig2_efield(data, out_dir)
    fig3_validation(analysis_dir, out_dir)
    fig4_charges(data, out_dir)

    print('\nAll figures saved to', out_dir)
    print('  PNG at 300 dpi (ChemRxiv)')
    print('  PDF vector (JPC Letters)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', default='analysis')
    args = parser.parse_args()
    main(args.analysis_dir)
