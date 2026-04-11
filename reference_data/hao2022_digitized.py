"""
Digitized reference data from:
Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022)
"Can electric fields drive chemistry for an aqueous microdroplet?"

DOI: 10.1038/s41467-021-27941-x (open access)

All data for the R40 (40 A radius, 8nm diameter) pure water droplet.
Values extracted from paper text (exact) and figures (approximate).
"""

# ---------------------------------------------------------------------------
# Exact values from paper text
# ---------------------------------------------------------------------------

# Section "Results", paragraph on electric fields:
# "electric field alignments along free O-H bonds at the surface are
#  ~16 MV/cm larger on average than that found for O-H bonds in the
#  interior of the water droplet"
SURFACE_FIELD_ENHANCEMENT_MV_PER_CM = 16.0

# The surface potential is POSITIVE with ReaxFF/C-GeM.
# Classical fixed-charge models (SPC/E, TIP4P) give NEGATIVE surface potential.
# This sign difference is the key finding of the paper.
SURFACE_POTENTIAL_SIGN = "positive"

# From Figure 2B / text: surface potential magnitude
SURFACE_POTENTIAL_EV = 1.0  # approximate, from Fig 2B

# Structural parameters
DROPLET_RADIUS_A = 40.0
N_WATER_APPROX = 9000
TEMPERATURE_K = 300.0
ENSEMBLE = "NVT"
FORCE_FIELD = "ReaxFF/C-GeM"

# From methods: simulation details
TIMESTEP_FS = 0.5
EQUILIBRATION_NS = 0.5
PRODUCTION_NS = 2.0

# From Figure 2A inset: surface layer definitions
# L0 = outermost molecular layer (surface)
# L1 = subsurface layer
# L2+ = interior/bulk
# Surface waters defined by instantaneous Willard-Chandler surface

# ---------------------------------------------------------------------------
# Digitized from Figure 2B: Radial electric potential profile
# Approximate values read from the figure for the C-GeM (reactive) model
# r in Angstrom, potential in eV
# Interior is roughly flat, rises sharply at the surface
# ---------------------------------------------------------------------------
FIG2B_RADIAL_POTENTIAL_CGEM = [
    # (r_angstrom, potential_eV)
    (0.0, 0.0),
    (5.0, 0.02),
    (10.0, 0.03),
    (15.0, 0.03),
    (20.0, 0.04),
    (25.0, 0.05),
    (30.0, 0.08),
    (32.0, 0.15),
    (34.0, 0.30),
    (36.0, 0.55),
    (38.0, 0.80),
    (39.0, 0.90),
    (40.0, 1.00),
    (41.0, 1.00),
    (42.0, 0.95),
]

# For comparison: SPC/Fw fixed-charge model gives NEGATIVE potential at surface
FIG2B_RADIAL_POTENTIAL_SPCFW = [
    # (r_angstrom, potential_eV)
    (0.0, 0.0),
    (10.0, 0.0),
    (20.0, 0.0),
    (30.0, -0.02),
    (34.0, -0.10),
    (36.0, -0.25),
    (38.0, -0.45),
    (39.0, -0.55),
    (40.0, -0.60),
    (41.0, -0.55),
    (42.0, -0.40),
]

# ---------------------------------------------------------------------------
# From Figure 4A: E-field projected onto OH bonds
# Mean and approximate width of distributions
# Interior OH: centered near 0 MV/cm, std ~30 MV/cm
# Surface free OH: shifted by +16 MV/cm, broader distribution
# ---------------------------------------------------------------------------
EFIELD_INTERIOR_MEAN_MV_PER_CM = 0.0    # approximately centered at zero
EFIELD_INTERIOR_STD_MV_PER_CM = 30.0    # approximate Gaussian width
EFIELD_SURFACE_MEAN_MV_PER_CM = 16.0    # shifted by the enhancement
EFIELD_SURFACE_STD_MV_PER_CM = 35.0     # broader than interior

# ---------------------------------------------------------------------------
# Key qualitative findings for validation
# ---------------------------------------------------------------------------

# 1. Surface OH bonds preferentially point outward (positive <cos theta>)
# 2. The E-field enhancement is driven by charge transfer at the surface,
#    not geometric reorientation alone
# 3. Fixed-charge models get the SIGN wrong for the surface potential
# 4. The E-field distribution has Lorentzian (heavy) tails, not Gaussian
# 5. Dangling OH bonds at the surface experience the strongest fields


def get_cgem_potential_interpolator():
    """Return an interpolation function for the C-GeM radial potential."""
    import numpy as np
    from scipy.interpolate import interp1d
    r = np.array([p[0] for p in FIG2B_RADIAL_POTENTIAL_CGEM])
    v = np.array([p[1] for p in FIG2B_RADIAL_POTENTIAL_CGEM])
    return interp1d(r, v, kind='cubic', bounds_error=False, fill_value=0.0)


def get_spcfw_potential_interpolator():
    """Return an interpolation function for the SPC/Fw radial potential."""
    import numpy as np
    from scipy.interpolate import interp1d
    r = np.array([p[0] for p in FIG2B_RADIAL_POTENTIAL_SPCFW])
    v = np.array([p[1] for p in FIG2B_RADIAL_POTENTIAL_SPCFW])
    return interp1d(r, v, kind='cubic', bounds_error=False, fill_value=0.0)
