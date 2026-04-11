# CLAUDE_CODE_INSTRUCTIONS.md
# Give this file to your local Claude Code instance.
# It contains everything needed to update the repo and prepare for the first run.

## Context

We're simulating an 8nm water nanodroplet to compute the radial electric field
profile at the air-water interface, using MACE-MP-0 large for dynamics and
PolarMACE for post-hoc charge assignment. The benchmark comparison is
Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022), who used a ReaxFF/C-GeM
reactive force field on the same size droplet (R=40 Å).

The paper has no public data repository. We need to digitize their Figure 2
and Figure 4 for comparison.

## Task 1: Update all scripts to MACE large

Already done. All scripts use model='large' and device='cuda'.

## Task 2: Update analyze_efield.py to use PolarMACE

Use `mace_polar(model='polar-1-s', ...)` from mace.calculators.
Requires mace-torch>=0.3.16 (develop branch) and graph_longrange.

## Task 3: Create reference data from Hao et al. 2022

See reference_data/hao2022_digitized.py.

## Task 4: Add structural validation to go/no-go checkpoint

Orientational order parameter <cos θ> for surface vs bulk.

## Task 5: Create compare_to_cgem.py

Publication-quality comparison figure.

## Task 6: Update PLAN.md

Add PolarMACE pipeline.
