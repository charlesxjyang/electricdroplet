# Paper Outline — JPC Letters

## Format Requirements
- **Words**: 1500–4000
- **Figures**: 3–5 (we plan 4)
- **Abstract**: ≤150 words
- **Structure**: no section headers except "Computational Methods"
- **TOC graphic**: 3.25 × 1.75 in

---

## Title (working)

Surface Electric Field Sign at the Water–Vacuum Interface Depends on
the Charge Model: Evidence from Machine Learning Molecular Dynamics

## Authors

Charles Yang

## Abstract (~150 words)

We compute the radial electric field at the air–water interface of a 5 nm
water nanodroplet (~2,100 molecules) using MACE-MP-0 machine learning
molecular dynamics with PolarMACE environment-dependent charges. The surface
field enhancement is −10.0 ± 1.0 MV/cm — the same magnitude as the +9 MV/cm
reported by Hao et al. using C-GeM, but with opposite sign. Fixed SPC/E
charges give −17 MV/cm, while PolarMACE reduces this to −10 MV/cm by
capturing environment-dependent polarization, but does not flip the sign.
PolarMACE charges are validated against DFT Mulliken charges on 289 water
clusters (dipole R² = 0.95). Our 5 nm droplet probes an untested curvature
regime below the R40–R80 range of prior work. We also introduce a
memory-efficient implementation of PolarMACE enabling 6,000+ atom systems
on commodity GPUs (48 GB).

---

## Introduction (1–2 paragraphs, no heading)

**Paragraph 1 — The question and prior work:**
- Electric fields at the air–water interface are central to microdroplet
  chemistry — rate enhancements, spontaneous redox, acid/base asymmetry
- Hao, Leven & Head-Gordon (Nat. Commun. 2022) computed the radial E-field
  of water nanodroplets using ReaxFF MD + C-GeM charges → ~9 MV/cm surface
  enhancement, positive, size-independent across R40–R80
- Subsequent work (Heindel, LaCour & Head-Gordon 2024; LaCour et al. 2025)
  linked surface charge and redox chemistry
- BUT: the sign of the water surface potential remains one of the most debated
  quantities in physical chemistry — different models give different signs
- No MLIP-based calculation of this quantity exists

**Paragraph 2 — What we do:**
- We use MACE-MP-0 (foundation MLIP, no water-specific parameterization) for
  MD on a 5 nm droplet (2,109 waters, 6,327 atoms)
- PolarMACE (MACE-POLAR-1) for environment-dependent charges → E-field
- Compare to fixed SPC/E charges and Hao et al.'s C-GeM result
- 5 nm is a novel curvature regime (R = 25 Å) below Hao et al.'s smallest (R40)
- Technical: memory-efficient PolarMACE implementation for large systems

---

## Results and Discussion (embedded figures)

### Result 1 — Droplet equilibration and structure (→ Figure 1)
- 0.5 ns NVT Langevin equilibration at 300 K, 0.5 fs timestep
- Early convergence detection at 0.2 ns (all structural observables plateau)
- Density profile: bulk-like core (~33.4 mol/nm³), smooth decay at surface
- Orientational order: surface <cos θ> = +0.06 (OH outward, correct physics)
- Droplet is compact, spherical (spread ratio 1.16), not fragmenting

### Result 2 — Surface E-field: main finding (→ Figure 2)
- Fixed SPC/E: enhancement = −17.5 ± 1.5 MV/cm (negative, as expected)
- PolarMACE (full droplet, 6,327 atoms): enhancement = −10.0 ± 1.0 MV/cm
- Hao et al. C-GeM reference: +9 MV/cm
- Same magnitude, opposite sign
- PolarMACE moves the field 7 MV/cm toward positive vs SPC/E (environment-
  dependent charges reduce the overestimated fixed-charge field) but does
  not flip the sign
- Validated: shell-size sensitivity test shows convergence at 5 Å shell
  thickness — result is independent of interior treatment
- SNR = 10.4 (20 frames) — statistically unambiguous

### Result 3 — PolarMACE validation (→ Figure 3)
- 289 DFT clusters (revPBE-D3/def2-SVP) from the MD trajectory
- PolarMACE dipoles vs DFT Mulliken dipoles: R² = 0.950
- Water monomer dipole: PolarMACE 1.58 D vs expt 1.86 D (15% under)
  — SPC/E gives 2.39 D (29% over)
- PolarMACE is closer to gas-phase DFT; SPC/E is parameterized for bulk
- Flat water slab: PolarMACE gives −0.83 V surface potential
  (negative, consistent with many DFT studies)

### Result 4 — Charge environment dependence (→ Figure 4)
- PolarMACE O charges: −0.46 ± 0.03 e (vs SPC/E fixed −0.85 e)
- Charges vary with local environment — surface vs bulk
- The smaller PolarMACE charges directly explain the reduced (but still
  negative) surface enhancement vs SPC/E
- Charge magnitude comparison across radial shells

### Discussion — Why opposite sign?
- C-GeM uses a polarizable model with explicit coarse-grained electrons
  that redistribute at the interface → positive surface potential
- PolarMACE uses environment-dependent point charges from an equivariant
  neural network trained on molecular dipoles → negative potential
- The sign depends on HOW charge polarization is modeled, not on the
  underlying MD (both use adequate force fields)
- This is consistent with the broader literature ambiguity: DFT slab
  calculations (PBE) give negative; some hybrid functionals give positive
- Our result adds a new data point: MLIP-based charges agree with DFT
  sign, not with C-GeM sign
- Implication: claims about microdroplet chemistry that depend on the
  sign of the surface field require specification of the charge model

---

## Computational Methods (heading required)

- **System**: 5 nm water nanodroplet, 2,109 water molecules (6,327 atoms),
  built on cubic grid + FIRE energy minimization (210 steps, max force < 1 eV/Å)
- **MD**: MACE-MP-0 medium (foundation model, no water-specific training),
  float32, NVT Langevin thermostat at 300 K, friction γ = 0.01 fs⁻¹,
  timestep 0.5 fs, 0.2 ns equilibration + ≥0.5 ns production
- **Charge models**: (1) PolarMACE (MACE-POLAR-1-S) for environment-dependent
  charges; (2) fixed SPC/E charges (O: −0.8476 e, H: +0.4238 e) as baseline
- **E-field**: Coulomb radial component E·r̂ at Fibonacci-spiral probe points
  (50 per shell, 40 radial bins), 1 Å exclusion radius, averaged over ≥20
  production frames
- **DFT validation**: 289 stratified clusters (120 surface / 90 interface /
  90 bulk, 6 Å cutoff), revPBE-D3/def2-SVP via PySCF, Mulliken charges +
  dipoles for PolarMACE cross-validation
- **Memory optimization**: Fused chunked electrostatics for PolarMACE
  (complete-graph construction + feature computation in O(chunk × N) memory
  instead of O(N²)), enabling 6,327-atom systems on 48 GB GPUs.
  Code: github.com/charlesxjyang/electricdroplet
- **Hardware**: NVIDIA L40S GPU (RunPod), ~0.03 ns/day MD throughput

---

## Figures

### Figure 1 — Droplet structure
- **(a)** Radial density profile ρ(r) with bulk water reference line
- **(b)** Orientational order parameter <cos θ>(r)
- Caption: "Radial profiles of the equilibrated 5 nm water nanodroplet
  (2,109 molecules). (a) Number density shows bulk-like core with smooth
  interfacial decay. (b) Orientational order parameter reveals preferential
  OH-outward orientation at the surface (<cos θ> > 0), consistent with
  prior simulations."

### Figure 2 — E-field profiles (MAIN RESULT)
- Radial E-field E·r̂ for SPC/E (blue) and PolarMACE (red) with shaded
  error bands. Dashed line at Hao et al. +9 MV/cm. Vertical line at r90.
- Caption: "Radial electric field (E·r̂) of the water nanodroplet from
  fixed SPC/E charges (blue) and PolarMACE environment-dependent charges
  (red). Both models show negative surface enhancement, in contrast to the
  positive +9 MV/cm reported by Hao et al. using C-GeM (dashed green).
  PolarMACE reduces the magnitude from −17 to −10 MV/cm by capturing
  environment-dependent polarization but does not change the sign."

### Figure 3 — PolarMACE validation
- **(a)** Dipole parity: PolarMACE |μ| vs DFT |μ| for 289 clusters.
  R² = 0.95. Color by stratum (surface/interface/bulk).
- **(b)** Water monomer + dimer dipoles: PolarMACE vs SPC/E vs experiment
- Caption: "Validation of PolarMACE charges against DFT. (a) Cluster
  dipole magnitudes (R² = 0.95, 289 clusters from MD trajectory at
  revPBE-D3/def2-SVP). (b) Gas-phase water monomer and dimer dipoles
  compared to experiment — PolarMACE (15% under) is closer to gas-phase
  DFT than SPC/E (29% over)."

### Figure 4 — Charge analysis
- PolarMACE O and H charges vs radial position, with SPC/E horizontal
  reference lines. Show charge variation at the interface.
- Caption: "PolarMACE per-atom charges as a function of radial position.
  Oxygen charges (red) and hydrogen charges (blue) show environment
  dependence — surface molecules have different charges than bulk,
  in contrast to the fixed SPC/E model (dashed lines)."

---

## Supporting Information

- Early convergence detection methodology and tolerances
- Shell-size sensitivity test for PolarMACE (Table: 4–8 Å shells converge)
- X23 benchmark validation of the chunked PolarMACE implementation (46 structures)
- DFT functional comparison: revPBE-D3 vs revPBE0-D3 on surface clusters
- Flat water slab surface potential (PolarMACE vs SPC/E vs DFT literature)
- Intermediate E-field convergence time series
- Full go/no-go report
- HOMO-LUMO gap distribution for DFT clusters

---

## Key References

1. Hao, Leven & Head-Gordon, Nat. Commun. 13, 280 (2022) — C-GeM E-field
2. Heindel, LaCour & Head-Gordon, Nat. Commun. 15, 3670 (2024) — charge + redox
3. LaCour et al., JACS 147, 6299 (2025) — interfaces + charge review
4. Batatia et al., JCP 163, 184110 (2025) — MACE-MP-0
5. Baldwin et al., arXiv:2602.19411 (2026) — MACE-POLAR-1
6. Xia et al., PNAS 122, e2519491122 (2025) — curvature-dependent reactivity
7. Kathmann et al., JPCB 115, 4369 (2011) — surface potential sign debate
8. Leung, JPC Lett. 1, 496 (2010) — DFT surface potential

---

## Estimated Word Count

- Abstract: ~150
- Introduction: ~400
- Results/Discussion: ~2000
- Computational Methods: ~500
- **Total: ~3050** (within 1500–4000 limit)
