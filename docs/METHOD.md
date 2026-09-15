# Implementation and scientific contract

## What is implemented

The package exposes one CLI with `prepare-data`, `train-reviewers`,
`sample-baselines`, `train-generator`, `generate`, `review`, and `run-all`.
Every output directory contains the resolved configuration, package and Python
versions, source registry, local input SHA-256 values, seeds, budget, an event
log, model cards, generated structures, metrics, review cards, and reports.

The family definitions in `config/default.json` contain fixed ground/charged
templates, reaction SMARTS, allowed attachment positions, and a versioned
synthon library.  `build_pair` canonicalizes both products with RDKit and
requires equal molecular formulae.  Enumeration is deduplicated by canonical
family/SMILES identity.

Search methods share the same enumerated chemical space and exact reviewer
call budget:

1. `prior_random` samples without replacement;
2. `weighted_retraining` updates synthon weights from observed scores;
3. `libinvent_rl` uses chemistry → spectrum → MOST → safety curriculum stages,
   exploration, a dense geometric reward, and ECFP cluster penalties.

The third method is a CPU-testable policy surrogate.  It does **not** claim to
be a trained REINVENT network.  `train-generator` exports one pinned-input
manifest per family for the real REINVENT4/LibInvent hand-off.  Production mode
fails unless the REINVENT revision, executable, and reaction prior are present.

## Data and model separation

`prepare-data` never changes the original XLSX, DOCX, or PPTX.  It records their
SHA-256 digests and writes derivatives under the experiment directory.  The
built-in labels are deterministic fixtures marked `synthetic_smoke_only`.
Production mode refuses that evidence tier.

Murcko scaffold groups, not rows, determine train/validation/test assignments.
The loader rejects scaffold overlap.  Reward reviewers are Random Forest
ensembles; independent evaluators are separate Extra Trees ensembles.  Spectrum
is predicted on a 5 nm grid from 290 to 400 nm.  MOST models are fit separately
for NBD/QC, Dewar-pyrimidinone, and spiropyran.  Applicability domains are
nearest-reference Morgan similarity and are reported separately for spectrum
and family-specific MOST.

M08–M10 are not represented as downloadable training corpora.  The source
registry treats literature-only resources as orientation, never bulk data.
M01/M03/M04, M11–M13, and U07–U17 require licensed or open local records with
validated endpoints before a production run.

## Pass rules

The spectral curve produces trapezoidal UVB AUC (290–320), UVA AUC (320–400),
the wavelength containing 90% of cumulative 290–400 AUC, and conditional
Beer–Lambert transmittance.  UV-pass requires both AUC lower confidence bounds
to meet the family reference medians and λc lower bound ≥370 nm.

MOST-pass requires positive energy LCB, energy LCB at least the family median,
and predicted half-life at 305 K between 4 and 24 hours.  Both applicability
domains must pass.  Results outside an AD do not pass regardless of mean.

The reward is a weighted geometric mean with a nonzero floor.  Continuous
sigmoid and interval components cover UVB, UVA, λc, energy, half-life,
phototoxicity, permeation, SA, and AD.  `reward_diagnostics.csv` records
nonzero fraction, effective sample size, cluster concentration, and component
means.

## Safety and claim boundaries

Safety vetoes run before reviewer scoring.  RDKit SMARTS catch configured
linear/angular furocoumarin cores, known phototoxic structures, reactive groups,
unsupported elements, invalid valence, and pair failure.  An uncertain
phototoxicity prediction cannot enter a shortlist.  Absence of an alert does
not establish safety.

Independent review writes a structured card per candidate.  The physical
oracle queue contains separately embedded and force-field-minimized ground and
charged XYZ structures and explicit GFN2-xTB/sTDA-xTB commands.  Missing tools
produce `not_run_external_tools_unavailable`; no deterministic surrogate is
silently substituted.  Consequently final `selected=true` is fail-closed until
the external oracle result is imported and complete.

Candidates are research hypotheses, not cosmetic ingredients. ISO 24444:2019
and ISO 24443:2021 concern finished products. Film/formulation testing and an
experimental phototoxicity method such as OECD TG 432 remain required.

## Production data adapter contract

A replacement `reviewer_training.csv` must keep these columns:

- identity/provenance: `candidate_id`, `smiles`, `charged_smiles`, `family`,
  `source_id`, `source_kind`, `evidence_tier`;
- split semantics: `scaffold`, `split`, `state`, `temperature_k`, `medium`,
  `label_level`;
- endpoints: `abs_290` … `abs_400`, `energy_kj_mol`,
  `specific_energy_wh_kg`, `log_half_life_h`, `kp_log_cm_s`, and
  `phototoxicity_probability`.

Rows must distinguish molecule-, state-, condition-, and film-level labels.
Do not copy a formulation endpoint onto a molecule. Units must be normalized
before training, while original units and values remain in immutable raw data.

