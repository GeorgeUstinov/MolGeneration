# De novo UV/MOST generation

The primary artifact is `UV_MOST_Joint_RL.ipynb`. It trains two property
experts on disjoint molecular datasets, compares a prior baseline, joint RL,
and a UV-only ablation, then writes `generated.csv` and auditable metrics.

Run the complete workflow from the repository root with one command:

```bash
./scripts/run_uv_most_joint.sh
```

Inputs are downloaded from pinned primary-source URLs and cached under
`data/cache/`. The Photoswitch Dataset is used for the thermal half-life
expert and UVVisML for the absorption-maximum expert. Source revisions and
SHA-256 hashes are recorded in `outputs/uv_most_joint/run_manifest.json`.

`generated.csv` contains computational screening candidates, not validated
UV absorbers, safe ingredients, or experimentally synthesised compounds.

## Applied photoswitch generation

`UV_Photoswitch_Application_RL.ipynb` extends the joint agent with an explicit
azo E/Z-pair constraint and transition-wavelength proxy models. Run it with:

```bash
./scripts/run_photoswitch_application.sh
```

The primary output is `generated_photoswitch.csv`. Every row contains a base
structure and two distinct RDKit-valid stereochemical states. PSS, quantum
yield, DeltaH, stored energy and cycling remain missing until physical or
experimental validation; the notebook does not manufacture those labels.

## One multiclass generator

`Unified_Multiclass_Photoswitch_RL.ipynb` trains one conditional SELFIES-GRU
with shared weights and the family tokens `<AZO>` and `<STILBENE>`. It does not
train separate generators for the two classes. Run it with:

```bash
./scripts/run_unified_multiclass_photoswitch.sh
```

The output is `generated_multiclass_photoswitch.csv`. Azo candidates use the
available M01 half-life and E/Z spectral proxies. Stilbene-like candidates use
an explicit diaryl-alkene E/Z pair and the UVVisML absorption proxy; unavailable
half-life, PSS, quantum yield, DeltaH and stored energy remain missing.
