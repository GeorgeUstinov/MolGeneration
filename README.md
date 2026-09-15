# MOSTGen

`mostgen` is a reproducible research prototype for reaction-constrained design
and conservative computational triage of single-molecule UV-absorbing
molecular solar-thermal (MOST) photoswitches.  It implements the complete
experiment contract—data provenance, family-aware generation, three matched
search strategies, independent reviewers, hard safety vetoes, applicability
domains, uncertainty-aware rewards, metrics, review cards, and reports—while
remaining runnable on a CPU with Python's standard library.

The built-in backend is an intentionally labelled **surrogate smoke backend**.
It is suitable for testing the pipeline, not for making chemical or safety
claims.  Production mode exports pinned REINVENT4/LibInvent job manifests and
queues top candidates for GFN2-xTB plus sTDA-xTB.  It fails closed when those
external tools, a reaction prior, or real labelled datasets are unavailable.

## Quick start

```bash
python -m mostgen run-all --mode smoke --output runs/smoke
python -m unittest discover -s tests -v
```

A full CPU benchmark uses the acceptance budget (three methods, three seeds,
at least 1,000 valid unique molecules per run):

```bash
python -m mostgen run-all --mode full --output runs/full
```

Individual commands are available through `python -m mostgen --help`:
`prepare-data`, `train-reviewers`, `sample-baselines`, `train-generator`,
`generate`, `review`, and `run-all`.

## Scientific boundary

Outputs are screening proxies and research candidates.  They are not evidence
of safety, cosmetic suitability, efficacy, or compliance with ISO 24444:2019
or ISO 24443:2021.  A finished formulation/film must be tested, and
phototoxicity requires experimental confirmation (for example OECD TG 432).
No candidate with an uncertain phototoxicity assessment is shortlisted.

See `docs/METHOD.md` for the implementation mapping, model-card limitations,
and the production REINVENT4/xTB hand-off.

