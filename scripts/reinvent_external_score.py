#!/usr/bin/env python3
"""REINVENT4 ExternalProcess bridge for MOSTGen reward reviewers.

REINVENT passes one full SMILES per stdin line.  The bridge deliberately scores
only canonical structures present in the versioned reaction library, thereby
keeping neural LibInvent generation inside the same commercial-synthon space
as both baselines. Unknown, invalid, vetoed, or wrong-family structures receive
zero. Output follows REINVENT4 ExternalProcess payload version 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mostgen.chemistry import ChemistryError, standardize_smiles  # noqa: E402
from mostgen.config import load_config  # noqa: E402
from mostgen.data import read_csv  # noqa: E402
from mostgen.reviewers import ReviewerBundle  # noqa: E402
from mostgen.scoring import ScoringContext  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=("chemistry", "spectrum", "most", "safety"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, mode="full")
    library = {}
    for row in read_csv(args.library):
        if row["family"] == args.family:
            library[row["smiles"]] = row
    requested = [line.strip() for line in sys.stdin if line.strip()]
    candidates = []
    positions = []
    scores = [0.0] * len(requested)
    for index, smiles in enumerate(requested):
        try:
            canonical = standardize_smiles(smiles)
        except ChemistryError:
            continue
        candidate = library.get(canonical)
        if candidate is not None:
            candidates.append(candidate)
            positions.append(index)
    if candidates:
        context = ScoringContext(config, ReviewerBundle.load(args.model))
        reviewed = context.review_candidates(candidates, "reinvent4_external", int(config["project"]["default_seed"]))
        for position, row in zip(positions, reviewed):
            if row.get("safety_veto") or not row.get("valid"):
                value = 0.0
            else:
                stages = json.loads(row.get("stage_rewards_json", "{}"))
                value = float(stages.get(args.stage, 0.0))
                if args.stage == "safety" and row.get("phototoxicity_uncertain"):
                    value = 0.0
            scores[position] = max(0.0, min(1.0, value))
    print(json.dumps({"version": 1, "payload": {"stage_score": scores}}))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
