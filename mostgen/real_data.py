from __future__ import annotations

"""Parsing of the versioned experimental sources used by production reviewers.

Every output row is long-form: one molecular endpoint under one set of
conditions.  No missing experimental value is imputed and no lambda-max record
is relabelled as a full spectrum.
"""

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import ChemistryError, murcko_scaffold, standardize_smiles
from .data import scaffold_split, write_csv
from .numerics import critical_wavelength, trapezoid_auc
from .provenance import sha256_file, stable_hash


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _standardize(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip().lower() in {"", "-", "nan", "none"}:
        return None
    try:
        return standardize_smiles(str(value))
    except (ChemistryError, TypeError, ValueError):
        return None


def _row(smiles: str, endpoint: str, target: float, source: str, **metadata: Any) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True) if mol is not None else ""
    scaffold = scaffold or smiles
    return {
        "record_id": stable_hash(f"{source}|{endpoint}|{smiles}|{metadata}|{target}", 24),
        "smiles": smiles,
        "family": metadata.pop("family", "unassigned"),
        "scaffold": scaffold,
        "split": scaffold_split(scaffold),
        "endpoint": endpoint,
        "target": target,
        "source_id": source,
        "evidence_tier": "experimental",
        **metadata,
    }


def parse_m13(path: Path, max_rows: int = 25_000) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, str]] = set()
    for record in frame.to_dict("records"):
        smiles = _standardize(record.get("smiles"))
        peak = _finite(record.get("peakwavs_max"))
        solvent = str(record.get("solvent") or "unknown")
        origin = str(record.get("source") or "unknown").strip().lower()
        # UV/VisML is the unified M13 distribution, but two of its constituent
        # tables are the M11/M12 sources from the project registry.  Preserve
        # that identity in every endpoint row instead of flattening all
        # provenance to M13.
        source_id = {"deep4chem": "M11", "cdex": "M12"}.get(origin, "M13")
        if smiles is None or peak is None or not 180.0 <= peak <= 900.0:
            continue
        key = (smiles, solvent, round(peak, 4), origin)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_row(smiles, "lambda_max_nm", peak, source_id, solvent=solvent, original_source=origin,
                         state="unspecified", label_level="molecule_condition"))
    # A content-derived selection is deterministic and independent of source order.
    rows.sort(key=lambda item: stable_hash(item["record_id"], 32))
    return rows[:max_rows]


def parse_m01(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    peak_columns = {
        "E isomer pi-pi* wavelength in nm": "E_pi_pi",
        "E isomer n-pi* wavelength in nm": "E_n_pi",
        "Z isomer pi-pi* wavelength in nm": "Z_pi_pi",
        "Z isomer n-pi* wavelength in nm": "Z_n_pi",
    }
    for record in frame.to_dict("records"):
        smiles = _standardize(record.get("SMILES"))
        if smiles is None:
            continue
        solvent = str(record.get("Irradiation solvent") or "unknown")
        for column, state in peak_columns.items():
            peak = _finite(record.get(column))
            if peak is not None and 180.0 <= peak <= 900.0:
                rows.append(_row(smiles, "lambda_max_nm", peak, "M01", solvent=solvent,
                                 state=state, label_level="molecule_state_condition"))
        rate = _finite(record.get("rate of thermal isomerisation from Z-E in s-1"))
        if rate is not None and rate > 0:
            half_life_h = math.log(2.0) / rate / 3600.0
            rows.append(_row(smiles, "log_half_life_h", math.log10(half_life_h), "M01",
                             solvent=str(record.get("Solvent used for thermal isomerisation rates") or "unknown"),
                             temperature_k="not_reported_in_table", state="Z_to_E",
                             label_level="molecule_state_condition"))
    return rows


def parse_u12_skinpix(path: Path) -> list[dict[str, Any]]:
    # The workbook has an incorrect A1 dimension; non-read-only pandas/openpyxl
    # parsing still exposes all 202 records.
    frame = pd.read_excel(path, sheet_name="default_1")
    rows = []
    for record in frame.to_dict("records"):
        smiles = _standardize(record.get("SMILES"))
        target = _finite(record.get("logKp (cm/s) (converted)"))
        if smiles is None or target is None:
            continue
        rows.append(_row(
            smiles, "kp_log_cm_s", target, "U12",
            cas=str(record.get("CAS number") or ""), compound_name=str(record.get("compound name") or ""),
            donor=str(record.get("category donor type") or "unknown"),
            acceptor=str(record.get("category acceptor type") or "unknown"),
            skin=str(record.get("used layer") or "unknown"),
            temperature_c=record.get("donor/skin surface temperature (°C)"),
            relation=str(record.get("Kp relation") or "="),
            doi=str(record.get("doi") or ""), label_level="molecule_condition",
        ))
    return rows


def parse_u13_permeation(path: Path, u09_path: Path, u12_path: Path) -> list[dict[str, Any]]:
    """Parse U13 and retain only exact CAS structures available locally.

    U13 reports log10(Kp / cm h-1) but no SMILES.  Structures are joined by
    exact CAS to the two already versioned workbooks and the target is converted
    to log10(cm s-1) by subtracting log10(3600).  Unmapped CAS are not guessed.
    """
    mappings: dict[str, str] = {}
    hppt = pd.read_excel(u09_path, sheet_name="HPPT Data")
    for record in hppt.to_dict("records"):
        cas = str(record.get("CASRN") or "").strip()
        smiles = _standardize(record.get("QSAR.Ready.SMILES") or record.get("SMILES"))
        if cas and smiles:
            mappings[cas] = smiles
    skinpix = pd.read_excel(u12_path, sheet_name="default_1")
    for record in skinpix.to_dict("records"):
        cas = str(record.get("CAS number") or "").strip()
        smiles = _standardize(record.get("SMILES"))
        if cas and smiles:
            mappings[cas] = smiles
    frame = pd.read_excel(path, sheet_name="Sheet1")
    rows = []
    for record in frame.to_dict("records"):
        cas = str(record.get("CAS No") or "").strip()
        smiles = mappings.get(cas)
        log_kp_cm_h = _finite(record.get("logkpl"))
        if not smiles or log_kp_cm_h is None:
            continue
        rows.append(_row(
            smiles, "kp_log_cm_s", log_kp_cm_h - math.log10(3600.0), "U13",
            cas=cas, compound_name=str(record.get("Compound") or ""),
            original_value=log_kp_cm_h, original_unit="log10(cm/h)",
            conversion_rule="log10(cm/s)=log10(cm/h)-log10(3600)",
            experimental_temperature_k=record.get("Texpi"),
            skin_thickness=record.get("Skin thicknessj"),
            skin_integrity_test=str(record.get("Skin Integrity testk") or ""),
            original_set=str(record.get("set") or ""), reference=str(record.get("Reference") or ""),
            label_level="molecule_condition",
        ))
    return rows


def parse_u09_hppt(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="HPPT Data")
    rows = []
    for record in frame.to_dict("records"):
        smiles = _standardize(record.get("QSAR.Ready.SMILES") or record.get("SMILES"))
        call = str(record.get("Call") or "").strip().lower()
        if smiles is None or call not in {"active", "inactive"}:
            continue
        rows.append(_row(
            smiles, "skin_sensitization", 1.0 if call == "active" else 0.0, "U09",
            cas=str(record.get("CASRN") or ""), dtxsid=str(record.get("DTXSID") or ""),
            test_type=str(record.get("Test.Type") or ""), vehicle=str(record.get("Vehicle") or ""),
            concentration_percent=record.get("Conc.(%)"), dose_ug_cm2=record.get("DSA.(μg/cm2)"),
            n_subjects=record.get("No.Test.Subjects"), label_level="molecule_condition_human",
        ))
    return rows


def parse_u07_irritation(path: Path, structure_source: Path) -> list[dict[str, Any]]:
    """Join NICE calls to structures by DTXSID without guessing missing IDs.

    The U07 workbook intentionally carries assay records rather than SMILES.
    U09 is an official NICE workbook containing DTXSID and QSAR-ready SMILES;
    only exact identifiers present in both sources are retained.
    """
    structures = pd.read_excel(structure_source, sheet_name="HPPT Data")
    dtxsid_to_smiles = {}
    for record in structures.to_dict("records"):
        dtxsid = str(record.get("DTXSID") or "").strip()
        smiles = _standardize(record.get("QSAR.Ready.SMILES") or record.get("SMILES"))
        if dtxsid and smiles:
            dtxsid_to_smiles[dtxsid] = smiles
    frame = pd.read_excel(path, sheet_name="Data")
    rows = []
    for record in frame.to_dict("records"):
        if str(record.get("Endpoint") or "").strip().lower() != "call":
            continue
        call = str(record.get("Response") or "").strip().lower()
        if call not in {"active", "inactive"}:
            continue
        dtxsid = str(record.get("DTXSID") or "").strip()
        smiles = dtxsid_to_smiles.get(dtxsid)
        if not smiles:
            continue
        rows.append(_row(
            smiles, "skin_irritation", 1.0 if call == "active" else 0.0, "U07",
            dtxsid=dtxsid, cas=str(record.get("CASRN") or ""),
            chemical_name=str(record.get("Chemical Name") or ""),
            assay=str(record.get("Assay") or ""), species=str(record.get("Species") or ""),
            route=str(record.get("Route") or ""), concentration=record.get("Concentration"),
            concentration_unit=str(record.get("Concentration Units") or ""),
            reference=str(record.get("Reference") or ""),
            label_level="molecule_condition_assay",
        ))
    return rows


def parse_u16_qsardb(directory: Path) -> list[dict[str, Any]]:
    namespace = {"q": "http://www.qsardb.org/QDB"}
    registry = ET.parse(directory / "compounds" / "compounds.xml").getroot()
    compounds: dict[int, dict[str, str]] = {}
    for node in registry.findall("q:Compound", namespace):
        cid = int(node.findtext("q:Id", namespaces=namespace))
        inchi = node.findtext("q:InChI", default="", namespaces=namespace)
        mol = Chem.MolFromInchi(inchi) if inchi else None
        smiles = _standardize(Chem.MolToSmiles(mol, isomericSmiles=True)) if mol is not None else None
        if smiles:
            compounds[cid] = {
                "smiles": smiles,
                "name": node.findtext("q:Name", default="", namespaces=namespace),
                "cas": node.findtext("q:Cas", default="", namespaces=namespace),
            }
    values = directory / "properties" / "3T3_NRU_Phototoxicity" / "values"
    rows = []
    with values.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) != 2 or int(fields[0]) not in compounds:
                continue
            cid, label = int(fields[0]), fields[1]
            compound = compounds[cid]
            rows.append(_row(
                compound["smiles"], "phototoxicity", 1.0 if label == "P" else 0.0, "U16",
                qsardb_compound_id=cid, cas=compound["cas"], compound_name=compound["name"],
                assay="3T3_NRU", species="Mus musculus", label_level="molecule_assay",
                original_split="validation" if cid >= 51 else "train",
            ))
    return rows


def _decimal(value: str) -> float | None:
    return _finite(value.strip().replace(",", "."))


def _read_instrument_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text(encoding="latin-1")
    records = list(csv.reader(text.splitlines()))
    for index, fields in enumerate(records):
        first = fields[0].strip().lower() if fields else ""
        if first in {"nm", "wavelength (nm)"}:
            return fields, records[index + 1:]
    return [], []


def parse_m05_full_spectra(directory: Path, pubchem_directory: Path) -> list[dict[str, Any]]:
    mappings = {}
    for cas, token in (("1498-88-0", "NO2_SP"), ("1498-89-1", "MeO_NO2_SP")):
        metadata = json.loads((pubchem_directory / f"{cas}.json").read_text())
        properties = metadata["PropertyTable"]["Properties"][0]
        mappings[token] = (standardize_smiles(properties.get("SMILES") or properties["ConnectivitySMILES"]), cas)
    rows = []
    for path in sorted(directory.rglob("*.csv")):
        header, records = _read_instrument_csv(path)
        if not header or len(header) < 2:
            continue
        token = "MeO_NO2_SP" if "MeO_NO2_SP" in path.name or "MeO_NO2_SP" in str(path.parent) else "NO2_SP"
        smiles, cas = mappings[token]
        curves: list[list[tuple[float, float]]] = [[] for _ in header[1:]]
        for record in records:
            wavelength = _decimal(record[0]) if record else None
            if wavelength is None:
                continue
            for index in range(min(len(curves), len(record) - 1)):
                absorbance = _decimal(record[index + 1])
                if absorbance is not None:
                    curves[index].append((wavelength, max(0.0, absorbance)))
        for curve_index in sorted({0, len(curves) - 1}):
            curve = curves[curve_index]
            wavelengths = [item[0] for item in curve if 290.0 <= item[0] <= 400.0]
            absorbance = [item[1] for item in curve if 290.0 <= item[0] <= 400.0]
            if len(wavelengths) < 20 or max(absorbance, default=0.0) <= 0:
                continue
            peak = wavelengths[max(range(len(absorbance)), key=absorbance.__getitem__)]
            rows.append({
                **_row(smiles, "full_spectrum", peak, "M05", family="spiropyran", cas=cas,
                       condition_file=str(path.relative_to(directory)), curve_index=curve_index,
                       label_level="molecule_state_condition_spectrum"),
                "wavelength_min_nm": min(wavelengths), "wavelength_max_nm": max(wavelengths),
                "uvb_auc": trapezoid_auc(wavelengths, absorbance, 290.0, 320.0),
                "uva_auc": trapezoid_auc(wavelengths, absorbance, 320.0, 400.0),
                "lambda_c_nm": critical_wavelength(wavelengths, absorbance),
                "spectrum_json": json.dumps([[w, a] for w, a in zip(wavelengths, absorbance)]),
            })
    return rows


def parse_source_catalog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workbook = pd.ExcelFile(path)
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        for index, record in enumerate(frame.to_dict("records"), start=2):
            clean = {str(key): value for key, value in record.items() if not pd.isna(value)}
            rows.append({"sheet": sheet, "excel_row": index, "record_json": json.dumps(clean, ensure_ascii=False, default=str)})
    return rows


def _source_status(path: Path, source_id: str, role: str) -> dict[str, Any]:
    return {
        "source_id": source_id, "role": role, "path": str(path.resolve()), "available": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def prepare_real_data(config: dict[str, Any], output_dir: str | Path, root: str | Path,
                      reaction_library: Iterable[dict[str, Any]]) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(root)
    inputs = {
        "M13": root / "data/raw/M13_uvvisml/uvvisml/data/processed/all_lambda_max_abs_including_duplicates.csv",
        "M01": root / "data/raw/M01_photoswitch/dataset/photoswitches.csv",
        "M05": root / "data/raw/M05_zenodo/files/data",
        "U09": root / "data/raw/U09_hppt/hppt_database_14feb2023.xlsx",
        "U07": root / "data/raw/U07_nice/Skin_Irritation_Corrosion.xlsx",
        "U12": root / "data/raw/U12_skinpix/20230620_cleanedDB.xlsx",
        "U16": root / "data/raw/U16_qsardb/files",
        "U13": root / "data/raw/U13_mendeley/Table_1.xlsx",
        "catalog": root / "database_matrix_MOST_UV_skin.xlsx",
    }
    missing = [key for key, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required real data sources are missing: {missing}")
    rows = []
    rows.extend(parse_m13(inputs["M13"], int(config.get("data", {}).get("m13_max_rows", 25_000))))
    rows.extend(parse_m01(inputs["M01"]))
    rows.extend(parse_u09_hppt(inputs["U09"]))
    rows.extend(parse_u07_irritation(inputs["U07"], inputs["U09"]))
    rows.extend(parse_u12_skinpix(inputs["U12"]))
    rows.extend(parse_u13_permeation(inputs["U13"], inputs["U09"], inputs["U12"]))
    rows.extend(parse_u16_qsardb(inputs["U16"]))
    full_spectra = parse_m05_full_spectra(inputs["M05"], root / "data/raw/M05_zenodo/pubchem")
    write_csv(destination / "reviewer_endpoints.csv", rows)
    write_csv(destination / "full_spectra.csv", full_spectra)
    write_csv(destination / "reaction_library.csv", reaction_library)
    catalog = parse_source_catalog(inputs["catalog"])
    write_csv(destination / "source_catalog_from_initial_excel.csv", catalog)
    positives = [row for row in rows if row["endpoint"] == "phototoxicity" and float(row["target"]) == 1.0]
    write_csv(destination / "known_phototoxic_registry.csv", positives)
    unique_structures = {row["smiles"] for row in rows}
    if len(unique_structures) > int(config["project"]["max_training_structures"]):
        raise ValueError("Real endpoint corpus exceeds the configured unique-structure cap")
    counts: dict[str, int] = defaultdict(int)
    structures_by_endpoint: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        counts[row["endpoint"]] += 1
        structures_by_endpoint[row["endpoint"]].add(row["smiles"])
    statuses = [
        _source_status(root / "data/raw/M13_uvvisml/uvvisml/data/original/joung/DB_for_chromophore_Sci_Data_rev02.csv",
                       "M11", "Deep4Chem lambda-max constituent of the M13 unified table"),
        _source_status(root / "data/raw/M13_uvvisml/uvvisml/data/original/jcole/paper_allDB.csv",
                       "M12", "CDEx lambda-max constituent of the M13 unified table"),
        _source_status(inputs["M13"], "M13", "unified UV/VisML lambda-max transfer table"),
        _source_status(inputs["M01"], "M01", "photoswitch lambda-max and thermal kinetics"),
        _source_status(root / "data/raw/M05_zenodo/data.zip", "M05", "condition-rich full spiropyran spectra"),
        _source_status(inputs["U09"], "U09", "human patch-test sensitization"),
        _source_status(inputs["U07"], "U07", "NICE irritation/corrosion calls joined by exact DTXSID"),
        _source_status(inputs["U12"], "U12", "skin permeation Kp"),
        _source_status(inputs["U13"], "U13", "human epidermis Kp/Jmax; exact CAS join and unit conversion"),
        _source_status(root / "data/raw/U16_qsardb/2011TIV324.qdb.zip", "U16", "3T3 NRU phototoxicity"),
        _source_status(inputs["catalog"], "initial_excel", "source registry and routing only"),
    ]
    manifest = {
        "schema_version": "2.0", "raw_inputs_mutated": False,
        "training_mode": "real_endpoint_specific", "endpoint_rows": dict(sorted(counts.items())),
        "unique_structures_by_endpoint": {key: len(value) for key, value in sorted(structures_by_endpoint.items())},
        "unique_structures_total": len(unique_structures), "full_spectrum_records": len(full_spectra),
        "reaction_library_rows": len(list(reaction_library)), "source_catalog_rows": len(catalog),
        "sources": statuses,
        "initial_excel_usage": "Used as a literature/source catalog and provenance input; it contains no molecular training labels.",
        "unavailable_labels": {
            "M03": "Metadata available, but Dryad file endpoint returned HTTP 401 without bearer token; empty partial file excluded.",
            "energy_kj_mol": "No sufficiently identified open molecular training table is available; no synthetic replacement is made.",
        },
        "scientific_limitations": [
            "Lambda-max observations are not full absorption spectra.",
            "M05 full spectra cover two spiropyran structures and are calibration/validation evidence, not a broad training corpus.",
            "M01 thermal kinetics are dominated by azo photoswitches and constitute class-shift evidence for MOST families.",
            "Safety models are triage only and do not establish cosmetic safety.",
        ],
    }
    (destination / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
