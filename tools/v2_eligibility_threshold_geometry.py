#!/usr/bin/env python3
"""Exact raw-BACI cohort geometry at 50/100/250 kUSD (no model scoring)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v2_product_space_density as common  # noqa: E402


SCHEMA = "upgrade-bench-v2-eligibility-threshold-geometry/1"
CONFIG_SCHEMA = "upgrade-bench-v2-eligibility-threshold-geometry-config/1"
STATUS = "complete_verified"
CONFIG_ROLE = "configs/v2_eligibility_threshold_geometry.json"
RUNNER_ROLE = "tools/v2_eligibility_threshold_geometry.py"
COMMON_ROLE = "tools/v2_product_space_density.py"
RAW_ROLE = "data/raw/BACI_HS92_V202401b.zip"
ATTESTATION_ROLE = "requirements/raw_source_attestation.json"
DEFAULT_CONFIG = ROOT / CONFIG_ROLE
DEFAULT_ARCHIVE = ROOT / RAW_ROLE
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "v2_eligibility_threshold_geometry.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "v2_eligibility_threshold_geometry.csv"
CHAINS = common.CHAINS
TASKS = ("a", "b1", "b2")
BAD_ISO = {"ANT", "SCG", "YUG", "SUN", "CSK", "DDR"}


class GeometryError(ValueError):
    pass


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = common._strict_json_load(path.resolve())
    if (
        cfg.get("schema_version") != CONFIG_SCHEMA
        or cfg.get("status") != "frozen_before_threshold_geometry_rebuild"
        or cfg.get("chains") != list(CHAINS)
        or cfg.get("thresholds_kusd") != [50.0, 100.0, 250.0]
        or cfg.get("reference_threshold_kusd") != 100.0
        or cfg.get("window", {}).get("early_years") != [2008, 2009, 2010, 2011, 2012]
        or cfg.get("window", {}).get("late_years") != [2018, 2019, 2020, 2021, 2022]
        or cfg.get("window", {}).get("comparison") != "strictly_greater_than_threshold"
        or cfg.get("scope")
        != "cohort geometry only; no model scoring, rerun, or performance claim"
    ):
        raise GeometryError("frozen threshold-geometry config changed")
    common._iso_datetime(cfg.get("frozen_at_utc"), "config.frozen_at_utc")
    common._assert_privacy(cfg)
    return cfg


def _load_registries() -> tuple[dict[str, Any], dict[str, str], set[int]]:
    registries: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    union: set[int] = set()
    for chain in CHAINS:
        role = f"chains/{chain}.json"
        reg = common._strict_json_load(ROOT / role)
        stages = reg.get("stages")
        upstream = reg.get("upstream_map")
        if reg.get("id") != chain or not isinstance(stages, Mapping) or not isinstance(upstream, Mapping):
            raise GeometryError(f"invalid registry {chain}")
        code_to_stage: dict[int, str] = {}
        for stage, codes in stages.items():
            if not isinstance(codes, list) or not codes:
                raise GeometryError(f"invalid stage {chain}/{stage}")
            for code in codes:
                if re.fullmatch(r"\d{6}", str(code)) is None:
                    raise GeometryError(f"invalid HS6 {chain}/{code}")
                numeric = int(code)
                if numeric in code_to_stage:
                    raise GeometryError(f"duplicate HS6 in {chain}: {code}")
                code_to_stage[numeric] = str(stage)
                union.add(numeric)
        if any(stage not in stages for stage in upstream):
            raise GeometryError(f"upstream_map target absent from stages for {chain}")
        if any(source not in stages for sources in upstream.values() for source in sources):
            raise GeometryError(f"upstream_map source absent from stages for {chain}")
        registries[chain] = {
            "code_to_stage": code_to_stage,
            "upstream_map": {str(k): tuple(str(x) for x in v) for k, v in upstream.items()},
        }
        hashes[role] = common._sha256(ROOT / role)
    return registries, hashes, union


def _read_filtered_raw(
    archive_path: Path, years: Sequence[int], hs6_union: set[int]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    attestation = common._strict_json_load(ROOT / ATTESTATION_ROLE)
    source = attestation.get("source", {})
    if (
        source.get("repository_path") != RAW_ROLE
        or source.get("size_bytes") != 2450783074
        or source.get("sha256")
        != "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a"
    ):
        raise GeometryError("raw source attestation changed")
    if archive_path.resolve() != DEFAULT_ARCHIVE.resolve():
        raise GeometryError("formal geometry audit requires canonical raw archive")
    if archive_path.stat().st_size != source["size_bytes"] or common._sha256(archive_path) != source["sha256"]:
        raise GeometryError("raw BACI archive differs from attestation")
    pieces: list[pd.DataFrame] = []
    member_records: list[dict[str, Any]] = []
    hs_values = np.asarray(sorted(hs6_union), dtype=np.int64)
    with zipfile.ZipFile(archive_path) as archive:
        countries = pd.read_csv(
            archive.open("country_codes_V202401b.csv"),
            usecols=["country_code", "country_iso3"],
        )
        iso = dict(zip(countries["country_code"].astype(int), countries["country_iso3"].astype(str)))
        for year in years:
            member = f"BACI_HS92_Y{year}_V202401b.csv"
            info = archive.getinfo(member)
            filtered_rows = 0
            print(f"reading threshold-geometry source {member} ...", flush=True)
            for chunk in pd.read_csv(
                archive.open(member), usecols=["i", "j", "k", "v"], chunksize=2_000_000
            ):
                k = pd.to_numeric(chunk["k"], errors="coerce").to_numpy(float)
                v = pd.to_numeric(chunk["v"], errors="coerce").to_numpy(float)
                if not np.isfinite(k).all() or not np.isfinite(v).all() or bool((v < 0).any()):
                    raise GeometryError(f"{member}: invalid k/v")
                numeric_k = k.astype(np.int64)
                keep = np.isin(numeric_k, hs_values)
                if not bool(keep.any()):
                    continue
                part = chunk.loc[keep, ["i", "j"]].copy()
                part["k"] = numeric_k[keep]
                part["v"] = v[keep]
                part["year"] = int(year)
                part["i_iso"] = pd.to_numeric(part["i"], errors="raise").astype(int).map(iso)
                part["j_iso"] = pd.to_numeric(part["j"], errors="raise").astype(int).map(iso)
                part = part.dropna(subset=["i_iso", "j_iso"])
                part = part.loc[
                    ~part["i_iso"].isin(BAD_ISO) & ~part["j_iso"].isin(BAD_ISO),
                    ["i_iso", "j_iso", "k", "year", "v"],
                ]
                filtered_rows += len(part)
                pieces.append(part)
            member_records.append(
                {
                    "member_name": member,
                    "crc32": f"{info.CRC:08x}",
                    "uncompressed_size_bytes": int(info.file_size),
                    "filtered_rows": int(filtered_rows),
                }
            )
    raw = pd.concat(pieces, ignore_index=True)
    return raw, member_records, {
        "path": RAW_ROLE,
        "sha256": source["sha256"],
        "size_bytes": source["size_bytes"],
        "attestation_path": ATTESTATION_ROLE,
        "attestation_sha256": common._sha256(ROOT / ATTESTATION_ROLE),
    }


def _aggregate_chain(
    raw: pd.DataFrame,
    years: Sequence[int],
    code_to_stage: Mapping[int, str],
) -> pd.DataFrame:
    work = raw.loc[raw["year"].isin(years) & raw["k"].isin(code_to_stage)].copy()
    work["stage"] = work["k"].map(code_to_stage)
    result = work.groupby(["i_iso", "j_iso", "stage"], as_index=False, sort=True)["v"].sum()
    result["v"] = result["v"] / float(len(years))
    return result


def _enumerate_geometry(
    early_values: pd.DataFrame,
    late_values: pd.DataFrame,
    upstream_map: Mapping[str, Sequence[str]],
    threshold: float,
) -> dict[str, dict[str, set[tuple[str, ...]]]]:
    early = early_values.loc[early_values["v"] > threshold, ["i_iso", "j_iso", "stage"]]
    late = late_values.loc[late_values["v"] > threshold, ["i_iso", "j_iso", "stage"]]
    early_set = {tuple(map(str, row)) for row in early.itertuples(index=False, name=None)}
    late_set = {tuple(map(str, row)) for row in late.itertuples(index=False, name=None)}
    early_exp = {
        stage: set(early.loc[early["stage"].eq(stage), "i_iso"].astype(str))
        for stage in upstream_map
    }
    early_imp = {
        stage: set(early.loc[early["stage"].eq(stage), "j_iso"].astype(str))
        for stage in upstream_map
    }
    for sources in upstream_map.values():
        for source in sources:
            early_exp.setdefault(
                source, set(early.loc[early["stage"].eq(source), "i_iso"].astype(str))
            )

    def lane_sets(first_time: bool) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
        candidates: set[tuple[str, ...]] = set()
        positives: set[tuple[str, ...]] = set()
        for stage, sources in upstream_map.items():
            upstream_exporters = set().union(*(early_exp.get(source, set()) for source in sources))
            pool = upstream_exporters - early_exp.get(stage, set()) if first_time else (
                upstream_exporters & early_exp.get(stage, set())
            )
            for exporter in sorted(pool):
                for importer in sorted(early_imp.get(stage, set())):
                    key = (exporter, importer, stage)
                    if importer == exporter or key in early_set:
                        continue
                    candidates.add(key)
                    if key in late_set:
                        positives.add(key)
        return candidates, positives

    a_candidates, a_positives = lane_sets(False)
    b_lanes, b_positive_lanes = lane_sets(True)
    b1_candidates = {(i, stage) for i, _j, stage in b_lanes}
    b1_positives = {(i, stage) for i, _j, stage in b_positive_lanes}
    b2_candidates = {
        key for key in b_lanes if (key[0], key[2]) in b1_positives
    }
    b2_positives = b_positive_lanes & b2_candidates
    return {
        "a": {"candidates": a_candidates, "positives": a_positives},
        "b1": {"candidates": b1_candidates, "positives": b1_positives},
        "b2": {"candidates": b2_candidates, "positives": b2_positives},
        "b_lanes": {"candidates": b_lanes, "positives": b_positive_lanes},
    }


def _key_hash(keys: set[tuple[str, ...]]) -> str:
    payload = "".join("|".join(key) + "\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _overlap(left: set[tuple[str, ...]], reference: set[tuple[str, ...]]) -> dict[str, Any]:
    intersection = len(left & reference)
    union = len(left | reference)
    return {
        "intersection": intersection,
        "left_only": len(left - reference),
        "reference_only": len(reference - left),
        "union": union,
        "jaccard": float(intersection / union) if union else 1.0,
        "retention_vs_reference": float(intersection / len(reference)) if reference else 1.0,
    }


def _canonical_gate(
    chain: str,
    geometry: Mapping[str, Mapping[str, set[tuple[str, ...]]]],
) -> dict[str, Any]:
    roles = {
        "a": (f"data/processed_v2/candidates_{chain}.csv", ["i_iso", "j_iso", "stage"], "y"),
        "b_lanes": (f"data/processed_v2/candidates_firsttime_{chain}.csv", ["i_iso", "j_iso", "stage"], "y"),
        "b1": (f"data/processed_v2/entries_firsttime_{chain}.csv", ["i_iso", "stage"], "z"),
        "b2": (f"data/processed_v2/destinations_given_entry_{chain}.csv", ["i_iso", "j_iso", "stage"], "y"),
    }
    audit: dict[str, Any] = {}
    for task, (role, columns, label) in roles.items():
        path = ROOT / role
        frame = pd.read_csv(path, usecols=[*columns, label])
        keys = {tuple(map(str, row)) for row in frame[columns].itertuples(index=False, name=None)}
        positive = {
            tuple(map(str, row))
            for row in frame.loc[pd.to_numeric(frame[label], errors="raise").eq(1), columns].itertuples(index=False, name=None)
        }
        if keys != geometry[task]["candidates"] or positive != geometry[task]["positives"]:
            raise GeometryError(
                f"100-kUSD raw reconstruction mismatch for {chain}/{task}: "
                f"candidate symmetric diff={len(keys ^ geometry[task]['candidates'])}, "
                f"positive symmetric diff={len(positive ^ geometry[task]['positives'])}"
            )
        audit[task] = {
            "path": role,
            "sha256": common._sha256(path),
            "candidate_keys_exact": True,
            "positive_keys_exact": True,
            "candidate_count": len(keys),
            "positive_count": len(positive),
        }
    return audit


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (50, 100, 250):
        key = str(threshold)
        for chain in CHAINS:
            for task in TASKS:
                item = payload["thresholds"][key]["chains"][chain][task]
                row = {
                    "threshold_kusd": threshold,
                    "chain": chain,
                    "task": task,
                    "candidate_count": item["candidate_count"],
                    "positive_count": item["positive_count"],
                    "base_rate": item["base_rate"],
                    "candidate_key_sha256": item["candidate_key_sha256"],
                    "positive_key_sha256": item["positive_key_sha256"],
                }
                for scope in ("candidate_overlap_vs_100", "positive_overlap_vs_100"):
                    for metric in (
                        "intersection",
                        "left_only",
                        "reference_only",
                        "union",
                        "jaccard",
                        "retention_vs_reference",
                    ):
                        row[f"{scope}_{metric}"] = item[scope][metric]
                rows.append(row)
    return rows


def _csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return pd.DataFrame(_csv_rows(payload)).to_csv(index=False, lineterminator="\n").encode("utf-8")


def run(config_path: Path = DEFAULT_CONFIG, archive_path: Path = DEFAULT_ARCHIVE) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise GeometryError("formal run requires canonical config")
    registries, registry_hashes, hs6_union = _load_registries()
    years = config["window"]["early_years"] + config["window"]["late_years"]
    raw, members, raw_source = _read_filtered_raw(archive_path, years, hs6_union)
    aggregated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for chain in CHAINS:
        aggregated[chain] = (
            _aggregate_chain(raw, config["window"]["early_years"], registries[chain]["code_to_stage"]),
            _aggregate_chain(raw, config["window"]["late_years"], registries[chain]["code_to_stage"]),
        )
    geometry: dict[int, dict[str, Any]] = {}
    for threshold in (50, 100, 250):
        geometry[threshold] = {
            chain: _enumerate_geometry(
                aggregated[chain][0], aggregated[chain][1], registries[chain]["upstream_map"], float(threshold)
            )
            for chain in CHAINS
        }
    canonical = {chain: _canonical_gate(chain, geometry[100][chain]) for chain in CHAINS}
    thresholds: dict[str, Any] = {}
    for threshold in (50, 100, 250):
        chains: dict[str, Any] = {}
        for chain in CHAINS:
            tasks: dict[str, Any] = {}
            for task in TASKS:
                current = geometry[threshold][chain][task]
                reference = geometry[100][chain][task]
                tasks[task] = {
                    "candidate_count": len(current["candidates"]),
                    "positive_count": len(current["positives"]),
                    "base_rate": float(len(current["positives"]) / len(current["candidates"]))
                    if current["candidates"] else 0.0,
                    "candidate_key_sha256": _key_hash(current["candidates"]),
                    "positive_key_sha256": _key_hash(current["positives"]),
                    "candidate_overlap_vs_100": _overlap(current["candidates"], reference["candidates"]),
                    "positive_overlap_vs_100": _overlap(current["positives"], reference["positives"]),
                }
            chains[chain] = tasks
        summary: dict[str, Any] = {}
        for task in TASKS:
            candidate_sets = [geometry[threshold][chain][task]["candidates"] for chain in CHAINS]
            positive_sets = [geometry[threshold][chain][task]["positives"] for chain in CHAINS]
            ref_candidate = [geometry[100][chain][task]["candidates"] for chain in CHAINS]
            ref_positive = [geometry[100][chain][task]["positives"] for chain in CHAINS]
            candidate_intersection = sum(len(a & b) for a, b in zip(candidate_sets, ref_candidate))
            positive_intersection = sum(len(a & b) for a, b in zip(positive_sets, ref_positive))
            candidate_union = sum(len(a | b) for a, b in zip(candidate_sets, ref_candidate))
            positive_union = sum(len(a | b) for a, b in zip(positive_sets, ref_positive))
            summary[task] = {
                "candidate_count": sum(map(len, candidate_sets)),
                "positive_count": sum(map(len, positive_sets)),
                "candidate_jaccard_vs_100": candidate_intersection / candidate_union if candidate_union else 1.0,
                "candidate_retention_vs_100": candidate_intersection / sum(map(len, ref_candidate)) if sum(map(len, ref_candidate)) else 1.0,
                "positive_jaccard_vs_100": positive_intersection / positive_union if positive_union else 1.0,
                "positive_retention_vs_100": positive_intersection / sum(map(len, ref_positive)) if sum(map(len, ref_positive)) else 1.0,
            }
        thresholds[str(threshold)] = {"chains": chains, "summary": summary}
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": config["scope"],
        "config": {"path": CONFIG_ROLE, "sha256": common._sha256(DEFAULT_CONFIG)},
        "protocol": {
            "aggregation": config["window"]["aggregation"],
            "comparison": config["window"]["comparison"],
            "early_window": "2008-2012",
            "late_window": "2018-2022",
            "thresholds_kusd": [50, 100, 250],
            "reference_threshold_kusd": 100,
            "model_scores_or_performance_computed": False,
        },
        "inputs": {
            "raw_baci": raw_source,
            "annual_members": members,
            "registry_hashes": registry_hashes,
            "source_hashes": {
                RUNNER_ROLE: common._sha256(ROOT / RUNNER_ROLE),
                COMMON_ROLE: common._sha256(ROOT / COMMON_ROLE),
                CONFIG_ROLE: common._sha256(ROOT / CONFIG_ROLE),
            },
            "filtered_raw_rows": int(len(raw)),
            "registered_hs6_union": int(len(hs6_union)),
        },
        "canonical_100kusd_gate": {
            "status": "PASS",
            "requirement": config["canonical_gate"],
            "chains": canonical,
        },
        "thresholds": thresholds,
        "runtime": {
            "device": "cpu",
            "gpu_used": False,
            "cpu_model": platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown-cpu"),
            "logical_cpu_cores": int(os.cpu_count() or 1),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "wall_elapsed_seconds": float(elapsed),
        },
    }
    common._assert_privacy(payload)
    return payload


def write_outputs(payload: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(common._canonical_json_bytes(payload))
    csv_path.write_bytes(_csv_bytes(payload))


def verify_output(json_path: Path = DEFAULT_JSON, csv_path: Path = DEFAULT_CSV) -> None:
    payload = common._strict_json_load(json_path)
    if json_path.read_bytes() != common._canonical_json_bytes(payload):
        raise GeometryError("geometry JSON is not canonical")
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != STATUS
        or payload.get("scope")
        != "cohort geometry only; no model scoring, rerun, or performance claim"
        or payload.get("canonical_100kusd_gate", {}).get("status") != "PASS"
        or payload.get("protocol", {}).get("model_scores_or_performance_computed") is not False
    ):
        raise GeometryError("geometry artifact identity/scope/gate changed")
    if csv_path.read_bytes() != _csv_bytes(payload):
        raise GeometryError("geometry CSV is stale")
    for role, digest in payload["inputs"]["source_hashes"].items():
        if common._sha256(ROOT / role) != digest:
            raise GeometryError(f"source hash mismatch: {role}")
    for role, digest in payload["inputs"]["registry_hashes"].items():
        if common._sha256(ROOT / role) != digest:
            raise GeometryError(f"registry hash mismatch: {role}")
    for chain, tasks in payload["canonical_100kusd_gate"]["chains"].items():
        for task, record in tasks.items():
            if common._sha256(ROOT / record["path"]) != record["sha256"]:
                raise GeometryError(f"canonical input hash mismatch: {chain}/{task}")
    common._assert_privacy(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_output:
            verify_output(args.json_out.resolve(), args.csv_out.resolve())
            print(f"verified eligibility-threshold geometry: {args.json_out}")
            return 0
        payload = run(args.config.resolve(), args.archive.resolve())
        write_outputs(payload, args.json_out.resolve(), args.csv_out.resolve())
        verify_output(args.json_out.resolve(), args.csv_out.resolve())
        print(f"wrote verified eligibility-threshold geometry: {args.json_out}")
        return 0
    except (GeometryError, OSError, zipfile.BadZipFile) as exc:
        print(f"eligibility-threshold geometry failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
