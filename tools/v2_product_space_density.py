#!/usr/bin/env python3
"""Leakage-audited prospective product-space density reference for B1.

This runner deliberately has no A or B2 output: a country--stage capability
score is constant across destinations and therefore cannot rank destination
lanes.  It builds two full-economy BACI country--HS92 matrices from early data,
freezes both density scorers and all registry mappings, and only then opens the
historical and main B1 outcome tables.

The standard Hidalgo product-space proximity is used.  For prospective entry,
the target product's diagonal proximity is removed from both numerator and
denominator.  That removal matters here because B1 absence is a nominal
100-kUSD stage criterion and does not imply RCA<1 for every constituent HS6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS))

import v2_rolling_cpu_baselines as cpu  # noqa: E402


SCHEMA_VERSION = "upgrade-bench-v2-product-space-density/1"
CONFIG_SCHEMA = "upgrade-bench-v2-product-space-density-config/1"
CONFIG_PROTOCOL = "early-full-economy-scorer-freeze-then-b1-evaluation-v1"
CONFIG_STATUS = "frozen_before_first_product_space_main_evaluation"
STATUS = "complete_verified"
BENCHMARK_VERSION = "2.1-dev"
MODEL_KEY = "prospective_product_space_density"

CONFIG_ROLE = "configs/v2_product_space_density.json"
RUNNER_ROLE = "tools/v2_product_space_density.py"
SHARED_METRICS_ROLE = "tools/v2_rolling_cpu_baselines.py"
RAW_ATTESTATION_ROLE = "requirements/raw_source_attestation.json"
RAW_ARCHIVE_ROLE = "data/raw/BACI_HS92_V202401b.zip"
SCORE_ARTIFACT_ROLE = "results_v2/scores/v2_product_space_density_scores.csv"

DEFAULT_CONFIG = ROOT / CONFIG_ROLE
DEFAULT_DATA = ROOT / "data" / "processed_v2"
DEFAULT_ARCHIVE = ROOT / RAW_ARCHIVE_ROLE
DEFAULT_JSON = ROOT / "results_v2" / "metrics" / "v2_product_space_density.json"
DEFAULT_CSV = ROOT / "results_v2" / "metrics" / "v2_product_space_density.csv"
DEFAULT_SCORES = ROOT / SCORE_ARTIFACT_ROLE

CHAINS = tuple(cpu.CHAINS)
COHORTS = ("historical", "main")
COHORT_WINDOWS = {
    "historical": ("1998-2002", "2008-2012", "_fold2", "history"),
    "main": ("2008-2012", "2018-2022", "", "target"),
}
EXPECTED_MAIN_ROWS = 1518
EXPECTED_MAIN_POSITIVES = 270
SCORE_COLUMNS = (
    "cohort",
    "chain",
    "i_iso",
    "stage",
    "entry_id",
    "density",
    "z",
    "entry_lateval",
    "early_window",
    "late_window",
    "candidate_source_sha256",
    "config_sha256",
    "freeze_sha256",
)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
UNIX_ABSOLUTE = re.compile(r"^/")
UNC_ABSOLUTE = re.compile(r"^\\\\")
REMOTE_ABSOLUTE = re.compile(r"^[^/\\\s:]+:[\\/]")


class ProductSpaceProtocolError(ValueError):
    """Raised when the frozen method or an artifact violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _strict_json_load(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductSpaceProtocolError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ProductSpaceProtocolError(f"non-finite JSON constant {value!r} in {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductSpaceProtocolError(f"cannot read strict JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ProductSpaceProtocolError(f"{path}: JSON root must be an object")
    return value


def _exact_keys(value: object, expected: Iterable[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductSpaceProtocolError(f"{where} must be an object")
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise ProductSpaceProtocolError(
            f"{where} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )
    return value


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductSpaceProtocolError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProductSpaceProtocolError(f"{where} must be finite")
    return result


def _integer(value: object, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductSpaceProtocolError(f"{where} must be an integer >= {minimum}")
    return value


def _probability(value: object, where: str) -> float:
    result = _finite(value, where)
    if not 0.0 <= result <= 1.0:
        raise ProductSpaceProtocolError(f"{where} must lie in [0,1]")
    return result


def _hex(value: object, where: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ProductSpaceProtocolError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _iso_datetime(value: object, where: str) -> datetime:
    if not isinstance(value, str):
        raise ProductSpaceProtocolError(f"{where} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProductSpaceProtocolError(f"{where} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProductSpaceProtocolError(f"{where} must be timezone-aware")
    return parsed


def _assert_privacy(value: object, where: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_privacy(key, f"{where}.<key>")
            _assert_privacy(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_privacy(item, f"{where}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.match(value)
        for pattern in (WINDOWS_ABSOLUTE, UNIX_ABSOLUTE, UNC_ABSOLUTE, REMOTE_ABSOLUTE)
    ):
        raise ProductSpaceProtocolError(f"private/absolute text is forbidden at {where}")


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _array_sha256(array: np.ndarray, *, kind: str) -> str:
    if kind == "float64-le":
        normalized = np.ascontiguousarray(array, dtype="<f8")
    elif kind == "uint8":
        normalized = np.ascontiguousarray(array, dtype=np.uint8)
    else:
        raise ProductSpaceProtocolError(f"unknown array hash kind {kind!r}")
    header = _canonical_json_bytes(
        {"kind": kind, "shape": [int(value) for value in normalized.shape]}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _sequence_sha256(values: Sequence[str], *, role: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes({"role": role, "values": [str(value) for value in values]})
    ).hexdigest()


def load_frozen_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _strict_json_load(path.resolve())
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol",
            "status",
            "frozen_at_utc",
            "chains",
            "task",
            "source",
            "formula",
            "selection",
            "evaluation",
            "uncertainty",
            "ranking",
            "claim_scope",
            "limitations",
        },
        "config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol"] != CONFIG_PROTOCOL
        or config["status"] != CONFIG_STATUS
    ):
        raise ProductSpaceProtocolError("frozen product-space config identity/status mismatch")
    frozen = _iso_datetime(config["frozen_at_utc"], "config.frozen_at_utc")
    if frozen.utcoffset() is None or frozen.utcoffset().total_seconds() != 0:
        raise ProductSpaceProtocolError("config.frozen_at_utc must be UTC")
    if config["chains"] != list(CHAINS):
        raise ProductSpaceProtocolError("config must cover the canonical six chains in order")
    task = config["task"]
    if task != {
        "id": "b1",
        "name": "processed_export_stage_entry",
        "unit": "exporter_stage",
        "directly_applicable_to": ["b1"],
        "not_reported_for": {
            "a": "exporter-stage density is constant across destination lanes",
            "b2": "exporter-stage density is constant within a realized entry",
        },
    }:
        raise ProductSpaceProtocolError("config task scope changed")
    source = config["source"]
    if (
        source.get("archive_role") != RAW_ARCHIVE_ROLE
        or source.get("raw_attestation_role") != RAW_ATTESTATION_ROLE
        or source.get("historical_early_years") != [1998, 1999, 2000, 2001, 2002]
        or source.get("main_early_years") != [2008, 2009, 2010, 2011, 2012]
        or source.get("product_universe")
        != "complete HS92 product dictionary, not the six-chain registry union"
        or source.get("nonfinite_flow_policy") != "reject"
        or source.get("negative_flow_policy") != "reject"
    ):
        raise ProductSpaceProtocolError("config source/universe contract changed")
    formula = config["formula"]
    if (
        formula.get("rca_threshold") != 1.0
        or formula.get("rca_comparison") != "greater_than_or_equal"
        or formula.get("prospective_target_self_relation")
        != "exclude q=p from numerator and denominator"
        or formula.get("stage_aggregation")
        != "equal_weight_mean_over_all_declared_target_stage_hs6"
        or formula.get("target_stages") != "keys of each chain registry upstream_map"
        or formula.get("missing_exporter_policy") != "score_zero"
        or formula.get("missing_target_product_policy")
        != "score_zero_and_retain_in_stage_mean"
        or formula.get("zero_proximity_denominator_policy") != "score_zero"
    ):
        raise ProductSpaceProtocolError("config density formula or missingness policy changed")
    selection = config["selection"]
    if selection != {
        "mode": "none_single_predeclared_formula",
        "candidate_formulas": 1,
        "historical_labels_used_for_selection": False,
        "historical_cohort_role": "diagnostic forward replication only",
        "main_labels_used_for_selection_or_calibration": False,
    }:
        raise ProductSpaceProtocolError("config selection policy changed")
    evaluation = config["evaluation"]
    if (
        evaluation.get("read_gate")
        != "build_and_freeze_both_full_economy_early_scorers_and_all_six_registry_stage_mappings_before_opening_any_b1_outcome_table"
        or evaluation.get("target_access")
        != "one_complete_main_cohort_evaluation_after_global_scorer_freeze"
        or evaluation.get("headline_metric") != "average_precision"
        or evaluation.get("value_budget") != 50
        or evaluation.get("candidate_file_template")
        != "data/processed_v2/entries_firsttime_{chain}{suffix}.csv"
    ):
        raise ProductSpaceProtocolError("config evaluation gate/reporting changed")
    uncertainty = config["uncertainty"]
    if uncertainty != {
        "method": "nonparametric_cluster_bootstrap",
        "cluster_unit": "exporter",
        "draws": 200,
        "rng_seed": 20260715,
        "interval": "percentile_95",
        "ap_only": True,
    }:
        raise ProductSpaceProtocolError("config uncertainty contract changed")
    ranking = config["ranking"]
    if ranking != {
        "score_direction": "higher_is_better",
        "canonical_candidate_order": ["i_iso", "stage", "entry_id"],
        "budget_tie_break": "canonical_candidate_order_ascending",
        "average_precision_ties": "threshold_block_invariant",
    }:
        raise ProductSpaceProtocolError("config ranking/tie policy changed")
    if config["claim_scope"] != (
        "reviewer-motivated post-hoc B1-only descriptive domain reference; not part of "
        "the original prespecified reference set"
    ):
        raise ProductSpaceProtocolError("config claim scope changed")
    if not isinstance(config["limitations"], list) or len(config["limitations"]) != 4:
        raise ProductSpaceProtocolError("config must retain the four declared limitations")
    _assert_privacy(config, "config")
    return config


def _config_reference(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if resolved != DEFAULT_CONFIG.resolve():
        raise ProductSpaceProtocolError(
            "formal runs require configs/v2_product_space_density.json at its canonical role"
        )
    return {"path": CONFIG_ROLE, "sha256": _sha256(resolved)}


def _load_raw_attestation(config: Mapping[str, Any]) -> dict[str, Any]:
    role = str(config["source"]["raw_attestation_role"])
    if role != RAW_ATTESTATION_ROLE:
        raise ProductSpaceProtocolError("raw attestation role is not canonical")
    attestation = _strict_json_load(ROOT / role)
    source = attestation.get("source")
    if (
        attestation.get("schema_version") != "upgrade-bench/raw-source-attestation/1"
        or not isinstance(source, Mapping)
        or source.get("archive_name") != "BACI_HS92_V202401b.zip"
        or source.get("repository_path") != RAW_ARCHIVE_ROLE
        or source.get("size_bytes") != 2450783074
        or source.get("sha256")
        != "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a"
    ):
        raise ProductSpaceProtocolError("raw BACI attestation identity changed")
    return attestation


def _zip_member_record(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "member_name": info.filename,
        "uncompressed_size_bytes": int(info.file_size),
        "compressed_size_bytes": int(info.compress_size),
        "crc32": f"{info.CRC:08x}",
    }


def _verify_raw_archive(
    archive_path: Path,
    config: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    if archive_path.resolve() != DEFAULT_ARCHIVE.resolve():
        raise ProductSpaceProtocolError("formal run refuses a noncanonical raw archive path")
    if not archive_path.is_file():
        raise ProductSpaceProtocolError(f"required BACI archive is missing: {RAW_ARCHIVE_ROLE}")
    expected = attestation["source"]
    size = archive_path.stat().st_size
    if size != expected["size_bytes"]:
        raise ProductSpaceProtocolError("raw BACI archive size differs from attestation")
    digest = _sha256(archive_path)
    if digest != expected["sha256"]:
        raise ProductSpaceProtocolError("raw BACI archive SHA-256 differs from attestation")
    required = [
        str(config["source"]["product_dictionary_member"]),
        str(config["source"]["country_dictionary_member"]),
    ]
    years = list(config["source"]["historical_early_years"]) + list(
        config["source"]["main_early_years"]
    )
    required.extend(
        str(config["source"]["annual_member_template"]).format(year=year)
        for year in years
    )
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(required) - names)
        if missing:
            raise ProductSpaceProtocolError(f"raw BACI archive is missing members {missing}")
        members = [_zip_member_record(archive.getinfo(name)) for name in required]
    return {
        "path": RAW_ARCHIVE_ROLE,
        "size_bytes": int(size),
        "sha256": digest,
        "attestation_path": RAW_ATTESTATION_ROLE,
        "attestation_sha256": _sha256(ROOT / RAW_ATTESTATION_ROLE),
        "hash_verification": "direct SHA-256 over current archive bytes",
        "required_members": members,
    }


def _load_dictionaries(
    archive: zipfile.ZipFile,
    config: Mapping[str, Any],
    *,
    require_formal_universe: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[int, int], dict[int, int]]:
    product_member = str(config["source"]["product_dictionary_member"])
    country_member = str(config["source"]["country_dictionary_member"])
    products_frame = pd.read_csv(
        archive.open(product_member), usecols=["code"], dtype={"code": "string"}
    )
    countries_frame = pd.read_csv(
        archive.open(country_member),
        usecols=["country_code", "country_iso3"],
        dtype={"country_iso3": "string"},
    )
    if bool(products_frame["code"].isna().any()):
        raise ProductSpaceProtocolError("product dictionary contains a missing code")
    product_codes = products_frame["code"].astype(str)
    if bool(product_codes.duplicated().any()) or bool((product_codes.str.len() != 6).any()):
        raise ProductSpaceProtocolError("product dictionary codes must be unique six-character keys")
    products = tuple(product_codes.tolist())

    if bool(countries_frame[["country_code", "country_iso3"]].isna().any().any()):
        raise ProductSpaceProtocolError("country dictionary contains a missing code or ISO3")
    numeric_country = pd.to_numeric(countries_frame["country_code"], errors="raise").astype(int)
    if bool(numeric_country.duplicated().any()):
        raise ProductSpaceProtocolError("country dictionary numeric codes are not unique")
    iso = countries_frame["country_iso3"].astype(str).str.strip()
    if bool((iso.str.len() != 3).any()):
        raise ProductSpaceProtocolError("country dictionary ISO3 values must have length three")
    countries = tuple(sorted(iso.unique().tolist()))
    country_index = {value: index for index, value in enumerate(countries)}
    numeric_country_to_index = {
        int(code): country_index[iso3] for code, iso3 in zip(numeric_country, iso)
    }
    numeric_product_to_index = {
        int(code): index
        for index, code in enumerate(products)
        if re.fullmatch(r"\d{6}", code) is not None
    }
    if require_formal_universe and (len(products) < 5000 or len(countries) < 200):
        raise ProductSpaceProtocolError(
            "formal BACI dictionaries are unexpectedly small; full product economy not loaded"
        )
    return countries, products, numeric_country_to_index, numeric_product_to_index


def _build_export_matrix(
    archive_path: Path,
    config: Mapping[str, Any],
    years: Sequence[int],
    *,
    chunksize: int = 2_000_000,
    require_formal_universe: bool = True,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
    """Build a dense full-dictionary country--product matrix from early years."""
    if not years or len(set(years)) != len(years):
        raise ProductSpaceProtocolError("early years must be a nonempty unique sequence")
    if chunksize < 1:
        raise ProductSpaceProtocolError("chunksize must be positive")
    template = str(config["source"]["annual_member_template"])
    member_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        countries, products, country_lookup_map, product_lookup_map = _load_dictionaries(
            archive, config, require_formal_universe=require_formal_universe
        )
        max_country = max(country_lookup_map)
        max_product = max(product_lookup_map)
        country_lookup = np.full(max_country + 1, -1, dtype=np.int32)
        product_lookup = np.full(max_product + 1, -1, dtype=np.int32)
        for code, index in country_lookup_map.items():
            country_lookup[code] = index
        for code, index in product_lookup_map.items():
            product_lookup[code] = index
        matrix = np.zeros((len(countries), len(products)), dtype=np.float64)
        flat_size = matrix.size
        for year in years:
            member = template.format(year=int(year))
            try:
                info = archive.getinfo(member)
            except KeyError as exc:
                raise ProductSpaceProtocolError(f"missing BACI annual member {member}") from exc
            member_records.append(_zip_member_record(info))
            rows = 0
            print(f"  reading full-economy {member} ...", flush=True)
            for chunk in pd.read_csv(
                archive.open(member),
                usecols=["i", "k", "v"],
                chunksize=chunksize,
            ):
                exporter = pd.to_numeric(chunk["i"], errors="coerce").to_numpy(float)
                product = pd.to_numeric(chunk["k"], errors="coerce").to_numpy(float)
                value = pd.to_numeric(chunk["v"], errors="coerce").to_numpy(float)
                if not (
                    np.isfinite(exporter).all()
                    and np.isfinite(product).all()
                    and np.isfinite(value).all()
                ):
                    raise ProductSpaceProtocolError(f"{member}: non-finite i/k/v value")
                if bool((value < 0).any()):
                    raise ProductSpaceProtocolError(f"{member}: negative BACI flow")
                exporter_i = exporter.astype(np.int64)
                product_i = product.astype(np.int64)
                if not (
                    np.array_equal(exporter, exporter_i.astype(float))
                    and np.array_equal(product, product_i.astype(float))
                ):
                    raise ProductSpaceProtocolError(f"{member}: nonintegral i or k code")
                in_range = (
                    (exporter_i >= 0)
                    & (exporter_i < len(country_lookup))
                    & (product_i >= 0)
                    & (product_i < len(product_lookup))
                )
                if not bool(in_range.all()):
                    raise ProductSpaceProtocolError(f"{member}: code outside dictionary lookup")
                ci = country_lookup[exporter_i]
                pi = product_lookup[product_i]
                known = (ci >= 0) & (pi >= 0)
                if not bool(known.all()):
                    raise ProductSpaceProtocolError(f"{member}: country/product absent from dictionary")
                positive = value > 0
                if bool(positive.any()):
                    flat = ci[positive].astype(np.int64) * len(products) + pi[positive]
                    matrix.ravel()[:] += np.bincount(
                        flat, weights=value[positive], minlength=flat_size
                    )
                rows += len(chunk)
            member_records[-1]["rows_read"] = int(rows)
    if not np.isfinite(matrix).all() or bool((matrix < 0).any()) or matrix.sum() <= 0:
        raise ProductSpaceProtocolError("full-economy export matrix is invalid")
    return matrix, countries, products, member_records


def _load_stage_registry() -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, str]]:
    mappings: dict[str, dict[str, tuple[str, ...]]] = {}
    hashes: dict[str, str] = {}
    for chain in CHAINS:
        role = f"chains/{chain}.json"
        path = ROOT / role
        registry = _strict_json_load(path)
        stages = registry.get("stages")
        upstream_map = registry.get("upstream_map")
        if not isinstance(stages, Mapping) or not isinstance(upstream_map, Mapping):
            raise ProductSpaceProtocolError(f"{role}: missing stages/upstream_map mappings")
        target: dict[str, tuple[str, ...]] = {}
        for stage in upstream_map:
            codes = stages.get(stage)
            if not isinstance(stage, str) or not isinstance(codes, list) or not codes:
                raise ProductSpaceProtocolError(f"{role}: invalid target stage {stage!r}")
            normalized = tuple(str(code) for code in codes)
            if len(set(normalized)) != len(normalized) or any(
                re.fullmatch(r"\d{6}", code) is None for code in normalized
            ):
                raise ProductSpaceProtocolError(f"{role}: target HS6 codes are invalid")
            target[stage] = normalized
        if not target:
            raise ProductSpaceProtocolError(f"{role}: no B1 target stages")
        mappings[chain] = target
        hashes[role] = _sha256(path)
    return mappings, hashes


@dataclass(frozen=True)
class ProductSpaceScorer:
    cohort: str
    countries: tuple[str, ...]
    products: tuple[str, ...]
    target_products: tuple[str, ...]
    membership: np.ndarray
    target_density: np.ndarray
    matrix_audit: dict[str, Any]
    annual_members: list[dict[str, Any]]


def _compute_product_space(
    export_matrix: np.ndarray,
    countries: Sequence[str],
    products: Sequence[str],
    target_products: Sequence[str],
    *,
    cohort: str,
    annual_members: list[dict[str, Any]] | None = None,
) -> ProductSpaceScorer:
    matrix = np.asarray(export_matrix, dtype=np.float64)
    countries_t = tuple(str(value) for value in countries)
    products_t = tuple(str(value) for value in products)
    targets_t = tuple(str(value) for value in target_products)
    if matrix.shape != (len(countries_t), len(products_t)):
        raise ProductSpaceProtocolError("export matrix shape differs from dictionaries")
    if not np.isfinite(matrix).all() or bool((matrix < 0).any()):
        raise ProductSpaceProtocolError("export matrix contains invalid flows")
    if len(set(countries_t)) != len(countries_t) or len(set(products_t)) != len(products_t):
        raise ProductSpaceProtocolError("country/product dictionaries must be unique")
    if not targets_t or len(set(targets_t)) != len(targets_t):
        raise ProductSpaceProtocolError("target product list must be nonempty and unique")
    product_index = {code: index for index, code in enumerate(products_t)}
    missing_targets = sorted(set(targets_t) - set(product_index))
    if missing_targets:
        raise ProductSpaceProtocolError(f"registry target products absent from dictionary: {missing_targets}")
    total = float(matrix.sum())
    row_total = matrix.sum(axis=1)
    column_total = matrix.sum(axis=0)
    if total <= 0:
        raise ProductSpaceProtocolError("full-economy export matrix has no positive flow")
    row_share = np.divide(
        matrix,
        row_total[:, None],
        out=np.zeros_like(matrix),
        where=row_total[:, None] > 0,
    )
    world_share = column_total / total
    rca = np.divide(
        row_share,
        world_share[None, :],
        out=np.zeros_like(matrix),
        where=world_share[None, :] > 0,
    )
    membership = rca >= 1.0
    target_indices = np.asarray([product_index[code] for code in targets_t], dtype=np.int64)
    float_membership = membership.astype(np.float64)
    ubiquity = float_membership.sum(axis=0)
    cooccurrence = float_membership[:, target_indices].T @ float_membership
    denominator = np.maximum(ubiquity[target_indices, None], ubiquity[None, :])
    proximity = np.divide(
        cooccurrence,
        denominator,
        out=np.zeros_like(cooccurrence),
        where=denominator > 0,
    )
    diagonal_before = proximity[np.arange(len(target_indices)), target_indices].copy()
    proximity[np.arange(len(target_indices)), target_indices] = 0.0
    if bool((proximity[np.arange(len(target_indices)), target_indices] != 0).any()):
        raise ProductSpaceProtocolError("failed to exclude target-product diagonal proximity")
    proximity_sum = proximity.sum(axis=1)
    numerator = float_membership @ proximity.T
    density = np.divide(
        numerator,
        proximity_sum[None, :],
        out=np.zeros_like(numerator),
        where=proximity_sum[None, :] > 0,
    )
    if not np.isfinite(density).all() or bool((density < -1e-12).any()) or bool(
        (density > 1 + 1e-12).any()
    ):
        raise ProductSpaceProtocolError("computed density lies outside [0,1]")
    density = np.clip(density, 0.0, 1.0)
    membership.flags.writeable = False
    density.flags.writeable = False
    audit = {
        "cohort": cohort,
        "countries": int(len(countries_t)),
        "country_iso3_identities": int(len(countries_t)),
        "countries_with_positive_exports": int((row_total > 0).sum()),
        "products": int(len(products_t)),
        "numeric_products": int(sum(re.fullmatch(r"\d{6}", code) is not None for code in products_t)),
        "zero_export_products_retained": int((column_total == 0).sum()),
        "positive_country_product_cells": int((matrix > 0).sum()),
        "rca_memberships": int(membership.sum()),
        "target_products": int(len(targets_t)),
        "target_products_with_positive_ubiquity": int((ubiquity[target_indices] > 0).sum()),
        "target_diagonal_nonzero_before_exclusion": int((diagonal_before > 0).sum()),
        "target_diagonal_max_after_exclusion": 0.0,
        "zero_density_denominator_targets": int((proximity_sum == 0).sum()),
        "total_early_exports_kusd": total,
        "formula": "prospective_hidalgo_density_target_diagonal_excluded",
        "country_vocabulary_sha256": _sequence_sha256(
            countries_t, role="country_iso3_vocabulary"
        ),
        "product_vocabulary_sha256": _sequence_sha256(
            products_t, role="full_hs92_product_vocabulary"
        ),
        "target_product_vocabulary_sha256": _sequence_sha256(
            targets_t, role="six_chain_target_hs6_vocabulary"
        ),
        "export_matrix_sha256": _array_sha256(matrix, kind="float64-le"),
        "rca_membership_sha256": _array_sha256(membership, kind="uint8"),
        "target_density_sha256": _array_sha256(density, kind="float64-le"),
    }
    return ProductSpaceScorer(
        cohort=cohort,
        countries=countries_t,
        products=products_t,
        target_products=targets_t,
        membership=membership,
        target_density=density,
        matrix_audit=audit,
        annual_members=list(annual_members or []),
    )


@dataclass(frozen=True)
class FrozenProtocol:
    stage_registry: dict[str, dict[str, tuple[str, ...]]]
    registry_hashes: dict[str, str]
    scorers: dict[str, ProductSpaceScorer]
    freeze_sha256: str
    sealed: bool


def _freeze_all_scorers(
    archive_path: Path,
    config: Mapping[str, Any],
    stage_registry: dict[str, dict[str, tuple[str, ...]]],
    registry_hashes: dict[str, str],
    *,
    matrix_builder: Callable[
        [Path, Mapping[str, Any], Sequence[int]],
        tuple[np.ndarray, tuple[str, ...], tuple[str, ...], list[dict[str, Any]]],
    ] = _build_export_matrix,
) -> FrozenProtocol:
    target_products = tuple(
        sorted(
            {
                code
                for chain_mapping in stage_registry.values()
                for codes in chain_mapping.values()
                for code in codes
            }
        )
    )
    scorers: dict[str, ProductSpaceScorer] = {}
    for cohort, years_key in (
        ("historical", "historical_early_years"),
        ("main", "main_early_years"),
    ):
        years = tuple(int(year) for year in config["source"][years_key])
        print(f"building {cohort} full-economy product-space scorer ...", flush=True)
        matrix, countries, products, annual_members = matrix_builder(
            archive_path, config, years
        )
        scorer = _compute_product_space(
            matrix,
            countries,
            products,
            target_products,
            cohort=cohort,
            annual_members=annual_members,
        )
        scorers[cohort] = scorer
    if set(scorers) != set(COHORTS):
        raise ProductSpaceProtocolError("both historical and main early scorers must exist")
    if scorers["historical"].countries != scorers["main"].countries or scorers[
        "historical"
    ].products != scorers["main"].products:
        raise ProductSpaceProtocolError("historical/main dictionary universes differ")
    freeze_record = {
        "config_sha256": _sha256(DEFAULT_CONFIG),
        "registry_hashes": registry_hashes,
        "target_products": list(target_products),
        "scorer_audits": {cohort: scorers[cohort].matrix_audit for cohort in COHORTS},
        "annual_members": {cohort: scorers[cohort].annual_members for cohort in COHORTS},
        "main_outcomes_opened": False,
    }
    freeze_sha256 = hashlib.sha256(_canonical_json_bytes(freeze_record)).hexdigest()
    return FrozenProtocol(
        stage_registry=stage_registry,
        registry_hashes=registry_hashes,
        scorers=scorers,
        freeze_sha256=freeze_sha256,
        sealed=True,
    )


def _entry_path(data_dir: Path, chain: str, cohort: str) -> Path:
    suffix = COHORT_WINDOWS[cohort][2]
    return data_dir / f"entries_firsttime_{chain}{suffix}.csv"


def _read_entry_table(data_dir: Path, chain: str, cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _entry_path(data_dir, chain, cohort)
    if not path.is_file():
        raise ProductSpaceProtocolError(f"missing B1 candidate table {path}")
    frame = pd.read_csv(path)
    required = {
        "i_iso",
        "stage",
        "z",
        "entry_lateval",
        "benchmark_version",
        "aggregation",
        "early_window",
        "late_window",
        "entry_id",
        "temporal_role",
        "task",
        "task_unit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ProductSpaceProtocolError(f"{path}: missing B1 columns {missing}")
    duplicate = frame.duplicated(["i_iso", "stage"], keep=False)
    if bool(duplicate.any()):
        raise ProductSpaceProtocolError(f"{path}: duplicate exporter-stage keys")
    z = pd.to_numeric(frame["z"], errors="raise")
    if not set(z.unique()).issubset({0, 1}):
        raise ProductSpaceProtocolError(f"{path}: z is not binary")
    lateval = pd.to_numeric(frame["entry_lateval"], errors="raise").to_numpy(float)
    if not np.isfinite(lateval).all() or bool((lateval < 0).any()):
        raise ProductSpaceProtocolError(f"{path}: entry_lateval is invalid")
    if bool(((z.to_numpy() == 0) & (lateval != 0)).any()) or bool(
        ((z.to_numpy() == 1) & (lateval <= 0)).any()
    ):
        raise ProductSpaceProtocolError(f"{path}: labels and realized values disagree")
    expected_early, expected_late, _, expected_role = COHORT_WINDOWS[cohort]
    expected_singletons = {
        "aggregation": "calendar_mean",
        "early_window": expected_early,
        "late_window": expected_late,
        "temporal_role": expected_role,
        "task": "processed_export_stage_entry",
        "task_unit": "exporter_stage",
    }
    for column, expected in expected_singletons.items():
        observed = set(frame[column].astype(str).unique())
        if observed != {expected}:
            raise ProductSpaceProtocolError(
                f"{path}: expected {column}={expected!r}, got {sorted(observed)}"
            )
    versions = set(frame["benchmark_version"].astype(str).unique())
    if not versions or any(not version.startswith("2.") for version in versions):
        raise ProductSpaceProtocolError(f"{path}: benchmark_version is not v2")
    expected_entry = frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)
    if not expected_entry.equals(frame["entry_id"].astype(str)):
        raise ProductSpaceProtocolError(f"{path}: entry_id does not match exporter-stage key")
    result = frame.copy()
    result["z"] = z.astype(np.int8)
    result["entry_lateval"] = lateval
    result = result.sort_values(
        ["i_iso", "stage", "entry_id"], kind="mergesort"
    ).reset_index(drop=True)
    role = str(path.relative_to(ROOT)).replace("\\", "/")
    audit = {
        "path": role,
        "sha256": _sha256(path),
        "rows": int(len(result)),
        "positives": int(result["z"].sum()),
        "exporters": int(result["i_iso"].nunique()),
        "stages": int(result["stage"].nunique()),
        "early_window": expected_early,
        "late_window": expected_late,
    }
    return result, audit


def _score_candidates(
    frame: pd.DataFrame,
    scorer: ProductSpaceScorer,
    stage_mapping: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    country_index = {country: index for index, country in enumerate(scorer.countries)}
    target_index = {code: index for index, code in enumerate(scorer.target_products)}
    product_index = {code: index for index, code in enumerate(scorer.products)}
    scores = np.zeros(len(frame), dtype=np.float64)
    exporter_covered = 0
    target_pairs = 0
    target_memberships = 0
    candidates_with_membership = 0
    missing_stage: set[str] = set()
    for row_index, row in enumerate(frame.itertuples(index=False)):
        exporter = str(row.i_iso)
        stage = str(row.stage)
        codes = stage_mapping.get(stage)
        if codes is None:
            missing_stage.add(stage)
            continue
        code_list = tuple(str(code) for code in codes)
        target_pairs += len(code_list)
        ci = country_index.get(exporter)
        if ci is None:
            continue
        exporter_covered += 1
        densities: list[float] = []
        membership_count = 0
        for code in code_list:
            ti = target_index.get(code)
            pi = product_index.get(code)
            if ti is None or pi is None:
                densities.append(0.0)
                continue
            densities.append(float(scorer.target_density[ci, ti]))
            membership_count += int(scorer.membership[ci, pi])
        scores[row_index] = float(np.mean(densities)) if densities else 0.0
        target_memberships += membership_count
        candidates_with_membership += int(membership_count > 0)
    if missing_stage:
        raise ProductSpaceProtocolError(f"candidate stages absent from registry: {sorted(missing_stage)}")
    if not np.isfinite(scores).all() or bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ProductSpaceProtocolError("candidate density scores are invalid")
    audit = {
        "candidate_rows": int(len(frame)),
        "exporter_dictionary_covered_rows": int(exporter_covered),
        "exporter_dictionary_coverage": float(exporter_covered / len(frame)) if len(frame) else 0.0,
        "candidate_target_hs6_pairs": int(target_pairs),
        "candidate_target_hs6_rca_memberships": int(target_memberships),
        "candidates_with_any_target_hs6_rca_membership": int(candidates_with_membership),
        "self_diagonal_exclusion_material": bool(candidates_with_membership > 0),
        "zero_score_rows": int((scores == 0).sum()),
        "unique_scores": int(np.unique(scores).size),
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
    }
    return scores, audit


def _evaluate_after_freeze(
    frozen: FrozenProtocol,
    data_dir: Path,
    config: Mapping[str, Any],
    *,
    reader: Callable[[Path, str, str], tuple[pd.DataFrame, dict[str, Any]]] = _read_entry_table,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not frozen.sealed or set(frozen.scorers) != set(COHORTS):
        raise ProductSpaceProtocolError("outcome tables cannot open before both scorers are sealed")
    results: dict[str, dict[str, Any]] = {cohort: {} for cohort in COHORTS}
    score_rows: list[dict[str, Any]] = []
    draws = int(config["uncertainty"]["draws"])
    base_seed = int(config["uncertainty"]["rng_seed"])
    budget = int(config["evaluation"]["value_budget"])
    for cohort in COHORTS:
        scorer = frozen.scorers[cohort]
        for chain in CHAINS:
            print(f"[{cohort}/{chain}] opening and evaluating frozen B1 cohort ...", flush=True)
            frame, input_audit = reader(data_dir, chain, cohort)
            observed_stages = set(frame["stage"].astype(str))
            expected_stages = set(frozen.stage_registry[chain])
            if observed_stages != expected_stages:
                raise ProductSpaceProtocolError(
                    f"{cohort}/{chain}: candidate stages differ from target registry "
                    f"(missing={sorted(expected_stages-observed_stages)}, "
                    f"extra={sorted(observed_stages-expected_stages)})"
                )
            score, score_audit = _score_candidates(
                frame, scorer, frozen.stage_registry[chain]
            )
            metrics = cpu._classification_metrics(
                frame,
                label="z",
                score=score,
                cluster=frame["i_iso"].astype(str).to_numpy(),
                cluster_unit="exporter",
                budgets=(budget,),
                bootstrap_draws=draws,
                seed=_stable_seed(base_seed, cohort, chain, MODEL_KEY),
            )
            for position, row in enumerate(frame.itertuples(index=False)):
                score_rows.append(
                    {
                        "cohort": cohort,
                        "chain": chain,
                        "i_iso": str(row.i_iso),
                        "stage": str(row.stage),
                        "entry_id": str(row.entry_id),
                        "density": float(score[position]),
                        "z": int(row.z),
                        "entry_lateval": float(row.entry_lateval),
                        "early_window": input_audit["early_window"],
                        "late_window": input_audit["late_window"],
                        "candidate_source_sha256": input_audit["sha256"],
                        "config_sha256": _sha256(DEFAULT_CONFIG),
                        "freeze_sha256": frozen.freeze_sha256,
                    }
                )
            results[cohort][chain] = {
                "input": input_audit,
                "score_audit": score_audit,
                "metrics": metrics,
            }
    main_rows = sum(results["main"][chain]["input"]["rows"] for chain in CHAINS)
    main_positives = sum(
        results["main"][chain]["input"]["positives"] for chain in CHAINS
    )
    if main_rows != EXPECTED_MAIN_ROWS or main_positives != EXPECTED_MAIN_POSITIVES:
        raise ProductSpaceProtocolError(
            f"canonical B1 main inventory changed: rows={main_rows}, positives={main_positives}"
        )
    if len(score_rows) != sum(
        results[cohort][chain]["input"]["rows"]
        for cohort in COHORTS
        for chain in CHAINS
    ):
        raise ProductSpaceProtocolError("keyed score row count differs from B1 cohorts")
    return results, score_rows


def _macro_summary(results: Mapping[str, Mapping[str, Any]], budget: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cohort in COHORTS:
        ap = {
            chain: float(results[cohort][chain]["metrics"]["average_precision"])
            for chain in CHAINS
        }
        value = {
            chain: float(
                results[cohort][chain]["metrics"]["budgets"][f"k_{budget}"][
                    "value_capture"
                ]
            )
            for chain in CHAINS
        }
        summary[cohort] = {
            "aggregation": "unweighted_mean_over_six_fixed_chains",
            "chain_registry": list(CHAINS),
            "headline_metric": "average_precision",
            "headline": {
                "per_chain": ap,
                "macro_mean": float(np.mean(list(ap.values()))),
                "std_across_chains": float(np.std(list(ap.values()), ddof=0)),
            },
            "value_metric": f"global_observed_late_value_capture_at_{budget}",
            "realized_value": {
                "per_chain": value,
                "macro_mean": float(np.mean(list(value.values()))),
                "std_across_chains": float(np.std(list(value.values()), ddof=0)),
            },
            "chain_level_ci95": None,
            "inference_note": "descriptive over the six fixed chains; no chain-population interval",
        }
    return summary


def _score_csv_bytes(score_rows: Sequence[Mapping[str, Any]]) -> bytes:
    frame = pd.DataFrame(score_rows, columns=list(SCORE_COLUMNS))
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _score_artifact_record(
    score_rows: Sequence[Mapping[str, Any]], freeze_sha256: str
) -> dict[str, Any]:
    raw = _score_csv_bytes(score_rows)
    return {
        "path": SCORE_ARTIFACT_ROLE,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": int(len(score_rows)),
        "columns": list(SCORE_COLUMNS),
        "canonical_order": {
            "cohort": list(COHORTS),
            "chain": list(CHAINS),
            "within_chain": ["i_iso", "stage", "entry_id"],
        },
        "config_sha256": _sha256(DEFAULT_CONFIG),
        "freeze_sha256": freeze_sha256,
        "purpose": "public metric recomputation without requiring the raw BACI archive",
    }


def run(
    data_dir: Path = DEFAULT_DATA,
    archive_path: Path = DEFAULT_ARCHIVE,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    config = load_frozen_config(config_path)
    config_ref = _config_reference(config_path)
    attestation = _load_raw_attestation(config)
    print("verifying the 2.45-GB raw BACI archive against its attestation ...", flush=True)
    raw_source = _verify_raw_archive(archive_path, config, attestation)

    # Registry mappings are frozen before any candidate/outcome file is opened.
    stage_registry, registry_hashes = _load_stage_registry()
    frozen = _freeze_all_scorers(
        archive_path,
        config,
        stage_registry,
        registry_hashes,
    )
    print("both early scorers and all stage mappings frozen; opening B1 outcomes ...", flush=True)
    results, score_rows = _evaluate_after_freeze(frozen, data_dir, config)
    budget = int(config["evaluation"]["value_budget"])

    cpu_model = platform.processor().strip() or os.environ.get(
        "PROCESSOR_IDENTIFIER", "unknown-cpu"
    ).strip()
    logical_cores = os.cpu_count()
    elapsed = time.perf_counter() - started
    if not cpu_model or not logical_cores or not math.isfinite(elapsed) or elapsed <= 0:
        raise ProductSpaceProtocolError("runtime CPU identity/core count/duration is unavailable")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": config["claim_scope"],
        "limitations": config["limitations"],
        "config": config_ref,
        "score_artifact": _score_artifact_record(score_rows, frozen.freeze_sha256),
        "protocol": {
            "task": "B1 processed exporter-stage entry only",
            "historical_transition": config["evaluation"]["historical_transition"],
            "main_transition": config["evaluation"]["main_transition"],
            "selection_mode": config["selection"]["mode"],
            "historical_labels_used_for_selection": False,
            "main_labels_used_for_selection_or_calibration": False,
            "read_gate": config["evaluation"]["read_gate"],
            "all_scorers_and_registry_mappings_frozen_before_any_outcome_read": True,
            "freeze_sha256": frozen.freeze_sha256,
            "target_self_relation": config["formula"]["prospective_target_self_relation"],
            "full_product_universe": config["source"]["product_universe"],
            "ranking": config["ranking"],
            "uncertainty": config["uncertainty"],
        },
        "inputs": {
            "raw_baci": raw_source,
            "public_sources": {
                RUNNER_ROLE: _sha256(ROOT / RUNNER_ROLE),
                SHARED_METRICS_ROLE: _sha256(ROOT / SHARED_METRICS_ROLE),
                CONFIG_ROLE: config_ref["sha256"],
                RAW_ATTESTATION_ROLE: raw_source["attestation_sha256"],
            },
            "chain_registries": registry_hashes,
        },
        "scorers": {
            cohort: {
                "matrix_audit": frozen.scorers[cohort].matrix_audit,
                "annual_members": frozen.scorers[cohort].annual_members,
            }
            for cohort in COHORTS
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "logical_cpu_cores": int(logical_cores),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "wall_elapsed_seconds": float(elapsed),
            "gpu_used": False,
        },
        "cohorts": results,
        "macro_summary": _macro_summary(results, budget),
    }
    _assert_privacy(payload)
    return payload, score_rows


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = load_frozen_config(ROOT / payload["config"]["path"])
    budget = int(config["evaluation"]["value_budget"])
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for chain in CHAINS:
            item = payload["cohorts"][cohort][chain]
            metrics = item["metrics"]
            ci = metrics["average_precision_ci95"]
            value = metrics["budgets"][f"k_{budget}"]
            audit = item["score_audit"]
            rows.append(
                {
                    "cohort": cohort,
                    "chain": chain,
                    "model": MODEL_KEY,
                    "early_window": item["input"]["early_window"],
                    "late_window": item["input"]["late_window"],
                    "n": metrics["n"],
                    "positives": metrics["positives"],
                    "exporters": item["input"]["exporters"],
                    "average_precision": metrics["average_precision"],
                    "average_precision_ci95_low": "" if ci is None else ci[0],
                    "average_precision_ci95_high": "" if ci is None else ci[1],
                    "roc_auc": metrics["roc_auc"],
                    "value_budget": budget,
                    "value_hits": value["hits"],
                    "value_capture": value["value_capture"],
                    "observed_late_value_kusd": value["observed_late_value_kusd"],
                    "total_observed_late_value_kusd": metrics[
                        "total_observed_late_value_kusd"
                    ],
                    "exporter_dictionary_coverage": audit[
                        "exporter_dictionary_coverage"
                    ],
                    "candidates_with_target_hs6_rca": audit[
                        "candidates_with_any_target_hs6_rca_membership"
                    ],
                    "candidate_target_hs6_rca_memberships": audit[
                        "candidate_target_hs6_rca_memberships"
                    ],
                    "zero_score_rows": audit["zero_score_rows"],
                    "unique_scores": audit["unique_scores"],
                }
            )
    return rows


def _csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return pd.DataFrame(_csv_rows(payload)).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")


def write_outputs(
    payload: Mapping[str, Any],
    score_rows: Sequence[Mapping[str, Any]],
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
    scores_path: Path = DEFAULT_SCORES,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(_canonical_json_bytes(payload))
    csv_path.write_bytes(_csv_bytes(payload))
    scores_path.write_bytes(_score_csv_bytes(score_rows))


def _verify_source(path: Path, digest: str, where: str) -> None:
    _hex(digest, where)
    if not path.is_file() or _sha256(path) != digest:
        raise ProductSpaceProtocolError(f"{where}: source missing or SHA-256 mismatch")


def _validate_interval(value: object, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ProductSpaceProtocolError(f"{where} must be null or [low,high]")
    low = _probability(value[0], f"{where}[0]")
    high = _probability(value[1], f"{where}[1]")
    if low > high:
        raise ProductSpaceProtocolError(f"{where} is reversed")


def validate_payload(
    payload: Mapping[str, Any],
    *,
    verify_sources: bool = True,
    verify_raw_archive: bool = False,
) -> None:
    config = load_frozen_config()
    _exact_keys(
        payload,
        {
            "schema_version",
            "benchmark_version",
            "status",
            "generated_at_utc",
            "claim_scope",
            "limitations",
            "config",
            "score_artifact",
            "protocol",
            "inputs",
            "scorers",
            "runtime",
            "cohorts",
            "macro_summary",
        },
        "result",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["benchmark_version"] != BENCHMARK_VERSION
        or payload["status"] != STATUS
        or payload["claim_scope"] != config["claim_scope"]
        or payload["limitations"] != config["limitations"]
    ):
        raise ProductSpaceProtocolError("result identity/status/scope changed")
    if _iso_datetime(payload["generated_at_utc"], "result.generated_at_utc") <= _iso_datetime(
        config["frozen_at_utc"], "config.frozen_at_utc"
    ):
        raise ProductSpaceProtocolError("result predates frozen configuration")
    expected_config = _config_reference(DEFAULT_CONFIG)
    if payload["config"] != expected_config:
        raise ProductSpaceProtocolError("result is not bound to current frozen config")
    score_artifact = payload["score_artifact"]
    expected_score_keys = {
        "path",
        "sha256",
        "rows",
        "columns",
        "canonical_order",
        "config_sha256",
        "freeze_sha256",
        "purpose",
    }
    _exact_keys(score_artifact, expected_score_keys, "result.score_artifact")
    if (
        score_artifact["path"] != SCORE_ARTIFACT_ROLE
        or score_artifact["columns"] != list(SCORE_COLUMNS)
        or score_artifact["canonical_order"]
        != {
            "cohort": list(COHORTS),
            "chain": list(CHAINS),
            "within_chain": ["i_iso", "stage", "entry_id"],
        }
        or score_artifact["config_sha256"] != expected_config["sha256"]
        or score_artifact["purpose"]
        != "public metric recomputation without requiring the raw BACI archive"
    ):
        raise ProductSpaceProtocolError("keyed score artifact contract changed")
    _hex(score_artifact["sha256"], "result.score_artifact.sha256")
    _hex(score_artifact["freeze_sha256"], "result.score_artifact.freeze_sha256")
    protocol = payload["protocol"]
    if (
        protocol.get("task") != "B1 processed exporter-stage entry only"
        or protocol.get("selection_mode") != "none_single_predeclared_formula"
        or protocol.get("historical_labels_used_for_selection") is not False
        or protocol.get("main_labels_used_for_selection_or_calibration") is not False
        or protocol.get("all_scorers_and_registry_mappings_frozen_before_any_outcome_read")
        is not True
        or protocol.get("read_gate") != config["evaluation"]["read_gate"]
        or protocol.get("target_self_relation")
        != "exclude q=p from numerator and denominator"
        or protocol.get("full_product_universe")
        != "complete HS92 product dictionary, not the six-chain registry union"
        or protocol.get("ranking") != config["ranking"]
        or protocol.get("uncertainty") != config["uncertainty"]
    ):
        raise ProductSpaceProtocolError("result protocol/read gate differs from config")
    _hex(protocol.get("freeze_sha256"), "result.protocol.freeze_sha256")
    if score_artifact["freeze_sha256"] != protocol["freeze_sha256"]:
        raise ProductSpaceProtocolError("score artifact and protocol freeze hashes differ")

    inputs = payload["inputs"]
    sources = inputs.get("public_sources")
    expected_source_roles = {
        RUNNER_ROLE,
        SHARED_METRICS_ROLE,
        CONFIG_ROLE,
        RAW_ATTESTATION_ROLE,
    }
    _exact_keys(sources, expected_source_roles, "result.inputs.public_sources")
    for role, digest in sources.items():
        _hex(digest, f"source {role}")
        if verify_sources:
            _verify_source(ROOT / role, digest, f"source {role}")
    registries = inputs.get("chain_registries")
    expected_registry_roles = {f"chains/{chain}.json" for chain in CHAINS}
    _exact_keys(registries, expected_registry_roles, "result.inputs.chain_registries")
    for role, digest in registries.items():
        _hex(digest, f"registry {role}")
        if verify_sources:
            _verify_source(ROOT / role, digest, f"registry {role}")
    raw = inputs.get("raw_baci")
    if (
        raw.get("path") != RAW_ARCHIVE_ROLE
        or raw.get("size_bytes") != 2450783074
        or raw.get("sha256")
        != "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a"
        or raw.get("attestation_path") != RAW_ATTESTATION_ROLE
        or raw.get("attestation_sha256") != sources[RAW_ATTESTATION_ROLE]
    ):
        raise ProductSpaceProtocolError("raw BACI provenance changed")
    if verify_sources and verify_raw_archive:
        _verify_source(ROOT / RAW_ARCHIVE_ROLE, raw["sha256"], "raw BACI archive")
    members = raw.get("required_members")
    if not isinstance(members, list) or len(members) != 12:
        raise ProductSpaceProtocolError("raw BACI member inventory must contain 12 members")

    scorers = _exact_keys(payload["scorers"], COHORTS, "result.scorers")
    for cohort in COHORTS:
        scorer = scorers[cohort]
        audit = scorer.get("matrix_audit", {})
        if (
            audit.get("cohort") != cohort
            or _integer(audit.get("countries"), f"{cohort}.countries", 200) < 200
            or _integer(audit.get("products"), f"{cohort}.products", 5000) < 5000
            or audit.get("formula")
            != "prospective_hidalgo_density_target_diagonal_excluded"
            or audit.get("target_diagonal_max_after_exclusion") != 0.0
        ):
            raise ProductSpaceProtocolError(f"{cohort}: scorer universe/formula audit changed")
        for field in (
            "country_vocabulary_sha256",
            "product_vocabulary_sha256",
            "target_product_vocabulary_sha256",
            "export_matrix_sha256",
            "rca_membership_sha256",
            "target_density_sha256",
        ):
            _hex(audit.get(field), f"{cohort}.{field}")
        annual = scorer.get("annual_members")
        if not isinstance(annual, list) or len(annual) != 5:
            raise ProductSpaceProtocolError(f"{cohort}: expected five early annual members")

    cohorts = _exact_keys(payload["cohorts"], COHORTS, "result.cohorts")
    all_rows = 0
    main_rows = 0
    main_positives = 0
    for cohort in COHORTS:
        chain_results = _exact_keys(cohorts[cohort], CHAINS, f"result.cohorts.{cohort}")
        for chain in CHAINS:
            item = chain_results[chain]
            _exact_keys(item, {"input", "score_audit", "metrics"}, f"{cohort}/{chain}")
            input_record = item["input"]
            expected_suffix = COHORT_WINDOWS[cohort][2]
            expected_path = f"data/processed_v2/entries_firsttime_{chain}{expected_suffix}.csv"
            if input_record.get("path") != expected_path:
                raise ProductSpaceProtocolError(f"{cohort}/{chain}: candidate role changed")
            digest = input_record.get("sha256")
            _hex(digest, f"{cohort}/{chain} candidate hash")
            if verify_sources:
                _verify_source(ROOT / expected_path, digest, f"{cohort}/{chain} candidate")
            rows = _integer(input_record.get("rows"), f"{cohort}/{chain}.rows", 1)
            positives = _integer(
                input_record.get("positives"), f"{cohort}/{chain}.positives", 1
            )
            if positives > rows:
                raise ProductSpaceProtocolError(f"{cohort}/{chain}: positives exceed rows")
            metrics = item["metrics"]
            if metrics.get("n") != rows or metrics.get("positives") != positives:
                raise ProductSpaceProtocolError(f"{cohort}/{chain}: metric counts differ")
            _probability(metrics.get("average_precision"), f"{cohort}/{chain}.AP")
            _validate_interval(
                metrics.get("average_precision_ci95"), f"{cohort}/{chain}.AP interval"
            )
            _probability(metrics.get("roc_auc"), f"{cohort}/{chain}.ROC AUC")
            budget = metrics.get("budgets", {}).get("k_50", {})
            if budget.get("requested_k") != 50 or budget.get("effective_k") != min(50, rows):
                raise ProductSpaceProtocolError(f"{cohort}/{chain}: value budget changed")
            _probability(budget.get("value_capture"), f"{cohort}/{chain}.value@50")
            score_audit = item["score_audit"]
            if (
                score_audit.get("candidate_rows") != rows
                or score_audit.get("exporter_dictionary_covered_rows", 0) > rows
                or score_audit.get("candidates_with_any_target_hs6_rca_membership", 0)
                > rows
            ):
                raise ProductSpaceProtocolError(f"{cohort}/{chain}: score audit counts invalid")
            _probability(
                score_audit.get("exporter_dictionary_coverage"),
                f"{cohort}/{chain}.coverage",
            )
            if cohort == "main":
                main_rows += rows
                main_positives += positives
            all_rows += rows
    if main_rows != EXPECTED_MAIN_ROWS or main_positives != EXPECTED_MAIN_POSITIVES:
        raise ProductSpaceProtocolError("main B1 inventory does not equal 1,518/270")
    if score_artifact["rows"] != all_rows:
        raise ProductSpaceProtocolError("keyed score artifact row count differs from cohorts")

    expected_macro = _macro_summary(cohorts, 50)
    macro = payload["macro_summary"]
    if macro != expected_macro:
        raise ProductSpaceProtocolError("macro summary is stale or inconsistent")
    runtime = payload["runtime"]
    if (
        runtime.get("device") != "cpu"
        or runtime.get("gpu_used") is not False
        or _integer(runtime.get("logical_cpu_cores"), "runtime.logical_cpu_cores", 1) < 1
        or _finite(runtime.get("wall_elapsed_seconds"), "runtime.wall_elapsed_seconds") <= 0
    ):
        raise ProductSpaceProtocolError("runtime hardware/duration audit changed")
    _assert_privacy(payload)


def _verify_keyed_scores(payload: Mapping[str, Any], scores_path: Path) -> None:
    artifact = payload["score_artifact"]
    try:
        raw = scores_path.read_bytes()
    except OSError as exc:
        raise ProductSpaceProtocolError(
            f"cannot read keyed product-space scores from {scores_path}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
        raise ProductSpaceProtocolError("keyed score CSV SHA-256 differs from result JSON")
    string_columns = {
        column: "string"
        for column in (
            "cohort",
            "chain",
            "i_iso",
            "stage",
            "entry_id",
            "early_window",
            "late_window",
            "candidate_source_sha256",
            "config_sha256",
            "freeze_sha256",
        )
    }
    try:
        frame = pd.read_csv(scores_path, dtype=string_columns)
    except (OSError, pd.errors.ParserError) as exc:
        raise ProductSpaceProtocolError("cannot parse keyed score CSV") from exc
    if list(frame.columns) != list(SCORE_COLUMNS):
        raise ProductSpaceProtocolError("keyed score CSV columns/order changed")
    if len(frame) != artifact["rows"] or bool(frame.isna().any().any()):
        raise ProductSpaceProtocolError("keyed score CSV row count/missingness changed")
    if bool(frame.duplicated(["cohort", "chain", "i_iso", "stage"], keep=False).any()):
        raise ProductSpaceProtocolError("keyed score CSV contains duplicate B1 keys")
    density = pd.to_numeric(frame["density"], errors="raise").to_numpy(float)
    z = pd.to_numeric(frame["z"], errors="raise").to_numpy(float)
    lateval = pd.to_numeric(frame["entry_lateval"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(density).all()
        or bool((density < 0).any())
        or bool((density > 1).any())
        or not np.isfinite(z).all()
        or not set(z).issubset({0.0, 1.0})
        or not np.isfinite(lateval).all()
        or bool((lateval < 0).any())
        or bool(((z == 0) & (lateval != 0)).any())
        or bool(((z == 1) & (lateval <= 0)).any())
    ):
        raise ProductSpaceProtocolError("keyed score CSV contains invalid score/label/value")
    expected_entry = frame["i_iso"].astype(str) + "|" + frame["stage"].astype(str)
    if not expected_entry.equals(frame["entry_id"].astype(str)):
        raise ProductSpaceProtocolError("keyed score CSV entry_id is inconsistent")
    if set(frame["cohort"].astype(str)) != set(COHORTS) or set(
        frame["chain"].astype(str)
    ) != set(CHAINS):
        raise ProductSpaceProtocolError("keyed score CSV cohort/chain coverage changed")

    ordered = frame.copy()
    ordered["_cohort_rank"] = ordered["cohort"].astype(str).map(
        {value: index for index, value in enumerate(COHORTS)}
    )
    ordered["_chain_rank"] = ordered["chain"].astype(str).map(
        {value: index for index, value in enumerate(CHAINS)}
    )
    expected_index = ordered.sort_values(
        ["_cohort_rank", "_chain_rank", "i_iso", "stage", "entry_id"],
        kind="mergesort",
    ).index.to_numpy()
    if not np.array_equal(expected_index, np.arange(len(frame))):
        raise ProductSpaceProtocolError("keyed score CSV is not in canonical order")

    config = load_frozen_config()
    config_hash = payload["config"]["sha256"]
    freeze_hash = payload["protocol"]["freeze_sha256"]
    if set(frame["config_sha256"].astype(str)) != {config_hash} or set(
        frame["freeze_sha256"].astype(str)
    ) != {freeze_hash}:
        raise ProductSpaceProtocolError("keyed scores are not bound to config/freeze hashes")
    for cohort in COHORTS:
        expected_early, expected_late, _, _ = COHORT_WINDOWS[cohort]
        for chain in CHAINS:
            group = frame.loc[
                frame["cohort"].astype(str).eq(cohort)
                & frame["chain"].astype(str).eq(chain)
            ].copy()
            item = payload["cohorts"][cohort][chain]
            source = item["input"]
            if (
                len(group) != source["rows"]
                or int(pd.to_numeric(group["z"], errors="raise").sum())
                != source["positives"]
                or set(group["candidate_source_sha256"].astype(str))
                != {source["sha256"]}
                or set(group["early_window"].astype(str)) != {expected_early}
                or set(group["late_window"].astype(str)) != {expected_late}
            ):
                raise ProductSpaceProtocolError(
                    f"keyed score cohort binding differs for {cohort}/{chain}"
                )
            metric_frame = pd.DataFrame(
                {
                    "i_iso": group["i_iso"].astype(str).to_numpy(),
                    "z": pd.to_numeric(group["z"], errors="raise").astype(np.int8),
                    "entry_lateval": pd.to_numeric(
                        group["entry_lateval"], errors="raise"
                    ).astype(float),
                }
            )
            recomputed = cpu._classification_metrics(
                metric_frame,
                label="z",
                score=pd.to_numeric(group["density"], errors="raise").to_numpy(float),
                cluster=metric_frame["i_iso"].to_numpy(str),
                cluster_unit="exporter",
                budgets=(int(config["evaluation"]["value_budget"]),),
                bootstrap_draws=int(config["uncertainty"]["draws"]),
                seed=_stable_seed(
                    int(config["uncertainty"]["rng_seed"]), cohort, chain, MODEL_KEY
                ),
            )
            if _canonical_json_bytes({"metrics": recomputed}) != _canonical_json_bytes(
                {"metrics": item["metrics"]}
            ):
                raise ProductSpaceProtocolError(
                    f"metrics do not recompute exactly from keyed scores for {cohort}/{chain}"
                )


def _rebuild_and_verify_scorers(
    payload: Mapping[str, Any], scores_path: Path
) -> None:
    """Full verifier: rebuild early matrices/densities from the attested raw BACI."""
    config = load_frozen_config()
    stage_registry, registry_hashes = _load_stage_registry()
    if registry_hashes != payload["inputs"]["chain_registries"]:
        raise ProductSpaceProtocolError("current registry hashes differ before raw rebuild")
    frozen = _freeze_all_scorers(
        DEFAULT_ARCHIVE,
        config,
        stage_registry,
        registry_hashes,
    )
    rebuilt_scorers = {
        cohort: {
            "matrix_audit": frozen.scorers[cohort].matrix_audit,
            "annual_members": frozen.scorers[cohort].annual_members,
        }
        for cohort in COHORTS
    }
    if _canonical_json_bytes(rebuilt_scorers) != _canonical_json_bytes(
        payload["scorers"]
    ):
        raise ProductSpaceProtocolError(
            "raw BACI rebuild differs from committed matrix/membership/density hashes"
        )
    if frozen.freeze_sha256 != payload["protocol"]["freeze_sha256"]:
        raise ProductSpaceProtocolError("raw BACI rebuild differs from committed freeze receipt")

    score_frame = pd.read_csv(
        scores_path,
        dtype={
            "cohort": "string",
            "chain": "string",
            "i_iso": "string",
            "stage": "string",
            "entry_id": "string",
        },
    )
    for cohort in COHORTS:
        for chain in CHAINS:
            group = score_frame.loc[
                score_frame["cohort"].astype(str).eq(cohort)
                & score_frame["chain"].astype(str).eq(chain)
            ].reset_index(drop=True)
            key_frame = group.loc[:, ["i_iso", "stage", "entry_id"]]
            rebuilt_scores, rebuilt_audit = _score_candidates(
                key_frame, frozen.scorers[cohort], stage_registry[chain]
            )
            committed_scores = pd.to_numeric(
                group["density"], errors="raise"
            ).to_numpy(float)
            if not np.allclose(
                rebuilt_scores, committed_scores, rtol=0.0, atol=5e-15
            ):
                raise ProductSpaceProtocolError(
                    f"raw BACI density rebuild differs for {cohort}/{chain}"
                )
            if _canonical_json_bytes(rebuilt_audit) != _canonical_json_bytes(
                payload["cohorts"][cohort][chain]["score_audit"]
            ):
                raise ProductSpaceProtocolError(
                    f"raw BACI score audit rebuild differs for {cohort}/{chain}"
                )


def verify_existing_output(
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
    scores_path: Path = DEFAULT_SCORES,
    *,
    verify_raw_archive: bool = False,
) -> None:
    payload = _strict_json_load(json_path.resolve())
    if json_path.read_bytes() != _canonical_json_bytes(payload):
        raise ProductSpaceProtocolError(f"{json_path}: JSON is not canonical")
    validate_payload(
        payload, verify_sources=True, verify_raw_archive=verify_raw_archive
    )
    if csv_path.read_bytes() != _csv_bytes(payload):
        raise ProductSpaceProtocolError(f"{csv_path}: CSV is stale or noncanonical")
    _verify_keyed_scores(payload, scores_path)
    if verify_raw_archive:
        _rebuild_and_verify_scorers(payload, scores_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--scores-out", type=Path, default=DEFAULT_SCORES)
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="verify JSON/CSV, source bindings, and recompute metrics from keyed scores; raw BACI is not required",
    )
    parser.add_argument(
        "--verify-raw",
        action="store_true",
        help="add direct raw-archive hashing and rebuild matrices, RCA memberships, densities, and keyed scores",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_output or args.verify_raw:
            verify_existing_output(
                args.json_out.resolve(),
                args.csv_out.resolve(),
                args.scores_out.resolve(),
                verify_raw_archive=bool(args.verify_raw),
            )
            scope = "including raw BACI provenance" if args.verify_raw else "from public keyed scores"
            print(f"verified product-space density artifact {scope}: {args.json_out}")
            return 0
        if args.config.resolve() != DEFAULT_CONFIG.resolve():
            raise ProductSpaceProtocolError("formal run refuses a noncanonical config path")
        payload, score_rows = run(
            args.data_dir.resolve(), args.archive.resolve(), args.config.resolve()
        )
        validate_payload(payload, verify_sources=True, verify_raw_archive=False)
        write_outputs(
            payload,
            score_rows,
            args.json_out.resolve(),
            args.csv_out.resolve(),
            args.scores_out.resolve(),
        )
        verify_existing_output(
            args.json_out.resolve(),
            args.csv_out.resolve(),
            args.scores_out.resolve(),
            verify_raw_archive=True,
        )
        print(f"wrote verified product-space density artifact: {args.json_out}")
        return 0
    except (ProductSpaceProtocolError, OSError, zipfile.BadZipFile) as exc:
        print(f"product-space density protocol failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
