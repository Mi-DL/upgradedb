"""Strict private cache for the audited BACI rows used by v2 cohort builds.

The cache is deliberately *not* a public data artifact.  It contains every raw
trade row whose HS6 code occurs in the six audited chain registries, retaining
only ``i,j,k,year,v``.  A cache is accepted only from a private/tmp path and only
after its source, registry snapshot, inventory, hashes, schema, years, and code
sets have been validated.

The source archive's complete SHA-256 is computed once by :func:`build_cache`.
Readers do not re-hash the multi-gigabyte archive: cache files are self-contained
and are fully verified, while the small country-code member used by
``temporal_backtest.py`` is verified separately against the manifest.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAINS_DIR = ROOT / "chains"
DEFAULT_AUDIT_PATH = ROOT / "docs" / "registry_audit.json"
DEFAULT_EVIDENCE_PATH = ROOT / "chains" / "evidence" / "registry_evidence.json"

SCHEMA_VERSION = "upgrade-bench/private-baci-filtered-cache/1"
MANIFEST_NAME = "manifest.json"
CACHE_FORMAT = "deterministic-csv-gzip"
CACHE_COLUMNS = ("i", "j", "k", "year", "v")
SOURCE_COLUMNS = ("i", "j", "k", "v")
COUNTRY_CODES_MEMBER = "country_codes_V202401b.csv"
TRADE_MEMBER_TEMPLATE = "BACI_HS92_Y{year}_V202401b.csv"
REQUIRED_YEARS = tuple(
    list(range(1998, 2003)) + list(range(2008, 2013)) + list(range(2018, 2023))
)
EXPECTED_CHAIN_COUNT = 6
AUDIT_SCHEMA = "upgrade-bench/registry-audit/3"
EVIDENCE_SCHEMA = "upgrade-bench/hs92-registry-evidence/3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HS6_RE = re.compile(r"^[0-9]{6}$")
_PRIVATE_COMPONENTS = frozenset({"private", ".private", "tmp", "temp"})


class CacheValidationError(ValueError):
    """Raised when a cache or one of its provenance inputs is not exact."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"{label} root must be a JSON object")
    return value


def _canonical_hs6(value: object, *, label: str) -> str:
    code = str(value)
    if not _HS6_RE.fullmatch(code):
        raise CacheValidationError(f"{label} is not a canonical six-digit HS6 code: {code!r}")
    return code


def _private_cache_path(path: Path) -> Path:
    """Resolve *path* and reject locations outside an explicit private/tmp tree."""

    resolved = path.expanduser().resolve(strict=False)
    temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    try:
        resolved.relative_to(temp_root)
        return resolved
    except ValueError:
        pass
    if not any(part.casefold() in _PRIVATE_COMPONENTS for part in resolved.parts):
        raise CacheValidationError(
            "BACI filtered caches are private inputs and may only live below a "
            "path component named private, .private, tmp, or temp"
        )
    return resolved


def _safe_registry_path(value: object, chain_id: str) -> str:
    if not isinstance(value, str):
        raise CacheValidationError(f"registry audit entry {chain_id!r} lacks registry_file")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix():
        raise CacheValidationError(f"unsafe registry_file for {chain_id!r}: {value!r}")
    if pure.name != f"{chain_id}.json":
        raise CacheValidationError(
            f"registry audit maps {chain_id!r} to unexpected file {value!r}"
        )
    return value


def registry_snapshot(
    *,
    chains_dir: Path = DEFAULT_CHAINS_DIR,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
) -> dict:
    """Return a validated, hash-bound snapshot of the six audited registries."""

    chains_dir = chains_dir.resolve()
    audit_path = audit_path.resolve()
    evidence_path = evidence_path.resolve()
    audit = _json_object(audit_path, "registry audit")
    evidence = _json_object(evidence_path, "registry evidence")

    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise CacheValidationError("BACI cache requires the full-ledger registry audit schema v3")
    if audit.get("status") != "PASS":
        raise CacheValidationError("registry audit status is not PASS")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise CacheValidationError("BACI cache requires the full-ledger registry evidence schema v3")
    required_checks = (
        "active_include_decisions_exact",
        "canonical_stage_definitions_complete",
        "per_code_stage_fit_supported_excluded_or_out_of_stage",
        "high_risk_stage_semantic_regressions_absent",
        "stage_and_hs6_relationship_references_valid",
    )
    checks = audit.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) != "PASS" for name in required_checks):
        raise CacheValidationError("registry audit v3 semantic/stage-fit gates are not all PASS")
    audit_chains = audit.get("chains")
    evidence_chains = evidence.get("chains")
    if not isinstance(audit_chains, dict) or not isinstance(evidence_chains, dict):
        raise CacheValidationError("registry audit/evidence lacks a chains object")
    if set(audit_chains) != set(evidence_chains):
        raise CacheValidationError("registry audit and evidence chain ids differ")
    if len(audit_chains) != EXPECTED_CHAIN_COUNT:
        raise CacheValidationError(
            f"expected exactly {EXPECTED_CHAIN_COUNT} audited chains, found {len(audit_chains)}"
        )
    if audit.get("summary", {}).get("chain_count") != EXPECTED_CHAIN_COUNT:
        raise CacheValidationError("registry audit summary chain_count is stale")
    if evidence.get("summary", {}).get("chain_count") != EXPECTED_CHAIN_COUNT:
        raise CacheValidationError("registry evidence summary chain_count is stale")

    actual_files = sorted(chains_dir.glob("*.json"))
    actual_names = {path.name for path in actual_files}
    expected_names = {f"{chain_id}.json" for chain_id in audit_chains}
    if actual_names != expected_names:
        raise CacheValidationError(
            "chains directory is not the exact audited six-file registry: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )

    chain_records: list[dict] = []
    union: set[str] = set()
    for chain_id in sorted(audit_chains):
        audit_entry = audit_chains[chain_id]
        evidence_entry = evidence_chains[chain_id]
        if not isinstance(audit_entry, dict) or not isinstance(evidence_entry, dict):
            raise CacheValidationError(f"invalid audit/evidence record for {chain_id!r}")
        if audit_entry.get("stage_semantic_integrity") != "PASS":
            raise CacheValidationError(
                f"registry audit stage-semantic gate is not PASS for {chain_id!r}"
            )
        logical_path = _safe_registry_path(audit_entry.get("registry_file"), chain_id)
        registry_path = chains_dir / f"{chain_id}.json"
        digest = sha256_file(registry_path)
        if audit_entry.get("registry_sha256") != digest:
            raise CacheValidationError(f"registry audit SHA-256 is stale for {chain_id!r}")
        registry = _json_object(registry_path, f"registry {chain_id}")
        if registry.get("id") != chain_id:
            raise CacheValidationError(f"registry id mismatch in {registry_path.name}")
        stages = registry.get("stages")
        if not isinstance(stages, dict) or not stages:
            raise CacheValidationError(f"registry {chain_id!r} has no stages")
        codes: list[str] = []
        for stage, stage_codes in stages.items():
            if not isinstance(stage, str) or not stage or not isinstance(stage_codes, list):
                raise CacheValidationError(f"registry {chain_id!r} has an invalid stage record")
            codes.extend(
                _canonical_hs6(code, label=f"{chain_id}.{stage}") for code in stage_codes
            )
        if len(codes) != len(set(codes)):
            raise CacheValidationError(f"registry {chain_id!r} assigns an HS6 more than once")
        codes = sorted(codes)
        if audit_entry.get("active_codes") != len(codes):
            raise CacheValidationError(f"registry audit active_codes is stale for {chain_id!r}")

        stage_definitions = evidence_entry.get("stage_definitions")
        if not isinstance(stage_definitions, dict) or set(stage_definitions) != set(stages):
            raise CacheValidationError(
                f"registry evidence stage definitions are stale for {chain_id!r}"
            )
        for stage, definition in stage_definitions.items():
            if not isinstance(definition, dict) or any(
                not isinstance(definition.get(field), str) or not definition[field].strip()
                for field in ("canonical_definition", "specificity", "fit_rule")
            ):
                raise CacheValidationError(
                    f"registry evidence has an incomplete stage definition for {chain_id}.{stage}"
                )

        decisions = evidence_entry.get("decisions")
        if not isinstance(decisions, list):
            raise CacheValidationError(f"registry evidence lacks decisions for {chain_id!r}")
        active_code_to_stage = {
            code: stage for stage, stage_codes in stages.items() for code in stage_codes
        }
        for decision in decisions:
            if not isinstance(decision, dict):
                raise CacheValidationError(f"registry evidence has an invalid decision for {chain_id!r}")
            code = _canonical_hs6(decision.get("code"), label=f"evidence {chain_id}")
            stage_fit = decision.get("stage_fit")
            if not isinstance(stage_fit, dict) or any(
                not isinstance(stage_fit.get(field), str) or not stage_fit[field].strip()
                for field in ("status", "canonical_definition", "evidence", "rationale")
            ):
                raise CacheValidationError(
                    f"registry evidence decision {chain_id}.{code} lacks a complete stage_fit"
                )
            if stage_fit["evidence"] != decision.get("description"):
                raise CacheValidationError(
                    f"registry evidence decision {chain_id}.{code} stage_fit evidence is stale"
                )
            if stage_fit["rationale"] != decision.get("rationale"):
                raise CacheValidationError(
                    f"registry evidence decision {chain_id}.{code} stage_fit rationale is stale"
                )
            if decision.get("decision") == "include":
                stage = decision.get("stage")
                definition = stage_definitions.get(stage)
                if (
                    stage != active_code_to_stage.get(code)
                    or stage_fit["status"] != "supported"
                    or not isinstance(definition, dict)
                    or stage_fit["canonical_definition"] != definition["canonical_definition"]
                    or decision.get("specificity") != definition["specificity"]
                    or decision.get("rationale") != definition["fit_rule"]
                ):
                    raise CacheValidationError(
                        f"registry evidence decision {chain_id}.{code} fails the supported stage_fit gate"
                    )
            elif decision.get("decision") == "exclude":
                if decision.get("stage") is not None or stage_fit["status"] != "unsupported":
                    raise CacheValidationError(
                        f"registry evidence decision {chain_id}.{code} fails the excluded stage_fit gate"
                    )
            elif decision.get("decision") == "out_of_stage":
                if decision.get("stage") is not None or stage_fit["status"] != "out_of_stage":
                    raise CacheValidationError(
                        f"registry evidence decision {chain_id}.{code} fails the out-of-stage gate"
                    )
            else:
                raise CacheValidationError(
                    f"registry evidence decision {chain_id}.{code} has invalid decision status"
                )
        included = sorted(
            _canonical_hs6(decision.get("code"), label=f"evidence {chain_id}")
            for decision in decisions
            if isinstance(decision, dict) and decision.get("decision") == "include"
        )
        if included != codes:
            raise CacheValidationError(
                f"active registry codes differ from included evidence for {chain_id!r}"
            )
        if evidence_entry.get("included_count") != len(codes):
            raise CacheValidationError(f"registry evidence included_count is stale for {chain_id!r}")
        union.update(codes)
        chain_records.append(
            {
                "id": chain_id,
                "path": logical_path,
                "sha256": digest,
                "active_hs6_codes": codes,
            }
        )

    union_codes = sorted(union)
    if audit.get("summary", {}).get("included_codes") != len(union_codes):
        raise CacheValidationError("registry audit included_codes is stale")
    if evidence.get("summary", {}).get("included_codes") != len(union_codes):
        raise CacheValidationError("registry evidence included_codes is stale")
    return {
        "audit": {
            "path": "docs/registry_audit.json",
            "sha256": sha256_file(audit_path),
            "schema_version": audit["schema_version"],
        },
        "evidence": {
            "path": "chains/evidence/registry_evidence.json",
            "sha256": sha256_file(evidence_path),
            "schema_version": evidence["schema_version"],
        },
        "chain_count": EXPECTED_CHAIN_COUNT,
        "chains": chain_records,
        "active_hs6_union": union_codes,
    }


def _normalize_source_chunk(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise CacheValidationError(
            f"BACI {year} source columns are not {list(SOURCE_COLUMNS)}: {list(frame.columns)}"
        )
    out = frame.copy()
    out["k"] = out["k"].astype("string").str.zfill(6)
    for column in ("i", "j"):
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or ((values % 1) != 0).any():
            raise CacheValidationError(f"BACI {year} has invalid {column} country codes")
        out[column] = values.astype("int64")
    values = pd.to_numeric(out["v"], errors="coerce")
    if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
        raise CacheValidationError(f"BACI {year} has non-numeric or non-finite trade values")
    out["v"] = values.astype("float64")
    out["year"] = int(year)
    return out[list(CACHE_COLUMNS)]


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic gzip member (fixed mtime/name and LF CSV)."""

    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="", write_through=True) as text:
                frame.to_csv(text, index=False, lineterminator="\n")
        raw.flush()
        os.fsync(raw.fileno())


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _stat_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0))


def build_cache(
    raw_archive: Path,
    output_dir: Path,
    *,
    chains_dir: Path = DEFAULT_CHAINS_DIR,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    years: Iterable[int] = REQUIRED_YEARS,
    chunk_rows: int = 500_000,
) -> dict:
    """Build an atomic, deterministic filtered cache and return its manifest.

    ``output_dir`` must not exist.  Production callers use the fixed
    :data:`REQUIRED_YEARS`; the ``years`` argument exists for small synthetic
    tests and is always recorded and enforced exactly.
    """

    raw_archive = raw_archive.expanduser().resolve()
    output_dir = _private_cache_path(output_dir)
    years = tuple(sorted({int(year) for year in years}))
    if not years:
        raise CacheValidationError("cache years must not be empty")
    if chunk_rows <= 0:
        raise CacheValidationError("chunk_rows must be positive")
    if not raw_archive.is_file():
        raise FileNotFoundError(raw_archive)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing cache directory: {output_dir}")

    snapshot = registry_snapshot(
        chains_dir=chains_dir, audit_path=audit_path, evidence_path=evidence_path
    )
    active_codes = set(snapshot["active_hs6_union"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    files: list[dict] = []
    source_members: list[dict] = []
    start_identity = _stat_identity(raw_archive)
    country_payload: bytes
    try:
        years_dir = stage / "years"
        years_dir.mkdir()
        with zipfile.ZipFile(raw_archive) as source_zip:
            names = source_zip.namelist()
            required_members = [COUNTRY_CODES_MEMBER] + [
                TRADE_MEMBER_TEMPLATE.format(year=year) for year in years
            ]
            for member_name in required_members:
                if names.count(member_name) != 1:
                    raise CacheValidationError(
                        f"raw BACI archive must contain exactly one {member_name!r} member"
                    )
            country_payload = source_zip.read(COUNTRY_CODES_MEMBER)
            country_info = source_zip.getinfo(COUNTRY_CODES_MEMBER)
            for year in years:
                member_name = TRADE_MEMBER_TEMPLATE.format(year=year)
                info = source_zip.getinfo(member_name)
                selected: list[pd.DataFrame] = []
                with source_zip.open(member_name) as member_handle:
                    chunks = pd.read_csv(
                        member_handle,
                        usecols=list(SOURCE_COLUMNS),
                        dtype={"k": "string"},
                        chunksize=chunk_rows,
                        low_memory=False,
                    )
                    for chunk in chunks:
                        # Normalize/filter HS6 first so numeric validation and
                        # conversion touch only rows this cache is authorized to
                        # retain. Unrelated BACI products cannot make the narrow
                        # audited cache fail or consume conversion time.
                        chunk["k"] = chunk["k"].astype("string").str.zfill(6)
                        keep = chunk[chunk["k"].isin(active_codes)]
                        if not keep.empty:
                            selected.append(_normalize_source_chunk(keep, year))
                frame = (
                    pd.concat(selected, ignore_index=True)
                    if selected
                    else pd.DataFrame(columns=list(CACHE_COLUMNS))
                )
                relative = f"years/baci_hs92_{year}.csv.gz"
                target = stage / PurePosixPath(relative)
                _write_gzip_csv(target, frame)
                observed_codes = sorted(set(frame["k"].astype(str))) if len(frame) else []
                files.append(
                    {
                        "year": year,
                        "path": relative,
                        "rows": int(len(frame)),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                        "observed_hs6_codes": observed_codes,
                    }
                )
                source_members.append(
                    {
                        "year": year,
                        "name": member_name,
                        "uncompressed_bytes": int(info.file_size),
                        "crc32": f"{info.CRC:08x}",
                    }
                )
        if _stat_identity(raw_archive) != start_identity:
            raise CacheValidationError("raw BACI archive changed while cache rows were read")
        archive_sha256 = sha256_file(raw_archive)
        if _stat_identity(raw_archive) != start_identity:
            raise CacheValidationError("raw BACI archive changed while its SHA-256 was computed")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "visibility": "private-never-publish",
            "format": CACHE_FORMAT,
            "columns": list(CACHE_COLUMNS),
            "years": list(years),
            "source": {
                "dataset": "CEPII BACI HS92 V202401b",
                "archive_name": raw_archive.name,
                "archive_bytes": raw_archive.stat().st_size,
                "archive_sha256": archive_sha256,
                "country_codes_member": {
                    "name": COUNTRY_CODES_MEMBER,
                    "bytes": len(country_payload),
                    "sha256": _sha256_bytes(country_payload),
                    "uncompressed_bytes": int(country_info.file_size),
                    "crc32": f"{country_info.CRC:08x}",
                },
                "trade_members": source_members,
            },
            "registry": snapshot,
            "files": files,
            "totals": {
                "files": len(files),
                "rows": sum(entry["rows"] for entry in files),
                "bytes": sum(entry["bytes"] for entry in files),
            },
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_atomic_bytes(stage / MANIFEST_NAME, payload)
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CacheValidationError(f"{label} must be an integer >= {minimum}")
    return value


class BaciFilteredCache:
    """Validated reader for complete audited-HS6 annual BACI rows.

    All manifest-listed files are opened and hash-checked during construction.
    ``requested_years`` controls which files are also decompressed,
    content-validated, and kept resident; :meth:`read_year` validates another
    manifest year on demand.  This preserves fail-closed whole-cache integrity
    without repeatedly parsing unused windows in every chain process.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        requested_years: Iterable[int] | None = None,
        expected_years: Iterable[int] = REQUIRED_YEARS,
        chains_dir: Path = DEFAULT_CHAINS_DIR,
        audit_path: Path = DEFAULT_AUDIT_PATH,
        evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    ) -> None:
        self.cache_dir = _private_cache_path(Path(cache_dir))
        if not self.cache_dir.is_dir():
            raise CacheValidationError(f"BACI cache directory does not exist: {self.cache_dir}")
        if self.cache_dir.is_symlink():
            raise CacheValidationError("BACI cache root must not be a symlink")
        manifest_path = self.cache_dir / MANIFEST_NAME
        self.manifest = _json_object(manifest_path, "BACI cache manifest")
        self._expected_years = tuple(sorted({int(year) for year in expected_years}))
        requested = (
            set(self._expected_years)
            if requested_years is None
            else {int(year) for year in requested_years}
        )
        if not requested <= set(self._expected_years):
            raise CacheValidationError(
                f"requested years are outside the exact cache protocol: {sorted(requested)}"
            )
        self._validate_manifest(
            chains_dir=chains_dir, audit_path=audit_path, evidence_path=evidence_path
        )
        self._frames: dict[int, pd.DataFrame] = {}
        for year in self._expected_years:
            payload = self._checked_payload(self._entries[year])
            if year in requested:
                self._frames[year] = self._decode_validated_entry(
                    self._entries[year], payload
                )

    @property
    def years(self) -> tuple[int, ...]:
        return self._expected_years

    @property
    def active_hs6_codes(self) -> tuple[str, ...]:
        return tuple(self.manifest["registry"]["active_hs6_union"])

    def _validate_manifest(
        self, *, chains_dir: Path, audit_path: Path, evidence_path: Path
    ) -> None:
        manifest = self.manifest
        expected_top_keys = {
            "schema_version",
            "visibility",
            "format",
            "columns",
            "years",
            "source",
            "registry",
            "files",
            "totals",
        }
        if set(manifest) != expected_top_keys:
            raise CacheValidationError(
                "BACI cache manifest top-level schema is not exact: "
                f"missing={sorted(expected_top_keys - set(manifest))}, "
                f"extra={sorted(set(manifest) - expected_top_keys)}"
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CacheValidationError("unsupported or missing BACI cache schema")
        if manifest.get("visibility") != "private-never-publish":
            raise CacheValidationError("BACI cache manifest is not marked private-never-publish")
        if manifest.get("format") != CACHE_FORMAT:
            raise CacheValidationError("unsupported BACI cache format")
        if manifest.get("columns") != list(CACHE_COLUMNS):
            raise CacheValidationError("BACI cache column schema differs from i,j,k,year,v")
        if manifest.get("years") != list(self._expected_years):
            raise CacheValidationError(
                "BACI cache years differ from the exact required protocol: "
                f"expected={list(self._expected_years)}, actual={manifest.get('years')!r}"
            )
        current_registry = registry_snapshot(
            chains_dir=chains_dir, audit_path=audit_path, evidence_path=evidence_path
        )
        if manifest.get("registry") != current_registry:
            raise CacheValidationError("BACI cache registry/evidence snapshot is stale")

        source = manifest.get("source")
        if not isinstance(source, dict):
            raise CacheValidationError("BACI cache manifest lacks source provenance")
        expected_source_keys = {
            "dataset",
            "archive_name",
            "archive_bytes",
            "archive_sha256",
            "country_codes_member",
            "trade_members",
        }
        if set(source) != expected_source_keys:
            raise CacheValidationError("BACI cache source provenance schema is not exact")
        archive_name = source.get("archive_name")
        if (
            not isinstance(archive_name, str)
            or not archive_name
            or PurePosixPath(archive_name).name != archive_name
        ):
            raise CacheValidationError("BACI cache raw archive name is unsafe")
        if not _SHA256_RE.fullmatch(str(source.get("archive_sha256", ""))):
            raise CacheValidationError("BACI cache lacks a strong raw-archive SHA-256")
        _require_int(source.get("archive_bytes"), label="source.archive_bytes", minimum=1)
        country = source.get("country_codes_member")
        if not isinstance(country, dict) or country.get("name") != COUNTRY_CODES_MEMBER:
            raise CacheValidationError("BACI cache country-code provenance is invalid")
        if set(country) != {"name", "bytes", "sha256", "uncompressed_bytes", "crc32"}:
            raise CacheValidationError("BACI cache country-code provenance schema is not exact")
        if not _SHA256_RE.fullmatch(str(country.get("sha256", ""))):
            raise CacheValidationError("BACI cache country-code member lacks SHA-256")
        country_bytes = _require_int(
            country.get("bytes"), label="country_codes_member.bytes", minimum=1
        )
        if _require_int(
            country.get("uncompressed_bytes"),
            label="country_codes_member.uncompressed_bytes",
            minimum=1,
        ) != country_bytes:
            raise CacheValidationError("BACI country-code member byte metadata is inconsistent")
        if not re.fullmatch(r"[0-9a-f]{8}", str(country.get("crc32", ""))):
            raise CacheValidationError("BACI country-code member CRC32 metadata is invalid")

        trade_members = source.get("trade_members")
        if not isinstance(trade_members, list) or len(trade_members) != len(
            self._expected_years
        ):
            raise CacheValidationError("BACI cache source must list every exact annual member")
        trade_years: set[int] = set()
        for member_entry in trade_members:
            if not isinstance(member_entry, dict) or set(member_entry) != {
                "year",
                "name",
                "uncompressed_bytes",
                "crc32",
            }:
                raise CacheValidationError("BACI cache annual source-member schema is invalid")
            member_year = _require_int(
                member_entry.get("year"), label="source trade-member year", minimum=1
            )
            if member_year in trade_years:
                raise CacheValidationError("BACI cache has duplicate annual source-member metadata")
            trade_years.add(member_year)
            if member_entry.get("name") != TRADE_MEMBER_TEMPLATE.format(year=member_year):
                raise CacheValidationError("BACI cache annual source-member name is stale")
            _require_int(
                member_entry.get("uncompressed_bytes"),
                label=f"source trade-member {member_year} bytes",
                minimum=1,
            )
            if not re.fullmatch(r"[0-9a-f]{8}", str(member_entry.get("crc32", ""))):
                raise CacheValidationError("BACI cache annual source-member CRC32 is invalid")
        if trade_years != set(self._expected_years):
            raise CacheValidationError("BACI cache annual source-member years are not exact")

        union = set(current_registry["active_hs6_union"])
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(self._expected_years):
            raise CacheValidationError("BACI cache manifest must list exactly one file per year")
        entries: dict[int, dict] = {}
        expected_paths: set[str] = {MANIFEST_NAME}
        for entry in files:
            if not isinstance(entry, dict):
                raise CacheValidationError("BACI cache file entry is not an object")
            if set(entry) != {
                "year",
                "path",
                "rows",
                "bytes",
                "sha256",
                "observed_hs6_codes",
            }:
                raise CacheValidationError("BACI cache file-entry schema is not exact")
            year = _require_int(entry.get("year"), label="cache file year", minimum=1)
            if year in entries:
                raise CacheValidationError(f"duplicate BACI cache file entry for {year}")
            expected_path = f"years/baci_hs92_{year}.csv.gz"
            if entry.get("path") != expected_path:
                raise CacheValidationError(f"unsafe or noncanonical BACI cache path for {year}")
            if not _SHA256_RE.fullmatch(str(entry.get("sha256", ""))):
                raise CacheValidationError(f"BACI cache file {year} lacks SHA-256")
            _require_int(entry.get("rows"), label=f"cache file {year} rows")
            _require_int(entry.get("bytes"), label=f"cache file {year} bytes", minimum=1)
            observed = entry.get("observed_hs6_codes")
            if not isinstance(observed, list) or observed != sorted(set(observed)):
                raise CacheValidationError(f"BACI cache {year} observed code list is not canonical")
            for code in observed:
                _canonical_hs6(code, label=f"cache {year}")
            extra = set(observed) - union
            if extra:
                raise CacheValidationError(
                    f"BACI cache {year} declares codes outside the audited union: {sorted(extra)}"
                )
            entries[year] = entry
            expected_paths.add(expected_path)
        if set(entries) != set(self._expected_years):
            raise CacheValidationError(
                f"BACI cache missing/extra years: {sorted(set(entries) ^ set(self._expected_years))}"
            )
        totals = manifest.get("totals")
        expected_totals = {
            "files": len(files),
            "rows": sum(entry["rows"] for entry in files),
            "bytes": sum(entry["bytes"] for entry in files),
        }
        if totals != expected_totals:
            raise CacheValidationError("BACI cache totals are stale or malformed")

        actual_paths: set[str] = set()
        for item in self.cache_dir.rglob("*"):
            if item.is_symlink():
                raise CacheValidationError(f"BACI cache contains a symlink: {item.name}")
            if item.is_file():
                actual_paths.add(item.relative_to(self.cache_dir).as_posix())
        if actual_paths != expected_paths:
            raise CacheValidationError(
                "BACI cache inventory differs from its manifest: "
                f"missing={sorted(expected_paths - actual_paths)}, "
                f"extra={sorted(actual_paths - expected_paths)}"
            )
        self._entries = entries

    def _checked_payload(self, entry: dict) -> bytes:
        year = int(entry["year"])
        path = self.cache_dir / PurePosixPath(entry["path"])
        payload = path.read_bytes()
        if len(payload) != entry["bytes"]:
            raise CacheValidationError(f"BACI cache file size mismatch for {year}")
        if _sha256_bytes(payload) != entry["sha256"]:
            raise CacheValidationError(f"BACI cache SHA-256 mismatch for {year}")
        return payload

    def _decode_validated_entry(self, entry: dict, payload: bytes) -> pd.DataFrame:
        year = int(entry["year"])
        try:
            csv_payload = gzip.decompress(payload)
            frame = pd.read_csv(io.BytesIO(csv_payload), dtype={"k": "string"})
        except (OSError, EOFError, pd.errors.ParserError) as exc:
            raise CacheValidationError(f"cannot decode BACI cache file for {year}: {exc}") from exc
        if list(frame.columns) != list(CACHE_COLUMNS):
            raise CacheValidationError(f"BACI cache columns are invalid for {year}")
        if len(frame) != entry["rows"]:
            raise CacheValidationError(f"BACI cache row count mismatch for {year}")
        normalized = _normalize_cached_frame(frame, year)
        observed = sorted(set(normalized["k"].astype(str))) if len(normalized) else []
        if observed != entry["observed_hs6_codes"]:
            raise CacheValidationError(f"BACI cache observed HS6 set mismatch for {year}")
        if not set(observed) <= set(self.active_hs6_codes):
            raise CacheValidationError(f"BACI cache contains an unaudited HS6 code for {year}")
        return normalized

    def read_year(self, year: int) -> pd.DataFrame:
        """Return a defensive copy of all filtered raw rows for ``year``."""

        year = int(year)
        if year not in self._entries:
            raise CacheValidationError(f"year {year} is not present in this BACI cache")
        if year not in self._frames:
            entry = self._entries[year]
            self._frames[year] = self._decode_validated_entry(
                entry, self._checked_payload(entry)
            )
        return self._frames[year].copy(deep=True)

    def country_codes_bytes(
        self, source_zip: zipfile.ZipFile, *, archive_path: Path | None = None
    ) -> bytes:
        """Read and verify the small country-code member used with cached rows.

        If ``archive_path`` is supplied, its filename and size are also checked.
        The manifest already records the whole-archive SHA-256 generated at build
        time; deliberately re-hashing that archive here would erase most of the
        cohort-rebuild speedup.
        """

        source = self.manifest["source"]
        if archive_path is not None:
            archive_path = Path(archive_path).expanduser().resolve()
            if archive_path.name != source.get("archive_name"):
                raise CacheValidationError("raw BACI archive filename differs from cache provenance")
            if not archive_path.is_file() or archive_path.stat().st_size != source["archive_bytes"]:
                raise CacheValidationError("raw BACI archive size differs from cache provenance")
        if source_zip.namelist().count(COUNTRY_CODES_MEMBER) != 1:
            raise CacheValidationError("raw BACI archive lacks an exact country-code member")
        payload = source_zip.read(COUNTRY_CODES_MEMBER)
        expected = source["country_codes_member"]
        if len(payload) != expected["bytes"] or _sha256_bytes(payload) != expected["sha256"]:
            raise CacheValidationError("raw BACI country-code member differs from cache provenance")
        return payload


def _normalize_cached_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    out = frame.copy()
    if len(out) == 0:
        return out[list(CACHE_COLUMNS)]
    for column in ("i", "j", "year"):
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or ((values % 1) != 0).any():
            raise CacheValidationError(f"BACI cache {year} has invalid integer column {column}")
        out[column] = values.astype("int64")
    if not out["year"].eq(year).all():
        raise CacheValidationError(f"BACI cache {year} contains a different year value")
    values = pd.to_numeric(out["v"], errors="coerce")
    if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
        raise CacheValidationError(f"BACI cache {year} has invalid trade values")
    out["v"] = values.astype("float64")
    out["k"] = out["k"].astype("string")
    bad_codes = [code for code in out["k"].astype(str).unique() if not _HS6_RE.fullmatch(code)]
    if bad_codes:
        raise CacheValidationError(f"BACI cache {year} has noncanonical HS6 codes: {bad_codes[:3]}")
    return out[list(CACHE_COLUMNS)]


def read_trade_year(source: object, year: int) -> pd.DataFrame:
    """Read one trade year from either a validated cache or a direct ZipFile."""

    if isinstance(source, BaciFilteredCache):
        return source.read_year(year)
    try:
        payload = source.open(TRADE_MEMBER_TEMPLATE.format(year=int(year))).read()
    except AttributeError as exc:
        raise TypeError("BACI source must be a BaciFilteredCache or ZipFile-like object") from exc
    return pd.read_csv(io.BytesIO(payload), dtype={"k": str})
