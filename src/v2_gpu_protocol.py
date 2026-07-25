"""Pure-Python protocol gate for strict rolling v2 GPU experiments.

The GPU runner is intentionally split into three separate invocations:

``select`` (fold2 only) -> ``freeze`` (hash every selection) -> ``evaluate``
(main only).  This module contains the file-format and hash checks without
importing torch, pandas, PyKEEN, NBFNet, or any dataset loader.  Consequently an
evaluation process can verify the complete freeze manifest before importing code
that is capable of opening the main target cohort.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROTOCOL = "strict_rolling_fold2_to_main"
SELECTION_SCHEMA = "upgrade-bench-v2/gpu-selection/1"
FREEZE_SCHEMA = "upgrade-bench-v2/gpu-freeze/1"
HISTORY_FOLD = "fold2"
TARGET_FOLD = "main"
TRACKS = ("a", "b1", "b2")
FAMILIES = ("kge", "nbfnet")


class ProtocolError(RuntimeError):
    """Raised when the rolling-protocol gate cannot be proven closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for stable hashes and tests."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, object], *, overwrite: bool = False) -> None:
    """Write a canonical JSON artifact without silently replacing an old run."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ProtocolError(f"refusing to overwrite existing protocol artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def selection_filename(chain: str, track: str, family: str) -> str:
    _validate_combo(chain, track, family)
    return f"selection_{chain}_track-{track}_{family}.json"


def selection_key(chain: str, track: str, family: str) -> str:
    _validate_combo(chain, track, family)
    return f"{chain}|{track}|{family}"


def _validate_combo(chain: str, track: str, family: str) -> None:
    if not chain or any(c in chain for c in "|/\\"):
        raise ProtocolError(f"unsafe/empty chain id: {chain!r}")
    if track not in TRACKS:
        raise ProtocolError(f"unknown track {track!r}; choose from {TRACKS}")
    if family not in FAMILIES:
        raise ProtocolError(f"unknown family {family!r}; choose from {FAMILIES}")


def validate_selection(payload: Mapping[str, object], *, expected=None) -> None:
    """Validate the fields that make a fold2 selection eligible for freezing."""
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise ProtocolError("selection has an unknown schema_version")
    if payload.get("protocol") != PROTOCOL:
        raise ProtocolError("selection is not a strict rolling v2 artifact")
    if payload.get("status") != "complete":
        raise ProtocolError("selection is not complete")
    if payload.get("selection_fold") != HISTORY_FOLD:
        raise ProtocolError("selection did not use fold2 history")
    if payload.get("target_fold") != TARGET_FOLD:
        raise ProtocolError("selection does not reserve main as the target fold")
    if payload.get("aggregation") != "calendar_mean":
        raise ProtocolError("selection did not use the v2 calendar_mean aggregation")
    if payload.get("main_target_labels_accessed") is not False:
        raise ProtocolError("selection does not attest that main target labels were unopened")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ProtocolError("selection has no immutable run_id")
    config_hash = payload.get("run_config_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ProtocolError("selection has no valid run_config_sha256")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or not selected.get("model"):
        raise ProtocolError("selection has no frozen model")
    if not isinstance(selected.get("hyperparameters"), Mapping):
        raise ProtocolError("selection has no frozen hyperparameter mapping")
    design = payload.get("selection_design")
    if not isinstance(design, Mapping) or design.get("orchestration") != "chain_multitask_shared_score_grid":
        raise ProtocolError("selection was not produced by the shared chain/task score grid")
    cache = payload.get("shared_score_cache")
    if not isinstance(cache, Mapping) or len(str(cache.get("context_sha256", ""))) != 64:
        raise ProtocolError("selection has no hash-locked shared score-cache provenance")
    chain = str(payload.get("chain", ""))
    track = str(payload.get("track", ""))
    family = str(payload.get("family", ""))
    _validate_combo(chain, track, family)
    if expected is not None and (chain, track, family) != tuple(expected):
        raise ProtocolError(
            f"selection identity mismatch: got {(chain, track, family)}, expected {tuple(expected)}"
        )


def load_selection(path: Path, *, expected=None) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read selection {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"selection must be a JSON object: {path}")
    validate_selection(payload, expected=expected)
    return payload


def expected_combinations(
    chains: Sequence[str], tracks: Sequence[str], families: Sequence[str]
) -> list[tuple[str, str, str]]:
    combos = []
    for chain in chains:
        for track in tracks:
            for family in families:
                _validate_combo(chain, track, family)
                combos.append((chain, track, family))
    if not combos:
        raise ProtocolError("freeze set is empty")
    return combos


def build_freeze_manifest(
    *,
    selection_dir: Path,
    manifest_path: Path,
    combinations: Iterable[tuple[str, str, str]],
) -> dict:
    """Validate every expected selection and return a hash-locked manifest."""
    selection_dir = Path(selection_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    entries = []
    seen = set()
    run_ids = set()
    config_hashes = set()
    for chain, track, family in combinations:
        key = selection_key(chain, track, family)
        if key in seen:
            raise ProtocolError(f"duplicate selection requested: {key}")
        seen.add(key)
        path = selection_dir / selection_filename(chain, track, family)
        payload = load_selection(path, expected=(chain, track, family))
        run_ids.add(payload["run_id"])
        config_hashes.add(payload["run_config_sha256"])
        try:
            relative = path.resolve().relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ProtocolError("selection files must live below the manifest directory") from exc
        entries.append(
            {
                "key": key,
                "chain": chain,
                "track": track,
                "family": family,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "selected_model": payload["selected"]["model"],
                "run_id": payload["run_id"],
                "run_config_sha256": payload["run_config_sha256"],
            }
        )
    if len(run_ids) != 1 or len(config_hashes) != 1:
        raise ProtocolError("all selections must share one run_id and run-config hash")
    entries.sort(key=lambda row: row["key"])
    return {
        "schema_version": FREEZE_SCHEMA,
        "protocol": PROTOCOL,
        "status": "frozen",
        "selection_fold": HISTORY_FOLD,
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "all_selections_frozen_before_main": True,
        "run_id": next(iter(run_ids)),
        "run_config_sha256": next(iter(config_hashes)),
        "entries": entries,
    }


def verify_freeze_manifest(manifest_path: Path) -> tuple[dict, dict[str, tuple[dict, dict]]]:
    """Verify every locked selection; return manifest and entries indexed by key.

    Call this before importing a target-data loader.  Any missing, changed, extra,
    or malformed selection closes the gate with :class:`ProtocolError`.
    """
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read freeze manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProtocolError("freeze manifest must be a JSON object")
    checks = {
        "schema_version": FREEZE_SCHEMA,
        "protocol": PROTOCOL,
        "status": "frozen",
        "selection_fold": HISTORY_FOLD,
        "target_fold": TARGET_FOLD,
        "aggregation": "calendar_mean",
        "all_selections_frozen_before_main": True,
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise ProtocolError(f"invalid freeze manifest field {field!r}")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ProtocolError("freeze manifest has no run_id")
    if not isinstance(manifest.get("run_config_sha256"), str) or len(manifest["run_config_sha256"]) != 64:
        raise ProtocolError("freeze manifest has no valid run_config_sha256")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("freeze manifest has no entries")
    indexed = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProtocolError("freeze entry must be a JSON object")
        combo = (str(entry.get("chain", "")), str(entry.get("track", "")), str(entry.get("family", "")))
        key = selection_key(*combo)
        if entry.get("key") != key or key in indexed:
            raise ProtocolError(f"invalid or duplicate freeze entry key: {entry.get('key')!r}")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ProtocolError(f"freeze entry {key} has no path")
        path = (manifest_path.parent / raw_path).resolve()
        try:
            path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ProtocolError(f"freeze entry escapes manifest root: {raw_path}") from exc
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ProtocolError(f"selection changed after freeze: {path}")
        selection = load_selection(path, expected=combo)
        if selection["run_id"] != manifest["run_id"] or selection["run_config_sha256"] != manifest["run_config_sha256"]:
            raise ProtocolError(f"run/config lock mismatch in freeze entry {key}")
        if selection["selected"]["model"] != entry.get("selected_model"):
            raise ProtocolError(f"selected model mismatch in freeze entry {key}")
        indexed[key] = (entry, selection)
    return manifest, indexed
