"""Stable, label-independent candidate splitting for Upgrade-Bench.

The benchmark prediction unit is a lane ``(exporter, stage, importer)``, but a
lane-level random/hash split lets the same future exporter-stage entry event
appear in both development and test through different importers.  The official
split therefore hashes *groups*, not rows, and defaults to ``exporter_stage``.

Supported split units
---------------------
``lane``
    Each candidate lane is assigned independently.  This reproduces the
    historical SHA-256 lane split and is retained only as an explicit legacy
    option.
``exporter_stage``
    All importers for one ``(exporter, stage)`` are assigned together.  This is
    the official default for both Track A and Track B.
``exporter``
    All rows for one exporter are assigned together.
``importer``
    All rows for one importer are assigned together.

Every assignment depends only on the requested identity fields, chain, salt,
and test fraction.  It never accepts labels, is independent of row order, and
is stable when unrelated candidates are added or removed.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


SPLIT_UNITS = ("lane", "exporter_stage", "exporter", "importer")
OFFICIAL_SPLIT_UNIT = "exporter_stage"
OFFICIAL_TRACK_SPLIT_UNITS = {"A": OFFICIAL_SPLIT_UNIT, "B": OFFICIAL_SPLIT_UNIT}
DEFAULT_SPLIT_SALT = 0
DEFAULT_TEST_FRACTION = 0.5


def official_split_unit(track: str) -> str:
    """Return the group-safe official unit for Track A or Track B."""
    key = str(track).strip().upper()
    if key not in OFFICIAL_TRACK_SPLIT_UNITS:
        raise ValueError(f"track must be 'A' or 'B', got {track!r}")
    return OFFICIAL_TRACK_SPLIT_UNITS[key]


def _identity_array(name: str, values: Iterable[object]) -> np.ndarray:
    """Normalize one identity field and reject missing/ambiguous values."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    out = raw.astype(str)
    bad = np.array([v.strip() == "" or v.strip().lower() in {"nan", "none"} for v in out])
    if bad.any():
        raise ValueError(f"{name} contains {int(bad.sum())} missing/empty identity values")
    return out


def _candidate_arrays(exporter, stage, importer):
    exporter = _identity_array("exporter", exporter)
    stage = _identity_array("stage", stage)
    importer = _identity_array("importer", importer)
    lengths = {len(exporter), len(stage), len(importer)}
    if len(lengths) != 1:
        raise ValueError(
            "exporter, stage, and importer must have the same length; got "
            f"{len(exporter)}, {len(stage)}, {len(importer)}"
        )
    return exporter, stage, importer


def split_group_keys(
    chain,
    exporter,
    stage,
    importer,
    *,
    unit: str = OFFICIAL_SPLIT_UNIT,
):
    """Build canonical hash keys for the requested evaluation unit.

    ``unit='lane'`` deliberately retains the historical key encoding
    ``chain|exporter|stage|importer`` so explicit legacy lane runs reproduce the
    previous SHA-256 assignments exactly.
    """
    if unit not in SPLIT_UNITS:
        raise ValueError(f"unknown split unit {unit!r}; choose from {SPLIT_UNITS}")
    chain = str(chain).strip()
    if not chain:
        raise ValueError("chain must be a non-empty identifier")
    exporter, stage, importer = _candidate_arrays(exporter, stage, importer)

    if unit == "lane":
        return [f"{chain}|{e}|{s}|{j}" for e, s, j in zip(exporter, stage, importer)]
    if unit == "exporter_stage":
        return [f"{chain}|exporter_stage|{e}|{s}" for e, s in zip(exporter, stage)]
    if unit == "exporter":
        return [f"{chain}|exporter|{e}" for e in exporter]
    return [f"{chain}|importer|{j}" for j in importer]


def _u01(key, salt):
    """Deterministic uniform[0,1) from a group key and salt via SHA-256."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def hash_uniform(keys, salt=DEFAULT_SPLIT_SALT):
    """Map pre-built keys to stable uniform values.

    This low-level function remains for backwards compatibility.  New callers
    should use :func:`split_test_mask`, which makes the evaluation unit explicit.
    """
    return np.array([_u01(str(key), salt) for key in keys], dtype=float)


def hash_test_mask(keys, salt=DEFAULT_SPLIT_SALT, test_frac=DEFAULT_TEST_FRACTION):
    """Legacy low-level mask over pre-built keys.

    Use :func:`split_test_mask` for official code paths.  Keeping this function
    preserves exact historical lane assignments for explicit legacy runs.
    """
    fraction = _validate_test_fraction(test_frac)
    return hash_uniform(keys, salt) < fraction


def split_test_mask(
    chain,
    exporter,
    stage,
    importer,
    *,
    unit: str = OFFICIAL_SPLIT_UNIT,
    salt=DEFAULT_SPLIT_SALT,
    test_frac: float = DEFAULT_TEST_FRACTION,
) -> np.ndarray:
    """Return a boolean test mask for an explicit candidate/group unit."""
    keys = split_group_keys(chain, exporter, stage, importer, unit=unit)
    return hash_test_mask(keys, salt=salt, test_frac=test_frac)


def split_labels(
    chain,
    exporter,
    stage,
    importer,
    *,
    unit: str = OFFICIAL_SPLIT_UNIT,
    salt=DEFAULT_SPLIT_SALT,
    test_frac: float = DEFAULT_TEST_FRACTION,
) -> np.ndarray:
    """Return a ``train``/``test`` label per candidate."""
    test = split_test_mask(
        chain,
        exporter,
        stage,
        importer,
        unit=unit,
        salt=salt,
        test_frac=test_frac,
    )
    return np.where(test, "test", "train")


def candidate_keys(chain, exporter, stage, importer):
    """Historical lane keys; equivalent to ``split_group_keys(..., unit='lane')``."""
    return split_group_keys(chain, exporter, stage, importer, unit="lane")


def _validate_test_fraction(test_frac):
    try:
        value = float(test_frac)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"test_frac must be numeric, got {test_frac!r}") from exc
    if not 0.0 < value < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac!r}")
    return value
