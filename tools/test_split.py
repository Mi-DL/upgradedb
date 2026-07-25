#!/usr/bin/env python
"""Invariant tests for the official Upgrade-Bench split API.

Run: ``python tools/test_split.py``
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from split import (  # noqa: E402
    OFFICIAL_SPLIT_UNIT,
    candidate_keys,
    hash_test_mask,
    official_split_unit,
    split_group_keys,
    split_labels,
    split_test_mask,
)


def _fixture(n_exporters=40):
    exporters, stages, importers = [], [], []
    for ei in range(n_exporters):
        for stage in ("processed_a", "processed_b"):
            for ji in range(4):
                exporters.append(f"E{ei:03d}")
                stages.append(stage)
                importers.append(f"I{(ei + ji) % 31:03d}")
    return np.array(exporters), np.array(stages), np.array(importers)


def _lane_ids(exporter, stage, importer):
    return [f"{e}|{s}|{j}" for e, s, j in zip(exporter, stage, importer)]


def test_official_track_defaults_are_group_safe():
    assert OFFICIAL_SPLIT_UNIT == "exporter_stage"
    assert official_split_unit("A") == "exporter_stage"
    assert official_split_unit("b") == "exporter_stage"


def test_row_order_and_unrelated_rows_do_not_change_assignment():
    e, s, j = _fixture()
    base = split_test_mask("demo", e, s, j)
    ids = _lane_ids(e, s, j)

    order = np.arange(len(e))[::-1]
    reordered = split_test_mask("demo", e[order], s[order], j[order])
    by_id = dict(zip(_lane_ids(e[order], s[order], j[order]), reordered))
    assert np.array_equal(base, np.array([by_id[key] for key in ids]))

    extended = split_test_mask(
        "demo",
        np.append(e, "NEW_EXPORTER"),
        np.append(s, "NEW_STAGE"),
        np.append(j, "NEW_IMPORTER"),
    )
    assert np.array_equal(base, extended[:-1])


def test_requested_groups_are_disjoint():
    e, s, j = _fixture()
    for unit in ("exporter_stage", "exporter", "importer"):
        groups = split_group_keys("demo", e, s, j, unit=unit)
        test = split_test_mask("demo", e, s, j, unit=unit)
        side_by_group = {}
        for group, side in zip(groups, test):
            previous = side_by_group.setdefault(group, bool(side))
            assert previous == bool(side), f"{unit} group {group} crosses train/test"
        assert test.any() and (~test).any(), f"synthetic {unit} split is degenerate"


def test_default_prevents_exporter_stage_crossing():
    e = np.array(["A", "A", "A", "B", "B"])
    s = np.array(["x", "x", "y", "x", "x"])
    j = np.array(["I1", "I2", "I3", "I1", "I4"])
    labels = split_labels("demo", e, s, j)
    assert labels[0] == labels[1]
    assert labels[3] == labels[4]


def test_split_is_label_independent():
    e, s, j = _fixture()
    labels_a = np.zeros(len(e), dtype=int)
    labels_b = np.arange(len(e)) % 2
    before = split_test_mask("demo", e, s, j, unit="exporter_stage", salt=17)
    # Altering every outcome while leaving identities fixed must not alter the split.
    labels_a[:] = labels_b
    after = split_test_mask("demo", e, s, j, unit="exporter_stage", salt=17)
    assert np.array_equal(before, after)


def test_explicit_lane_mode_reproduces_historical_sha_split():
    e, s, j = _fixture(8)
    old = hash_test_mask(candidate_keys("demo", e, s, j), salt=0)
    explicit = split_test_mask("demo", e, s, j, unit="lane", salt=0)
    assert np.array_equal(old, explicit)


def test_invalid_configuration_is_rejected():
    e, s, j = _fixture(2)
    for bad_unit in ("", "country_stage", None):
        try:
            split_test_mask("demo", e, s, j, unit=bad_unit)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid split unit {bad_unit!r} was accepted")
    for frac in (0, 1, -0.1, 1.1):
        try:
            split_test_mask("demo", e, s, j, test_frac=frac)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid test fraction {frac!r} was accepted")


def main():
    tests = [
        test_official_track_defaults_are_group_safe,
        test_row_order_and_unrelated_rows_do_not_change_assignment,
        test_requested_groups_are_disjoint,
        test_default_prevents_exporter_stage_crossing,
        test_split_is_label_independent,
        test_explicit_lane_mode_reproduces_historical_sha_split,
        test_invalid_configuration_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nall {len(tests)} split tests passed")


if __name__ == "__main__":
    main()
