from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.v2_b1_coverage import (
    ROOT,
    _assert_production_protocol_literals,
    _candidate_reconciliation,
    _path_from_report,
    candidate_universe_from_windows,
    coverage_from_windows,
    stage_window_from_cache,
)


class B1CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The reconciliation fixtures must remain below the repository root so
        # public-path guards are exercised. A clean clone need not already have
        # the gitignored tmp/ directory.
        (ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _entry_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
        early = pd.DataFrame(
            [
                ("A", "M1", "raw", 200.0),
                ("B", "M1", "raw", 200.0),
                ("C", "M1", "raw", 200.0),
                ("D", "M1", "raw", 200.0),
                ("D", "E1", "processed", 200.0),
                ("X", "E2", "processed", 200.0),
            ],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        late = pd.DataFrame(
            [
                ("A", "E1", "processed", 300.0),
                ("A", "N1", "processed", 200.0),
                ("B", "N2", "processed", 300.0),
                ("C", "E2", "processed", 200.0),
                ("D", "N3", "processed", 999.0),
            ],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        return early, late

    @staticmethod
    def _write_release_views(
        root: Path, lanes: pd.DataFrame, entries: pd.DataFrame
    ) -> None:
        lane_view = lanes.copy()
        lane_view = lane_view.merge(
            entries[["i_iso", "stage", "z"]].rename(columns={"z": "entry_y"}),
            on=["i_iso", "stage"],
            how="left",
            validate="many_to_one",
        )
        lane_view["entry_id"] = (
            lane_view.i_iso.astype(str) + "|" + lane_view.stage.astype(str)
        )
        lane_view["aggregation"] = "calendar_mean"
        lane_view["early_window"] = "2000-2004"
        lane_view["late_window"] = "2010-2014"
        lane_view["temporal_role"] = "target"
        lane_view["task"] = "processed_export_entry_candidate_lane"
        lane_view["task_unit"] = "exporter_stage_destination"

        entry_view = entries.copy()
        entry_view["entry_id"] = (
            entry_view.i_iso.astype(str) + "|" + entry_view.stage.astype(str)
        )
        entry_view["aggregation"] = "calendar_mean"
        entry_view["early_window"] = "2000-2004"
        entry_view["late_window"] = "2010-2014"
        entry_view["temporal_role"] = "target"
        entry_view["task"] = "processed_export_stage_entry"
        entry_view["task_unit"] = "exporter_stage"

        lane_view.to_csv(root / "candidates_firsttime_demo.csv", index=False)
        entry_view.to_csv(root / "entries_firsttime_demo.csv", index=False)

    def test_raw_cache_window_uses_complete_calendar_denominator(self) -> None:
        class FakeCache:
            def __init__(self) -> None:
                self.frames = {
                    2000: pd.DataFrame(
                        [
                            (1, 2, "010410", 2000, 300.0),
                            # Exactly 100 after the two-year denominator: inactive.
                            (1, 2, "020410", 2000, 200.0),
                            # A dissolved-state endpoint is removed independently.
                            (3, 2, "010410", 2000, 999.0),
                        ],
                        columns=["i", "j", "k", "year", "v"],
                    ),
                    2001: pd.DataFrame(
                        columns=["i", "j", "k", "year", "v"]
                    ),
                }

            def read_year(self, year: int) -> pd.DataFrame:
                return self.frames[year].copy()

        frame = stage_window_from_cache(
            FakeCache(),  # type: ignore[arg-type]
            {1: "A", 2: "B", 3: "SUN"},
            "sheep",
            [2000, 2001],
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].stage, "exp_live")
        self.assertAlmostEqual(float(frame.iloc[0].v), 150.0)

    def test_duplicated_protocol_literals_match_candidate_generator(self) -> None:
        _assert_production_protocol_literals()

    def test_reports_covered_inactive_only_and_mixed_entries(self) -> None:
        early, late = self._entry_windows()

        result = coverage_from_windows(early, late, {"processed": ["raw"]})
        totals = result["totals"]
        self.assertEqual(
            totals["n_upstream_qualified_nonincumbent_exporter_stage_pairs"], 3
        )
        self.assertEqual(totals["n_released_candidate_entries"], 3)
        self.assertEqual(totals["n_all_realized_entries"], 3)
        self.assertEqual(totals["n_covered_realized_entries"], 2)
        self.assertEqual(totals["n_inactive_only_realized_entries"], 1)
        self.assertEqual(totals["n_mixed_realized_entries"], 1)
        self.assertEqual(totals["n_all_late_start_lanes"], 4)
        self.assertEqual(totals["n_eligible_market_late_start_lanes"], 2)
        self.assertEqual(totals["n_previously_inactive_market_late_start_lanes"], 2)
        self.assertAlmostEqual(totals["realized_entry_coverage"], 2 / 3)
        self.assertAlmostEqual(totals["late_start_lane_coverage"], 0.5)
        self.assertAlmostEqual(totals["late_start_value_coverage"], 0.5)

    def test_exact_candidate_universe_matches_entry_view_semantics(self) -> None:
        early, late = self._entry_windows()
        lanes, entries = candidate_universe_from_windows(
            early, late, {"processed": ["raw"]}
        )
        self.assertEqual(len(lanes), 6)
        self.assertEqual(int(lanes.y.sum()), 2)
        self.assertAlmostEqual(float(lanes.lateval.sum()), 500.0)
        self.assertEqual(len(entries), 3)
        self.assertEqual(int(entries.z.sum()), 2)
        by_exporter = entries.set_index("i_iso")
        self.assertEqual(int(by_exporter.loc["A", "n_candidate_destinations"]), 2)
        self.assertEqual(int(by_exporter.loc["A", "n_materialized_destinations"]), 1)
        self.assertAlmostEqual(float(by_exporter.loc["A", "entry_lateval"]), 300.0)
        self.assertEqual(int(by_exporter.loc["B", "z"]), 0)

    def test_reconciliation_rejects_count_preserving_entry_identity_swap(self) -> None:
        early, late = self._entry_windows()
        coverage = coverage_from_windows(early, late, {"processed": ["raw"]})
        lanes, entries = candidate_universe_from_windows(
            early, late, {"processed": ["raw"]}
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            self._write_release_views(root, lanes, entries)
            path = root / "entries_firsttime_demo.csv"
            swapped = pd.read_csv(path, dtype={"i_iso": str, "stage": str})
            swapped.loc[0, "i_iso"] = "Q"
            swapped.loc[0, "entry_id"] = "Q|processed"
            swapped.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "entry identities differ"):
                _candidate_reconciliation(
                    root,
                    "demo",
                    "",
                    coverage["totals"],
                    lanes,
                    entries,
                    early_years=range(2000, 2005),
                    late_years=range(2010, 2015),
                    temporal_role="target",
                )

    def test_reconciliation_accepts_exact_lane_and_entry_views(self) -> None:
        early, late = self._entry_windows()
        coverage = coverage_from_windows(early, late, {"processed": ["raw"]})
        lanes, entries = candidate_universe_from_windows(
            early, late, {"processed": ["raw"]}
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            self._write_release_views(root, lanes, entries)
            reconciliation, hashes = _candidate_reconciliation(
                root,
                "demo",
                "",
                coverage["totals"],
                lanes,
                entries,
                early_years=range(2000, 2005),
                late_years=range(2010, 2015),
                temporal_role="target",
            )
            self.assertTrue(reconciliation["pass"])
            self.assertTrue(
                reconciliation["exact_lane_identity_label_value_reconciliation"]
            )
            self.assertEqual(len(hashes), 2)

    def test_reconciliation_rejects_value_drift_with_unchanged_counts(self) -> None:
        early, late = self._entry_windows()
        coverage = coverage_from_windows(early, late, {"processed": ["raw"]})
        lanes, entries = candidate_universe_from_windows(
            early, late, {"processed": ["raw"]}
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            self._write_release_views(root, lanes, entries)
            path = root / "candidates_firsttime_demo.csv"
            drifted = pd.read_csv(path, dtype={"i_iso": str, "j_iso": str, "stage": str})
            positive = drifted.index[drifted.y.eq(1)][0]
            drifted.loc[positive, "lateval"] += 1.0
            drifted.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "lateval values"):
                _candidate_reconciliation(
                    root,
                    "demo",
                    "",
                    coverage["totals"],
                    lanes,
                    entries,
                    early_years=range(2000, 2005),
                    late_years=range(2010, 2015),
                    temporal_role="target",
                )

    def test_report_path_rejects_parent_traversal_before_file_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe repository-relative"):
            _path_from_report("../private.csv", label="candidate input")

    def test_zero_realized_starts_have_explicit_null_rates(self) -> None:
        early = pd.DataFrame(
            [("A", "M", "raw", 200.0), ("X", "E", "processed", 200.0)],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        late = pd.DataFrame(
            [("Z", "E", "processed", 200.0)],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        totals = coverage_from_windows(early, late, {"processed": ["raw"]})["totals"]
        self.assertEqual(totals["n_all_realized_entries"], 0)
        self.assertIsNone(totals["realized_entry_coverage"])
        self.assertIsNone(totals["late_start_lane_coverage"])
        self.assertIsNone(totals["late_start_value_coverage"])

    def test_self_market_is_excluded_from_released_candidates_and_coverage(self) -> None:
        early = pd.DataFrame(
            [
                ("A", "M", "raw", 200.0),
                # A is the processed stage's only early-demand destination.
                ("X", "A", "processed", 200.0),
            ],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        late = pd.DataFrame(
            [("A", "B", "processed", 300.0)],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        result = coverage_from_windows(early, late, {"processed": ["raw"]})
        lanes, entries = candidate_universe_from_windows(
            early, late, {"processed": ["raw"]}
        )
        self.assertTrue(lanes.empty)
        self.assertTrue(entries.empty)
        self.assertEqual(result["totals"]["n_released_candidate_entries"], 0)
        self.assertEqual(result["totals"]["n_all_realized_entries"], 1)
        self.assertEqual(result["totals"]["n_inactive_only_realized_entries"], 1)
        self.assertEqual(result["totals"]["realized_entry_coverage"], 0.0)

    def test_rejects_duplicate_stage_lanes(self) -> None:
        early = pd.DataFrame(
            [("A", "M", "raw", 200.0), ("A", "M", "raw", 300.0)],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        late = pd.DataFrame(
            [("A", "E", "processed", 200.0)],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            coverage_from_windows(early, late, {"processed": ["raw"]})


if __name__ == "__main__":
    unittest.main()
