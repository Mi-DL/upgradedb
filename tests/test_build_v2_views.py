import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_v2_views import build_views  # noqa: E402


class BuildV2ViewsTests(unittest.TestCase):
    @staticmethod
    def _candidate_rows() -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("AAA", "X", "processed", 1, 10.0, 2.0),
                ("AAA", "Y", "processed", 0, 0.0, 2.0),
                ("BBB", "X", "processed", 0, 0.0, 3.0),
            ],
            columns=[
                "i_iso",
                "j_iso",
                "stage",
                "y",
                "lateval",
                "log_exporter_capacity",
            ],
        ).assign(
            aggregation="calendar_mean",
            benchmark_version="2.1-dev",
            early_window="2008-2012",
            late_window="2013-2017",
        )

    def test_fold_suffix_creates_history_views_without_touching_target_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            # Deliberately write non-canonical order; the materializer must
            # normalize it so release hashes do not depend on set iteration.
            track_a = self._candidate_rows().iloc[::-1].reset_index(drop=True)
            track_b = self._candidate_rows().iloc[::-1].reset_index(drop=True)
            track_a.to_csv(data_root / "candidates_sheep_fold2.csv", index=False)
            track_b.to_csv(data_root / "candidates_firsttime_sheep_fold2.csv", index=False)
            sentinel = data_root / "candidates_sheep.csv"
            sentinel.write_text("target-sentinel\n", encoding="utf-8")

            payload = build_views(data_root, suffix="_fold2", chains=["sheep"])

            self.assertEqual(payload["temporal_role"], "history")
            self.assertEqual(payload["source_suffix"], "_fold2")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "target-sentinel\n")
            self.assertTrue((data_root / "dataset_summary_fold2.json").exists())
            self.assertTrue((data_root / "dataset_summary_fold2.csv").exists())
            self.assertFalse((data_root / "dataset_summary.json").exists())

            lanes = pd.read_csv(data_root / "candidates_firsttime_sheep_fold2.csv")
            entries = pd.read_csv(data_root / "entries_firsttime_sheep_fold2.csv")
            conditional = pd.read_csv(
                data_root / "destinations_given_entry_sheep_fold2.csv"
            )
            summary = json.loads(
                (data_root / "dataset_summary_fold2.json").read_text(encoding="utf-8")
            )

            self.assertEqual(set(lanes.temporal_role), {"history"})
            self.assertEqual(set(lanes.task), {"processed_export_entry_candidate_lane"})
            self.assertEqual(set(lanes.task_unit), {"exporter_stage_destination"})
            self.assertEqual(
                lanes[["i_iso", "j_iso", "stage"]].values.tolist(),
                lanes.sort_values(["i_iso", "j_iso", "stage"], kind="mergesort")
                [["i_iso", "j_iso", "stage"]]
                .values.tolist(),
            )
            self.assertEqual(set(entries.temporal_role), {"history"})
            self.assertEqual(set(entries.task_unit), {"exporter_stage"})
            self.assertEqual(set(conditional.temporal_role), {"history"})
            self.assertTrue(
                (lanes.groupby("group_id").transductive_split.nunique() == 1).all()
            )
            self.assertTrue((lanes.groupby("entry_id").entry_y.nunique() == 1).all())
            self.assertEqual(len(entries), lanes.entry_id.nunique())
            self.assertEqual(set(conditional.entry_y), {1})
            self.assertEqual(summary["chains"][0]["track_b_positive_entries"], 1)
            self.assertEqual(summary["chains"][0]["track_b2_conditional_lanes"], 2)
            self.assertEqual(summary["chains"][0]["track_b2_positive_lanes"], 1)

    def test_rejects_unsafe_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "--suffix"):
                build_views(Path(tmp), suffix="../fold2", chains=["sheep"])


if __name__ == "__main__":
    unittest.main()
