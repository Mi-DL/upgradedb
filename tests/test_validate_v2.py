import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from split import OFFICIAL_SPLIT_UNIT, split_labels  # noqa: E402
from validate_v2 import SNAPSHOTS, V2ValidationError, validate_release  # noqa: E402


class ValidateV2Tests(unittest.TestCase):
    @staticmethod
    def _lanes(snapshot, *, track_b: bool) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                ("AAA", "X", "processed", 1, 150.0, 2.0, 3.0),
                ("AAA", "Y", "processed", 0, 0.0, 2.0, 4.0),
                ("BBB", "X", "processed", 0, 0.0, 3.0, 3.0),
            ],
            columns=[
                "i_iso",
                "j_iso",
                "stage",
                "y",
                "lateval",
                "log_exporter_capacity",
                "log_importer_demand",
            ],
        )
        frame["size"] = frame["log_exporter_capacity"] + frame["log_importer_demand"]
        frame["size_basis"] = (
            "registered_upstream_exporter_plus_processed_importer"
            if track_b
            else "processed_exporter_plus_processed_importer"
        )
        frame["grav"] = 0.1
        # Shipped dev tables may be unscored (NaN), while trained GNN columns
        # contain signed logits; both are valid release states.
        frame["gnn"] = [-1.0, 0.5, np.nan]
        frame["benchmark_version"] = "2.1-dev"
        frame["aggregation"] = "calendar_mean"
        frame["early_window"] = snapshot.early_window
        frame["late_window"] = snapshot.late_window
        frame["group_id"] = frame["i_iso"] + "|" + frame["stage"]
        frame["transductive_split_unit"] = OFFICIAL_SPLIT_UNIT
        frame["transductive_split"] = split_labels(
            "sheep",
            frame["i_iso"],
            frame["stage"],
            frame["j_iso"],
            unit=OFFICIAL_SPLIT_UNIT,
        )
        frame["temporal_role"] = snapshot.role
        frame["task"] = (
            "processed_export_entry_candidate_lane" if track_b else "destination_extension"
        )
        frame["task_unit"] = "exporter_stage_destination"
        if track_b:
            frame["entry_id"] = frame["group_id"]
            frame["entry_y"] = frame.groupby(["i_iso", "stage"])["y"].transform("max")
        return frame

    @classmethod
    def _write_snapshot(cls, root: Path, snapshot) -> None:
        track_a = cls._lanes(snapshot, track_b=False)
        track_b = cls._lanes(snapshot, track_b=True)
        track_a.to_csv(root / f"candidates_sheep{snapshot.suffix}.csv", index=False)
        track_b.to_csv(root / f"candidates_firsttime_sheep{snapshot.suffix}.csv", index=False)

        entries = (
            track_b.groupby(["i_iso", "stage"], as_index=False, sort=True)
            .agg(
                z=("y", "max"),
                size=("log_exporter_capacity", "first"),
                log_upstream_capacity=("log_exporter_capacity", "first"),
                entry_lateval=("lateval", "sum"),
                n_candidate_destinations=("j_iso", "size"),
                n_materialized_destinations=("y", "sum"),
                benchmark_version=("benchmark_version", "first"),
                aggregation=("aggregation", "first"),
                early_window=("early_window", "first"),
                late_window=("late_window", "first"),
                transductive_split=("transductive_split", "first"),
            )
        )
        entries["entry_id"] = entries["i_iso"] + "|" + entries["stage"]
        entries["transductive_split_unit"] = OFFICIAL_SPLIT_UNIT
        entries["temporal_role"] = snapshot.role
        entries["task"] = "processed_export_stage_entry"
        entries["task_unit"] = "exporter_stage"
        entries.to_csv(root / f"entries_firsttime_sheep{snapshot.suffix}.csv", index=False)

        conditional = track_b.loc[track_b["entry_y"] == 1].copy()
        conditional["task"] = "conditional_destination_given_entry"
        conditional["task_unit"] = "exporter_stage_destination"
        conditional.to_csv(
            root / f"destinations_given_entry_sheep{snapshot.suffix}.csv", index=False
        )

    @classmethod
    def _write_release(cls, root: Path) -> None:
        for snapshot in SNAPSHOTS:
            cls._write_snapshot(root, snapshot)

    def test_accepts_consistent_main_and_historical_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_release(root)
            report = validate_release(root, chains=("sheep",), check_summaries=False)
            self.assertEqual(set(report), {"main", "fold2"})
            self.assertEqual(report["main"][0]["track_b_unique_entries"], 2)

    def test_rejects_group_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_release(root)
            path = root / "candidates_firsttime_sheep_fold2.csv"
            lanes = pd.read_csv(path)
            lanes.loc[0, "transductive_split"] = (
                "test" if lanes.loc[0, "transductive_split"] == "train" else "train"
            )
            lanes.to_csv(path, index=False)
            with self.assertRaisesRegex(V2ValidationError, "groups cross splits"):
                validate_release(root, chains=("sheep",), check_summaries=False)

    def test_rejects_stale_entry_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_release(root)
            path = root / "entries_firsttime_sheep.csv"
            entries = pd.read_csv(path)
            entries.loc[entries["z"] == 1, "entry_lateval"] += 1.0
            entries.to_csv(path, index=False)
            with self.assertRaisesRegex(V2ValidationError, "entry aggregation mismatch"):
                validate_release(root, chains=("sheep",), check_summaries=False)


if __name__ == "__main__":
    unittest.main()
