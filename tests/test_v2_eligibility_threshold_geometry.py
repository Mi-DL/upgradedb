from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v2_eligibility_threshold_geometry as geometry  # noqa: E402


FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"


class EligibilityThresholdGeometryTest(unittest.TestCase):
    def test_calendar_mean_divides_by_all_five_years(self) -> None:
        raw = pd.DataFrame(
            {
                "i_iso": ["AAA"],
                "j_iso": ["BBB"],
                "k": [1],
                "year": [2008],
                "v": [500.0],
            }
        )
        result = geometry._aggregate_chain(
            raw, [2008, 2009, 2010, 2011, 2012], {1: "s"}
        )
        self.assertEqual(result.loc[0, "v"], 100.0)

    def test_threshold_is_strict_and_b1_b2_are_nested(self) -> None:
        early = pd.DataFrame(
            {
                "i_iso": ["EXP", "INC"],
                "j_iso": ["XXX", "DST"],
                "stage": ["u", "s"],
                "v": [100.0, 101.0],
            }
        )
        late = pd.DataFrame(
            {
                "i_iso": ["EXP"],
                "j_iso": ["DST"],
                "stage": ["s"],
                "v": [101.0],
            }
        )
        below = geometry._enumerate_geometry(early, late, {"s": ("u",)}, 99.0)
        at = geometry._enumerate_geometry(early, late, {"s": ("u",)}, 100.0)
        self.assertEqual(below["b1"]["candidates"], {("EXP", "s")})
        self.assertEqual(below["b1"]["positives"], {("EXP", "s")})
        self.assertEqual(below["b2"]["candidates"], {("EXP", "DST", "s")})
        self.assertEqual(below["b2"]["positives"], {("EXP", "DST", "s")})
        self.assertFalse(at["b1"]["candidates"])
        self.assertFalse(at["b2"]["candidates"])

    def test_committed_100kusd_raw_gate_matches_released_geometry(self) -> None:
        payload = geometry.common._strict_json_load(geometry.DEFAULT_JSON)
        self.assertEqual(payload["canonical_100kusd_gate"]["status"], "PASS")
        expected = {
            "a": (317624, 12273),
            "b1": (1518, 270),
            "b2": (33433, 556),
        }
        for task, (candidates, positives) in expected.items():
            item = payload["thresholds"]["100"]["summary"][task]
            self.assertEqual(item["candidate_count"], candidates)
            self.assertEqual(item["positive_count"], positives)
        self.assertFalse(payload["protocol"]["model_scores_or_performance_computed"])

    @unittest.skipUnless(
        FULL_PAYLOAD_TESTS_ENABLED,
        "requires the externally mounted full data payload",
    )
    def test_committed_outputs_verify_without_raw_rebuild(self) -> None:
        geometry.verify_output()


if __name__ == "__main__":
    unittest.main()
