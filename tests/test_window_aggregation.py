import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from window_aggregation import (  # noqa: E402
    LEGACY_PRESENT_HS6_MEAN,
    aggregate_trade_window,
    stage_year_totals,
)


KEYS = ["i_iso", "j_iso", "stage"]
YEARS = [2018, 2019, 2020, 2021, 2022]


def rows(*values):
    return pd.DataFrame(
        values,
        columns=["i_iso", "j_iso", "stage", "k", "year", "v"],
    )


class WindowAggregationTests(unittest.TestCase):
    def test_intermittent_hs6_is_divided_by_full_calendar_window(self):
        raw = rows(("AAA", "BBB", "processed", "100001", 2018, 500.0))

        canonical = aggregate_trade_window(raw, YEARS, KEYS)
        legacy = aggregate_trade_window(
            raw, YEARS, KEYS, mode=LEGACY_PRESENT_HS6_MEAN
        )
        panel = stage_year_totals(raw, YEARS, KEYS)

        self.assertEqual(panel.v.tolist(), [500.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(canonical.v.iloc[0], 100.0)
        self.assertAlmostEqual(legacy.v.iloc[0], 500.0)

    def test_multiple_hs6_in_different_years_do_not_stack_present_year_means(self):
        raw = rows(
            ("AAA", "BBB", "processed", "100001", 2018, 200.0),
            ("AAA", "BBB", "processed", "100002", 2019, 300.0),
        )

        canonical = aggregate_trade_window(raw, YEARS, KEYS)
        legacy = aggregate_trade_window(
            raw, YEARS, KEYS, mode=LEGACY_PRESENT_HS6_MEAN
        )
        panel = stage_year_totals(raw, YEARS, KEYS)

        self.assertEqual(panel.v.tolist(), [200.0, 300.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(canonical.v.iloc[0], 100.0)
        self.assertAlmostEqual(legacy.v.iloc[0], 500.0)

    def test_all_missing_expected_key_materializes_as_zero(self):
        raw = pd.DataFrame(
            columns=["i_iso", "j_iso", "stage", "k", "year", "v"]
        )
        expected = pd.DataFrame(
            [("AAA", "BBB", "processed")], columns=KEYS
        )

        canonical = aggregate_trade_window(
            raw, YEARS, KEYS, expected_keys=expected
        )
        panel = stage_year_totals(raw, YEARS, KEYS, expected_keys=expected)

        self.assertEqual(len(panel), len(YEARS))
        self.assertEqual(panel.v.tolist(), [0.0] * len(YEARS))
        self.assertAlmostEqual(canonical.v.iloc[0], 0.0)

    def test_stage_year_sums_all_hs6_before_window_average(self):
        raw = rows(
            ("AAA", "BBB", "processed", "100001", 2018, 100.0),
            ("AAA", "BBB", "processed", "100002", 2018, 150.0),
            ("AAA", "BBB", "processed", "100001", 2018, 50.0),
        )

        panel = stage_year_totals(raw, YEARS, KEYS)
        canonical = aggregate_trade_window(raw, YEARS, KEYS)

        self.assertEqual(panel.v.tolist(), [300.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(canonical.v.iloc[0], 60.0)


if __name__ == "__main__":
    unittest.main()
