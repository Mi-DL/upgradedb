import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task_features import build_size_lookups, candidate_size_components  # noqa: E402


class TaskFeatureTests(unittest.TestCase):
    def setUp(self):
        self.early = pd.DataFrame(
            [
                ("AAA", "X", "raw", 200.0),
                ("AAA", "Y", "raw", 300.0),
                ("BBB", "X", "processed", 400.0),
                ("BBB", "Y", "processed", 100.0),
            ],
            columns=["i_iso", "j_iso", "stage", "v"],
        )
        self.processed_out, self.processed_in, self.upstream_out = build_size_lookups(
            self.early, ["processed"], {"processed": ["raw"]}
        )

    def test_first_time_size_uses_upstream_capacity(self):
        exporter, importer, size = candidate_size_components(
            "AAA",
            "X",
            "processed",
            first_time=True,
            processed_out=self.processed_out,
            processed_in=self.processed_in,
            upstream_out=self.upstream_out,
        )
        self.assertAlmostEqual(exporter, np.log1p(500.0))
        self.assertAlmostEqual(importer, np.log1p(400.0))
        self.assertAlmostEqual(size, exporter + importer)
        self.assertGreater(exporter, 0.0)

    def test_destination_extension_size_uses_processed_capacity(self):
        exporter, importer, size = candidate_size_components(
            "BBB",
            "X",
            "processed",
            first_time=False,
            processed_out=self.processed_out,
            processed_in=self.processed_in,
            upstream_out=self.upstream_out,
        )
        self.assertAlmostEqual(exporter, np.log1p(500.0))
        self.assertAlmostEqual(importer, np.log1p(400.0))
        self.assertAlmostEqual(size, exporter + importer)


if __name__ == "__main__":
    unittest.main()
