import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2_ultra import (  # noqa: E402
    CLAIM_STATUS,
    FORBIDDEN_MAIN_COLUMNS,
    UltraProtocolError,
    align_scores_exact,
    build_parser,
    install_torch_scatter_compat,
    read_candidate_identities,
    validate_graph_coverage,
)


def candidate_frame():
    return pd.DataFrame(
        {
            "i_iso": ["AAA", "AAA", "BBB"],
            "j_iso": ["BBB", "CCC", "AAA"],
            "stage": ["exp_x", "exp_x", "exp_x"],
        }
    )


def write_candidate(path: Path, *, aggregation="calendar_mean") -> None:
    frame = candidate_frame()
    frame["y"] = ["MUST_NOT_PARSE", "MUST_NOT_PARSE", "MUST_NOT_PARSE"]
    frame["lateval"] = ["PRIVATE_LABEL", "PRIVATE_LABEL", "PRIVATE_LABEL"]
    frame["size"] = ["LABEL_DERIVED", "LABEL_DERIVED", "LABEL_DERIVED"]
    frame["benchmark_version"] = "2.1-dev"
    frame["aggregation"] = aggregation
    frame["early_window"] = "2008-2012"
    frame["late_window"] = "2018-2022"
    frame["task"] = "destination_extension"
    frame["task_unit"] = "exporter_stage_destination"
    frame.to_csv(path, index=False)


class V2UltraProtocolTest(unittest.TestCase):
    def test_identity_reader_never_requests_main_labels_or_label_derived_features(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidates.csv"
            write_candidate(path)
            calls = []
            real_read_csv = pd.read_csv

            def audited_read_csv(*args, **kwargs):
                calls.append(tuple(kwargs.get("usecols") or ()))
                return real_read_csv(*args, **kwargs)

            with mock.patch("pandas.read_csv", side_effect=audited_read_csv):
                identities, metadata = read_candidate_identities(path)
            self.assertEqual(list(identities.columns), ["i_iso", "j_iso", "stage"])
            self.assertEqual(metadata["task"], "destination_extension")
            self.assertTrue(calls)
            for requested in calls:
                self.assertTrue(set(requested).isdisjoint(FORBIDDEN_MAIN_COLUMNS))

    def test_identity_reader_rejects_protocol_metadata_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidates.csv"
            write_candidate(path, aggregation="legacy_present_hs6_mean")
            with self.assertRaisesRegex(UltraProtocolError, "aggregation mismatch"):
                read_candidate_identities(path)

    def test_exact_alignment_preserves_candidate_order(self):
        candidates = candidate_frame()
        scored = candidates.iloc[[2, 0, 1]].copy()
        scored["ultra_score"] = [0.3, 0.1, 0.2]
        aligned = align_scores_exact(candidates, scored)
        pd.testing.assert_frame_equal(aligned.loc[:, list(candidates.columns)], candidates)
        np.testing.assert_allclose(aligned["ultra_score"], [0.1, 0.2, 0.3])

    def test_exact_alignment_rejects_missing_extra_duplicate_and_nonfinite_scores(self):
        candidates = candidate_frame()
        complete = candidates.copy()
        complete["ultra_score"] = [0.1, 0.2, 0.3]
        with self.assertRaisesRegex(UltraProtocolError, "missing=1"):
            align_scores_exact(candidates, complete.iloc[:2])
        extra = pd.concat(
            [
                complete,
                pd.DataFrame(
                    [{"i_iso": "ZZZ", "j_iso": "AAA", "stage": "exp_x", "ultra_score": 0.4}]
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(UltraProtocolError, "extra=1"):
            align_scores_exact(candidates, extra)
        duplicate = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(UltraProtocolError, "duplicated"):
            align_scores_exact(candidates, duplicate)
        nonfinite = complete.copy()
        nonfinite.loc[1, "ultra_score"] = np.nan
        with self.assertRaisesRegex(UltraProtocolError, "non-finite"):
            align_scores_exact(candidates, nonfinite)

    def test_graph_coverage_requires_every_candidate_vocabulary_item(self):
        candidates = candidate_frame()
        triples = np.array(
            [["AAA", "exp_x", "BBB"], ["BBB", "exp_x", "CCC"]], dtype=str
        )
        coverage = validate_graph_coverage(candidates, triples)
        self.assertEqual(coverage["vocabulary_coverage"], 1.0)
        bad = candidates.copy()
        bad.loc[0, "j_iso"] = "MISSING"
        with self.assertRaisesRegex(UltraProtocolError, "outside the early graph"):
            validate_graph_coverage(bad, triples)

    def test_scatter_compat_has_numeric_sum_semantics(self):
        import torch

        backend = install_torch_scatter_compat()
        from torch_scatter import scatter, scatter_add

        source = torch.tensor([1.0, 2.0, 4.0, 8.0])
        index = torch.tensor([0, 1, 0, 1])
        expected = torch.tensor([5.0, 10.0])
        torch.testing.assert_close(scatter_add(source, index, dim=0), expected)
        torch.testing.assert_close(scatter(source, index, dim=0, reduce="sum"), expected)
        self.assertIn(backend, {"native_torch_scatter", "torch_geometric_utils_scatter_compat"})

    def test_cli_exposes_one_fixed_checkpoint_and_no_selection_mode(self):
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("checkpoint", destinations)
        self.assertNotIn("ckpts", destinations)
        self.assertNotIn("selection", destinations)
        self.assertEqual(CLAIM_STATUS, "FEASIBILITY_ONLY_NOT_A_PAPER_RESULT")


if __name__ == "__main__":
    unittest.main()
