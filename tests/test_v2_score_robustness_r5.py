import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_score_robustness_r5 as r5  # noqa: E402


def _config() -> dict:
    payload = json.loads((ROOT / "configs" / "v2_score_robustness_r5.json").read_text())
    payload["chains"] = ["c0", "c1", "c2", "c3", "c4", "c5"]
    payload["bootstrap"]["iterations"] = 100
    return payload


def _candidate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i_iso": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "j_iso": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "stage": ["s"] * 8,
            "y": [1, 0, 0, 0, 1, 0, 0, 1],
            "lateval": [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        }
    )


class WeightedAveragePrecisionTests(unittest.TestCase):
    def test_matches_sklearn_with_ties_and_weights(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        score = np.array([0.9, 0.9, 0.4, 0.2, 0.2, 0.1])
        weight = np.array([2.0, 2.0, 0.0, 3.0, 1.0, 3.0])
        self.assertAlmostEqual(
            r5.weighted_average_precision(y, score, weight),
            average_precision_score(y, score, sample_weight=weight),
            places=14,
        )

    def test_single_class_bootstrap_is_nonfinite(self) -> None:
        self.assertTrue(
            np.isnan(r5.weighted_average_precision([1, 1], [0.2, 0.1], [1.0, 1.0]))
        )

    def test_presorted_plan_matches_direct_weighted_ap(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        score = np.array([0.9, 0.9, 0.4, 0.2, 0.2, 0.1])
        weight = np.array([2.0, 2.0, 0.0, 3.0, 1.0, 3.0])
        self.assertAlmostEqual(
            r5.WeightedAPPlan(y, score).evaluate(weight),
            r5.weighted_average_precision(y, score, weight),
            places=14,
        )


class PoolingTests(unittest.TestCase):
    def test_midrank_ecdf_is_affine_invariant_and_tie_aware(self) -> None:
        score = np.array([1.0, 1.0, 4.0, 8.0])
        first = r5.midrank_ecdf(score)
        second = r5.midrank_ecdf(7.0 * score + 13.0)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first[0], first[1])
        self.assertAlmostEqual(first[0], 0.25)

    def test_official_and_normalized_variants_keep_identical_entry_universe(self) -> None:
        candidate = _candidate()
        score = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.7, 0.6])
        config = _config()
        entries = [
            r5._build_b1_entries(candidate, score, method)
            for method in config["b1_pooling_methods"]
        ]
        for entry in entries[1:]:
            self.assertTrue(
                entry[["i_iso", "stage", "y", "lateval", "lane_count"]].equals(
                    entries[0][["i_iso", "stage", "y", "lateval", "lane_count"]]
                )
            )
        self.assertEqual(entries[0]["score"].tolist(), [0.9, 0.4, 0.8, 0.7])
        expected_mean_a = r5.midrank_ecdf(score)[:2].mean()
        self.assertAlmostEqual(entries[1].loc[0, "score"], expected_mean_a)
        self.assertNotAlmostEqual(entries[1].loc[0, "score"], r5.midrank_ecdf(score)[:2].sum())

    def test_ecdf_pooling_entry_ranking_is_affine_invariant(self) -> None:
        candidate = _candidate()
        score = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.7, 0.6])
        method = _config()["b1_pooling_methods"][1]
        first = r5._build_b1_entries(candidate, score, method)
        second = r5._build_b1_entries(candidate, 9.0 * score - 4.0, method)
        np.testing.assert_allclose(first["score"], second["score"])


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_a_scores_have_zero_paired_interval(self) -> None:
        candidate = _candidate()
        score = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.7, 0.6])
        scores = {seed: score + seed * 0.001 for seed in r5.EXPECTED_SEEDS}
        record, draws = r5._paired_chain_task(
            "c0",
            "a",
            candidate,
            scores,
            scores,
            {"kge": "M", "nbfnet": "N"},
            _config(),
        )
        self.assertEqual(record["cluster_unit"], "exporter")
        self.assertEqual(record["cluster_count"], 4)
        np.testing.assert_allclose(draws, 0.0)
        for seed in record["per_seed"]:
            self.assertEqual(seed["delta"], 0.0)
            self.assertEqual(seed["lower_95"], 0.0)
            self.assertEqual(seed["upper_95"], 0.0)

    def test_b2_resamples_complete_positive_entries_only(self) -> None:
        candidate = _candidate()
        score = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.6, 0.7])
        scores = {seed: score for seed in r5.EXPECTED_SEEDS}
        record, _ = r5._paired_chain_task(
            "c0",
            "b2",
            candidate,
            scores,
            scores,
            {"kge": "M", "nbfnet": "N"},
            _config(),
        )
        self.assertEqual(record["cluster_unit"], "exporter_stage")
        self.assertEqual(record["cluster_count"], 3)
        self.assertTrue(record["same_cluster_draws_for_both_families"])


class TwoStageTests(unittest.TestCase):
    def test_missed_positive_entry_contributes_zero(self) -> None:
        candidate = _candidate()
        # B1 selects A only. A and C/D are positive entries, so two positive
        # entries are missed and contribute zero to the macro denominator.
        b1 = np.array([0.99, 0.98, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5])
        b2 = np.array([0.9, 0.1, 0.4, 0.3, 0.8, 0.2, 0.1, 0.9])
        metrics = r5._two_stage_condition(
            candidate,
            b1,
            b2,
            b1_budget=1,
            b2_budget=1,
            official_method=_config()["b1_pooling_methods"][0],
        )
        self.assertEqual(metrics["positive_entries_denominator"], 3.0)
        self.assertAlmostEqual(metrics["positive_entry_gate_recall"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["e2e_macro_destination_recall"], 1.0 / 3.0)
        self.assertEqual(metrics["selected_positive_entries"], 1.0)


class ProtocolTests(unittest.TestCase):
    def test_threshold_geometry_cannot_be_silently_relabelled(self) -> None:
        config = _config()
        config["eligibility_threshold_geometry"]["status"] = "outcome_only_relabel"
        with self.assertRaisesRegex(r5.R5Error, "must not approximate"):
            r5.validate_config(config)

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
            with self.assertRaisesRegex(r5.R5Error, "duplicate JSON key"):
                r5.load_json(path)

    def test_receipt_book_rejects_hash_mismatch(self) -> None:
        path = ROOT / "configs" / "v2_score_robustness_r5.json"
        with self.assertRaisesRegex(r5.R5Error, "input hash mismatch"):
            r5.ReceiptBook().add(path, "test", "0" * 64)

    @unittest.skipUnless(
        os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1",
        "full governed-score recomputation is opt-in",
    )
    def test_full_r5_output_recomputes(self) -> None:
        r5.verify(ROOT / "configs" / "v2_score_robustness_r5.json")


if __name__ == "__main__":
    unittest.main()
