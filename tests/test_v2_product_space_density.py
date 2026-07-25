from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_product_space_density as ps  # noqa: E402


FULL_PAYLOAD_TESTS_ENABLED = os.environ.get("UPGRADE_BENCH_FULL_PAYLOAD_TESTS") == "1"


def _toy_matrix() -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    # Every product has ubiquity two.  For target p, proximity to q and r is
    # 0.5.  A has q+r, while B/C have p plus one neighbor.
    matrix = np.asarray(
        [
            [0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
        ]
    )
    return matrix, ("A", "B", "C"), ("p", "q", "r")


class ProductSpaceFormulaTest(unittest.TestCase):
    def test_frozen_config_excludes_target_product_diagonal(self) -> None:
        config = ps.load_frozen_config()
        self.assertEqual(
            config["formula"]["prospective_target_self_relation"],
            "exclude q=p from numerator and denominator",
        )
        self.assertEqual(config["selection"]["candidate_formulas"], 1)
        self.assertFalse(config["selection"]["historical_labels_used_for_selection"])

        tampered = copy.deepcopy(config)
        tampered["formula"]["prospective_target_self_relation"] = "include q=p"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ps.ProductSpaceProtocolError, "formula"):
                ps.load_frozen_config(path)

    def test_rca_proximity_and_prospective_density_remove_self_term(self) -> None:
        matrix, countries, products = _toy_matrix()
        scorer = ps._compute_product_space(
            matrix, countries, products, ("p",), cohort="synthetic"
        )
        np.testing.assert_array_equal(
            scorer.membership,
            np.asarray(
                [
                    [False, True, True],
                    [True, True, False],
                    [True, False, True],
                ]
            ),
        )
        # Excluding phi_pp changes B/C from 0.75 (self included) to 0.5.
        np.testing.assert_allclose(scorer.target_density[:, 0], [1.0, 0.5, 0.5])
        self.assertEqual(
            scorer.matrix_audit["target_diagonal_nonzero_before_exclusion"], 1
        )
        self.assertEqual(scorer.matrix_audit["target_diagonal_max_after_exclusion"], 0.0)

    def test_zero_denominator_target_scores_zero(self) -> None:
        scorer = ps._compute_product_space(
            np.asarray([[10.0], [10.0]]),
            ("A", "B"),
            ("p",),
            ("p",),
            cohort="synthetic",
        )
        np.testing.assert_array_equal(scorer.target_density[:, 0], [0.0, 0.0])
        self.assertEqual(scorer.matrix_audit["zero_density_denominator_targets"], 1)

    def test_missing_exporter_scores_zero_and_target_membership_is_audited(self) -> None:
        matrix, countries, products = _toy_matrix()
        scorer = ps._compute_product_space(
            matrix, countries, products, ("p",), cohort="synthetic"
        )
        frame = pd.DataFrame(
            {
                "i_iso": ["A", "B", "ZZZ"],
                "stage": ["s", "s", "s"],
                "entry_id": ["A|s", "B|s", "ZZZ|s"],
            }
        )
        scores, audit = ps._score_candidates(frame, scorer, {"s": ("p",)})
        np.testing.assert_allclose(scores, [1.0, 0.5, 0.0])
        self.assertEqual(audit["candidate_target_hs6_rca_memberships"], 1)
        self.assertEqual(audit["candidates_with_any_target_hs6_rca_membership"], 1)
        self.assertTrue(audit["self_diagonal_exclusion_material"])
        self.assertAlmostEqual(audit["exporter_dictionary_coverage"], 2 / 3)

    def test_budget_ties_follow_canonical_exporter_stage_key_order(self) -> None:
        frame = pd.DataFrame(
            {
                "i_iso": ["B", "A"],
                "stage": ["s", "s"],
                "entry_id": ["B|s", "A|s"],
                "z": [0, 1],
                "entry_lateval": [0.0, 10.0],
                "score": [0.5, 0.5],
            }
        ).sort_values(["i_iso", "stage", "entry_id"], kind="mergesort")
        metrics = ps.cpu._classification_metrics(
            frame,
            label="z",
            score=frame["score"].to_numpy(float),
            cluster=frame["i_iso"].to_numpy(str),
            cluster_unit="exporter",
            budgets=(1,),
            bootstrap_draws=0,
            seed=0,
        )
        self.assertEqual(metrics["budgets"]["k_1"]["hits"], 1)
        self.assertEqual(metrics["budgets"]["k_1"]["value_capture"], 1.0)


class ProductUniverseAndRegistryTest(unittest.TestCase):
    def test_matrix_uses_dictionary_universe_and_retains_zero_trade_product(self) -> None:
        config = ps.load_frozen_config()
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "toy.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    config["source"]["product_dictionary_member"],
                    "code,description\n000001,one\n000002,two\n000003,unused\n",
                )
                archive.writestr(
                    config["source"]["country_dictionary_member"],
                    "country_code,country_name,country_iso2,country_iso3\n"
                    "1,A,AA,AAA\n2,B,BB,BBB\n",
                )
                archive.writestr(
                    config["source"]["annual_member_template"].format(year=2000),
                    "t,i,j,k,v,q\n2000,1,2,1,10,1\n2000,2,1,2,20,1\n",
                )
            matrix, countries, products, records = ps._build_export_matrix(
                archive_path,
                config,
                (2000,),
                chunksize=1,
                require_formal_universe=False,
            )
        self.assertEqual(countries, ("AAA", "BBB"))
        self.assertEqual(products, ("000001", "000002", "000003"))
        self.assertEqual(matrix.shape, (2, 3))
        np.testing.assert_array_equal(matrix[:, 2], [0.0, 0.0])
        self.assertEqual(records[0]["rows_read"], 2)

    def test_registry_target_mapping_is_exactly_upstream_map_keys(self) -> None:
        mappings, hashes = ps._load_stage_registry()
        self.assertEqual(set(mappings), set(ps.CHAINS))
        for chain in ps.CHAINS:
            raw = json.loads((ROOT / "chains" / f"{chain}.json").read_text())
            self.assertEqual(set(mappings[chain]), set(raw["upstream_map"]))
            self.assertRegex(hashes[f"chains/{chain}.json"], r"^[0-9a-f]{64}$")

    @unittest.skipUnless(
        FULL_PAYLOAD_TESTS_ENABLED,
        "requires the externally mounted full data payload",
    )
    def test_canonical_main_b1_inventory_is_1518_unique_keys_and_270_entries(self) -> None:
        keys: set[tuple[str, str, str]] = set()
        rows = positives = 0
        for chain in ps.CHAINS:
            frame, _ = ps._read_entry_table(ps.DEFAULT_DATA, chain, "main")
            rows += len(frame)
            positives += int(frame["z"].sum())
            chain_keys = {(chain, str(row.i_iso), str(row.stage)) for row in frame.itertuples()}
            self.assertEqual(len(chain_keys), len(frame))
            self.assertFalse(keys.intersection(chain_keys))
            keys.update(chain_keys)
            observed_order = list(
                frame[["i_iso", "stage", "entry_id"]].itertuples(index=False, name=None)
            )
            self.assertEqual(observed_order, sorted(observed_order))
        self.assertEqual(rows, ps.EXPECTED_MAIN_ROWS)
        self.assertEqual(positives, ps.EXPECTED_MAIN_POSITIVES)
        self.assertEqual(len(keys), ps.EXPECTED_MAIN_ROWS)


class ProductSpaceReadGateTest(unittest.TestCase):
    def test_both_early_scorers_freeze_before_first_outcome_read(self) -> None:
        config = ps.load_frozen_config()
        matrix, countries, products = _toy_matrix()
        events: list[str] = []

        def builder(
            _archive: Path, _config: object, years: tuple[int, ...]
        ) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], list[dict[str, object]]]:
            events.append(f"build:{years[0]}")
            return matrix, countries, products, []

        stage_registry = {chain: {"s": ("p",)} for chain in ps.CHAINS}
        registry_hashes = {f"chains/{chain}.json": "0" * 64 for chain in ps.CHAINS}
        frozen = ps._freeze_all_scorers(
            Path("synthetic.zip"),
            config,
            stage_registry,
            registry_hashes,
            matrix_builder=builder,
        )

        def reader(
            _data: Path, chain: str, cohort: str
        ) -> tuple[pd.DataFrame, dict[str, object]]:
            events.append(f"read:{cohort}:{chain}")
            frame = pd.DataFrame(
                {
                    "i_iso": ["A", "B"],
                    "stage": ["s", "s"],
                    "entry_id": ["A|s", "B|s"],
                    "z": [1, 0],
                    "entry_lateval": [10.0, 0.0],
                }
            )
            return frame, {
                "path": f"synthetic/{cohort}/{chain}.csv",
                "sha256": "0" * 64,
                "rows": 2,
                "positives": 1,
                "exporters": 2,
                "stages": 1,
                "early_window": "synthetic",
                "late_window": "synthetic",
            }

        with (
            patch.object(ps, "EXPECTED_MAIN_ROWS", 12),
            patch.object(ps, "EXPECTED_MAIN_POSITIVES", 6),
        ):
            results, score_rows = ps._evaluate_after_freeze(
                frozen, Path("synthetic"), config, reader=reader
            )

        self.assertEqual(events[:2], ["build:1998", "build:2008"])
        self.assertTrue(all(event.startswith("read:") for event in events[2:]))
        self.assertTrue(frozen.sealed)
        self.assertEqual(len(score_rows), 24)
        self.assertEqual(set(results), set(ps.COHORTS))

    def test_unsealed_or_incomplete_scorer_cannot_open_outcomes(self) -> None:
        config = ps.load_frozen_config()
        frozen = ps.FrozenProtocol({}, {}, {}, "0" * 64, False)
        with self.assertRaisesRegex(ps.ProductSpaceProtocolError, "cannot open"):
            ps._evaluate_after_freeze(frozen, Path("synthetic"), config)

    def test_privacy_guard_rejects_absolute_or_remote_paths(self) -> None:
        unix_private = "/" + "home/private/raw.zip"
        for value in (r"C:\\private\\raw.zip", unix_private, "host:/private"):
            with self.assertRaises(ps.ProductSpaceProtocolError):
                ps._assert_privacy({"value": value})


class ProductSpaceArtifactVerificationTest(unittest.TestCase):
    @unittest.skipUnless(
        FULL_PAYLOAD_TESTS_ENABLED,
        "requires the externally mounted full data payload",
    )
    def test_public_verifier_recomputes_metrics_without_touching_raw_archive(self) -> None:
        original = ps._verify_source

        def guard(path: Path, digest: str, where: str) -> None:
            if path.resolve() == ps.DEFAULT_ARCHIVE.resolve():
                raise AssertionError("public verification unexpectedly touched raw BACI")
            original(path, digest, where)

        with patch.object(ps, "_verify_source", side_effect=guard):
            ps.verify_existing_output(
                ps.DEFAULT_JSON,
                ps.DEFAULT_CSV,
                ps.DEFAULT_SCORES,
                verify_raw_archive=False,
            )

    def test_keyed_score_tamper_is_rejected_before_metric_recomputation(self) -> None:
        payload = ps._strict_json_load(ps.DEFAULT_JSON)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            raw = ps.DEFAULT_SCORES.read_text(encoding="utf-8")
            path.write_text(raw + "#tamper\n", encoding="utf-8")
            with self.assertRaisesRegex(ps.ProductSpaceProtocolError, "SHA-256"):
                ps._verify_keyed_scores(payload, path)


if __name__ == "__main__":
    unittest.main()
