import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_registry_human_validation_sample as sampling  # noqa: E402


class RegistryHumanValidationSampleTests(unittest.TestCase):
    def test_canonical_artifact_is_the_deterministic_rebuild(self) -> None:
        observed = sampling.load_plan(sampling.OUTPUT_PATH, root=ROOT)
        rebuilt = sampling.build_plan(ROOT)
        self.assertEqual(
            sampling._canonical_json_bytes(observed),
            sampling._canonical_json_bytes(rebuilt),
        )
        self.assertEqual(observed["frame"]["decision_records"], 610)
        self.assertEqual(observed["frame"]["unique_hs6"], 588)
        self.assertEqual(observed["sample"]["decision_records"], 212)
        self.assertEqual(observed["sample"]["unique_hs6"], 212)
        self.assertEqual(observed["sample"]["certainty_records"], 74)
        self.assertEqual(observed["sample"]["probability_selected_records"], 138)
        self.assertEqual(len(observed["strata"]), 21)

    def test_every_declared_certainty_unit_is_selected(self) -> None:
        plan = sampling.build_plan(ROOT)
        selected = {row["record_id"]: row for row in plan["selected_records"]}
        frame = sampling._load_frame(ROOT)
        stratum_sizes = Counter(sampling._stratum_id(row) for row in frame)

        expected_certainty: dict[str, list[str]] = {}
        for row in frame:
            reasons = sampling._certainty_reasons(
                row, stratum_sizes[sampling._stratum_id(row)]
            )
            if reasons:
                expected_certainty[sampling._record_id(row)] = reasons

        self.assertEqual(len(expected_certainty), 74)
        self.assertLessEqual(set(expected_certainty), set(selected))
        for record_id, reasons in expected_certainty.items():
            with self.subTest(record_id=record_id):
                record = selected[record_id]
                self.assertTrue(record["certainty"])
                self.assertEqual(record["certainty_reasons"], reasons)
                self.assertEqual(record["inclusion_probability"], {"numerator": 1, "denominator": 1})
                self.assertEqual(record["analysis_weight"], {"numerator": 1, "denominator": 1})
                self.assertIsNone(record["selection_score_sha256"])
                self.assertIsNone(record["selection_rank_within_random_pool"])

        boundary_ids = {
            f"CODE-{chain_id}-{code}"
            for chain_id, code in sampling.BOUNDARY_IDENTITIES
        }
        self.assertLessEqual(boundary_ids, set(selected))

    def test_random_units_store_exact_inverse_probability_weights(self) -> None:
        plan = sampling.build_plan(ROOT)
        strata = {row["stratum_id"]: row for row in plan["strata"]}
        random_records = [
            row for row in plan["selected_records"] if not row["certainty"]
        ]
        self.assertEqual(len(random_records), 138)
        ranks_by_stratum: dict[str, set[int]] = {}
        for record in random_records:
            with self.subTest(record_id=record["record_id"]):
                probability = record["inclusion_probability"]
                weight = record["analysis_weight"]
                stratum = strata[record["stratum_id"]]
                self.assertEqual(
                    probability,
                    {
                        "numerator": stratum["random_sample_records"],
                        "denominator": stratum["random_pool_records"],
                    },
                )
                self.assertEqual(
                    weight,
                    {
                        "numerator": probability["denominator"],
                        "denominator": probability["numerator"],
                    },
                )
                self.assertGreaterEqual(record["selection_rank_within_random_pool"], 1)
                self.assertLessEqual(
                    record["selection_rank_within_random_pool"],
                    stratum["random_sample_records"],
                )
                self.assertRegex(record["selection_score_sha256"], r"^[0-9a-f]{64}$")
                ranks_by_stratum.setdefault(record["stratum_id"], set()).add(
                    record["selection_rank_within_random_pool"]
                )
        for stratum_id, ranks in ranks_by_stratum.items():
            self.assertEqual(
                ranks,
                set(range(1, strata[stratum_id]["random_sample_records"] + 1)),
            )

    def test_selected_record_or_seed_tampering_fails_rebuild_validation(self) -> None:
        plan = sampling.build_plan(ROOT)

        changed_record = copy.deepcopy(plan)
        changed_record["selected_records"][0]["decision"] = "tampered"
        with self.assertRaisesRegex(sampling.SamplePlanError, "deterministic rebuild"):
            sampling.validate_plan(changed_record, root=ROOT)

        changed_seed = copy.deepcopy(plan)
        changed_seed["design"]["seed_material_sha256"] = "0" * 64
        with self.assertRaisesRegex(sampling.SamplePlanError, "deterministic rebuild"):
            sampling.validate_plan(changed_seed, root=ROOT)

    def test_noncanonical_serialization_is_rejected(self) -> None:
        plan = sampling.build_plan(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "sample.json"
            temporary.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(sampling.SamplePlanError, "canonical strict JSON"):
                sampling.load_plan(temporary, root=ROOT)


if __name__ == "__main__":
    unittest.main()
