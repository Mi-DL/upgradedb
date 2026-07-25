import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v2_robustness as robustness  # noqa: E402


def _historical_selection(track: str, *, selected_c: float = 0.01) -> dict:
    spec = robustness.ROLLING_CHOICE_SPECS[track]
    scores = {float(value): 1.0 - index * 0.1 for index, value in enumerate(robustness.cpu.C_GRID)}
    candidates = [
        {
            "C": float(value),
            "fold_objective_values": [scores[float(value)]] * 5,
            "fold_objective_units": [2 if track == "B2" else 1] * 5,
            "mean_objective": scores[float(value)],
            "std_objective": 0.0,
        }
        for value in robustness.cpu.C_GRID
    ]
    return {
        "feature_names": list(spec["feature_names"]),
        "objective": spec["objective"],
        "objective_definition": spec["objective_definition"],
        "objective_aggregation": spec["objective_aggregation"],
        "group_unit": spec["group_unit"],
        "ranking_tie_break": spec["ranking_tie_break"],
        "train_validation_group_overlap_checked": True,
        "hyperparameter_tie_break": "maximize_mean_objective_then_smaller_C",
        "c_grid": list(robustness.cpu.C_GRID),
        "n_splits": 5,
        "candidates": candidates,
        "selected_C": selected_c,
        "selected_mean_objective": scores[selected_c],
        "refit_rows": 100,
        "refit_positives": 10,
    }


def _rolling_payload() -> dict:
    chains = {}
    for chain in robustness.CHAINS:
        paths = {
            "history_track_a": f"data/processed_v2/candidates_{chain}_fold2.csv",
            "history_track_b": (
                f"data/processed_v2/candidates_firsttime_{chain}_fold2.csv"
            ),
            "target_track_a": f"data/processed_v2/candidates_{chain}.csv",
            "target_track_b": f"data/processed_v2/candidates_firsttime_{chain}.csv",
        }
        chain_payload = {
            "protocol_audit": {
                role: {"path": paths[role], "sha256": "a" * 64}
                for role in robustness.ROLLING_INPUT_ROLES
            },
        }
        chain_payload["protocol_audit"].update(
            {
                "target_loaded_after_all_models_frozen": True,
                "target_labels_used_for_training_selection_imputation_or_calibration": False,
                "transductive_split_used": False,
            }
        )
        for track, spec in robustness.ROLLING_CHOICE_SPECS.items():
            models = {model: {} for model in spec["model_keys"]}
            models[spec["model_key"]] = {
                "model": {"selection": _historical_selection(track)}
            }
            chain_payload[spec["track_key"]] = {
                "history_rows": 100,
                "history_positives": 10,
                "models": models,
            }
        chains[chain] = chain_payload
    return {
        "protocol": {
            "selection_window": "1998-2002 -> 2008-2012",
            "frozen_target_window": "2008-2012 -> 2018-2022",
            "selection_source": "fold2 only",
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_imputation_scaling_or_calibration": False,
            "transductive_split_used": False,
            "main_target_models_compared_without_post_hoc_champion_selection": True,
            "selection_objectives": {
                "track_a": "historical_exporter_group_cv_average_precision",
                "track_b1": "historical_exporter_group_cv_average_precision",
                "track_b2": (
                    "historical_exporter_stage_entry_group_cv_per_positive_entry_"
                    "macro_recall_at_3"
                ),
            },
        },
        "chains": chains,
    }


class RollingChoiceFreezeTests(unittest.TestCase):
    def _write(self, directory: str, payload: object) -> Path:
        path = Path(directory) / "rolling.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_verified_artifact_is_the_only_choice_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, _rolling_payload())
            expected_hash = robustness.sha256_file(path)
            with mock.patch.object(robustness.cpu, "verify_existing_output") as verify:
                frozen = robustness._freeze_choices_from_verified_rolling(path)
        verify.assert_called_once_with(path.resolve())
        self.assertEqual(frozen.verified_input_hashes, 24)
        self.assertEqual(frozen.selected_c("sheep", "A"), 0.01)
        self.assertEqual(frozen.artifact_sha256, expected_hash)

    def test_stale_artifact_failure_precedes_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling.json"
            path.write_text("not JSON", encoding="utf-8")
            with mock.patch.object(
                robustness.cpu,
                "verify_existing_output",
                side_effect=ValueError("stale input hash"),
            ):
                with self.assertRaisesRegex(ValueError, "stale input hash"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_missing_choice_task_is_rejected(self) -> None:
        payload = _rolling_payload()
        del payload["chains"]["sheep"][
            robustness.ROLLING_CHOICE_SPECS["B2"]["track_key"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "missing=.*track_b2"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_extra_choice_task_is_rejected(self) -> None:
        payload = _rolling_payload()
        payload["chains"]["sheep"]["unexpected_task"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "extra=.*unexpected_task"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_nonhistorical_selection_source_is_rejected(self) -> None:
        payload = _rolling_payload()
        payload["protocol"]["selection_source"] = "main target"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "nonhistorical or unsafe"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_nonhistorical_input_role_swap_is_rejected(self) -> None:
        payload = _rolling_payload()
        audit = payload["chains"]["sheep"]["protocol_audit"]
        audit["history_track_a"]["path"] = audit["target_track_a"]["path"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "noncanonical or nonhistorical"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_selected_c_must_match_deterministic_historical_winner(self) -> None:
        payload = _rolling_payload()
        spec = robustness.ROLLING_CHOICE_SPECS["A"]
        selection = payload["chains"]["sheep"][spec["track_key"]]["models"][
            spec["model_key"]
        ]["model"]["selection"]
        selection["selected_C"] = 10.0
        selection["selected_mean_objective"] = 0.7
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "selected_C mismatch"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_all_24_input_hash_records_are_required(self) -> None:
        payload = _rolling_payload()
        del payload["chains"]["sheep"]["protocol_audit"]["history_track_a"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            with mock.patch.object(robustness.cpu, "verify_existing_output"):
                with self.assertRaisesRegex(ValueError, "missing history_track_a"):
                    robustness._freeze_choices_from_verified_rolling(path)

    def test_run_stops_at_choice_gate_before_any_candidate_reader(self) -> None:
        args = SimpleNamespace(
            data_dir=robustness.DEFAULT_DATA,
            baci_zip=Path("not-opened.zip"),
            rolling_result=Path("rolling.json"),
            chains=list(robustness.CHAINS),
        )
        with mock.patch.object(
            robustness,
            "_freeze_choices_from_verified_rolling",
            side_effect=ValueError("choice gate stopped"),
        ), mock.patch.object(robustness, "_fit_frozen_chain") as fit:
            with self.assertRaisesRegex(ValueError, "choice gate stopped"):
                robustness.run(args)
        fit.assert_not_called()


class RawAnnualLabelTests(unittest.TestCase):
    def test_persistence_is_built_from_annual_stage_totals_not_window_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "tiny_baci.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for year in range(1, 6):
                    if year <= 3:
                        csv = "i,j,k,v\n1,2,000001,150\n"
                    else:
                        csv = "i,j,k,v\n"
                    zf.writestr(f"BACI_HS92_Y{year}_V202401b.csv", csv)

            keys = pd.DataFrame(
                [
                    {"chain": "toy", "i_iso": "AAA", "j_iso": "BBB", "stage": "processed"},
                    {"chain": "toy", "i_iso": "AAA", "j_iso": "CCC", "stage": "processed"},
                ]
            )
            with zipfile.ZipFile(archive) as zf:
                labels, audit = robustness.aggregate_candidate_late_years(
                    zf,
                    keys,
                    iso={1: "AAA", 2: "BBB", 3: "CCC"},
                    hs_to_chain={"000001": "toy"},
                    hs_to_stage={"000001": "processed"},
                    years=(1, 2, 3, 4, 5),
                    chunk_size=1,
                )

            lane = labels.loc[labels["j_iso"].eq("BBB")].reset_index(drop=True)
            self.assertAlmostEqual(float(lane.loc[0, "raw_late_calendar_mean_kusd"]), 90.0)
            self.assertEqual(int(lane.loc[0, "active_years_100kusd"]), 3)
            threshold_y, _ = robustness.labels_at_threshold(lane, 100.0)
            persistence_y, persistence_value = robustness.persistence_labels(lane)
            self.assertEqual(threshold_y.tolist(), [0])
            self.assertEqual(persistence_y.tolist(), [1])
            self.assertEqual(persistence_value.tolist(), [90.0])
            self.assertEqual(audit["missing_stage_year_value_kusd"], 0.0)

            absent = labels.loc[labels["j_iso"].eq("CCC")].iloc[0]
            self.assertEqual(float(absent["raw_late_calendar_mean_kusd"]), 0.0)
            self.assertEqual(int(absent["active_years_100kusd"]), 0)

    def test_threshold_rule_is_strictly_greater_than(self) -> None:
        frame = pd.DataFrame({"raw_late_calendar_mean_kusd": [50.0, 50.01, 250.0, 250.01]})
        y50, value50 = robustness.labels_at_threshold(frame, 50.0)
        y250, _ = robustness.labels_at_threshold(frame, 250.0)
        self.assertEqual(y50.tolist(), [0, 1, 1, 1])
        self.assertEqual(value50.tolist(), [0.0, 50.01, 250.0, 250.01])
        self.assertEqual(y250.tolist(), [0, 0, 0, 1])


class SliceMetricTests(unittest.TestCase):
    def test_b2_slice_reconditions_on_positive_entries_after_filtering(self) -> None:
        frame = pd.DataFrame(
            {
                "entry_id": ["A|s", "A|s", "B|s", "B|s"],
                "i_iso": ["A", "A", "B", "B"],
                "j_iso": ["X", "Y", "X", "Y"],
                "stage": ["s"] * 4,
            }
        )
        metrics = robustness._conditional_metrics_for_slice(
            frame,
            np.array([0.9, 0.1, 0.8, 0.2]),
            label=np.array([1, 0, 0, 1]),
            lateval=np.array([10.0, 0.0, 0.0, 20.0]),
            mask=np.array([True, False, True, False]),
            bootstrap=0,
            seed=1,
        )
        self.assertEqual(metrics["entry_groups_before_entry_reconditioning"], 2)
        self.assertEqual(metrics["entry_groups_after_entry_reconditioning"], 1)
        self.assertEqual(metrics["dropped_zero_positive_entry_groups"], 1)
        self.assertEqual(metrics["n_entry_groups"], 1)


class PrespecValidationTests(unittest.TestCase):
    def _minimal_payload(self) -> dict:
        choices = {
            chain: {track: float(robustness.cpu.C_GRID[0]) for track in robustness.TRACKS}
            for chain in robustness.CHAINS
        }
        choice_digest = robustness.RollingChoiceFreeze(
            path=Path("."),
            artifact_sha256="",
            choices=robustness._choice_tuple(choices, context="test choices"),
            verified_input_hashes=24,
        ).choices_sha256
        expected = {
            "identity": {
                "exporter_seen": {},
                "exporter_unseen": {},
                "importer_seen": {},
                "importer_unseen": {},
            },
            "entity_exclusion": {
                "exclude_hubs": {},
                "exclude_bad_iso": {},
                "exclude_hubs_and_bad_iso": {},
            },
            "threshold_outcome_only": {
                "threshold_50_kusd": {},
                "threshold_100_kusd": {},
                "threshold_250_kusd": {},
            },
            "persistence": {"active_at_least_3_of_5_years_above_100_kusd": {}},
        }
        return {
            "schema_version": robustness.SCHEMA_VERSION,
            "prespecification": {
                "chains": list(robustness.CHAINS),
                "thresholds_kusd": list(robustness.THRESHOLDS_KUSD),
                "hubs": sorted(robustness.HUBS),
                "bad_iso": sorted(robustness.BAD_ISO),
                "frozen_C": choices,
                "freeze_digest_sha256": choice_digest,
                "persistence": {
                    "minimum_active_years": robustness.PERSISTENCE_MIN_ACTIVE_YEARS
                },
            },
            "protocol": {
                "rolling_artifact_verified_before_any_candidate_label_parse": True,
                "all_chain_choices_frozen_before_any_candidate_label_parse": True,
                "all_chain_models_frozen_before_main_open": True,
                "main_labels_used_for_model_or_hyperparameter_selection": False,
                "main_labels_used_for_feature_fitting_imputation_or_calibration": False,
                "main_model_champion_selected": False,
                "raw_persistence_or_threshold_labels_use_production_aggregation_helper": False,
                "transductive_split_used": False,
            },
            "chains": {
                chain: {
                    "frozen_choices": {
                        track: {
                            "selected_C": choices[chain][track],
                            "source": "verified rolling-CPU fold2 grouped-CV artifact",
                        }
                        for track in robustness.TRACKS
                    },
                    "raw_reconciliation": {
                        "track_a": {"pass": True},
                        "track_b_lanes": {"pass": True},
                    },
                    "sensitivities": expected,
                }
                for chain in robustness.CHAINS
            },
            "selection_artifact": {
                "sha256": "b" * 64,
                "choices_sha256": choice_digest,
                "frozen_choices": json.loads(json.dumps(choices)),
                "verified_input_hashes": 24,
                "verified_before_any_candidate_label_parse": True,
                "selected_C_fields_checked_before_historical_fit": True,
            },
        }

    def test_validator_rejects_changed_thresholds(self) -> None:
        payload = self._minimal_payload()
        robustness._validate_prespec_payload(payload)
        payload["prespecification"]["thresholds_kusd"] = [100.0]
        with self.assertRaisesRegex(ValueError, "threshold prespecification"):
            robustness._validate_prespec_payload(payload)

    def test_validator_rejects_output_choice_mismatch(self) -> None:
        payload = self._minimal_payload()
        payload["selection_artifact"]["frozen_choices"]["sheep"]["A"] = 0.1
        with self.assertRaisesRegex(ValueError, "choices disagree"):
            robustness._validate_prespec_payload(payload)


if __name__ == "__main__":
    unittest.main()
