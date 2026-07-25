import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from v2_gpu_protocol import (  # noqa: E402
    PROTOCOL,
    SELECTION_SCHEMA,
    ProtocolError,
    build_freeze_manifest,
    canonical_json_bytes,
    selection_filename,
    verify_freeze_manifest,
    write_json_atomic,
)
from v2_gpu_rolling import (  # noqa: E402
    DEFAULT_CHAINS,
    FAMILIES,
    FORMAL_EXECUTION_STATUS,
    TRACKS,
    _resolve_run_lock,
    build_parser as build_rolling_parser,
    main as rolling_main,
)
from v2_gpu_env_check import (  # noqa: E402
    BASE_DIRECT_MODULES,
    BASE_PYKEEN_NUMERICAL_MODULES,
    OVERLAY_DIRECT_MODULES,
    OVERLAY_TRANSITIVE_MODULES,
    PYKEEN_DIRECT_DISTRIBUTIONS,
    _read_pins,
)


def selection(chain="sheep", track="a", family="kge", config_hash="a" * 64):
    return {
        "schema_version": SELECTION_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete",
        "chain": chain,
        "track": track,
        "family": family,
        "selection_fold": "fold2",
        "target_fold": "main",
        "aggregation": "calendar_mean",
        "main_target_labels_accessed": False,
        "run_id": "unit-test-run",
        "run_config_sha256": config_hash,
        "selected": {
            "model": "RotatE" if family == "kge" else "NBFNet",
            "hyperparameters": {"epochs": 1},
        },
        "selection_design": {
            "orchestration": "chain_multitask_shared_score_grid",
            "evaluation_seeds": [0],
        },
        "shared_score_cache": {"context_sha256": "b" * 64},
    }


def formal_run_config(*, run_id="unit-test-run", evaluation_seeds=(0,)):
    config = json.loads(
        (ROOT / "configs" / "v2_gpu_rolling.json").read_text(encoding="utf-8")
    )
    config["run_id"] = run_id
    config["selection"]["evaluation_seeds"] = list(evaluation_seeds)
    return config


class V2GpuProtocolTest(unittest.TestCase):
    def test_gpu_overlay_lock_covers_audited_pykeen_runtime_dependencies(self):
        lock_path = ROOT / "requirements" / "v2-gpu-nodeps-lock.txt"
        pins = _read_pins(lock_path)
        expected_direct = (
            set(OVERLAY_DIRECT_MODULES).difference({"pykeen"})
            | set(BASE_DIRECT_MODULES)
            | set(BASE_PYKEEN_NUMERICAL_MODULES)
        )
        self.assertEqual(expected_direct, set(PYKEEN_DIRECT_DISTRIBUTIONS))
        self.assertTrue(set(OVERLAY_DIRECT_MODULES).issubset(pins))
        self.assertEqual(pins["tabulate"], "0.10.0")
        self.assertTrue(set(OVERLAY_TRANSITIVE_MODULES).issubset(pins))
        self.assertEqual(pins["packaging"], "26.2")
        self.assertTrue(set(BASE_DIRECT_MODULES).isdisjoint(pins))
        self.assertTrue(set(BASE_PYKEEN_NUMERICAL_MODULES).isdisjoint(pins))

    def test_gpu_worker_checks_active_overlay_and_attempt_scopes_artifacts(self):
        worker = (ROOT / "jobs" / "v2_gpu_nohup_worker.sh").read_text(encoding="utf-8")
        self.assertIn('--forbid-prefix "$OVERLAY"', worker)
        self.assertIn('ENV_REPORT="$MANIFEST_ROOT/env_${WORKER_ID}.json"', worker)
        self.assertIn('LOG="$LOG_ROOT/${WORKER_ID}_${CHAIN}_tasks-A-B1-B2.log"', worker)
        self.assertIn('OUTPUT_ROOT="results_v2/gpu_smoke/$WORKER_ID"', worker)
        self.assertIn('PILOT_INVALIDATED.json', worker)

    def test_freeze_and_verify_all_expected_selections(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selection_dir = root / "selections"
            combos = [("sheep", "a", "kge"), ("sheep", "a", "nbfnet")]
            for combo in combos:
                write_json_atomic(
                    selection_dir / selection_filename(*combo), selection(*combo)
                )
            manifest_path = root / "frozen_manifest.json"
            manifest = build_freeze_manifest(
                selection_dir=selection_dir,
                manifest_path=manifest_path,
                combinations=combos,
            )
            write_json_atomic(manifest_path, manifest)
            verified, indexed = verify_freeze_manifest(manifest_path)
            self.assertTrue(verified["all_selections_frozen_before_main"])
            self.assertEqual(verified["run_id"], "unit-test-run")
            self.assertEqual(verified["run_config_sha256"], "a" * 64)
            self.assertEqual(set(indexed), {"sheep|a|kge", "sheep|a|nbfnet"})

    def test_post_freeze_tampering_closes_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selection_dir = root / "selections"
            combo = ("sheep", "a", "kge")
            selection_path = selection_dir / selection_filename(*combo)
            write_json_atomic(selection_path, selection(*combo))
            manifest_path = root / "frozen_manifest.json"
            manifest = build_freeze_manifest(
                selection_dir=selection_dir,
                manifest_path=manifest_path,
                combinations=[combo],
            )
            write_json_atomic(manifest_path, manifest)
            selection_path.write_bytes(canonical_json_bytes({**selection(*combo), "tampered": True}))
            with self.assertRaisesRegex(ProtocolError, "changed after freeze"):
                verify_freeze_manifest(manifest_path)

    def test_main_label_attestation_is_required(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            combo = ("sheep", "a", "kge")
            bad = selection(*combo)
            bad["main_target_labels_accessed"] = True
            path = root / "selections" / selection_filename(*combo)
            write_json_atomic(path, bad)
            with self.assertRaisesRegex(ProtocolError, "main target labels"):
                build_freeze_manifest(
                    selection_dir=path.parent,
                    manifest_path=root / "frozen_manifest.json",
                    combinations=[combo],
                )

    def test_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            combo = ("sheep", "a", "kge")
            path = root / "selections" / selection_filename(*combo)
            write_json_atomic(path, selection(*combo))
            manifest_path = root / "frozen_manifest.json"
            manifest = build_freeze_manifest(
                selection_dir=path.parent,
                manifest_path=manifest_path,
                combinations=[combo],
            )
            manifest["entries"][0]["path"] = "../outside.json"
            write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(ProtocolError, "escapes manifest root"):
                verify_freeze_manifest(manifest_path)

    def test_protocol_artifacts_do_not_silently_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "selection.json"
            write_json_atomic(path, {"one": 1})
            with self.assertRaisesRegex(ProtocolError, "refusing to overwrite"):
                write_json_atomic(path, {"two": 2})

    def test_full_three_task_manifest_and_evaluate_chain_dry_run_open_no_data(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "gpu"
            config = root / "run_config.json"
            write_json_atomic(config, formal_run_config())
            import hashlib

            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            combos = [
                (chain, track, family)
                for chain in DEFAULT_CHAINS
                for track in TRACKS
                for family in FAMILIES
            ]
            self.assertEqual(len(combos), 36)
            for combo in combos:
                write_json_atomic(
                    output / "selections" / selection_filename(*combo),
                    selection(*combo, config_hash=config_hash),
                )
            manifest_path = output / "frozen_manifest.json"
            manifest = build_freeze_manifest(
                selection_dir=output / "selections",
                manifest_path=manifest_path,
                combinations=combos,
            )
            write_json_atomic(manifest_path, manifest)
            rc = rolling_main(
                [
                    "evaluate-chain",
                    "--chain",
                    "sheep",
                    "--family",
                    "kge",
                    "--manifest",
                    str(manifest_path),
                    "--run-config",
                    str(config),
                    "--candidate-root",
                    str(root / "definitely-missing"),
                    "--output-root",
                    str(output),
                    "--seeds",
                    "0",
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse((output / "MAIN_EVALUATION_STARTED.json").exists())

    def test_cli_exposes_only_shared_chain_formal_phases(self):
        parser = build_rolling_parser()
        command_action = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(command_action.choices),
            {"select-chain", "freeze", "evaluate-chain"},
        )
        self.assertNotIn("select", command_action.choices)
        self.assertNotIn("evaluate", command_action.choices)

    def test_select_chain_rejects_every_effective_cli_grid_or_seed_mismatch(self):
        cases = (
            (["--models", "TransE"], "kge.models"),
            (["--dims", "64"], "kge.embedding_dims"),
            (["--learning-rates", "0.005"], "kge.learning_rates"),
            (["--epochs", "149"], "kge.epochs"),
            (["--kge-batch-size", "1024"], "kge.batch_size"),
            (["--split-salt", "different-salt"], "selection.split_salt"),
            (["--selection-seed", "1"], "selection.selection_seed"),
            (["--seeds", "0,1"], "selection.evaluation_seeds"),
        )
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "run_config.json"
            write_json_atomic(config, formal_run_config())
            base = [
                "select-chain",
                "--chain",
                "sheep",
                "--family",
                "kge",
                "--run-config",
                str(config),
                "--seeds",
                "0",
                "--dry-run",
            ]
            for override, field in cases:
                with self.subTest(field=field):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                        rc = rolling_main([*base, *override])
                    self.assertEqual(rc, 2)
                    self.assertIn(field, stderr.getvalue())

    def test_select_chain_rejects_nbfnet_grid_mismatches(self):
        cases = (
            (["--layers", "4"], "nbfnet.layers"),
            (["--nbfnet-learning-rates", "0.001"], "nbfnet.learning_rates"),
            (["--epochs", "24"], "nbfnet.epochs"),
            (["--nbfnet-batch-size", "32"], "nbfnet.batch_size"),
            (["--nbfnet-negatives", "16"], "nbfnet.negatives"),
        )
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "run_config.json"
            write_json_atomic(config, formal_run_config())
            base = [
                "select-chain",
                "--chain",
                "sheep",
                "--family",
                "nbfnet",
                "--run-config",
                str(config),
                "--seeds",
                "0",
                "--dry-run",
            ]
            for override, field in cases:
                with self.subTest(field=field):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                        rc = rolling_main([*base, *override])
                    self.assertEqual(rc, 2)
                    self.assertIn(field, stderr.getvalue())

    def test_formal_scope_is_checked_and_irrelevant_family_args_are_ignored(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            valid = root / "valid.json"
            write_json_atomic(valid, formal_run_config())
            base = [
                "select-chain",
                "--chain",
                "sheep",
                "--family",
                "kge",
                "--run-config",
                str(valid),
                "--seeds",
                "0",
                "--dry-run",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    rolling_main([*base, "--layers", "999", "--nbfnet-negatives", "999"]),
                    0,
                )

            for field, value in (
                ("chains", ["sheep"]),
                ("tracks", ["a", "b1"]),
                ("families", ["kge"]),
            ):
                config = formal_run_config()
                config[field] = value
                path = root / f"bad_{field}.json"
                write_json_atomic(path, config)
                argv = list(base)
                argv[argv.index(str(valid))] = str(path)
                stderr = io.StringIO()
                with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    rc = rolling_main(argv)
                self.assertEqual(rc, 2)
                self.assertIn(field, stderr.getvalue())

            outside_chain = list(base)
            outside_chain[outside_chain.index("sheep")] = "not-configured"
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                rc = rolling_main(outside_chain)
            self.assertEqual(rc, 2)
            self.assertIn("outside the formal run config domain", stderr.getvalue())

    def test_select_chain_checks_all_configured_primary_metrics_before_data_access(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = formal_run_config()
            # Tamper B2 while invoking a KGE chain job.  The gate must validate
            # the complete three-task mapping, not only the first/current task.
            config["selection"]["primary_metric_by_task"]["b2"] = "recall_at_5"
            config_path = root / "run_config.json"
            write_json_atomic(config_path, config)
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                rc = rolling_main(
                    [
                        "select-chain",
                        "--chain",
                        "sheep",
                        "--family",
                        "kge",
                        "--run-config",
                        str(config_path),
                        "--candidate-root",
                        str(root / "definitely-missing"),
                        "--seeds",
                        "0",
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertIn("selection.primary_metric_by_task", stderr.getvalue())
            self.assertNotIn("candidate", stderr.getvalue().lower())

    def test_evaluate_chain_checks_cli_seeds_before_opening_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "run_config.json"
            write_json_atomic(config, formal_run_config())
            missing_manifest = root / "missing_manifest.json"
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                rc = rolling_main(
                    [
                        "evaluate-chain",
                        "--chain",
                        "sheep",
                        "--family",
                        "kge",
                        "--run-config",
                        str(config),
                        "--manifest",
                        str(missing_manifest),
                        "--seeds",
                        "1",
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertIn("selection.evaluation_seeds", stderr.getvalue())
            self.assertNotIn("cannot read freeze manifest", stderr.getvalue())

    def test_evaluate_chain_rejects_bootstrap_overrides_before_opening_manifest(self):
        cases = (
            (["--bootstrap-iters", "499"], "--bootstrap-iters=500"),
            (["--bootstrap-seed", "20260713"], "--bootstrap-seed=20260712"),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "run_config.json"
            write_json_atomic(config, formal_run_config())
            missing_manifest = root / "missing_manifest.json"
            base = [
                "evaluate-chain",
                "--chain",
                "sheep",
                "--family",
                "kge",
                "--run-config",
                str(config),
                "--manifest",
                str(missing_manifest),
                "--seeds",
                "0",
                "--dry-run",
            ]
            for override, message in cases:
                with self.subTest(message=message):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                        rc = rolling_main([*base, *override])
                    self.assertEqual(rc, 2)
                    self.assertIn(message, stderr.getvalue())
                    self.assertNotIn("cannot read freeze manifest", stderr.getvalue())

    def test_run_lock_rejects_non_authorized_execution_status(self):
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "run_config.json"
            write_json_atomic(
                config,
                {
                    "run_id": "blocked-run",
                    "execution_status": "BLOCKED_PENDING_CORRECTED_DATA",
                },
            )
            args = SimpleNamespace(run_config=config, run_id=None)
            with self.assertRaisesRegex(ProtocolError, "not authorized"):
                _resolve_run_lock(args)


if __name__ == "__main__":
    unittest.main()
