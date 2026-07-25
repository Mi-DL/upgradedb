import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
# Construct a deliberately private fixture path without embedding a real-looking
# home path in the public test source itself.  The sanitizer must still remove
# the runtime value from generated summaries.
PRIVATE_FIXTURE_ROOT = "/" + "home" + "/" + "private-user" + "/project"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from summarize_v2_gpu_results import (  # noqa: E402
    BOOTSTRAP_SPECS,
    BENCHMARK_VERSION,
    CHAINS,
    COUNT_METRICS,
    EVALUATION_SCHEMA,
    FAMILIES,
    FORMAL_GRIDS,
    FORMAL_EXECUTION_STATUS,
    KGE_MODELS,
    PRIMARY_METRICS,
    NBFNET_ATTESTATION_TOOL_SHA256,
    NBFNET_FORMAL_GATE_SCHEMA,
    NBFNET_MAIN_ROLES,
    NBFNET_RECEIPT_ROLES,
    ResultValidationError,
    RUN_CONFIG_SCHEMA,
    SEEDS,
    SUMMARY_SCHEMA,
    TRACKS,
    TRACK_METRICS,
    _mean,
    _std,
    _validate_current_run_config,
    build_summary,
    render_csv,
    render_json,
    verify_outputs,
    verify_nbfnet_public_binding,
    write_outputs,
)
from v2_gpu_protocol import (  # noqa: E402
    PROTOCOL,
    SELECTION_SCHEMA,
    build_freeze_manifest,
    selection_filename,
    write_json_atomic,
)
import v2_gpu_rolling as gpu_runner  # noqa: E402
import build_nbfnet_source_attestation as nbfnet_attestation  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


CANDIDATE_ROWS = (
    ("A", "U", 0),
    ("B", "V", 0),
    ("C", "W", 0),
    ("D", "X", 1),
    ("E", "Y", 1),
    ("F", "Z", 1),
)


def fast_fixture_bootstrap(track, identities, labels, score, *, iters, seed):
    """Cheap deterministic stand-in; production calls the real shared helper."""

    cluster_unit, metric = BOOTSTRAP_SPECS[track]
    center = 0.5 + (int(seed) % 5) * 0.01
    return {
        "cluster_unit": cluster_unit,
        "metric": metric,
        "iterations": int(iters),
        "seed": int(seed),
        "lower_95": center - 0.1,
        "upper_95": center + 0.1,
    }


class SyntheticFormalRun:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "synthetic-formal-run"
        self.deployment_root = root.parents[1]
        self.run_config = self.deployment_root / "configs" / "v2_gpu_rolling.json"
        self.candidate_root = self.deployment_root / "data" / "processed_v2"
        deployed_runner = self.deployment_root / "src" / "v2_gpu_rolling.py"
        deployed_runner.parent.mkdir(parents=True, exist_ok=True)
        deployed_runner.write_bytes((ROOT / "src" / "v2_gpu_rolling.py").read_bytes())
        deployed_attestation = (
            self.deployment_root / "tools" / "build_nbfnet_source_attestation.py"
        )
        deployed_attestation.parent.mkdir(parents=True, exist_ok=True)
        deployed_attestation.write_bytes(
            (ROOT / "tools" / "build_nbfnet_source_attestation.py").read_bytes()
        )
        private_attestation = (
            self.deployment_root
            / "private"
            / "build_nbfnet_source_attestation_v2.py"
        )
        private_attestation.parent.mkdir(parents=True, exist_ok=True)
        private_attestation.write_bytes(deployed_attestation.read_bytes())
        self._write_run_config()
        self._write_candidates()
        self.selections = {}
        self._write_selections_and_freeze()
        self._write_step3_manifest()
        self._write_nbfnet_evidence()
        self._write_main_marker()
        self._write_evaluations()
        self._write_claims()

    def _write_run_config(self) -> None:
        write_json(
            self.run_config,
            {
                "schema_version": RUN_CONFIG_SCHEMA,
                "run_id": self.run_id,
                "benchmark_version": BENCHMARK_VERSION,
                "execution_status": FORMAL_EXECUTION_STATUS,
                "formal_authorization_value": FORMAL_EXECUTION_STATUS,
                "protocol": PROTOCOL,
                "chains": list(CHAINS),
                "tracks": list(TRACKS),
                "families": list(FAMILIES),
                "expected_selection_count": 36,
                "selection_orchestration_job_count": 12,
                "selection_fold": "fold2",
                "target_fold": "main",
                "aggregation": "calendar_mean",
                "selection": {
                    "split_unit": "exporter_stage",
                    "split_salt": "v2-history-0",
                    "selection_seed": 0,
                    "evaluation_seeds": list(SEEDS),
                },
            },
        )
        self.config_hash = hashlib.sha256(self.run_config.read_bytes()).hexdigest()

    def _candidate_path(self, chain: str, track: str, fold: str) -> Path:
        stem = "candidates" if track == "a" else "candidates_firsttime"
        suffix = "_fold2" if fold == "fold2" else ""
        return self.candidate_root / f"{stem}_{chain}{suffix}.csv"

    def _write_candidates(self) -> None:
        fields = [
            "i_iso",
            "j_iso",
            "stage",
            "y",
            "size",
            "lateval",
            "benchmark_version",
            "aggregation",
            "early_window",
            "late_window",
            "temporal_role",
        ]
        for chain in CHAINS:
            for fold in ("fold2", "main"):
                early, late, temporal_role = (
                    ("1998-2002", "2008-2012", "history")
                    if fold == "fold2"
                    else ("2008-2012", "2018-2022", "target")
                )
                for track in ("a", "b1"):
                    path = self._candidate_path(chain, track, fold)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                        writer.writeheader()
                        for exporter, importer, label in CANDIDATE_ROWS:
                            writer.writerow(
                                {
                                    "i_iso": exporter,
                                    "j_iso": importer,
                                    "stage": "stage",
                                    "y": label,
                                    "size": 10.0,
                                    "lateval": 100.0 if label else 0.0,
                                    "benchmark_version": BENCHMARK_VERSION,
                                    "aggregation": "calendar_mean",
                                    "early_window": early,
                                    "late_window": late,
                                    "temporal_role": temporal_role,
                                }
                            )

    def _candidate_sha256(self, chain: str, track: str, fold: str) -> str:
        return hashlib.sha256(self._candidate_path(chain, track, fold).read_bytes()).hexdigest()

    def _selection(self, chain: str, track: str, family: str) -> dict:
        models = KGE_MODELS if family == "kge" else ("NBFNet",)
        model_rows = []
        for model_index, model in enumerate(models):
            chosen_hp = dict(FORMAL_GRIDS[model][0])
            grid = []
            for hp_index, hp in enumerate(FORMAL_GRIDS[model]):
                grid.append(
                    {
                        "hyperparameters": dict(hp),
                        "status": "complete",
                        "history_dev_selection_metric": 0.1 + model_index * 0.05 + hp_index * 0.001,
                        "score_cache_key": digest(f"grid|{chain}|{family}|{model}|{hp_index}"),
                        "score_cache_hit": False,
                    }
                )
            values = [0.2 + model_index * 0.05 + seed * 0.001 for seed in SEEDS]
            per_seed = [
                {
                    "seed": seed,
                    "status": "complete",
                    "history_holdout_selection_metric": value,
                    "score_cache_key": digest(f"holdout|{chain}|{family}|{model}|{seed}"),
                    "score_cache_hit": seed == 0,
                }
                for seed, value in zip(SEEDS, values)
            ]
            model_rows.append(
                {
                    "model": model,
                    "status": "complete",
                    "selected_hyperparameters": chosen_hp,
                    "history_dev_selection_metric": grid[0]["history_dev_selection_metric"],
                    "history_holdout_selection_metric_mean": _mean(values),
                    "history_holdout_selection_metric_std": _std(values),
                    "per_seed": per_seed,
                    "grid": grid,
                }
            )
        winner = model_rows[-1]
        source = "candidates" if track == "a" else "candidates_firsttime"
        payload = {
            "schema_version": SELECTION_SCHEMA,
            "protocol": PROTOCOL,
            "status": "complete",
            "created_at_utc": "2026-07-12T00:00:00+00:00",
            "chain": chain,
            "track": track,
            "family": family,
            "run_id": self.run_id,
            "run_config": f"{PRIVATE_FIXTURE_ROOT}/configs/v2_gpu_rolling.json",
            "run_config_sha256": self.config_hash,
            "selection_fold": "fold2",
            "target_fold": "main",
            "aggregation": "calendar_mean",
            "main_target_labels_accessed": False,
            "selection_design": {
                "orchestration": "chain_multitask_shared_score_grid",
                "hp_partition": "fold2 exporter_stage dev",
                "model_partition": "fold2 exporter_stage holdout",
                "split_unit": "exporter_stage",
                "split_salt": "v2-history-0",
                "primary_metric": {
                    "a": "track_a_lane_average_precision",
                    "b1": "track_b1_entry_average_precision_max_lane_score",
                    "b2": "track_b2_positive_entry_macro_recall_at_3",
                }[track],
                "selection_seed": 0,
                "evaluation_seeds": list(SEEDS),
            },
            "history_candidate": {
                "path": f"data/processed_v2/{source}_{chain}_fold2.csv",
                "sha256": self._candidate_sha256(chain, track, "fold2"),
                "rows": len(CANDIDATE_ROWS),
                "positive_lanes": sum(row[2] for row in CANDIDATE_ROWS),
            },
            "shared_score_cache": {
                "root": f"{PRIVATE_FIXTURE_ROOT}/cache/{chain}/{family}",
                "context_sha256": digest(f"selection-context|{chain}|{family}"),
                "combined_rows": 2 * len(CANDIDATE_ROWS),
            },
            "models": model_rows,
            "representation_policy": "refit selected label-free model from scratch on main early graph per seed",
            "raw_score_policy": "one column per seed; no cross-seed raw-score average",
            "selected": {
                "model": winner["model"],
                "hyperparameters": winner["selected_hyperparameters"],
                "history_holdout_selection_metric_mean": winner[
                    "history_holdout_selection_metric_mean"
                ],
            },
        }
        return payload

    def _write_selections_and_freeze(self) -> None:
        combos = []
        for chain in CHAINS:
            for track in TRACKS:
                for family in FAMILIES:
                    combo = (chain, track, family)
                    combos.append(combo)
                    payload = self._selection(*combo)
                    self.selections[combo] = payload
                    write_json_atomic(
                        self.root / "selections" / selection_filename(*combo), payload
                    )
        manifest_path = self.root / "frozen_manifest.json"
        manifest = build_freeze_manifest(
            selection_dir=self.root / "selections",
            manifest_path=manifest_path,
            combinations=combos,
        )
        write_json_atomic(manifest_path, manifest)
        self.manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def _write_step3_manifest(self) -> None:
        self.step3_manifest = (
            self.root
            / "runs"
            / self.run_id
            / "STEP3_SYNC_MANIFEST.sha256"
        )
        self.step3_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.step3_manifest.write_text(
            f"{digest('fixture runner')}  src/v2_gpu_rolling.py\n",
            encoding="utf-8",
        )
        self.step3_sha = hashlib.sha256(
            self.step3_manifest.read_bytes()
        ).hexdigest()

    def _write_nbfnet_evidence(self) -> None:
        selection_source = self.deployment_root / "private" / "nbfnet_selection_source"
        selection_source.mkdir(parents=True)
        (selection_source / "README.md").write_text(
            "synthetic NBFNet source fixture\n", encoding="utf-8"
        )
        (selection_source / "nbfnet").mkdir()
        (selection_source / "nbfnet" / "model.py").write_text(
            "class NBFNet: pass\n", encoding="utf-8"
        )
        old_mtime = 1_577_836_800
        for path in selection_source.rglob("*"):
            os.utime(path, (old_mtime, old_mtime))

        frozen_source = self.deployment_root / "private" / "nbfnet_source_frozen"
        shutil.copytree(selection_source, frozen_source, copy_function=shutil.copy2)
        runtime = (
            self.deployment_root / "torch_extensions" / "rspmm" / "rspmm.so"
        )
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b"synthetic-rspmm-extension")
        os.utime(runtime, (old_mtime, old_mtime))
        runtime_sha = nbfnet_attestation.sha256_file(runtime)
        identity = nbfnet_attestation.RunIdentityInputs(
            run_id=self.run_id,
            frozen_manifest=self.root / "frozen_manifest.json",
            frozen_manifest_sha256=self.manifest_sha,
            step3_manifest=self.step3_manifest,
            step3_manifest_sha256=self.step3_sha,
        )
        private_root = self.root / "nbfnet_attestation"
        private_root.mkdir()
        receipt_paths = {
            role: private_root / f"{role}.json" for role in NBFNET_RECEIPT_ROLES
        }
        receipts = {}
        for role in NBFNET_RECEIPT_ROLES:
            is_selection = role.startswith("selection-")
            source = selection_source if is_selection else frozen_source
            with mock.patch.dict(
                os.environ,
                {nbfnet_attestation.SOURCE_ENVIRONMENT_VARIABLE: str(source)},
            ):
                receipts[role] = nbfnet_attestation.build_private_receipt(
                    identity,
                    source_root=source,
                    observed_at_utc=(
                        "2026-07-12T00:00:30+00:00"
                        if is_selection
                        else "2026-07-12T00:00:45+00:00"
                    ),
                    runtime_artifacts=(
                        [] if is_selection else [("rspmm-extension", runtime)]
                    ),
                    expected_runtime_sha256=(
                        None
                        if is_selection
                        else {"rspmm-extension": runtime_sha}
                    ),
                    host_role=role,
                    selection_started_at_utc=(
                        "2026-07-12T00:00:15+00:00"
                        if is_selection
                        else None
                    ),
                )
            receipt_paths[role].write_bytes(
                nbfnet_attestation.render_json(receipts[role])
            )

        evidence_root = self.deployment_root / "chains" / "evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        for role, private_path in receipt_paths.items():
            public = nbfnet_attestation.project_public_receipt(private_path)
            (evidence_root / f"nbfnet_{role}.public.json").write_bytes(
                nbfnet_attestation.render_json(public)
            )

        source_comparison = nbfnet_attestation.compare_source_receipts(
            [receipt_paths[role] for role in NBFNET_RECEIPT_ROLES]
        )
        source_bytes = nbfnet_attestation.render_json(source_comparison)
        source_private = private_root / "source-comparison.json"
        source_private.write_bytes(source_bytes)
        (evidence_root / "nbfnet_source_comparison.json").write_bytes(source_bytes)

        runtime_comparison = nbfnet_attestation.compare_runtime_receipts(
            [receipt_paths[role] for role in NBFNET_MAIN_ROLES]
        )
        runtime_bytes = nbfnet_attestation.render_json(runtime_comparison)
        runtime_private = private_root / "runtime-comparison.json"
        runtime_private.write_bytes(runtime_bytes)
        (evidence_root / "nbfnet_runtime_comparison.json").write_bytes(runtime_bytes)

        receipt_hashes = {
            role: nbfnet_attestation.sha256_file(path)
            for role, path in receipt_paths.items()
        }
        shared_runtime = runtime_comparison["hosts"][0]["runtime_artifacts"]
        for role in NBFNET_MAIN_ROLES:
            write_json(
                private_root / f"formal_gate_{role}.json",
                {
                    "schema_version": NBFNET_FORMAL_GATE_SCHEMA,
                    "status": "PASS",
                    "run_id": self.run_id,
                    "host_role": role,
                    "created_at_utc": "2026-07-12T00:00:50Z",
                    "verified_before_main_marker": True,
                    "attestation_tool": {
                        "role": "private/build_nbfnet_source_attestation_v2.py",
                        "sha256": NBFNET_ATTESTATION_TOOL_SHA256,
                    },
                    "frozen_manifest": {
                        "role": "results_v2/gpu_rolling/frozen_manifest.json",
                        "sha256": self.manifest_sha,
                    },
                    "step3_sync_manifest": {
                        "role": (
                            "results_v2/gpu_rolling/runs/"
                            f"{self.run_id}/STEP3_SYNC_MANIFEST.sha256"
                        ),
                        "sha256": self.step3_sha,
                    },
                    "main_marker": {
                        "role": "results_v2/gpu_rolling/MAIN_EVALUATION_STARTED.json",
                        "absent_at_verification": True,
                    },
                    "source_snapshot": {
                        "role": "private/nbfnet_source_frozen",
                        "tree_sha256": source_comparison["receipts"][0]["tree_sha256"],
                        "mode_read_only": True,
                        "unattested_python_bytecode_absent": True,
                    },
                    "private_receipts": [
                        {
                            "role": (
                                "results_v2/gpu_rolling/nbfnet_attestation/"
                                f"{receipt_role}.json"
                            ),
                            "sha256": receipt_hashes[receipt_role],
                        }
                        for receipt_role in sorted(NBFNET_RECEIPT_ROLES)
                    ],
                    "comparisons": {
                        "source": {
                            "role": (
                                "results_v2/gpu_rolling/nbfnet_attestation/"
                                "source-comparison.json"
                            ),
                            "sha256": nbfnet_attestation.sha256_file(source_private),
                            "status": "PASS",
                        },
                        "runtime": {
                            "role": (
                                "results_v2/gpu_rolling/nbfnet_attestation/"
                                "runtime-comparison.json"
                            ),
                            "sha256": nbfnet_attestation.sha256_file(runtime_private),
                            "status": "PASS",
                        },
                    },
                    "runtime_artifacts": shared_runtime,
                    "gate_result": {
                        "source_peer_count": 3,
                        "runtime_peer_count": 1,
                        "all_source_trees_match": True,
                        "all_runtime_artifacts_match": True,
                    },
                },
            )

    def _write_main_marker(self) -> None:
        write_json(
            self.root / "MAIN_EVALUATION_STARTED.json",
            {
                "schema_version": "upgrade-bench-v2/main-start/1",
                "run_id": self.run_id,
                "manifest_sha256": self.manifest_sha,
                "main_started_at_utc": "2026-07-12T00:01:00+00:00",
                "policy": "freeze and selections are immutable; main outputs never overwrite",
            },
        )

    def _timestamps(self, chain: str, family: str) -> dict:
        seen = set()
        events = []
        for track in TRACKS:
            selected = self.selections[(chain, track, family)]["selected"]
            key = (selected["model"], json.dumps(selected["hyperparameters"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            for seed in SEEDS:
                events.append(
                    {
                        "model": selected["model"],
                        "hyperparameters": selected["hyperparameters"],
                        "seed": seed,
                        "score_cache_key": digest(f"main-cache|{chain}|{family}|{key}|{seed}"),
                        "score_cache_hit": False,
                        "finished_at_utc": f"2026-07-12T00:02:0{seed}+00:00",
                    }
                )
        return {
            "full_freeze_gate_verified_at_utc": "2026-07-12T00:01:30+00:00",
            "representations_finished": events,
            "main_target_labels_read_at_utc": "2026-07-12T00:03:00+00:00",
            "ordering_attestation": "all unique representation scores precede the target-label read",
        }

    def _write_evaluations(self) -> None:
        (self.root / "metrics").mkdir(parents=True)
        (self.root / "scores").mkdir(parents=True)
        identities = pd.DataFrame(
            [
                {"i_iso": exporter, "j_iso": importer, "stage": "stage"}
                for exporter, importer, _ in CANDIDATE_ROWS
            ]
        )
        labels = pd.DataFrame(
            {
                "y": [label for _, _, label in CANDIDATE_ROWS],
                "size": [10.0] * len(CANDIDATE_ROWS),
                "lateval": [100.0 if label else 0.0 for _, _, label in CANDIDATE_ROWS],
            }
        )
        for chain in CHAINS:
            for family in FAMILIES:
                timestamps = self._timestamps(chain, family)
                for track in TRACKS:
                    selection = self.selections[(chain, track, family)]
                    selection_path = self.root / "selections" / selection_filename(
                        chain, track, family
                    )
                    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
                    score_name = f"scores_{chain}_track-{track}_{family}.csv"
                    score_path = self.root / "scores" / score_name
                    model = selection["selected"]["model"]
                    with score_path.open("w", encoding="utf-8", newline="") as handle:
                        fields = [
                            "i_iso",
                            "j_iso",
                            "stage",
                            *(f"score_{model}_s{seed}" for seed in SEEDS),
                            "selection_sha256",
                            "protocol",
                        ]
                        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                        writer.writeheader()
                        for row_index, (exporter, importer, _) in enumerate(CANDIDATE_ROWS):
                            row = {
                                "i_iso": exporter,
                                "j_iso": importer,
                                "stage": "stage",
                                "selection_sha256": selection_sha,
                                "protocol": PROTOCOL,
                            }
                            row.update(
                                {
                                    f"score_{model}_s{seed}": 0.1 + seed + row_index * 0.1
                                    for seed in SEEDS
                                }
                            )
                            writer.writerow(row)

                    score_vectors = {
                        seed: [0.1 + seed + row_index * 0.1 for row_index in range(len(CANDIDATE_ROWS))]
                        for seed in SEEDS
                    }
                    metric_runs = [
                        gpu_runner._ranking_metrics(
                            track, identities, labels, score_vectors[seed]
                        )
                        for seed in SEEDS
                    ]
                    per_seed = [
                        {"seed": seed, **metrics}
                        for seed, metrics in zip(SEEDS, metric_runs)
                    ]
                    summary = gpu_runner._summarize_runs(metric_runs)
                    bootstrap = [
                        {
                            "seed": seed,
                            **gpu_runner._cluster_bootstrap(
                                track,
                                identities,
                                labels,
                                score_vectors[seed],
                                iters=500,
                                seed=20260712 + seed,
                            ),
                        }
                        for seed in SEEDS
                    ]
                    source = "candidates" if track == "a" else "candidates_firsttime"
                    metric_name = f"metrics_{chain}_track-{track}_{family}.json"
                    write_json(
                        self.root / "metrics" / metric_name,
                        {
                            "schema_version": EVALUATION_SCHEMA,
                            "protocol": PROTOCOL,
                            "status": "complete",
                            "created_at_utc": "2026-07-12T00:04:00+00:00",
                            "chain": chain,
                            "track": track,
                            "family": family,
                            "selection_manifest": (
                                f"{PRIVATE_FIXTURE_ROOT}/results_v2/gpu_rolling/"
                                "frozen_manifest.json"
                            ),
                            "manifest_sha256": self.manifest_sha,
                            "selection_sha256": selection_sha,
                            "run_id": self.run_id,
                            "run_config": f"{PRIVATE_FIXTURE_ROOT}/configs/v2_gpu_rolling.json",
                            "run_config_sha256": self.config_hash,
                            "selected": selection["selected"],
                            "target_fold": "main",
                            "aggregation": "calendar_mean",
                            "cohort_policy": "complete main cohort; no same-window dev/test split",
                            "orchestration": "chain_multitask_unique_config_training",
                            "protocol_timestamps": timestamps,
                            "target_candidate": {
                                "path": f"data/processed_v2/{source}_{chain}.csv",
                                "sha256": self._candidate_sha256(chain, track, "main"),
                                "rows": len(CANDIDATE_ROWS),
                                "positive_lanes": sum(row[2] for row in CANDIDATE_ROWS),
                            },
                            "seeds": list(SEEDS),
                            "per_seed": per_seed,
                            "cluster_bootstrap_by_seed": bootstrap,
                            "summary": summary,
                            "score_artifact": f"{PRIVATE_FIXTURE_ROOT}/results_v2/gpu_rolling/scores/{score_name}",
                        },
                    )

    def _write_claims(self) -> None:
        claims_root = self.root / "main_job_claims"
        log_root = self.root / "logs" / "main"
        log_root.mkdir(parents=True)
        step3_sha256 = self.step3_sha
        for index, (chain, family) in enumerate(
            (chain, family) for chain in CHAINS for family in FAMILIES
        ):
            worker_id = f"main_{chain}_{family}_fixture_gpu0_{index}"
            claim = claims_root / f"{chain}_{family}.lock"
            claim.mkdir(parents=True)
            (claim / "pid").write_text(f"{1000 + index}\n", encoding="utf-8")
            (claim / "worker.env").write_text(
                "\n".join(
                    (
                        f"worker_id={worker_id}",
                        "claimed_at=2026-07-12T00:01:00+00:00",
                        "host=private-fixture-host",
                        "physical_gpu=0",
                        f"chain={chain}",
                        f"family={family}",
                        f"manifest={PRIVATE_FIXTURE_ROOT}/results_v2/gpu_rolling/frozen_manifest.json",
                        f"manifest_sha256={self.manifest_sha}",
                        f"sync_manifest={PRIVATE_FIXTURE_ROOT}/results_v2/gpu_rolling/runs/{self.run_id}/STEP3_SYNC_MANIFEST.sha256",
                        f"sync_manifest_sha256={step3_sha256}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            run_log = log_root / f"{worker_id}.log"
            run_log.write_text("formal fixture completed\n", encoding="utf-8")
            (claim / "status").write_text(
                "\n".join(
                    (
                        "finished_at=2026-07-12T00:05:00+00:00",
                        "exit_code=0",
                        f"run_log={PRIVATE_FIXTURE_ROOT}/results_v2/gpu_rolling/logs/main/{worker_id}.log",
                    )
                )
                + "\n",
                encoding="utf-8",
            )


class V2GpuSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "results_v2" / "gpu_rolling"
        self.bootstrap_patcher = mock.patch(
            "v2_gpu_rolling._cluster_bootstrap", side_effect=fast_fixture_bootstrap
        )
        self.bootstrap_mock = self.bootstrap_patcher.start()
        self.fixture = SyntheticFormalRun(self.root)
        self.attestation_binding = {
            "artifact_role": "chains/evidence/gpu_step3_postfreeze_semantic_attestation.json",
            "sha256": digest("synthetic post-freeze attestation"),
            "schema_version": "upgrade-bench-v2/gpu-step3-post-freeze-semantic-attestation/1",
            "step3_manifest_sha256": self.fixture.step3_sha,
            "allowed_changed_file_count": 6,
            "machine_semantics_and_candidate_bytes_unchanged": True,
        }
        self.attestation_verifier = mock.patch(
            "summarize_v2_gpu_results._verify_postfreeze_attestation",
            return_value=self.attestation_binding,
        )
        self.attestation_mock = self.attestation_verifier.start()

    def tearDown(self):
        self.attestation_verifier.stop()
        self.bootstrap_patcher.stop()
        self.temporary.cleanup()

    def test_invalidated_run_namespace_is_never_promoted(self):
        write_json(self.root / "PILOT_INVALIDATED.json", {"status": "invalidated"})
        with self.assertRaisesRegex(ResultValidationError, "explicitly invalidated"):
            build_summary(self.root)

    def test_canonical_run_config_bytes_and_run_id_are_bound(self):
        self.fixture.run_config.write_bytes(self.fixture.run_config.read_bytes() + b"\n")
        with self.assertRaisesRegex(ResultValidationError, "canonical run config: byte hash"):
            build_summary(self.root)

        payload = json.loads(self.fixture.run_config.read_text(encoding="utf-8"))
        payload["run_id"] = "different-current-run"
        write_json(self.fixture.run_config, payload)
        current_hash = hashlib.sha256(self.fixture.run_config.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ResultValidationError, "invalid or stale run_id"):
            _validate_current_run_config(
                self.fixture.run_config,
                {"run_id": self.fixture.run_id, "run_config_sha256": current_hash},
            )

    def test_current_candidate_byte_mismatch_is_rejected(self):
        candidate = self.fixture._candidate_path("sheep", "a", "fold2")
        candidate.write_bytes(candidate.read_bytes() + b"\n")
        with self.assertRaisesRegex(ResultValidationError, "current candidate sha256 differs"):
            build_summary(self.root)

    def test_recorded_target_positive_count_is_recomputed(self):
        metric = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        payload = json.loads(metric.read_text(encoding="utf-8"))
        payload["target_candidate"]["positive_lanes"] = 0
        write_json(metric, payload)
        with self.assertRaisesRegex(
            ResultValidationError, "current candidate positive_lanes differs"
        ):
            build_summary(self.root)

    def test_same_count_wrong_score_identities_are_rejected(self):
        score = self.root / "scores" / "scores_sheep_track-a_kge.csv"
        raw = score.read_text(encoding="utf-8")
        self.assertIn("B,V,stage,", raw)
        score.write_text(raw.replace("B,V,stage,", "B,T,stage,", 1), encoding="utf-8")
        with self.assertRaisesRegex(
            ResultValidationError, "score identities do not exactly match"
        ):
            build_summary(self.root)

    def test_paired_metric_and_summary_tamper_is_rejected_by_recomputation(self):
        metric = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        payload = json.loads(metric.read_text(encoding="utf-8"))
        name = PRIMARY_METRICS["a"]
        payload["per_seed"][0][name] = float(payload["per_seed"][0][name]) - 0.125
        values = [float(row[name]) for row in payload["per_seed"]]
        payload["summary"][name] = {
            "mean": _mean(values),
            "std": _std(values),
            "n": len(values),
        }
        write_json(metric, payload)
        with self.assertRaisesRegex(ResultValidationError, "mechanically recomputed"):
            build_summary(self.root)

    def test_bootstrap_interval_tamper_is_rejected_by_recomputation(self):
        metric = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        payload = json.loads(metric.read_text(encoding="utf-8"))
        original = float(payload["cluster_bootstrap_by_seed"][0]["lower_95"])
        payload["cluster_bootstrap_by_seed"][0]["lower_95"] = max(0.0, original - 0.125)
        write_json(metric, payload)
        with self.assertRaisesRegex(ResultValidationError, "mechanically recomputed"):
            build_summary(self.root)

    def test_exact_successful_claim_inventory_and_test_injection(self):
        claims = self.root / "main_job_claims"
        extra = claims / "extra_kge.lock"
        extra.mkdir()
        with self.assertRaisesRegex(ResultValidationError, "exactly the 12"):
            build_summary(self.root)
        extra.rmdir()

        status = claims / "sheep_kge.lock" / "status"
        original = status.read_text(encoding="utf-8")
        status.write_text(original.replace("exit_code=0", "exit_code=9"), encoding="utf-8")
        with self.assertRaisesRegex(ResultValidationError, "exit_code=0"):
            build_summary(self.root)
        status.write_text(original, encoding="utf-8")

        injected = self.fixture.deployment_root / "private" / "injected-main-claims"
        injected.parent.mkdir(parents=True, exist_ok=True)
        claims.rename(injected)
        summary = build_summary(self.root, claims_root=injected)
        self.assertEqual(summary["complete_chain_family_jobs"], 12)

    def test_complete_run_is_sanitized_and_never_selects_a_main_champion(self):
        summary = build_summary(self.root)
        self.assertEqual(summary["schema_version"], SUMMARY_SCHEMA)
        self.assertEqual(summary["complete_chain_family_jobs"], 12)
        self.assertEqual(summary["complete_task_evaluations"], 36)
        self.assertEqual(len(summary["records"]), 36)
        self.assertEqual(len(summary["macro_summary"]), 6)
        self.assertEqual(
            summary["post_freeze_semantic_attestation"], self.attestation_binding
        )
        nbfnet = summary["nbfnet_source_evidence"]
        self.assertEqual(
            [row["role"] for row in nbfnet["private_receipts"]],
            list(NBFNET_RECEIPT_ROLES),
        )
        self.assertEqual(len(nbfnet["formal_main_gates"]), 2)
        self.assertFalse(
            nbfnet["claim_boundary"][
                "selection_prehash_or_contemporaneous_freeze_proved"
            ]
        )
        self.assertTrue(
            nbfnet["claim_boundary"]["main_read_only_pre_main_snapshot_verified"]
        )
        self.assertEqual(
            summary["metric_recomputation"]["source_sha256"],
            hashlib.sha256((ROOT / "src" / "v2_gpu_rolling.py").read_bytes()).hexdigest(),
        )
        self.assertTrue(
            summary["metric_recomputation"][
                "all_per_seed_metrics_and_summaries_recomputed"
            ]
        )
        self.attestation_mock.assert_called_with(
            self.fixture.deployment_root, self.fixture.run_id
        )
        self.assertFalse(summary["reporting_policy"]["main_test_champion_selected"])
        grouped = {
            (row["chain"], row["family"]): set()
            for row in summary["records"]
        }
        for row in summary["records"]:
            grouped[(row["chain"], row["family"])].add(row["track"])
        self.assertEqual(len(grouped), 12)
        self.assertTrue(all(tracks == set(TRACKS) for tracks in grouped.values()))

        rendered = render_json(summary).decode("utf-8")
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("private-user", rendered)
        self.assertNotIn("score_cache.root", rendered)
        csv_rows = list(csv.DictReader(render_csv(summary).decode("utf-8").splitlines()))
        self.assertEqual(len(csv_rows), 42)
        self.assertEqual(sum(row["scope"] == "macro" for row in csv_rows), 6)
        self.assertEqual(
            {row["post_freeze_semantic_attestation_sha256"] for row in csv_rows},
            {self.attestation_binding["sha256"]},
        )
        self.assertEqual(
            {row["metric_recomputation_source_sha256"] for row in csv_rows},
            {summary["metric_recomputation"]["source_sha256"]},
        )
        self.assertEqual(
            len({row["nbfnet_source_evidence_sha256"] for row in csv_rows}), 1
        )
        self.assertRegex(
            csv_rows[0]["nbfnet_source_evidence_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            {(row["track"], row["family"]) for row in csv_rows if row["scope"] == "macro"},
            {(track, family) for track in TRACKS for family in FAMILIES},
        )

    def test_incomplete_run_never_writes_public_outputs(self):
        missing = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        missing.unlink()
        json_out = Path(self.temporary.name) / "public" / "summary.json"
        csv_out = Path(self.temporary.name) / "public" / "summary.csv"
        with self.assertRaisesRegex(ResultValidationError, "exactly the 36"):
            write_outputs(self.root, json_out, csv_out)
        self.assertFalse(json_out.exists())
        self.assertFalse(csv_out.exists())

    def test_nbfnet_private_inventory_is_exact_and_receipts_are_hash_bound(self):
        private_root = self.root / "nbfnet_attestation"
        missing = private_root / "selection-host-a.json"
        missing.unlink()
        with self.assertRaisesRegex(ResultValidationError, "exactly eight evidence files"):
            build_summary(self.root)

    def test_nbfnet_private_run_tool_copy_is_exactly_hash_bound(self):
        tool = (
            self.fixture.deployment_root
            / "private"
            / "build_nbfnet_source_attestation_v2.py"
        )
        tool.write_bytes(tool.read_bytes() + b"\n")
        with self.assertRaisesRegex(ResultValidationError, "private formal-run tool copy"):
            build_summary(self.root)

    def test_nbfnet_private_receipt_and_public_projection_tamper_fail_closed(self):
        private = self.root / "nbfnet_attestation" / "selection-host-a.json"
        payload = json.loads(private.read_text(encoding="utf-8"))
        payload["observed_at_utc"] = "2026-07-12T00:00:31+00:00"
        private.write_bytes(nbfnet_attestation.render_json(payload))
        with self.assertRaisesRegex(ResultValidationError, "public.*projection"):
            build_summary(self.root)

    def test_nbfnet_formal_gate_must_precede_marker_and_bind_read_only_snapshot(self):
        gate = self.root / "nbfnet_attestation" / "formal_gate_main-host-a.json"
        original = json.loads(gate.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(original)
        tampered["source_snapshot"]["mode_read_only"] = False
        write_json(gate, tampered)
        with self.assertRaisesRegex(ResultValidationError, "source_snapshot binding"):
            build_summary(self.root)

        write_json(gate, original)
        tampered = copy.deepcopy(original)
        tampered["created_at_utc"] = "2026-07-12T00:00:44Z"
        write_json(gate, tampered)
        with self.assertRaisesRegex(ResultValidationError, "predates a private receipt"):
            build_summary(self.root)

        write_json(gate, original)
        tampered = copy.deepcopy(original)
        tampered["created_at_utc"] = "2026-07-12T00:01:01Z"
        write_json(gate, tampered)
        with self.assertRaisesRegex(ResultValidationError, "created after main started"):
            build_summary(self.root)

    def test_nbfnet_public_gate_rejects_selection_prehash_overclaim(self):
        summary = build_summary(self.root)
        summary["nbfnet_source_evidence"]["claim_boundary"][
            "selection_prehash_or_contemporaneous_freeze_proved"
        ] = True
        with self.assertRaisesRegex(ResultValidationError, "claim boundary"):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_nbfnet_public_inventory_is_exactly_six_files(self):
        summary = build_summary(self.root)
        extra = (
            self.fixture.deployment_root
            / "chains"
            / "evidence"
            / "nbfnet_unreviewed_extra.json"
        )
        write_json(extra, {"status": "PASS"})
        with self.assertRaisesRegex(ResultValidationError, "exactly six"):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_nbfnet_public_gate_rejects_private_path_leak_even_if_rehashed(self):
        summary = build_summary(self.root)
        public = (
            self.fixture.deployment_root
            / "chains"
            / "evidence"
            / "nbfnet_selection-host-a.public.json"
        )
        payload = json.loads(public.read_text(encoding="utf-8"))
        payload["claim_boundary"]["supported"] = PRIVATE_FIXTURE_ROOT
        public.write_bytes(nbfnet_attestation.render_json(payload))
        binding_row = summary["nbfnet_source_evidence"]["private_receipts"][0]
        binding_row["public_projection_sha256"] = hashlib.sha256(
            public.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(
            ResultValidationError, "absolute path|private host/user token"
        ):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_nbfnet_public_projection_cannot_overclaim_selection_prehash(self):
        summary = build_summary(self.root)
        public = (
            self.fixture.deployment_root
            / "chains"
            / "evidence"
            / "nbfnet_selection-host-a.public.json"
        )
        payload = json.loads(public.read_text(encoding="utf-8"))
        payload["claim_boundary"]["supported"] = (
            "The selection-time source was prehashed and frozen."
        )
        public.write_bytes(nbfnet_attestation.render_json(payload))
        summary["nbfnet_source_evidence"]["private_receipts"][0][
            "public_projection_sha256"
        ] = hashlib.sha256(public.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ResultValidationError, "claim boundary"):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_nbfnet_public_gate_rejects_runtime_identity_tamper(self):
        summary = build_summary(self.root)
        summary["nbfnet_source_evidence"]["runtime_identity"]["artifacts"][0][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ResultValidationError, "runtime identities differ"):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_nbfnet_public_gate_requires_two_actual_runtime_host_rows(self):
        summary = build_summary(self.root)
        comparison = (
            self.fixture.deployment_root
            / "chains"
            / "evidence"
            / "nbfnet_runtime_comparison.json"
        )
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        payload["hosts"] = []
        comparison.write_bytes(nbfnet_attestation.render_json(payload))
        digest_value = hashlib.sha256(comparison.read_bytes()).hexdigest()
        binding = summary["nbfnet_source_evidence"]["comparisons"]["runtime"]
        binding["private_artifact_sha256"] = digest_value
        binding["public_artifact_sha256"] = digest_value
        with self.assertRaisesRegex(ResultValidationError, "host runtime identities"):
            verify_nbfnet_public_binding(
                summary, root=self.fixture.deployment_root
            )

    def test_hand_edited_main_number_is_rejected(self):
        path = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["per_seed"][0][PRIMARY_METRICS["a"]] -= 0.1
        write_json(path, payload)
        with self.assertRaisesRegex(ResultValidationError, "mechanically recomputed"):
            build_summary(self.root)

    def test_selection_hash_mismatch_and_extra_champion_file_are_rejected(self):
        path = self.root / "metrics" / "metrics_sheep_track-a_kge.json"
        original = path.read_bytes()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selection_sha256"] = "0" * 64
        write_json(path, payload)
        with self.assertRaisesRegex(ResultValidationError, "selection_sha256"):
            build_summary(self.root)

        # Restore the fixture metric, then prove a hand-picked extra main result
        # cannot be silently ignored by the exact-set gate.
        path.write_bytes(original)
        extra = self.root / "metrics" / "metrics_handpicked_main_champion.json"
        write_json(extra, {"metric": 1.0})
        with self.assertRaisesRegex(ResultValidationError, "exactly the 36"):
            build_summary(self.root)

    def test_verify_output_is_byte_exact_and_detects_staleness(self):
        json_out = Path(self.temporary.name) / "public" / "summary.json"
        csv_out = Path(self.temporary.name) / "public" / "summary.csv"
        write_outputs(self.root, json_out, csv_out)
        verified = verify_outputs(self.root, json_out, csv_out)
        self.assertEqual(verified["complete_task_evaluations"], 36)
        csv_out.write_bytes(csv_out.read_bytes() + b"\n")
        with self.assertRaisesRegex(ResultValidationError, "stale or non-deterministic"):
            verify_outputs(self.root, json_out, csv_out)


if __name__ == "__main__":
    unittest.main()
