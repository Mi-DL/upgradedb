import argparse
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PROVENANCE_TESTS_ENABLED = (
    os.environ.get("UPGRADE_BENCH_PRIVATE_PROVENANCE_TESTS") == "1"
)
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "NBFNet-PyG"))

import v2_loco_transfer as loco  # noqa: E402
from v2_gpu_protocol import ProtocolError  # noqa: E402
from v2_loco_transfer import (  # noqa: E402
    CHAINS,
    KEYS,
    PROFILES,
    GraphBundle,
    _align_union_scores,
    _candidate_union,
    _code_provenance,
    _component_relative_artifact_path,
    _coverage_audit,
    _parser,
    _read_identities,
    _score_candidates,
    _assert_outputs_absent,
    _build_graph,
    _finalize_input_snapshot,
    _graph_contract,
    _raw_provenance,
    _relation_registry,
    _stable_json_hash,
    _stable_supervision_subset,
    _validate_run_contract,
    frozen_input_snapshot,
    verify_component,
)


class _FakeChain:
    def __init__(self, tiers):
        self._tiers = dict(tiers)

    def tiers(self):
        return dict(self._tiers)


class _FakeUniverse:
    CHAINS = {"x": _FakeChain({"s": 1, "t": 2})}


class V2LocoTransferTest(unittest.TestCase):
    def test_six_chains_match_current_registry_files(self):
        registry = {path.stem for path in (ROOT / "chains").glob("*.json")}
        self.assertEqual(set(CHAINS), registry)

    def test_directional_form_of_contributes_to_shared_loco_tier(self):
        import universe as U

        tiers = U.CHAINS["sheep"].tiers()
        stage_registry = loco._stage_registry(U)
        self.assertEqual(tiers["exp_live"], 0)
        self.assertEqual(tiers["exp_meat"], 1)
        self.assertEqual(
            stage_registry["sheep.exp_meat"], ("sheep", "exp_meat", 1)
        )

    def test_processing_tier_cycle_is_rejected(self):
        import universe as U

        cyclic = U.Chain(
            {
                "id": "cycle-test",
                "stages": {"stage_a": ["000001"], "stage_b": ["000002"]},
                "form_of": [["stage_a", "stage_b"], ["stage_b", "stage_a"]],
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "cycle in processing-tier DAG: stage_a -> stage_b -> stage_a",
        ):
            cyclic.tiers()

    def test_profiles_are_source_locked_and_smoke_is_not_paper_eligible(self):
        self.assertFalse(PROFILES["smoke-fixed-v1"]["formal_component_eligible"])
        self.assertTrue(PROFILES["formal-fixed-v1"]["formal_component_eligible"])
        self.assertIsNone(PROFILES["formal-fixed-v1"]["max_supervised_train_edges"])
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                ["evaluate", "--holdout", "sheep", "--epochs", "2"]
            )

    def test_artifacts_are_immutable(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "already-there.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "refusing to overwrite"):
                _assert_outputs_absent([path])

    def test_smoke_supervision_cap_is_deterministic_and_context_independent(self):
        rows = [(f"H{i}", "exp_tier1", f"T{i}") for i in range(20)]
        forward = _stable_supervision_subset(rows, 7)
        reverse_input = _stable_supervision_subset(list(reversed(rows)), 7)
        self.assertEqual(forward, reverse_input)
        self.assertEqual(len(forward), 7)
        self.assertEqual(_stable_supervision_subset(rows, None), rows)

    def test_cache_provenance_never_opens_or_hashes_raw_archive(self):
        import baci_filtered_cache

        original = loco.sha256_file
        original_cache = baci_filtered_cache.BaciFilteredCache
        original_env = os.environ.get("VCU_BACI_CACHE")
        calls = []
        source_sha = "1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a"
        country_path = ROOT / "requirements" / "baci_country_codes_V202401b.csv"
        country_sha = original(country_path)

        def controlled_hash(path):
            path = Path(path)
            calls.append(path)
            if path.name == "BACI_HS92_V202401b.zip":
                raise AssertionError("strict provenance attempted to hash the raw BACI ZIP")
            return original(path)

        with tempfile.TemporaryDirectory() as raw:
            cache_dir = Path(raw) / "private" / "cache"
            cache_dir.mkdir(parents=True)
            source = {
                "dataset": "CEPII BACI HS92 V202401b",
                "archive_name": "BACI_HS92_V202401b.zip",
                "archive_bytes": 2450783074,
                "archive_sha256": source_sha,
                "country_codes_member": {
                    "name": "country_codes_V202401b.csv",
                    "bytes": country_path.stat().st_size,
                    "sha256": country_sha,
                    "uncompressed_bytes": country_path.stat().st_size,
                    "crc32": "00000000",
                },
                "trade_members": [],
            }
            manifest = {
                "schema_version": "upgrade-bench/private-baci-filtered-cache/1",
                "visibility": "private-never-publish",
                "years": [
                    1998, 1999, 2000, 2001, 2002,
                    2008, 2009, 2010, 2011, 2012,
                    2018, 2019, 2020, 2021, 2022,
                ],
                "source": source,
                "registry": {},
                "files": [],
                "totals": {"files": 0, "rows": 0, "bytes": 0},
            }
            (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            class FakeCache:
                def __init__(self, path, **_kwargs):
                    self.cache_dir = Path(path).resolve()
                    self.manifest = manifest

            os.environ["VCU_BACI_CACHE"] = str(cache_dir)
            baci_filtered_cache.BaciFilteredCache = FakeCache
            loco.sha256_file = controlled_hash
            try:
                provenance = _raw_provenance()
            finally:
                loco.sha256_file = original
                baci_filtered_cache.BaciFilteredCache = original_cache
                if original_env is None:
                    os.environ.pop("VCU_BACI_CACHE", None)
                else:
                    os.environ["VCU_BACI_CACHE"] = original_env
        self.assertEqual(provenance["source"]["archive_sha256"], source_sha)
        self.assertFalse(provenance["raw_archive_opened_or_hashed"])
        self.assertTrue(provenance["source_attestation"]["path"].endswith("raw_source_attestation.json"))
        self.assertFalse(any(path.name in {"BACI_HS92_V202401b.zip", "raw_label_audit.json"} for path in calls))

    @unittest.skipUnless(
        PRIVATE_PROVENANCE_TESTS_ENABLED,
        "vendored NBFNet and formal-controller provenance are private staging inputs",
    )
    def test_code_provenance_recursively_binds_all_vendored_nbfnet_sources(self):
        provenance = _code_provenance()
        vendored_root = ROOT / "third_party" / "NBFNet-PyG"
        suffixes = (".py", ".cpp", ".cu", ".h", ".cuh") + tuple(
            loco.importlib.machinery.EXTENSION_SUFFIXES
        ) + (".so", ".pyd", ".dll", ".dylib", ".pyc")
        expected = {
            path.relative_to(ROOT).as_posix()
            for path in vendored_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name.lower().endswith(suffixes)
        }
        observed = {
            path for path in provenance if path.startswith("third_party/NBFNet-PyG/")
        }
        self.assertEqual(observed, expected)
        self.assertIn("third_party/NBFNet-PyG/nbfnet/rspmm/rspmm.py", observed)
        self.assertIn("third_party/NBFNet-PyG/nbfnet/rspmm/source/rspmm.cpp", observed)
        self.assertIn("third_party/NBFNet-PyG/nbfnet/rspmm/source/rspmm.cu", observed)
        self.assertIn("src/task_features.py", provenance)
        self.assertIn("src/split.py", provenance)
        self.assertIn("tools/v2_loco_formal.py", provenance)
        expected_local = {
            "tools/v2_loco_transfer.py",
            "tools/v2_loco_formal.py",
            "src/gap_discovery.py",
            "src/benchmark.py",
            "src/temporal_backtest.py",
            "src/universe.py",
            "src/baci_filtered_cache.py",
            "src/window_aggregation.py",
            "src/task_features.py",
            "src/split.py",
            "src/v2_gpu_rolling.py",
            "src/v2_gpu_protocol.py",
        }
        self.assertEqual(
            {path for path in provenance if not path.startswith("third_party/")},
            expected_local,
        )
        self.assertEqual(provenance, _code_provenance())

    def test_canonical_path_precedence_and_shadow_rejection(self):
        loco._set_canonical_sys_path()
        src = str((ROOT / "src").resolve())
        tools = str((ROOT / "tools").resolve())
        vendor = str((ROOT / "third_party" / "NBFNet-PyG").resolve())
        self.assertLess(sys.path.index(src), sys.path.index(tools))
        self.assertLess(sys.path.index(tools), sys.path.index(vendor))

        original = sys.modules.get("v2_gpu_rolling")
        sys.modules["v2_gpu_rolling"] = types.SimpleNamespace(
            __name__="v2_gpu_rolling",
            __file__=str(ROOT / "third_party" / "NBFNet-PyG" / "v2_gpu_rolling.py"),
        )
        try:
            with self.assertRaisesRegex(ProtocolError, "shadowed"):
                loco._import_canonical_module(
                    "v2_gpu_rolling", ROOT / "src" / "v2_gpu_rolling.py"
                )
        finally:
            if original is None:
                sys.modules.pop("v2_gpu_rolling", None)
            else:
                sys.modules["v2_gpu_rolling"] = original

    def test_nbfnet_path_override_is_fail_closed(self):
        original = os.environ.get("NBFNET_PATH")
        try:
            os.environ["NBFNET_PATH"] = str(ROOT / "src")
            with self.assertRaisesRegex(ProtocolError, "may not redirect"):
                loco._validate_nbfnet_path_environment()
            os.environ["NBFNET_PATH"] = str(ROOT / "third_party" / "NBFNet-PyG")
            loco._validate_nbfnet_path_environment()
        finally:
            if original is None:
                os.environ.pop("NBFNET_PATH", None)
            else:
                os.environ["NBFNET_PATH"] = original

    def test_repository_module_cannot_claim_scatter_fallback_exemption(self):
        original = sys.modules.get("torch_scatter")
        fake = types.ModuleType("torch_scatter")
        fake.__file__ = str(ROOT / "src" / "torch_scatter.py")
        fake.__upgrade_bench_fallback__ = True
        sys.modules["torch_scatter"] = fake
        try:
            with self.assertRaisesRegex(ProtocolError, "unauthorized"):
                loco._require_external_module("torch_scatter", optional=True)
        finally:
            if original is None:
                sys.modules.pop("torch_scatter", None)
            else:
                sys.modules["torch_scatter"] = original

    @unittest.skipUnless(
        PRIVATE_PROVENANCE_TESTS_ENABLED,
        "formal runtime-origin checks require private controller provenance",
    )
    def test_frozen_input_snapshot_is_six_chain_and_outcome_blind(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate_root = Path(raw) / "candidates"
            candidate_root.mkdir()
            for chain in CHAINS:
                for track, stem in (("a", "candidates"), ("b", "candidates_firsttime")):
                    frame = pd.DataFrame(
                        {
                            "i_iso": ["AAA"],
                            "j_iso": ["BBB"],
                            "stage": [f"{chain}.s"],
                            "y": [1],
                            "size": [2.0],
                            "lateval": [3.0],
                            "aggregation": ["calendar_mean"],
                            "early_window": ["2008-2012"],
                            "late_window": ["2018-2022"],
                        }
                    )
                    frame.to_csv(candidate_root / f"{stem}_{chain}.csv", index=False)

            requested = []
            original_read_csv = pd.read_csv
            original_raw = loco._raw_provenance

            def recording_read_csv(*args, **kwargs):
                requested.extend(kwargs.get("usecols", []))
                return original_read_csv(*args, **kwargs)

            pd.read_csv = recording_read_csv
            loco._raw_provenance = lambda: {"cache": "strict", "raw_opened": False}
            try:
                snapshot = frozen_input_snapshot(candidate_root, fold="main")
            finally:
                pd.read_csv = original_read_csv
                loco._raw_provenance = original_raw
        self.assertEqual(snapshot["chains"], list(CHAINS))
        self.assertEqual(set(snapshot["candidate_identities"]), set(CHAINS))
        self.assertTrue(
            all(set(records) == {"A", "B"} for records in snapshot["candidate_identities"].values())
        )
        self.assertTrue({"y", "size", "lateval"}.isdisjoint(requested))
        self.assertEqual(snapshot["raw_baci"], {"cache": "strict", "raw_opened": False})

    def test_formal_relative_artifacts_never_prefer_repository_root(self):
        with tempfile.TemporaryDirectory() as raw:
            component_dir = Path(raw) / "component"
            component_dir.mkdir()
            component = component_dir / "component.json"
            local = component_dir / "README.md"
            local.write_text("component-local", encoding="utf-8")
            self.assertTrue((ROOT / "README.md").is_file())
            self.assertEqual(
                _component_relative_artifact_path("README.md", component), local.resolve()
            )
            with self.assertRaisesRegex(ProtocolError, "escapes"):
                _component_relative_artifact_path("../README.md", component)

    def test_supplies_demands_do_not_expand_stage_country_vocabulary(self):
        import universe as U

        relations, _ = _relation_registry(U)
        product = U.CHAINS["sheep"].all_hs[0]
        early = pd.DataFrame(
            [{"i_iso": "AAA", "j_iso": "BBB", "stage": "sheep.exp_live"}]
        )
        early_hs6 = pd.DataFrame(
            [
                {"i_iso": "AAA", "j_iso": "BBB", "k": product},
                {"i_iso": "OUT", "j_iso": "SIDE", "k": product},
            ]
        )
        graph = _build_graph(
            U=U,
            early=early,
            early_hs6=early_hs6,
            chains_subset=["sheep"],
            relation_to_id=relations,
            device="cpu",
        )
        self.assertIn("AAA", graph.entity_to_id)
        self.assertIn("BBB", graph.entity_to_id)
        self.assertNotIn("OUT", graph.entity_to_id)
        self.assertNotIn("SIDE", graph.entity_to_id)
        self.assertEqual(graph.provenance["country_vocabulary_size"], 2)

    def test_real_six_chain_registry_excludes_target_stages_and_hs6(self):
        import universe as U

        relations, _ = _relation_registry(U)
        early_rows = []
        hs6_rows = []
        for index, chain in enumerate(CHAINS):
            origin, destination = f"O{index}", f"D{index}"
            registry = U.CHAINS[chain]
            for stage in registry.stages:
                early_rows.append(
                    {
                        "i_iso": origin,
                        "j_iso": destination,
                        "stage": f"{chain}.{stage}",
                    }
                )
            for product in registry.all_hs:
                hs6_rows.append(
                    {"i_iso": origin, "j_iso": destination, "k": product}
                )
        target = "sheep"
        train = _build_graph(
            U=U,
            early=pd.DataFrame(early_rows),
            early_hs6=pd.DataFrame(hs6_rows),
            chains_subset=sorted(set(CHAINS) - {target}),
            relation_to_id=relations,
            device="cpu",
        )
        self.assertEqual(set(train.provenance["chains"]), set(CHAINS) - {target})
        self.assertFalse(
            any(stage.startswith(f"{target}.") for stage in train.source_namespaced_stages)
        )
        self.assertTrue(set(U.CHAINS[target].all_hs).isdisjoint(train.registered_hs6))
        self.assertTrue(set(U.CHAINS[target].all_hs).isdisjoint(train.entity_to_id))

    def test_tier_protocol_only_allows_matched_in_domain_drop(self):
        contract = _graph_contract()
        self.assertTrue(contract["independent_protocol"])
        self.assertIn("loco versus in_domain", contract["permitted_transfer_drop"])
        self.assertIn("never compare directly", contract["forbidden_transfer_drop"])

    def test_formal_component_requires_freeze_binding_and_exact_path(self):
        good = argparse.Namespace(
            mode="loco",
            profile="formal-fixed-v1",
            fold="main",
            run_id="run",
            config_sha256="a" * 64,
            freeze_sha256="b" * 64,
            component_output=Path(
                "results_v2/loco_formal/components/sheep/loco/seed_0/component.json"
            ),
            formal_manifest=Path("results_v2/loco_formal/frozen_manifest.json"),
            holdout="sheep",
            seed=0,
        )
        original = loco._formal_authorization
        loco._formal_authorization = lambda _args: {"authorized": "test"}
        try:
            _validate_run_contract(good)
        finally:
            loco._formal_authorization = original
        missing_manifest = argparse.Namespace(**vars(good))
        missing_manifest.formal_manifest = None
        with self.assertRaisesRegex(ProtocolError, "formal-manifest"):
            _validate_run_contract(missing_manifest)
        bad = argparse.Namespace(**vars(good))
        bad.component_output = Path("results_v2/run/component.json")
        with self.assertRaisesRegex(ProtocolError, "must end with"):
            _validate_run_contract(bad)

    def test_formal_evaluation_rejects_missing_main_start_marker_before_prepare(self):
        args = argparse.Namespace(
            mode="loco",
            profile="formal-fixed-v1",
            fold="main",
            run_id="run",
            config_sha256="a" * 64,
            freeze_sha256="b" * 64,
            component_output=Path(
                "results_v2/loco_formal/components/sheep/loco/seed_0/component.json"
            ),
            formal_manifest=Path("results_v2/loco_formal/frozen_manifest.json"),
            holdout="sheep",
            seed=0,
        )
        canonical = (ROOT / args.formal_manifest).resolve()

        def reject_marker(_manifest_path, _manifest):
            raise ProtocolError("main start marker is missing")

        fake_formal = types.SimpleNamespace(
            CANONICAL_MANIFEST=canonical,
            verify_freeze=lambda _path: {},
            _verify_main_marker=reject_marker,
        )
        original = loco._import_canonical_module
        loco._import_canonical_module = lambda name, _path: (
            fake_formal if name == "v2_loco_formal" else original(name, _path)
        )
        try:
            with self.assertRaisesRegex(ProtocolError, "marker is missing"):
                _validate_run_contract(args)
        finally:
            loco._import_canonical_module = original

    def test_formal_authorization_is_bound_into_score_rows(self):
        args = argparse.Namespace(
            mode="loco",
            profile="formal-fixed-v1",
            fold="main",
            run_id="run",
            config_sha256="a" * 64,
            freeze_sha256="b" * 64,
            component_output=Path(
                "results_v2/loco_formal/components/sheep/loco/seed_0/component.json"
            ),
            formal_manifest=Path("results_v2/loco_formal/frozen_manifest.json"),
            holdout="sheep",
            seed=0,
        )
        canonical = (ROOT / args.formal_manifest).resolve()
        manifest = {
            "run_id": args.run_id,
            "config_sha256": args.config_sha256,
            "freeze_sha256": args.freeze_sha256,
            "expected_components": [{"component_id": "sheep|loco|seed0"}],
        }
        marker = {
            "marker_sha256": "c" * 64,
            "global_claim_file_sha256": "d" * 64,
            "global_claim_sha256": "e" * 64,
        }
        fake_formal = types.SimpleNamespace(
            CANONICAL_MANIFEST=canonical,
            verify_freeze=lambda _path: manifest,
            _verify_main_marker=lambda _path, _manifest: marker,
        )
        original_import = loco._import_canonical_module
        original_hash = loco.sha256_file
        loco._import_canonical_module = lambda name, _path: (
            fake_formal if name == "v2_loco_formal" else original_import(name, _path)
        )
        loco.sha256_file = lambda path: (
            "f" * 64 if Path(path).name == "frozen_manifest.json" else "1" * 64
        )
        try:
            authorization = loco._formal_authorization(args)
        finally:
            loco._import_canonical_module = original_import
            loco.sha256_file = original_hash

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "score_A.csv"
            identities = pd.DataFrame(
                {"i_iso": ["AAA"], "j_iso": ["BBB"], "stage": ["s"]}
            )
            loco._write_scores(
                path,
                identities,
                np.array([0.5]),
                seed=0,
                profile=args.profile,
                mode=args.mode,
                run_id=args.run_id,
                config_sha256=args.config_sha256,
                freeze_sha256=args.freeze_sha256,
                formal_manifest_file_sha256=authorization["formal_manifest_file_sha256"],
                main_start_marker_file_sha256=authorization[
                    "main_start_marker_file_sha256"
                ],
                main_start_marker_sha256=authorization["main_start_marker_sha256"],
            )
            score = pd.read_csv(path, dtype=str)
        self.assertEqual(set(score["formal_manifest_file_sha256"]), {"f" * 64})
        self.assertEqual(set(score["main_start_marker_file_sha256"]), {"1" * 64})
        self.assertEqual(set(score["main_start_marker_sha256"]), {"c" * 64})

    def test_score_csv_round_trips_exact_float64_for_scoring_audit(self):
        """Persisted scores must reproduce the pre-outcome vector byte-for-byte."""
        identities = pd.DataFrame(
            {
                "i_iso": ["AAA", "AAA", "CCC", "DDD", "EEE"],
                "j_iso": ["BBB", "CCC", "DDD", "EEE", "FFF"],
                "stage": ["s", "s", "t", "t", "u"],
            }
        )
        # The first value is a real failure-case decimal: Pandas 3's default
        # parser reads it one ULP away, while round_trip recovers it exactly.
        scores = np.array(
            [
                -3.6373486518859863,
                -2.0438144207000732,
                np.nextafter(1.0, 2.0),
                -0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "score_A.csv"
            loco._write_scores(
                path,
                identities,
                scores,
                seed=0,
                profile="formal-fixed-v1",
                mode="loco",
                run_id="run",
                config_sha256="a" * 64,
                freeze_sha256="b" * 64,
                formal_manifest_file_sha256="c" * 64,
                main_start_marker_file_sha256="d" * 64,
                main_start_marker_sha256="e" * 64,
            )
            restored = pd.read_csv(
                path,
                dtype={"seed_0": np.float64},
                float_precision="round_trip",
            )["seed_0"].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(restored.view(np.uint64), scores.view(np.uint64))
        self.assertEqual(
            hashlib.sha256(restored.astype("<f8").tobytes()).hexdigest(),
            hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest(),
        )

    def test_snapshot_change_is_fail_closed(self):
        original = loco._capture_input_snapshot
        first = {"x": 1}
        loco._capture_input_snapshot = lambda _args: ({"x": 2}, None, None, None)
        try:
            with self.assertRaisesRegex(ProtocolError, "changed during execution"):
                _finalize_input_snapshot(object(), first)
        finally:
            loco._capture_input_snapshot = original

    def test_component_verifier_rejects_score_hash_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identities = pd.DataFrame(
                {"i_iso": ["AAA"], "j_iso": ["BBB"], "stage": ["exp_meat"]}
            )
            for task in ("A", "B"):
                frame = identities.copy()
                frame["y"] = [1]
                frame["size"] = [1.0]
                frame["lateval"] = [2.0]
                frame["aggregation"] = ["calendar_mean"]
                frame["early_window"] = ["2008-2012"]
                frame["late_window"] = ["2018-2022"]
                frame.to_csv(root / f"candidate_{task}.csv", index=False)
            frozen_inputs = {
                "candidate_identities": {
                    "sheep": {
                        "A": {
                            "path": str(root / "candidate_A.csv"),
                            "identity_sha256": "id-a",
                        },
                        "B": {
                            "path": str(root / "candidate_B.csv"),
                            "identity_sha256": "id-b",
                        },
                    }
                }
            }
            frozen_inputs_sha256 = _stable_json_hash(frozen_inputs)
            snapshot = {
                "code_sha256": {"runner": "a" * 64},
                "chain_registry_sha256": {"chain": "b" * 64},
                "raw_baci": {"sha256": "c" * 64},
                "dependency_versions": {"python": "test"},
                "module_origins": {"torch": "test"},
                "frozen_input_snapshot": frozen_inputs,
                "frozen_input_snapshot_sha256": frozen_inputs_sha256,
            }
            snapshot_sha = _stable_json_hash(snapshot)
            coverage = {
                "A": {"candidate_identity_sha256": "id-a"},
                "B": {"candidate_identity_sha256": "id-b"},
                "union": {"coverage_fraction": 1.0},
            }
            component = root / "component.json"
            payload = {
                "schema_version": loco.COMPONENT_SCHEMA,
                "protocol": loco.PROTOCOL,
                "command": "evaluate",
                "status": "SMOKE_COMPLETE",
                "mode": "loco",
                "chain": "sheep",
                "holdout_chain": "sheep",
                "fold": "main",
                "aggregation": "calendar_mean",
                "seed": 0,
                "component_id": "sheep|loco|seed0",
                "run_id": "UNBOUND",
                "config_sha256": "UNBOUND",
                "freeze_sha256": "UNBOUND",
                "main_outcomes_used_for_training_or_selection": False,
                "paper_eligible": False,
                "formal_component_eligible": False,
                "profile_name": "smoke-fixed-v1",
                "profile": PROFILES["smoke-fixed-v1"],
                "profile_sha256": _stable_json_hash(PROFILES["smoke-fixed-v1"]),
                "graph_contract": _graph_contract(),
                "graph_contract_sha256": _stable_json_hash(_graph_contract()),
                "frozen_input_snapshot_sha256": frozen_inputs_sha256,
                "candidate_inputs": {
                    task: {
                        "path": str(root / f"candidate_{task}.csv"),
                        "identity_sha256": coverage[task]["candidate_identity_sha256"],
                        "full_file_sha256_computed_after_scoring": loco.sha256_file(
                            root / f"candidate_{task}.csv"
                        ),
                    }
                    for task in ("A", "B")
                },
                "immutability_snapshot": {
                    "start": snapshot,
                    "end": snapshot,
                    "start_sha256": snapshot_sha,
                    "end_sha256": snapshot_sha,
                    "unchanged": True,
                },
                "code_sha256": snapshot["code_sha256"],
                "chain_registry_sha256": snapshot["chain_registry_sha256"],
                "raw_baci": snapshot["raw_baci"],
                "dependency_versions": snapshot["dependency_versions"],
                "module_origins": snapshot["module_origins"],
                "train_graph": {},
                "target_inference_graph": {},
                "target_exclusion_audit": {},
                "coverage": coverage,
                "score_artifacts": {
                    "A": {"path": str(root / "missing_A.csv"), "sha256": "d" * 64, "n_rows": 1},
                    "B": {"path": str(root / "missing_B.csv"), "sha256": "e" * 64, "n_rows": 1},
                },
                "scoring": {},
                "metrics": {},
            }
            original = loco._prepare_problem
            loco._prepare_problem = lambda _args, device: {
                "input_snapshot_start": snapshot,
                "train_graph": GraphBundle(None, {}, {}, {}),
                "inference_graph": GraphBundle(None, {}, {}, {}),
                "exclusion_audit": {},
                "coverage": coverage,
                "identities": {"A": identities, "B": identities},
                "union": identities,
            }
            try:
                payload["frozen_input_snapshot_sha256"] = "0" * 64
                component.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ProtocolError, "frozen input snapshot"):
                    verify_component(component)
                payload["frozen_input_snapshot_sha256"] = frozen_inputs_sha256
                component.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ProtocolError, "score artifact hash mismatch"):
                    verify_component(component)
            finally:
                loco._prepare_problem = original

    def test_component_verifier_never_accepts_component_as_paper_eligible(self):
        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw) / "component.json"
            component.write_text(
                json.dumps(
                    {
                        "schema_version": loco.COMPONENT_SCHEMA,
                        "protocol": loco.PROTOCOL,
                        "command": "evaluate",
                        "mode": "loco",
                        "chain": "sheep",
                        "holdout_chain": "sheep",
                        "seed": 0,
                        "component_id": "sheep|loco|seed0",
                        "aggregation": "calendar_mean",
                        "main_outcomes_used_for_training_or_selection": False,
                        "paper_eligible": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "must never be paper-eligible"):
                verify_component(component)

    def test_formal_component_output_is_publish_location_independent(self):
        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw) / "component.json"
            payload = {
                "schema_version": loco.COMPONENT_SCHEMA,
                "protocol": loco.PROTOCOL,
                "command": "evaluate",
                "status": "FIXED_PROFILE_COMPONENT_COMPLETE",
                "mode": "loco",
                "chain": "sheep",
                "holdout_chain": "sheep",
                "fold": "main",
                "aggregation": "calendar_mean",
                "seed": 0,
                "component_id": "sheep|loco|seed0",
                "run_id": "run",
                "config_sha256": "a" * 64,
                "freeze_sha256": "b" * 64,
                "main_outcomes_used_for_training_or_selection": False,
                "paper_eligible": False,
                "formal_component_eligible": True,
                "profile_name": "formal-fixed-v1",
                "profile": PROFILES["formal-fixed-v1"],
                "profile_sha256": _stable_json_hash(PROFILES["formal-fixed-v1"]),
                "graph_contract": _graph_contract(),
                "graph_contract_sha256": _stable_json_hash(_graph_contract()),
                "component_output": "attempts/attempt_1/component.json",
            }
            component.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "canonical basename"):
                verify_component(component)

    def test_identity_reader_never_requests_outcome_columns(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.csv"
            pd.DataFrame(
                {
                    "i_iso": ["A", "A"],
                    "j_iso": ["B", "C"],
                    "stage": ["s", "s"],
                    "y": [1, 0],
                    "size": [1.0, 2.0],
                    "lateval": [3.0, 0.0],
                    "aggregation": ["calendar_mean"] * 2,
                    "early_window": ["2008-2012"] * 2,
                    "late_window": ["2018-2022"] * 2,
                }
            ).to_csv(path, index=False)
            requested = []
            original = pd.read_csv

            def recording_read_csv(*args, **kwargs):
                requested.extend(kwargs.get("usecols", []))
                return original(*args, **kwargs)

            pd.read_csv = recording_read_csv
            try:
                identities = _read_identities(path, "main")
            finally:
                pd.read_csv = original
            self.assertEqual(list(identities.columns), list(KEYS))
            self.assertTrue({"y", "size", "lateval"}.isdisjoint(requested))

    def test_union_alignment_preserves_each_task_and_requires_full_coverage(self):
        a = pd.DataFrame(
            {"i_iso": ["A", "A"], "j_iso": ["B", "C"], "stage": ["s", "s"]}
        )
        b = pd.DataFrame(
            {"i_iso": ["D", "D"], "j_iso": ["B", "C"], "stage": ["t", "t"]}
        )
        union = _candidate_union(a, b)
        scores = np.arange(len(union), dtype=float) + 0.5
        aligned = _align_union_scores(union, scores, b)
        expected_locations = pd.MultiIndex.from_frame(union[list(KEYS)]).get_indexer(
            pd.MultiIndex.from_frame(b[list(KEYS)])
        )
        np.testing.assert_array_equal(aligned, scores[expected_locations])

    def test_coverage_audit_rejects_any_candidate_filtering(self):
        identities = pd.DataFrame(
            {"i_iso": ["A", "A"], "j_iso": ["B", "C"], "stage": ["s", "s"]}
        )
        graph = GraphBundle(
            data=None,
            entity_to_id={"A": 0, "B": 1},
            relation_to_id={"exp_tier1": 0},
            provenance={},
        )
        with self.assertRaisesRegex(ProtocolError, "not 100%"):
            _coverage_audit(identities, chain="x", U=_FakeUniverse, graph=graph)
        graph.entity_to_id["C"] = 2
        audit = _coverage_audit(identities, chain="x", U=_FakeUniverse, graph=graph)
        self.assertEqual(audit["coverage_fraction"], 1.0)
        self.assertEqual(audit["n_covered"], 2)

    @unittest.skipUnless(
        PRIVATE_PROVENANCE_TESTS_ENABLED,
        "candidate-scoring integration requires the private vendored NBFNet tree",
    )
    def test_candidate_scoring_fills_every_row_with_shared_tier_queries(self):
        import torch
        from torch_geometric.data import Data
        from v2_loco_transfer import _install_portable_scatter

        _install_portable_scatter()
        entities = {"A": 0, "B": 1, "C": 2}
        # A single context edge plus inverse is sufficient for all_negative.
        data = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_type=torch.tensor([0, 1], dtype=torch.long),
            num_nodes=3,
            target_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            target_edge_type=torch.tensor([0], dtype=torch.long),
            num_relations=2,
        )
        graph = GraphBundle(data, entities, {"exp_tier1": 0}, {})
        identities = pd.DataFrame(
            {"i_iso": ["A", "A"], "j_iso": ["B", "C"], "stage": ["s", "s"]}
        )

        class TailIdModel:
            def eval(self):
                return self

            def __call__(self, _data, batch):
                return batch[..., 1].float()

        scores, audit = _score_candidates(
            TailIdModel(),
            graph,
            identities,
            chain="x",
            U=_FakeUniverse,
            device="cpu",
            query_batch_size=4,
        )
        np.testing.assert_array_equal(scores, np.array([1.0, 2.0]))
        self.assertEqual(audit["coverage_fraction"], 1.0)
        self.assertEqual(audit["n_unique_exporter_tier_queries"], 1)

    def test_current_task_metrics_use_lane_entry_and_conditional_units(self):
        from v2_gpu_rolling import _ranking_metrics

        identities = pd.DataFrame(
            {
                "i_iso": ["A", "A", "B", "B"],
                "j_iso": ["X", "Y", "X", "Y"],
                "stage": ["s"] * 4,
            }
        )
        labels = pd.DataFrame(
            {"y": [0, 1, 0, 0], "size": [1.0] * 4, "lateval": [0.0, 2.0, 0.0, 0.0]}
        )
        score = np.array([0.2, 0.9, 0.7, 0.1])
        a = _ranking_metrics("a", identities, labels, score)
        b1 = _ranking_metrics("b1", identities, labels, score)
        b2 = _ranking_metrics("b2", identities, labels, score)
        self.assertIn("lane_average_precision", a)
        self.assertEqual(b1["entry_average_precision"], 1.0)
        self.assertEqual(b2["conditional_recall_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
