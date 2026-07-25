#!/usr/bin/env python3
"""Read-only, CPU-only smoke checks for the frozen release artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_PACKAGE = ROOT / "benchmark" / "upgrade-bench-v2"


def fail(message: str) -> None:
    raise AssertionError(message)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_v2_package(*, full: bool) -> None:
    loader = load_module("upgrade_bench_v2_loader", V2_PACKAGE / "loader.py")
    previous_loader = sys.modules.get("loader")
    sys.modules["loader"] = loader
    try:
        evaluator = load_module("upgrade_bench_v2_eval", V2_PACKAGE / "eval.py")
    finally:
        if previous_loader is None:
            sys.modules.pop("loader", None)
        else:
            sys.modules["loader"] = previous_loader

    if loader.filename_for("B1", "sheep", "fold2") != "entries_firsttime_sheep_fold2.csv":
        fail("v2 standalone loader filename contract changed")
    if not full:
        print("v2 standalone package import/API OK (external payload not required)")
        return

    data_root = loader.resolve_data_root()
    counts = 0
    sample = None
    for snapshot in loader.SNAPSHOTS:
        for chain in loader.CHAINS:
            for track in loader.TRACKS:
                frame = loader.load(track, chain, snapshot, data_root=data_root, validate=True)
                counts += len(frame)
                if sample is None and track == "A" and snapshot == "main":
                    sample = frame
    if sample is None:
        fail("v2 standalone package did not load a main Track-A sample")
    score = evaluator.builtin_scores(sample, "A", "size")
    metrics = evaluator.evaluate("A", sample, score, budgets=(10,))
    if metrics.get("rows") != len(sample):
        fail("v2 standalone evaluator returned an inconsistent row count")
    print(f"v2 standalone loader/evaluator OK ({counts:,} validated table rows)")


def check_gbdt_baseline(module, *, full: bool) -> None:
    """Verify the public GBDT pair with profile-appropriate source checks.

    A clean repository intentionally lacks the externally distributed candidate
    tables.  In that profile the public manifest/receipt bind source bytes, while
    this check still verifies canonical JSON/CSV, schema, aggregates, and privacy.
    The full profile additionally re-hashes all 24 historical/main candidate
    tables through the runner's complete verifier.
    """

    json_path = module.DEFAULT_JSON.resolve()
    csv_path = module.DEFAULT_CSV.resolve()
    if full:
        module.verify_existing_output(json_path, csv_path)
        print("formal GBDT reference artifact OK (full source verification)")
        return

    payload = module._strict_json_load(json_path)
    if json_path.read_bytes() != module._strict_json_bytes(payload):
        fail("GBDT result JSON is not canonical")
    module.validate_payload(payload, verify_sources=False)
    if csv_path.read_bytes() != module._csv_bytes(payload):
        fail("GBDT result CSV is stale or noncanonical")
    print(
        "formal GBDT reference artifact OK "
        "(repository structure; external candidate hashes deferred to full profile)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-manifest",
        action="store_true",
        help="run functional checks without hashes (useful before the final concurrent freeze)",
    )
    ap.add_argument(
        "--profile",
        choices=("repository", "full"),
        default="full",
        help=(
            "repository permits externally distributed v2 tables to be absent; "
            "full requires and validates every frozen data payload"
        ),
    )
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import sklearn
    except ImportError as exc:
        print(
            "missing minimal dependency; run: "
            "python -m pip install -r benchmark/upgrade-bench-v2/requirements.txt scikit-learn",
            file=sys.stderr,
        )
        print(f"import error: {exc}", file=sys.stderr)
        return 2

    try:
        import artifact_bundles
        import audit_chain_registry
        import build_gpu_step3_postfreeze_attestation as gpu_postfreeze
        import generate_v2_benchmark_profile
        import public_release_audit
        import registry_human_review_receipt
        import release_manifest
        import resolve_v2_invalidation
        import summarize_v2_gpu_results as gpu_summary_verifier

        print(
            f"dependencies: numpy={np.__version__} pandas={pd.__version__} "
            f"sklearn={sklearn.__version__}; profile={args.profile}"
        )
        # Public smoke must not require historical/internal nested manifests,
        # which are intentionally absent from a clean public export.
        if not args.skip_manifest and not release_manifest.verify_all(scope="release", strict=True):
            return 1
        if not artifact_bundles.verify_index(
            allow_missing=args.profile == "repository"
        ):
            return 1
        if not public_release_audit.audit():
            return 1

        receipt = resolve_v2_invalidation.verify_public_receipt(
            ROOT,
            profile=args.profile,
        )
        print(
            "public invalidation receipt OK "
            f"({len(receipt['replacement_sha256'])} exact replacements; {args.profile} profile)"
        )
        registry_report = audit_chain_registry.verify_outputs()
        print(
            "strict registry evidence/audit OK "
            f"({registry_report['summary']['included_codes']} included, "
            f"{registry_report['summary']['excluded_codes']} excluded)"
        )
        review_receipt = registry_human_review_receipt.verify_release_gate(ROOT)
        print(
            "outcome-blind registry human-review gate OK "
            f"(audit_id={review_receipt['audit_id']}; "
            f"disposition={review_receipt['disposition']['kind']})"
        )
        gpu_summary = json.loads(
            (ROOT / "results_v2/metrics/v2_gpu_rolling_summary.json").read_text(
                encoding="utf-8"
            )
        )
        gpu_postfreeze.verify_summary_binding(
            gpu_summary,
            artifact_path=ROOT / gpu_postfreeze.ARTIFACT_ROLE,
            root=ROOT,
            require_full_inventory=args.profile == "full",
        )
        print(
            "GPU post-freeze registry equivalence OK "
            f"({args.profile} profile; four provenance-only JSON changes)"
        )
        gpu_summary_verifier.verify_nbfnet_public_binding(gpu_summary, root=ROOT)
        print(
            "GPU NBFNet source/runtime evidence OK "
            "(four path-redacted source receipts, two-host runtime equality, "
            "and two pre-main formal gates)"
        )
        generate_v2_benchmark_profile.verify_outputs(mode=args.profile)
        print(f"benchmark scale/compute profile OK ({args.profile} profile)")
        check_v2_package(full=args.profile == "full")
        import summarize_v2_loco_results
        import summarize_v2_ultra_results
        import v2_gbdt_baselines

        summarize_v2_loco_results.verify_outputs(
            summarize_v2_loco_results.DEFAULT_JSON_OUT,
            summarize_v2_loco_results.DEFAULT_CSV_OUT,
        )
        if args.profile == "full":
            summarize_v2_ultra_results.verify_outputs(
                summarize_v2_ultra_results.DEFAULT_JSON_OUT,
                summarize_v2_ultra_results.DEFAULT_CSV_OUT,
            )
        else:
            # The repository profile has already verified these exact public
            # JSON/CSV bytes through the frozen artifact index.  Replaying the
            # trained-reference bridge additionally requires the external
            # candidate inventory and therefore belongs to the full profile.
            print(
                "formal ULTRA reference artifact OK "
                "(repository bytes verified by frozen index; "
                "external trained-reference bridge deferred to full profile)"
            )
        check_gbdt_baseline(
            v2_gbdt_baselines,
            full=args.profile == "full",
        )
        if args.profile == "full":
            import audit_v2
            import v2_rolling_cpu_baselines
            import validate_v2

            validate_v2.validate_release()
            v2_rolling_cpu_baselines.verify_existing_output()
            audit_v2.verify_existing_output()
            # The public receipt above verifies the exact value-diagnostic
            # JSON/CSV, generators, current JSON/TeX interface, and all public
            # source hashes.  Re-running their maintainer verifiers here would
            # additionally require private GPU score/selection artifacts and
            # the private frozen-run manifest, none of which may be shipped in
            # a public clean clone.
            print(
                "current value/paper interfaces OK via public receipt "
                "(private GPU run provenance intentionally not redistributed)"
            )
    except (AssertionError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"RELEASE SMOKE PASSED ({args.profile} profile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
