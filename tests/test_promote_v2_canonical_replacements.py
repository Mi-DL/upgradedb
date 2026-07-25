from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import promote_v2_canonical_replacements as promotion  # noqa: E402


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


class ReplacementFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.candidate = self.root / "candidate_bundle"
        self.candidate.mkdir()
        self.old_bytes: dict[str, bytes] = {}
        self.new_bytes: dict[str, bytes] = {}

        for role in promotion.BOUND_SOURCE_ROLES:
            path = self.root.joinpath(*Path(role).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"bound:{role}\n".encode("utf-8"))

        self.profile_inputs: dict[str, Path] = {}
        profile_root = self.root / "formal_profile_inputs"
        for name in promotion.PROFILE_INPUT_NAMES:
            path = profile_root / name
            if name in promotion.PROFILE_FILE_INPUTS:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"formal-file:{name}\n".encode("utf-8"))
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "receipt.json").write_bytes(
                    f"formal-directory-member:{name}\n".encode("utf-8")
                )
            self.profile_inputs[name] = path

        self.loco_dir = self.root / "results_v2" / "loco_formal"
        self.loco_dir.mkdir(parents=True, exist_ok=True)
        (self.loco_dir / "formal_receipt.json").write_bytes(b"formal LOCO fixture\n")

        hold = self.root.joinpath(*Path(promotion.HOLD_ROLE).parts)
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.write_bytes(
            canonical_json(
                {
                    "schema_version": "fixture-hold/1",
                    "status": promotion.REQUIRED_HOLD_STATUS,
                }
            )
        )

        for role in promotion.ROLES:
            old = f"old:{role}\n".encode("utf-8")
            new = f"new:{role}\n".encode("utf-8")
            self.old_bytes[role] = old
            self.new_bytes[role] = new
            target = self.root.joinpath(*Path(role).parts)
            staged = self.candidate.joinpath(*Path(role).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old)
            staged.write_bytes(new)

        self.baseline = self.root / "expected_before.json"
        self.baseline.write_bytes(
            canonical_json(
                {
                    "schema_version": promotion.BASELINE_SCHEMA,
                    "files": {
                        role: promotion._file_info(self.old_bytes[role])
                        for role in promotion.ROLES
                    },
                }
            )
        )
        self.transaction_id = "fixture-r1"

    def close(self) -> None:
        self.temporary.cleanup()

    def target(self, role: str) -> Path:
        return self.root.joinpath(*Path(role).parts)

    def transaction_directory(self) -> Path:
        return (
            self.root
            / "private"
            / "canonical_replacement_transactions"
            / self.transaction_id
        )

    def journal(self) -> dict:
        return json.loads(
            (self.transaction_directory() / "transaction.json").read_text(
                encoding="utf-8"
            )
        )

    def assert_old(self, case: unittest.TestCase) -> None:
        for role in promotion.ROLES:
            case.assertEqual(self.target(role).read_bytes(), self.old_bytes[role], role)

    def assert_new(self, case: unittest.TestCase) -> None:
        for role in promotion.ROLES:
            case.assertEqual(self.target(role).read_bytes(), self.new_bytes[role], role)


def fixture_validator(root: Path, bundle: Path) -> None:
    del root
    for role in promotion.ROLES:
        content = bundle.joinpath(*Path(role).parts).read_bytes()
        if not content.startswith(b"new:"):
            raise promotion.TransactionError(f"fixture candidate rejected: {role}")


def fixture_profile_gate(
    root: Path, bundle: Path, profile_inputs: dict[str, Path] | object
) -> None:
    del root
    if not isinstance(profile_inputs, dict) or set(profile_inputs) != set(
        promotion.PROFILE_INPUT_NAMES
    ):
        raise promotion.TransactionError("fixture formal profile inputs rejected")
    for path in profile_inputs.values():
        if not Path(path).exists():
            raise promotion.TransactionError("fixture formal profile input is missing")
    for role in promotion.PROFILE_ROLES:
        if not bundle.joinpath(*Path(role).parts).read_bytes().startswith(b"new:"):
            raise promotion.TransactionError("fixture rebuilt profile differs")


def fixture_loco_gate(root: Path, bundle: Path, loco_dir: Path) -> None:
    del root
    if not loco_dir.is_dir():
        raise promotion.TransactionError("fixture LOCO formal input is missing")
    for role in promotion.LOCO_ROLES:
        if not bundle.joinpath(*Path(role).parts).read_bytes().startswith(b"new:"):
            raise promotion.TransactionError("fixture rebuilt LOCO differs")


def fixture_ultra_gate(root: Path, bundle: Path, ultra_dir: Path) -> None:
    del root
    if not ultra_dir.is_dir():
        raise promotion.TransactionError("fixture ULTRA formal input is missing")
    for role in promotion.ULTRA_ROLES:
        if not bundle.joinpath(*Path(role).parts).read_bytes().startswith(b"new:"):
            raise promotion.TransactionError("fixture rebuilt ULTRA differs")


class CanonicalReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ReplacementFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def prepare(self) -> dict:
        return promotion.prepare_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            self.fixture.candidate,
            self.fixture.baseline,
            self.fixture.profile_inputs,
            self.fixture.loco_dir,
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )

    def test_happy_path_is_marker_last_and_verifiable(self) -> None:
        prepared = self.prepare()
        self.assertEqual(prepared["status"], "PREPARED")
        self.fixture.assert_old(self)

        promoted = promotion.apply_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            prepared["plan_sha256"],
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        self.assertEqual(promoted["status"], "PROMOTED")
        self.assertEqual(promoted["applied_roles"], list(promotion.ROLES))
        self.assertIsNotNone(promoted["committed_at_utc"])
        self.fixture.assert_new(self)

        verified = promotion.verify_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        self.assertEqual(verified, promoted)
        raw = (self.fixture.transaction_directory() / "transaction.json").read_bytes()
        self.assertEqual(raw, promotion._json_bytes(promoted))

    def test_prepare_rejects_unexpected_before_bytes_without_transaction(self) -> None:
        first = promotion.ROLES[0]
        self.fixture.target(first).write_bytes(b"unreviewed drift\n")
        with self.assertRaisesRegex(
            promotion.TransactionError, "expected-before inventory"
        ):
            self.prepare()
        self.assertFalse(self.fixture.transaction_directory().exists())

    def test_prepare_rejects_partial_candidate_inventory(self) -> None:
        self.fixture.candidate.joinpath(*Path(promotion.ROLES[1]).parts).unlink()
        with self.assertRaisesRegex(promotion.TransactionError, "fixed six-role scope"):
            self.prepare()
        self.fixture.assert_old(self)

    def test_prepare_rejects_a_candidate_with_an_external_hard_link(self) -> None:
        source = self.fixture.candidate.joinpath(*Path(promotion.ROLES[0]).parts)
        alias = self.fixture.root / "external-hard-link-alias"
        try:
            os.link(source, alias)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"hard-link creation is unavailable: {exc}")
        with self.assertRaisesRegex(promotion.TransactionError, "hard-link aliases"):
            self.prepare()
        self.fixture.assert_old(self)

    def test_prepare_validation_failure_writes_no_canonical_byte(self) -> None:
        def reject(root: Path, bundle: Path) -> None:
            del root, bundle
            raise promotion.TransactionError("candidate rejected")

        with self.assertRaisesRegex(promotion.TransactionError, "candidate rejected"):
            promotion.prepare_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                self.fixture.candidate,
                self.fixture.baseline,
                self.fixture.profile_inputs,
                self.fixture.loco_dir,
                validator=reject,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertFalse(self.fixture.transaction_directory().exists())

    def test_prepare_requires_independent_full_profile_gate(self) -> None:
        def reject_profile(
            root: Path, bundle: Path, profile_inputs: object
        ) -> None:
            del root, bundle, profile_inputs
            raise promotion.TransactionError("formal profile rebuild rejected")

        with self.assertRaisesRegex(
            promotion.TransactionError, "formal profile rebuild rejected"
        ):
            promotion.prepare_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                self.fixture.candidate,
                self.fixture.baseline,
                self.fixture.profile_inputs,
                self.fixture.loco_dir,
                validator=fixture_validator,
                profile_gate=reject_profile,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertFalse(self.fixture.transaction_directory().exists())

    def test_production_profile_gate_rebuilds_full_and_byte_compares(self) -> None:
        score_seal = "a" * 64
        rebuilt = {
            "profile": "rebuilt-from-formal-inputs",
            "provenance": {
                "formal_evidence_sha256": {
                    "ultra_score_seal_sha256": score_seal
                }
            },
        }
        json_bytes = canonical_json(rebuilt)
        digest = promotion._sha256_bytes(json_bytes)
        tex_bytes = f"profile:{digest}\n".encode("utf-8")
        self.fixture.candidate.joinpath(*Path(promotion.PROFILE_ROLES[0]).parts).write_bytes(
            json_bytes
        )
        self.fixture.candidate.joinpath(*Path(promotion.PROFILE_ROLES[1]).parts).write_bytes(
            tex_bytes
        )
        self.fixture.candidate.joinpath(*Path(promotion.ULTRA_ROLES[0]).parts).write_bytes(
            canonical_json({"provenance": {"score_seal_sha256": score_seal}})
        )
        module = mock.Mock()
        module.build_profile.return_value = rebuilt
        module.render_tex.side_effect = lambda profile, profile_sha: (
            f"profile:{profile_sha}\n"
        )
        with mock.patch.object(promotion, "_load_module", return_value=module):
            promotion.validate_full_profile_candidate(
                self.fixture.root,
                self.fixture.candidate,
                self.fixture.profile_inputs,
            )
            module.build_profile.assert_called_once()
            arguments = module.build_profile.call_args.args[0]
            self.assertEqual(
                arguments.ultra_receipts_dir,
                self.fixture.profile_inputs["ultra_receipts_dir"],
            )
            module.validate_profile.assert_called_once_with(rebuilt, mode="full")
            module.render_tex.assert_called_once_with(rebuilt, digest)

            self.fixture.candidate.joinpath(
                *Path(promotion.PROFILE_ROLES[0]).parts
            ).write_bytes(json_bytes + b"drift")
            with self.assertRaisesRegex(
                promotion.TransactionError, "differs from the formal-input rebuild"
            ):
                promotion.validate_full_profile_candidate(
                    self.fixture.root,
                    self.fixture.candidate,
                    self.fixture.profile_inputs,
                )

    def test_production_loco_gate_rebuilds_and_rejects_mismatch(self) -> None:
        summary = {"formal": "loco"}
        module = mock.Mock()
        module.build_summary.return_value = summary
        module.render_json.return_value = b"formal-loco-json\n"
        module.render_csv.return_value = b"formal-loco-csv\n"
        self.fixture.candidate.joinpath(*Path(promotion.LOCO_ROLES[0]).parts).write_bytes(
            module.render_json.return_value
        )
        self.fixture.candidate.joinpath(*Path(promotion.LOCO_ROLES[1]).parts).write_bytes(
            module.render_csv.return_value
        )
        with mock.patch.object(promotion, "_load_module", return_value=module):
            promotion.validate_full_loco_candidate(
                self.fixture.root, self.fixture.candidate, self.fixture.loco_dir
            )
            module.build_summary.assert_called_once_with(
                self.fixture.loco_dir / "summary.json",
                self.fixture.loco_dir / "verification_receipt.json",
            )
            self.fixture.candidate.joinpath(
                *Path(promotion.LOCO_ROLES[1]).parts
            ).write_bytes(b"different LOCO CSV\n")
            with self.assertRaisesRegex(
                promotion.TransactionError, "differs from the live formal rebuild"
            ):
                promotion.validate_full_loco_candidate(
                    self.fixture.root, self.fixture.candidate, self.fixture.loco_dir
                )

    def test_production_ultra_gate_rebuilds_and_rejects_mismatch(self) -> None:
        summary = {"formal": "ultra"}
        module = mock.Mock()
        module.build_summary.return_value = summary
        module.render_json.return_value = b"formal-ultra-json\n"
        module.render_csv.return_value = b"formal-ultra-csv\n"
        self.fixture.candidate.joinpath(*Path(promotion.ULTRA_ROLES[0]).parts).write_bytes(
            module.render_json.return_value
        )
        self.fixture.candidate.joinpath(*Path(promotion.ULTRA_ROLES[1]).parts).write_bytes(
            module.render_csv.return_value
        )
        ultra_dir = self.fixture.profile_inputs["ultra_dir"]
        with mock.patch.object(promotion, "_load_module", return_value=module):
            promotion.validate_full_ultra_candidate(
                self.fixture.root, self.fixture.candidate, ultra_dir
            )
            module.build_summary.assert_called_once_with(ultra_dir)
            self.fixture.candidate.joinpath(
                *Path(promotion.ULTRA_ROLES[0]).parts
            ).write_bytes(b"different ULTRA JSON\n")
            with self.assertRaisesRegex(
                promotion.TransactionError, "differs from the sealed formal-tree rebuild"
            ):
                promotion.validate_full_ultra_candidate(
                    self.fixture.root, self.fixture.candidate, ultra_dir
                )

    def test_profile_ultra_score_seal_mismatch_is_rejected(self) -> None:
        profile_seal = "a" * 64
        ultra_seal = "b" * 64
        rebuilt = {
            "provenance": {
                "formal_evidence_sha256": {
                    "ultra_score_seal_sha256": profile_seal
                }
            }
        }
        json_bytes = canonical_json(rebuilt)
        module = mock.Mock()
        module.build_profile.return_value = rebuilt
        module.render_tex.side_effect = lambda profile, profile_sha: "tex\n"
        self.fixture.candidate.joinpath(*Path(promotion.PROFILE_ROLES[0]).parts).write_bytes(
            json_bytes
        )
        self.fixture.candidate.joinpath(*Path(promotion.PROFILE_ROLES[1]).parts).write_bytes(
            b"tex\n"
        )
        self.fixture.candidate.joinpath(*Path(promotion.ULTRA_ROLES[0]).parts).write_bytes(
            canonical_json({"provenance": {"score_seal_sha256": ultra_seal}})
        )
        with mock.patch.object(promotion, "_load_module", return_value=module):
            with self.assertRaisesRegex(
                promotion.TransactionError, "bind different score seals"
            ):
                promotion.validate_full_profile_candidate(
                    self.fixture.root,
                    self.fixture.candidate,
                    self.fixture.profile_inputs,
                )

    def test_journal_seals_every_formal_profile_input_inventory(self) -> None:
        prepared = self.prepare()
        self.assertEqual(
            set(prepared["profile_inputs"]), set(promotion.PROFILE_INPUT_NAMES)
        )
        self.assertIn("ultra_receipts_dir", prepared["profile_inputs"])
        for name in promotion.PROFILE_FILE_INPUTS:
            row = prepared["profile_inputs"][name]
            self.assertEqual(row["kind"], "file")
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(row["size"], 0)
        for name in promotion.PROFILE_DIRECTORY_INPUTS:
            row = prepared["profile_inputs"][name]
            self.assertEqual(row["kind"], "directory")
            self.assertEqual(row["file_count"], len(row["files"]))
            self.assertGreater(row["file_count"], 0)
            self.assertRegex(row["inventory_sha256"], r"^[0-9a-f]{64}$")
        loco = prepared["loco_formal_input"]
        self.assertEqual(loco["role"], promotion.CANONICAL_LOCO_FORMAL_ROLE)
        self.assertEqual(loco["file_count"], len(loco["files"]))
        self.assertGreater(loco["file_count"], 0)
        self.assertRegex(loco["inventory_sha256"], r"^[0-9a-f]{64}$")

    def test_prepare_cli_requires_and_exposes_ultra_receipts_directory(self) -> None:
        parser = promotion._build_parser()
        common = [
            "prepare",
            "--transaction-id",
            "fixture-r1",
            "--candidate-root",
            "candidate",
            "--expected-before",
            "baseline.json",
            "--dataset-summary",
            "dataset.json",
            "--freeze-manifest",
            "freeze.json",
            "--claims-dir",
            "claims",
            "--metrics-dir",
            "metrics",
            "--selections-dir",
            "selections",
            "--gpu-inventory",
            "gpu.json",
            "--ultra-dir",
            "ultra",
            "--loco-dir",
            "loco",
        ]
        with mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            parser.parse_args(common)
        receipts = Path("ultra-receipts")
        args = parser.parse_args(
            [*common, "--ultra-receipts-dir", str(receipts)]
        )
        self.assertEqual(args.ultra_receipts_dir, receipts)

        prepared = {
            "transaction_id": "fixture-r1",
            "plan_sha256": "a" * 64,
        }
        with (
            mock.patch.object(
                promotion, "prepare_transaction", return_value=prepared
            ) as prepare,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(
                promotion.main(
                    [
                        "--root",
                        str(self.fixture.root),
                        *common,
                        "--ultra-receipts-dir",
                        str(receipts),
                    ]
                ),
                0,
            )
        forwarded_inputs = prepare.call_args.args[4]
        self.assertEqual(forwarded_inputs["ultra_receipts_dir"], receipts)

    def test_apply_rejects_formal_profile_input_drift_before_replacement(self) -> None:
        prepared = self.prepare()
        receipt = self.fixture.profile_inputs["ultra_dir"] / "receipt.json"
        receipt.write_bytes(receipt.read_bytes() + b"formal drift\n")
        with self.assertRaisesRegex(
            promotion.TransactionError, "profile inputs changed|profile input.*changed"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_apply_rejects_ultra_receipts_input_drift_before_replacement(self) -> None:
        prepared = self.prepare()
        receipt = self.fixture.profile_inputs["ultra_receipts_dir"] / "receipt.json"
        receipt.write_bytes(receipt.read_bytes() + b"orchestration drift\n")
        with self.assertRaisesRegex(
            promotion.TransactionError, "profile inputs changed|profile input.*changed"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_apply_rejects_loco_formal_tree_drift_before_replacement(self) -> None:
        prepared = self.prepare()
        receipt = self.fixture.loco_dir / "formal_receipt.json"
        receipt.write_bytes(receipt.read_bytes() + b"formal drift\n")
        with self.assertRaisesRegex(promotion.TransactionError, "LOCO formal input tree"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_apply_rejects_hold_drift_before_any_replacement(self) -> None:
        prepared = self.prepare()
        hold = self.fixture.root.joinpath(*Path(promotion.HOLD_ROLE).parts)
        hold.write_bytes(
            canonical_json(
                {
                    "schema_version": "fixture-hold/1",
                    "status": promotion.REQUIRED_HOLD_STATUS,
                    "note": "changed bytes",
                }
            )
        )
        with self.assertRaisesRegex(promotion.TransactionError, "hold changed"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_apply_rejects_bound_source_drift_before_any_replacement(self) -> None:
        prepared = self.prepare()
        source = self.fixture.root.joinpath(*Path(promotion.BOUND_SOURCE_ROLES[0]).parts)
        source.write_bytes(source.read_bytes() + b"drift\n")
        with self.assertRaisesRegex(promotion.TransactionError, "bound generator"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_apply_rejects_candidate_snapshot_drift_before_any_replacement(self) -> None:
        prepared = self.prepare()
        candidate = (
            self.fixture.transaction_directory()
            / "candidate"
            / Path(promotion.ROLES[0])
        )
        candidate.write_bytes(candidate.read_bytes() + b"tamper\n")
        with self.assertRaisesRegex(promotion.TransactionError, "candidate inventory"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)

    def test_bad_confirmation_is_read_only(self) -> None:
        self.prepare()
        with self.assertRaisesRegex(promotion.TransactionError, "confirmation"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                "0" * 64,
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "PREPARED")

    def test_foreign_live_writer_lock_is_never_removed(self) -> None:
        prepared = self.prepare()
        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.parent.mkdir(parents=True, exist_ok=True)
        content = promotion._json_bytes(
            {
                "schema_version": promotion.LOCK_SCHEMA,
                "transaction_id": "foreign-r1",
                "host": promotion._host_identity(),
                "pid": os.getpid(),
                "created_at_utc": promotion._utc_now(),
            }
        )
        lock.write_bytes(content)
        with self.assertRaisesRegex(promotion.TransactionError, "writer lock"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.assertEqual(lock.read_bytes(), content)
        self.fixture.assert_old(self)

    def test_each_injected_replace_failure_rolls_back_exactly(self) -> None:
        # One fresh repository per failure position avoids making rollback
        # success at one position an input to the next subtest.
        self.fixture.close()
        for failure_position in range(1, len(promotion.ROLES) + 1):
            with self.subTest(failure_position=failure_position):
                fixture = ReplacementFixture()
                try:
                    prepared = promotion.prepare_transaction(
                        fixture.root,
                        fixture.transaction_id,
                        fixture.candidate,
                        fixture.baseline,
                        fixture.profile_inputs,
                        fixture.loco_dir,
                        validator=fixture_validator,
                        profile_gate=fixture_profile_gate,
                        loco_gate=fixture_loco_gate,
                        ultra_gate=fixture_ultra_gate,
                    )
                    calls = 0

                    def fail_once(source: Path, target: Path) -> None:
                        nonlocal calls
                        calls += 1
                        if calls == failure_position:
                            raise OSError("injected replacement failure")
                        os.replace(source, target)

                    with self.assertRaisesRegex(
                        promotion.TransactionError, "all canonical bytes were restored"
                    ):
                        promotion.apply_transaction(
                            fixture.root,
                            fixture.transaction_id,
                            prepared["plan_sha256"],
                            validator=fixture_validator,
                            profile_gate=fixture_profile_gate,
                            loco_gate=fixture_loco_gate,
                            ultra_gate=fixture_ultra_gate,
                            replace_func=fail_once,
                        )
                    fixture.assert_old(self)
                    self.assertEqual(fixture.journal()["status"], "ROLLED_BACK")
                finally:
                    fixture.close()
        # tearDown must not attempt to clean the fixture closed above twice on
        # platforms that object to a repeated TemporaryDirectory cleanup.
        self.fixture = ReplacementFixture()

    def test_post_apply_verifier_failure_rolls_back(self) -> None:
        prepared = self.prepare()

        def reject_canonical(root: Path, bundle: Path) -> None:
            fixture_validator(root, bundle)
            if bundle.resolve() == root.resolve():
                raise promotion.TransactionError("post-apply verification rejected")

        with self.assertRaisesRegex(
            promotion.TransactionError, "all canonical bytes were restored"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=reject_canonical,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "ROLLED_BACK")

    def test_rollback_failure_enters_recovery_required(self) -> None:
        prepared = self.prepare()
        application_calls = 0

        def fail_apply_and_rollback(source: Path, target: Path) -> None:
            nonlocal application_calls
            if ".rollback-" in source.name:
                raise OSError("injected rollback failure")
            application_calls += 1
            if application_calls == 2:
                raise OSError("injected apply failure")
            os.replace(source, target)

        with self.assertRaisesRegex(promotion.TransactionError, "requires recovery"):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
                replace_func=fail_apply_and_rollback,
            )
        self.assertEqual(self.fixture.journal()["status"], "RECOVERY_REQUIRED")
        self.assertNotEqual(
            self.fixture.target(promotion.ROLES[0]).read_bytes(),
            self.fixture.old_bytes[promotion.ROLES[0]],
        )

    def test_recovery_required_transaction_blocks_a_different_apply(self) -> None:
        first = self.prepare()
        promotion._update_journal(
            self.fixture.transaction_directory(),
            first,
            status="RECOVERY_REQUIRED",
            error="InjectedFailure",
        )
        second_id = "fixture-r2"
        second = promotion.prepare_transaction(
            self.fixture.root,
            second_id,
            self.fixture.candidate,
            self.fixture.baseline,
            self.fixture.profile_inputs,
            self.fixture.loco_dir,
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        with self.assertRaisesRegex(
            promotion.TransactionError, "requires recovery before any new apply"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                second_id,
                second["plan_sha256"],
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)

    def test_candidate_storage_mutation_during_post_gate_rolls_back(self) -> None:
        prepared = self.prepare()
        calls = 0

        def mutate_stored_candidate(
            root: Path, bundle: Path, profile_inputs: object
        ) -> None:
            nonlocal calls
            fixture_profile_gate(root, bundle, profile_inputs)
            calls += 1
            if bundle.resolve() == root.resolve():
                stored = (
                    self.fixture.transaction_directory()
                    / "candidate"
                    / Path(promotion.ROLES[0])
                )
                stored.write_bytes(stored.read_bytes() + b"mid-commit tamper\n")

        with self.assertRaisesRegex(
            promotion.TransactionError, "all canonical bytes were restored"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=mutate_stored_candidate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.assertGreaterEqual(calls, 2)
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "ROLLED_BACK")

    def test_canonical_mutation_during_post_gate_rolls_back(self) -> None:
        prepared = self.prepare()

        def mutate_canonical(
            root: Path, bundle: Path, profile_inputs: object
        ) -> None:
            fixture_profile_gate(root, bundle, profile_inputs)
            if bundle.resolve() == root.resolve():
                target = root.joinpath(*Path(promotion.ROLES[0]).parts)
                target.write_bytes(target.read_bytes() + b"external drift\n")

        with self.assertRaisesRegex(
            promotion.TransactionError, "all canonical bytes were restored"
        ):
            promotion.apply_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                prepared["plan_sha256"],
                validator=fixture_validator,
                profile_gate=mutate_canonical,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )
        self.fixture.assert_old(self)
        self.assertEqual(self.fixture.journal()["status"], "ROLLED_BACK")

    def test_recover_restores_an_interrupted_prefix_and_stale_owned_lock(self) -> None:
        prepared = self.prepare()
        directory = self.fixture.transaction_directory()
        journal = promotion._update_journal(
            directory,
            prepared,
            status="APPLYING",
            applied_roles=list(promotion.ROLES[:3]),
        )
        for role in promotion.ROLES[:3]:
            self.fixture.target(role).write_bytes(self.fixture.new_bytes[role])

        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            promotion._json_bytes(
                {
                    "schema_version": promotion.LOCK_SCHEMA,
                    "transaction_id": self.fixture.transaction_id,
                    "host": promotion._host_identity(),
                    "pid": 999_999_999,
                    "created_at_utc": promotion._utc_now(),
                }
            )
        )
        recovered = promotion.recover_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            journal["before_inventory_sha256"],
        )
        self.assertEqual(recovered["status"], "ROLLED_BACK")
        self.assertIsNotNone(recovered["recovered_at_utc"])
        self.fixture.assert_old(self)
        self.assertFalse(lock.exists())
        self.assertEqual(
            promotion.verify_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )["status"],
            "ROLLED_BACK",
        )

    def test_recovery_does_not_depend_on_candidate_or_verifier_bytes(self) -> None:
        prepared = self.prepare()
        directory = self.fixture.transaction_directory()
        applying = promotion._update_journal(
            directory,
            prepared,
            status="APPLYING",
            applied_roles=list(promotion.ROLES[:1]),
        )
        first = promotion.ROLES[0]
        self.fixture.target(first).write_bytes(self.fixture.new_bytes[first])
        candidate = directory / "candidate" / Path(promotion.ROLES[-1])
        candidate.write_bytes(b"corrupt candidate no longer needed for recovery\n")
        source = self.fixture.root.joinpath(*Path(promotion.BOUND_SOURCE_ROLES[0]).parts)
        source.write_bytes(source.read_bytes() + b"post-crash drift\n")

        recovered = promotion.recover_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            applying["before_inventory_sha256"],
        )
        self.assertEqual(recovered["status"], "ROLLED_BACK")
        self.fixture.assert_old(self)

    def test_recover_refuses_a_live_owned_lock(self) -> None:
        prepared = self.prepare()
        directory = self.fixture.transaction_directory()
        applying = promotion._update_journal(
            directory, prepared, status="APPLYING", applied_roles=[]
        )
        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            promotion._json_bytes(
                {
                    "schema_version": promotion.LOCK_SCHEMA,
                    "transaction_id": self.fixture.transaction_id,
                    "host": promotion._host_identity(),
                    "pid": os.getpid(),
                    "created_at_utc": promotion._utc_now(),
                }
            )
        )
        with self.assertRaisesRegex(promotion.TransactionError, "still alive"):
            promotion.recover_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                applying["before_inventory_sha256"],
            )
        self.assertTrue(lock.exists())

    def test_recover_refuses_a_lock_from_another_host(self) -> None:
        prepared = self.prepare()
        directory = self.fixture.transaction_directory()
        applying = promotion._update_journal(
            directory, prepared, status="APPLYING", applied_roles=[]
        )
        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.parent.mkdir(parents=True, exist_ok=True)
        content = promotion._json_bytes(
            {
                "schema_version": promotion.LOCK_SCHEMA,
                "transaction_id": self.fixture.transaction_id,
                "host": promotion._host_identity() + "-other",
                "pid": 999_999_999,
                "created_at_utc": promotion._utc_now(),
            }
        )
        lock.write_bytes(content)
        with self.assertRaisesRegex(promotion.TransactionError, "cross-host"):
            promotion.recover_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                applying["before_inventory_sha256"],
            )
        self.assertEqual(lock.read_bytes(), content)
        self.fixture.assert_old(self)

    def test_recover_cancels_prepared_after_pre_apply_crash(self) -> None:
        prepared = self.prepare()
        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            promotion._json_bytes(
                {
                    "schema_version": promotion.LOCK_SCHEMA,
                    "transaction_id": self.fixture.transaction_id,
                    "host": promotion._host_identity(),
                    "pid": 999_999_999,
                    "created_at_utc": promotion._utc_now(),
                }
            )
        )
        recovered = promotion.recover_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            prepared["before_inventory_sha256"],
        )
        self.assertEqual(recovered["status"], "ROLLED_BACK")
        self.fixture.assert_old(self)
        self.assertFalse(lock.exists())

    def test_recover_terminal_promotion_only_clears_dead_owned_lock(self) -> None:
        prepared = self.prepare()
        promoted = promotion.apply_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            prepared["plan_sha256"],
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        lock = self.fixture.root.joinpath(*Path(promotion.LOCK_ROLE).parts)
        lock.write_bytes(
            promotion._json_bytes(
                {
                    "schema_version": promotion.LOCK_SCHEMA,
                    "transaction_id": self.fixture.transaction_id,
                    "host": promotion._host_identity(),
                    "pid": 999_999_999,
                    "created_at_utc": promotion._utc_now(),
                }
            )
        )
        recovered = promotion.recover_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            promoted["before_inventory_sha256"],
        )
        self.assertEqual(recovered["status"], "PROMOTED")
        self.fixture.assert_new(self)
        self.assertFalse(lock.exists())

    def test_promoted_receipt_remains_verifiable_after_hold_resolution(self) -> None:
        prepared = self.prepare()
        promoted = promotion.apply_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            prepared["plan_sha256"],
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        hold = self.fixture.root.joinpath(*Path(promotion.HOLD_ROLE).parts)
        hold.write_bytes(
            canonical_json(
                {
                    "schema_version": "fixture-hold/1",
                    "status": "RESOLVED_REBUILT_VERIFIED",
                    "original_status": promotion.REQUIRED_HOLD_STATUS,
                }
            )
        )
        verified = promotion.verify_transaction(
            self.fixture.root,
            self.fixture.transaction_id,
            validator=fixture_validator,
            profile_gate=fixture_profile_gate,
            loco_gate=fixture_loco_gate,
            ultra_gate=fixture_ultra_gate,
        )
        self.assertEqual(verified["status"], "PROMOTED")
        self.assertEqual(verified["plan_sha256"], promoted["plan_sha256"])

    def test_verify_refuses_unfinished_state(self) -> None:
        prepared = self.prepare()
        promotion._update_journal(
            self.fixture.transaction_directory(),
            prepared,
            status="APPLYING",
            applied_roles=[],
        )
        with self.assertRaisesRegex(promotion.TransactionError, "explicit recovery"):
            promotion.verify_transaction(
                self.fixture.root,
                self.fixture.transaction_id,
                validator=fixture_validator,
                profile_gate=fixture_profile_gate,
                loco_gate=fixture_loco_gate,
                ultra_gate=fixture_ultra_gate,
            )

    def test_symlink_candidate_is_rejected_when_supported(self) -> None:
        role = promotion.ROLES[0]
        path = self.fixture.candidate.joinpath(*Path(role).parts)
        target = path.with_suffix(path.suffix + ".real")
        path.rename(target)
        try:
            path.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic-link creation is unavailable")
        with self.assertRaisesRegex(promotion.TransactionError, "non-regular|symbolic"):
            self.prepare()
        self.fixture.assert_old(self)


if __name__ == "__main__":
    unittest.main()
