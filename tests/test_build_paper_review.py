from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "tools"))

import build_paper_review as review  # noqa: E402


def _numbers() -> dict[str, str]:
    return {
        "VTwoEligibilityThresholdStatus": "COMPLETE",
        "VTwoGBDTStatus": "COMPLETE",
        "VTwoGPUStatus": "COMPLETE",
        "VTwoLOCOStatus": "COMPLETE",
        "VTwoProductSpaceStatus": "COMPLETE",
        "VTwoScoreRobustnessRFiveStatus": "COMPLETE",
        "VTwoULTRAStatus": "COMPLETE",
    }


class BuildPaperReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.cache = self.root / review.CACHE_RELATIVE
        self.binding = {
            "active_invalidation_notice_sha256": "a" * 64,
            "invalidation_notice_status": review.resolution.ACTIVE_STATUS,
            "fixed_replacement_sha256": {"fixed": "b" * 64},
            "paper_source_sha256": {"source": "c" * 64},
            "benchmark_profile_sha256": {"profile": "d" * 64},
            "resolution_verifier_sha256": {"verifier": "e" * 64},
        }

    def _write_notice(self, payload: dict[str, object]) -> Path:
        path = self.root / review.resolution.NOTICE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(review._canonical_json_bytes(payload))
        return path

    def _write_cache(self) -> review.ReviewInterface:
        numbers = _numbers()
        sources = dict(self.binding["paper_source_sha256"])
        generated = review._generated_cache_paths(self.cache)
        for path in generated.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        generated[review.resolution.PAPER_JSON_PATH].write_text(
            review.paper_numbers.render_json(numbers, sources),
            encoding="utf-8",
            newline="\n",
        )
        generated[review.resolution.PAPER_TEX_PATH].write_text(
            review.paper_numbers.render_tex(numbers, sources),
            encoding="utf-8",
            newline="\n",
        )
        manifest = {
            "schema_version": review.CACHE_SCHEMA,
            "status": review.CACHE_STATUS,
            "binding": self.binding,
            "generated_sha256": {
                relative: review._sha256_file(path)
                for relative, path in generated.items()
            },
            "number_key_count": len(numbers),
            "number_keys_sha256": review.public_policy._paper_number_key_digest(numbers),
            "number_values_sha256": review.public_policy._paper_number_value_digest(numbers),
        }
        review._manifest_path(self.cache).write_bytes(review._canonical_json_bytes(manifest))
        with mock.patch.object(
            review, "_current_governed_binding", return_value=self.binding
        ), mock.patch.object(
            review.public_policy, "_validate_paper_numbers", return_value=numbers
        ):
            return review.validate_review_cache(self.root, self.cache)

    def test_text_only_changes_reuse_exact_review_interface(self) -> None:
        interface = self._write_cache()
        paper = self.root / "paper"
        paper.mkdir(exist_ok=True)
        abstract = paper / "abstract.tex"
        abstract.write_text("first wording\n", encoding="utf-8")

        with mock.patch.object(
            review, "_current_governed_binding", return_value=self.binding
        ), mock.patch.object(
            review.public_policy, "_validate_paper_numbers", return_value=_numbers()
        ):
            first = review.validate_review_cache(self.root, self.cache)
            abstract.write_text("updated author wording\n", encoding="utf-8")
            second = review.validate_review_cache(self.root, self.cache)

        self.assertEqual(first.tex_path.read_bytes(), second.tex_path.read_bytes())
        self.assertEqual(second.tex_path, interface.tex_path)

    def test_resolved_notice_uses_public_receipt_verifier(self) -> None:
        receipt = {"status": review.resolution.RESOLVED_STATUS}
        path = self._write_notice(receipt)

        with mock.patch.object(
            review.resolution,
            "verify_public_receipt",
            return_value=receipt,
        ) as verify, mock.patch.object(
            review.resolution,
            "_validate_active_notice",
        ) as validate_active:
            observed, observed_bytes = review._validated_review_notice(self.root)

        self.assertEqual(observed, receipt)
        self.assertEqual(observed_bytes, path.read_bytes())
        verify.assert_called_once_with(self.root, profile="full")
        validate_active.assert_not_called()

    def test_unknown_notice_status_fails_closed(self) -> None:
        self._write_notice({"status": "UNKNOWN"})

        with mock.patch.object(
            review.resolution,
            "verify_public_receipt",
        ) as verify, self.assertRaisesRegex(
            review.ReviewBuildError,
            "unsupported invalidation notice status",
        ):
            review._validated_review_notice(self.root)

        verify.assert_not_called()

    def test_review_snapshot_keeps_active_preview_path(self) -> None:
        preview = mock.sentinel.preview
        destination = self.root / "preview"

        with mock.patch.object(
            review.resolution,
            "write_paper_preview",
            return_value=preview,
        ) as write_active, mock.patch.object(
            review,
            "_write_resolved_paper_snapshot",
        ) as write_resolved:
            observed = review._write_review_snapshot(
                self.root,
                preview_root=destination,
                notice_status=review.resolution.ACTIVE_STATUS,
            )

        self.assertIs(observed, preview)
        write_active.assert_called_once_with(self.root, preview_root=destination)
        write_resolved.assert_not_called()

    def test_resolved_snapshot_copies_receipt_bound_canonical_bytes(self) -> None:
        numbers = _numbers()
        sources = {"source": "c" * 64}
        canonical = {
            review.resolution.PAPER_JSON_PATH: review.paper_numbers.render_json(
                numbers,
                sources,
            ).encode("utf-8"),
            review.resolution.PAPER_TEX_PATH: review.paper_numbers.render_tex(
                numbers,
                sources,
            ).encode("utf-8"),
        }
        for relative, content in canonical.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        receipt = {
            "status": review.resolution.RESOLVED_STATUS,
            "replacement_sha256": {
                relative: review._sha256_bytes(content)
                for relative, content in canonical.items()
            },
            "resolution_source_sha256": sources,
        }
        destination = self.root / "resolved-preview"

        with mock.patch.object(
            review.resolution,
            "verify_public_receipt",
            return_value=receipt,
        ) as verify, mock.patch.object(
            review.public_policy,
            "_validate_paper_numbers",
            return_value=numbers,
        ):
            preview = review._write_resolved_paper_snapshot(
                self.root,
                preview_root=destination,
            )

        verify.assert_called_once_with(self.root, profile="full")
        self.assertEqual(preview.generated_bytes, canonical)
        self.assertEqual(preview.source_sha256, sources)
        self.assertEqual(preview.number_key_count, len(numbers))
        for relative, content in canonical.items():
            self.assertEqual((destination / relative).read_bytes(), content)

    def test_resolved_snapshot_rejects_canonical_byte_drift(self) -> None:
        numbers = _numbers()
        sources = {"source": "c" * 64}
        for relative, content in {
            review.resolution.PAPER_JSON_PATH: review.paper_numbers.render_json(
                numbers,
                sources,
            ).encode("utf-8"),
            review.resolution.PAPER_TEX_PATH: review.paper_numbers.render_tex(
                numbers,
                sources,
            ).encode("utf-8"),
        }.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        receipt = {
            "status": review.resolution.RESOLVED_STATUS,
            "replacement_sha256": {
                relative: "f" * 64 for relative in review.resolution.GENERATED_PATHS
            },
            "resolution_source_sha256": sources,
        }

        with mock.patch.object(
            review.resolution,
            "verify_public_receipt",
            return_value=receipt,
        ), self.assertRaisesRegex(
            review.ReviewBuildError,
            "canonical paper output hash mismatch",
        ):
            review._write_resolved_paper_snapshot(
                self.root,
                preview_root=self.root / "resolved-preview",
            )

    def test_cache_rejects_binding_or_generated_drift(self) -> None:
        interface = self._write_cache()
        changed = dict(self.binding)
        changed["active_invalidation_notice_sha256"] = "f" * 64
        with mock.patch.object(
            review, "_current_governed_binding", return_value=changed
        ), self.assertRaisesRegex(review.ReviewBuildError, "cache is stale"):
            review.validate_review_cache(self.root, self.cache)

        interface.tex_path.write_text("tampered\n", encoding="utf-8")
        with mock.patch.object(
            review, "_current_governed_binding", return_value=self.binding
        ), self.assertRaisesRegex(review.ReviewBuildError, "changed after verification"):
            review.validate_review_cache(self.root, self.cache)

    def test_stale_binding_is_optional_but_cache_integrity_is_not(self) -> None:
        interface = self._write_cache()
        changed = dict(self.binding)
        changed["active_invalidation_notice_sha256"] = "f" * 64
        with mock.patch.object(
            review, "_current_governed_binding", return_value=changed
        ), mock.patch.object(
            review.public_policy, "_validate_paper_numbers", return_value=_numbers()
        ):
            stale = review.validate_review_cache(
                self.root,
                self.cache,
                allow_stale_binding=True,
            )
        self.assertFalse(stale.binding_current)
        self.assertEqual(stale.tex_path.read_bytes(), interface.tex_path.read_bytes())

        interface.tex_path.write_text("tampered\n", encoding="utf-8")
        with mock.patch.object(
            review, "_current_governed_binding", return_value=changed
        ), self.assertRaisesRegex(review.ReviewBuildError, "changed after verification"):
            review.validate_review_cache(
                self.root,
                self.cache,
                allow_stale_binding=True,
            )

    def test_staging_overlays_review_numbers_without_copying_canonical_numbers(self) -> None:
        interface = self._write_cache()
        paper = self.root / "paper"
        paper.mkdir(exist_ok=True)
        for name in review.PAPER_SOURCE_FILES:
            (paper / name).write_text(f"source:{name}\n", encoding="utf-8")
        figures = paper / "figures"
        figures.mkdir()
        (figures / "figure.pdf").write_bytes(b"fixture")
        generated = paper / "generated"
        generated.mkdir()
        canonical = generated / "v2_numbers.tex"
        canonical.write_bytes(b"canonical numbers\n")
        profile = self.root / review.PROFILE_TEX_RELATIVE
        profile.write_bytes(b"profile\n")
        contemporary = self.root / review.CONTEMPORARY_TEX_RELATIVE
        contemporary.write_bytes(b"contemporary references\n")

        canonical_before = canonical.read_bytes()
        build = self.root / "build"
        review._copy_paper_tree(self.root, build, interface)

        self.assertEqual(canonical.read_bytes(), canonical_before)
        self.assertEqual(
            (build / "generated/v2_numbers.tex").read_bytes(),
            interface.tex_path.read_bytes(),
        )
        self.assertNotEqual(
            (build / "generated/v2_numbers.tex").read_bytes(),
            canonical_before,
        )
        self.assertEqual(
            (build / "generated/v2_contemporary_references.tex").read_bytes(),
            contemporary.read_bytes(),
        )

    def test_contemporary_interface_and_verifier_inputs_are_drift_guarded(self) -> None:
        self.assertTrue(
            {
                review.CONTEMPORARY_CONFIG_RELATIVE,
                review.CONTEMPORARY_JSON_RELATIVE,
                review.CONTEMPORARY_CSV_RELATIVE,
                review.CONTEMPORARY_TEX_RELATIVE,
                review.CONTEMPORARY_TOOL_RELATIVE,
            }.issubset(review.CANONICAL_GUARD_PATHS)
        )

    def test_failed_tex_never_overwrites_last_good_pdf(self) -> None:
        interface = self._write_cache()
        output = self.root / review.OUTPUT_RELATIVE
        output.parent.mkdir(parents=True)
        output.write_bytes(b"last-good-pdf")

        def prepare(_root: Path, build_root: Path, _interface: review.ReviewInterface) -> None:
            build_root.mkdir(parents=True)

        with mock.patch.object(review, "_verify_benchmark_profile"), mock.patch.object(
            review, "_verify_contemporary_references"
        ), mock.patch.object(
            review, "_copy_paper_tree", side_effect=prepare
        ), mock.patch.object(
            review,
            "_compile_acm_pdf",
            side_effect=review.ReviewBuildError("fixture TeX failure"),
        ), self.assertRaisesRegex(review.ReviewBuildError, "fixture TeX failure"):
            review.build_review_pdf(
                self.root,
                interface,
                output=output,
                pdflatex=Path("pdflatex"),
                bibtex=Path("bibtex"),
                canonical_before={},
            )

        self.assertEqual(output.read_bytes(), b"last-good-pdf")

    def test_failed_refresh_preserves_last_good_cache(self) -> None:
        self._write_cache()
        manifest = review._manifest_path(self.cache)
        before = manifest.read_bytes()

        with mock.patch.object(
            review, "_current_governed_binding", return_value=self.binding
        ), mock.patch.object(
            review.resolution,
            "write_paper_preview",
            side_effect=review.resolution.ResolutionError("fixture verifier failure"),
        ), self.assertRaisesRegex(review.ReviewBuildError, "fixture verifier failure"):
            review.refresh_review_cache(self.root, self.cache)

        self.assertEqual(manifest.read_bytes(), before)

    def test_failed_resolved_refresh_preserves_last_good_cache(self) -> None:
        self._write_cache()
        manifest = review._manifest_path(self.cache)
        before = manifest.read_bytes()
        resolved_binding = dict(self.binding)
        resolved_binding["invalidation_notice_status"] = review.resolution.RESOLVED_STATUS

        with mock.patch.object(
            review,
            "_current_governed_binding",
            return_value=resolved_binding,
        ), mock.patch.object(
            review,
            "_write_resolved_paper_snapshot",
            side_effect=review.ReviewBuildError("fixture resolved verifier failure"),
        ), self.assertRaisesRegex(
            review.ReviewBuildError,
            "fixture resolved verifier failure",
        ):
            review.refresh_review_cache(self.root, self.cache)

        self.assertEqual(manifest.read_bytes(), before)

    def test_cache_switch_during_compile_never_overwrites_pdf(self) -> None:
        interface = self._write_cache()
        output = self.root / review.OUTPUT_RELATIVE
        output.parent.mkdir(parents=True)
        output.write_bytes(b"last-good-pdf")

        changed_manifest = dict(interface.manifest)
        changed_manifest["number_values_sha256"] = "f" * 64
        changed = review.ReviewInterface(
            cache_root=interface.cache_root,
            json_path=interface.json_path,
            tex_path=interface.tex_path,
            manifest=changed_manifest,
        )

        def prepare(_root: Path, build_root: Path, _interface: review.ReviewInterface) -> None:
            build_root.mkdir(parents=True)

        def compile_pdf(
            build_root: Path, _pdflatex: Path, _bibtex: Path
        ) -> Path:
            pdf = build_root / "main-acm.pdf"
            pdf.write_bytes(b"new-review-pdf")
            return pdf

        with mock.patch.object(review, "_verify_benchmark_profile"), mock.patch.object(
            review, "_verify_contemporary_references"
        ), mock.patch.object(
            review, "_copy_paper_tree", side_effect=prepare
        ), mock.patch.object(
            review, "_compile_acm_pdf", side_effect=compile_pdf
        ), mock.patch.object(
            review, "validate_review_cache", return_value=changed
        ), mock.patch.object(
            review, "_canonical_guard_snapshot", return_value={}
        ), self.assertRaisesRegex(review.ReviewBuildError, "cache changed"):
            review.build_review_pdf(
                self.root,
                interface,
                output=output,
                pdflatex=Path("pdflatex"),
                bibtex=Path("bibtex"),
                canonical_before={},
            )

        self.assertEqual(output.read_bytes(), b"last-good-pdf")

    def test_temp_cleanup_failure_never_overwrites_pdf(self) -> None:
        interface = self._write_cache()
        output = self.root / review.OUTPUT_RELATIVE
        output.parent.mkdir(parents=True)
        output.write_bytes(b"last-good-pdf")
        temporary = self.root / "cleanup-fixture"

        class CleanupFailure:
            def __enter__(self) -> str:
                temporary.mkdir()
                return str(temporary)

            def __exit__(self, *_args: object) -> None:
                raise PermissionError("fixture cleanup failure")

        def prepare(_root: Path, build_root: Path, _interface: review.ReviewInterface) -> None:
            build_root.mkdir(parents=True)

        def compile_pdf(
            build_root: Path, _pdflatex: Path, _bibtex: Path
        ) -> Path:
            pdf = build_root / "main-acm.pdf"
            pdf.write_bytes(b"new-review-pdf")
            return pdf

        with mock.patch.object(review, "_verify_benchmark_profile"), mock.patch.object(
            review, "_verify_contemporary_references"
        ), mock.patch.object(
            review.tempfile, "TemporaryDirectory", return_value=CleanupFailure()
        ), mock.patch.object(
            review, "_copy_paper_tree", side_effect=prepare
        ), mock.patch.object(
            review, "_compile_acm_pdf", side_effect=compile_pdf
        ), mock.patch.object(
            review, "validate_review_cache", return_value=interface
        ), mock.patch.object(
            review, "_canonical_guard_snapshot", return_value={}
        ), self.assertRaisesRegex(PermissionError, "fixture cleanup failure"):
            review.build_review_pdf(
                self.root,
                interface,
                output=output,
                pdflatex=Path("pdflatex"),
                bibtex=Path("bibtex"),
                canonical_before={},
            )

        self.assertEqual(output.read_bytes(), b"last-good-pdf")

    def test_state_change_during_publish_restores_last_good_pdf(self) -> None:
        interface = self._write_cache()
        output = self.root / review.OUTPUT_RELATIVE
        output.parent.mkdir(parents=True)
        output.write_bytes(b"last-good-pdf")

        changed_manifest = dict(interface.manifest)
        changed_manifest["number_values_sha256"] = "f" * 64
        changed = review.ReviewInterface(
            cache_root=interface.cache_root,
            json_path=interface.json_path,
            tex_path=interface.tex_path,
            manifest=changed_manifest,
        )

        def prepare(_root: Path, build_root: Path, _interface: review.ReviewInterface) -> None:
            build_root.mkdir(parents=True)

        def compile_pdf(
            build_root: Path, _pdflatex: Path, _bibtex: Path
        ) -> Path:
            pdf = build_root / "main-acm.pdf"
            pdf.write_bytes(b"new-review-pdf")
            return pdf

        with mock.patch.object(review, "_verify_benchmark_profile"), mock.patch.object(
            review, "_verify_contemporary_references"
        ), mock.patch.object(
            review, "_copy_paper_tree", side_effect=prepare
        ), mock.patch.object(
            review, "_compile_acm_pdf", side_effect=compile_pdf
        ), mock.patch.object(
            review,
            "validate_review_cache",
            side_effect=(interface, interface, changed),
        ), mock.patch.object(
            review, "_canonical_guard_snapshot", return_value={}
        ), self.assertRaisesRegex(review.ReviewBuildError, "prior PDF was restored"):
            review.build_review_pdf(
                self.root,
                interface,
                output=output,
                pdflatex=Path("pdflatex"),
                bibtex=Path("bibtex"),
                canonical_before={},
            )

        self.assertEqual(output.read_bytes(), b"last-good-pdf")

    def test_pdf_output_must_not_overlap_cache(self) -> None:
        with self.assertRaisesRegex(review.ReviewBuildError, "must not overlap"):
            review._require_disjoint_review_paths(
                self.cache,
                self.cache / review.resolution.PAPER_TEX_PATH,
            )
        control = self.root / "output/paper-review-cache"
        with self.assertRaisesRegex(review.ReviewBuildError, "must not overlap"):
            review._require_disjoint_review_paths(
                control,
                control / ".review-build.lock",
            )

    def test_cache_cannot_equal_reserved_lock_directory(self) -> None:
        result = review.main(
            (
                "--root",
                str(self.root),
                "--cache-dir",
                "output/paper-review-cache",
            )
        )
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
