from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_v2_number_alignment as alignment  # noqa: E402


def profile_fixture() -> dict:
    hs6 = {
        "sheep": 1,
        "cotton": 2,
        "aluminium": 3,
        "nickel": 4,
        "cocoa": 5,
        "oilseed-soy": 6,
    }
    return {
        "benchmark_version": alignment.paper_numbers.BENCHMARK_VERSION,
        "totals": {
            "b1_candidate_entries": 100,
            "b1_positive_entries": 20,
            "b2_candidate_lanes": 300,
            "b2_positive_entry_groups": 20,
            "b2_positive_lanes": 40,
        },
        "chains": [
            {"chain": chain, "graph": {"hs6_products": count}}
            for chain, count in hs6.items()
        ],
    }


def paper_fixture() -> dict[str, str]:
    result = {
        "VTwoTrackACandidates": "1{,}000",
        "VTwoTrackAPositives": "50",
        "VTwoTrackBOneCandidates": "100",
        "VTwoTrackBOnePositives": "20",
        "VTwoTrackBTwoCandidates": "300",
        "VTwoTrackBTwoPositives": "40",
        "VTwoBOneCoverageMainAllRealizedEntries": "22",
        "VTwoBOneCoverageMainCoveredRealizedEntries": "20",
        "VTwoBOneCoverageMainAllLateStartLanes": "45",
        "VTwoBOneCoverageMainEligibleMarketLateStartLanes": "40",
        "VTwoBOneCoverageMainInactiveMarketLateStartLanes": "5",
        "VTwoProductSpaceMainBOneCandidates": "100",
        "VTwoProductSpaceMainBOnePositives": "20",
        "VTwoEligibilityThresholdHundredTrackACandidates": "1{,}000",
        "VTwoEligibilityThresholdHundredTrackAPositives": "50",
        "VTwoEligibilityThresholdHundredTrackBOneCandidates": "100",
        "VTwoEligibilityThresholdHundredTrackBOnePositives": "20",
        "VTwoEligibilityThresholdHundredTrackBTwoCandidates": "300",
        "VTwoEligibilityThresholdHundredTrackBTwoPositives": "40",
        "VTwoRegistryIncludedCodes": "21",
    }
    for index, chain in enumerate(alignment.CHAINS, start=1):
        result[f"VTwoRegistry{alignment.CHAIN_MACRO[chain]}ActiveCodes"] = str(index)
    return result


def evidence_fixture() -> dict:
    return {
        "chains": {
            chain: {
                "decisions": [
                    {
                        "candidate_source": "observable_regex",
                        "decision": "include",
                    }
                ]
            }
            for chain in alignment.CHAINS
        }
    }


class ProfilePaperJoinTests(unittest.TestCase):
    def test_exact_task_and_hs6_joins_pass(self) -> None:
        alignment._profile_paper_crosscheck(profile_fixture(), paper_fixture())

    def test_task_count_mismatch_fails_closed(self) -> None:
        numbers = paper_fixture()
        numbers["VTwoTrackBTwoCandidates"] = "301"
        with self.assertRaisesRegex(alignment.NumberAlignmentError, "profile/paper count"):
            alignment._profile_paper_crosscheck(profile_fixture(), numbers)

    def test_chain_hs6_mismatch_fails_closed(self) -> None:
        numbers = paper_fixture()
        numbers["VTwoRegistrySheepActiveCodes"] = "2"
        with self.assertRaisesRegex(alignment.NumberAlignmentError, "HS6 mismatch"):
            alignment._profile_paper_crosscheck(profile_fixture(), numbers)

    def test_coverage_positive_join_mismatch_fails_closed(self) -> None:
        numbers = paper_fixture()
        numbers["VTwoBOneCoverageMainCoveredRealizedEntries"] = "19"
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "cohort-count join mismatch"
        ):
            alignment._profile_paper_crosscheck(profile_fixture(), numbers)

    def test_reference_threshold_join_mismatch_fails_closed(self) -> None:
        numbers = paper_fixture()
        numbers["VTwoEligibilityThresholdHundredTrackBTwoPositives"] = "39"
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "cohort-count join mismatch"
        ):
            alignment._profile_paper_crosscheck(profile_fixture(), numbers)

    def test_coverage_lane_accounting_must_close(self) -> None:
        numbers = paper_fixture()
        numbers["VTwoBOneCoverageMainInactiveMarketLateStartLanes"] = "4"
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "lane accounting does not close"
        ):
            alignment._profile_paper_crosscheck(profile_fixture(), numbers)


class PaperTexInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tex = Path(self.temporary.name) / "v2_numbers.tex"
        self.sources = {
            "z/source.json": "f" * 64,
            "a/source.json": "0" * 64,
        }
        self.logical_numbers = {
            "VTwoTrackACandidates": "317{,}624",
            "VTwoAluminiumGBDTTrackAAP": "0.1830",
        }
        self.json_order_numbers = dict(sorted(self.logical_numbers.items()))

    def _write(self, content: str) -> None:
        self.tex.write_bytes(content.encode("utf-8"))

    def test_logical_tex_order_matches_alphabetic_json_mapping(self) -> None:
        self._write(
            alignment.paper_numbers.render_tex(self.logical_numbers, self.sources)
        )
        alignment._verify_paper_tex_interface(
            self.tex,
            self.json_order_numbers,
            self.sources,
        )

    def test_changed_missing_and_extra_macros_fail_closed(self) -> None:
        cases = (
            {**self.logical_numbers, "VTwoTrackACandidates": "1"},
            {"VTwoTrackACandidates": "317{,}624"},
            {**self.logical_numbers, "VTwoUnexpected": "1"},
        )
        for observed in cases:
            with self.subTest(observed=observed):
                self._write(alignment.paper_numbers.render_tex(observed, self.sources))
                with self.assertRaisesRegex(
                    alignment.NumberAlignmentError,
                    "JSON and TeX interfaces differ",
                ):
                    alignment._verify_paper_tex_interface(
                        self.tex,
                        self.json_order_numbers,
                        self.sources,
                    )

    def test_duplicate_and_malformed_macros_fail_closed(self) -> None:
        valid = alignment.paper_numbers.render_tex(
            self.logical_numbers,
            self.sources,
        )
        macro = "\\newcommand{\\VTwoTrackACandidates}{317{,}624}\n"
        for content, message in (
            (valid + macro, "repeats macro"),
            (valid.replace("\\newcommand", "\\renewcommand", 1), "malformed macro"),
        ):
            with self.subTest(message=message):
                self._write(content)
                with self.assertRaisesRegex(alignment.NumberAlignmentError, message):
                    alignment._verify_paper_tex_interface(
                        self.tex,
                        self.json_order_numbers,
                        self.sources,
                    )

    def test_source_order_and_canonical_newline_fail_closed(self) -> None:
        valid = alignment.paper_numbers.render_tex(
            self.logical_numbers,
            self.sources,
        )
        source_lines = valid.splitlines()
        source_lines[2], source_lines[3] = source_lines[3], source_lines[2]
        cases = (
            ("\n".join(source_lines) + "\n", "source maps differ"),
            (valid.replace("\n", "\r\n"), "canonical LF"),
            (valid.rstrip("\n"), "canonical LF"),
        )
        for content, message in cases:
            with self.subTest(message=message):
                self._write(content)
                with self.assertRaisesRegex(alignment.NumberAlignmentError, message):
                    alignment._verify_paper_tex_interface(
                        self.tex,
                        self.json_order_numbers,
                        self.sources,
                    )


class ManuscriptNumberLintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = {
            "summary": {
                "included_codes": 6,
                "active_stages": 3,
                "observable_candidate_records": 6,
                "legacy_only_records": 4,
                "decision_records": 10,
                "unique_reviewed_hs6": 9,
                "excluded_codes": 3,
                "out_of_stage_codes": 1,
                "historical_active_retained": 2,
                "new_active_added": 4,
            }
        }
        self.protocol = {
            "quality_controls": {"source_rows_automatically_regex_scanned": 5022}
        }
        self.evidence = evidence_fixture()

    def test_governed_registry_literals_pass_when_aligned(self) -> None:
        text = (
            "The current registry contains 6 included HS6 codes across 3 families. "
            "The scan yields 6 records, split into 6 included, 0 excluded, and 0 out of stage. "
            "Adding 4 legacy-only records gives a 10-decision ledger covering 9 unique HS6 "
            "codes; its full split is 6 included, 3 excluded, and 1 out of stage. "
            "The revision retains all 2 previously active codes and adds 4. "
            "The scan applies to all 5{,}022 source rows."
        )
        alignment._lint_registry_literals(
            text, self.audit, self.evidence, self.protocol
        )

    def test_conflicting_hard_coded_registry_literal_fails(self) -> None:
        text = "The current registry contains 7 included HS6 codes across 3 families."
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "hard-coded manuscript claim conflicts"
        ):
            alignment._lint_registry_literals(
                text, self.audit, self.evidence, self.protocol
            )

    def test_release_mode_rejects_stale_status_language(self) -> None:
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "stale/pending status language"
        ):
            alignment._reject_stale_release_language(
                "This working-draft state still uses a historical layout placeholder."
            )

    def test_current_release_prose_has_no_stale_marker_only_after_rewrite(self) -> None:
        alignment._reject_stale_release_language(
            "All result interfaces and review receipts are verified and current."
        )

    def test_bare_decimal_in_results_section_fails(self) -> None:
        body = (
            r"\section{Reference results on the frozen future cohort}" + "\n"
            r"The copied AP is 0.1234." + "\n"
            r"\section{Limitations and open questions}"
        )
        appendix = (
            r"\section{Additional reference results}" + "\n"
            r"\setlength{\tabcolsep}{4.5pt}" + "\n"
            r"\section{Reference implementation details}"
        )
        with self.assertRaisesRegex(
            alignment.NumberAlignmentError, "bare decimal literals"
        ):
            alignment._reject_hardcoded_result_decimals(body, appendix)

    def test_layout_decimal_is_not_an_empirical_claim(self) -> None:
        body = (
            r"\section{Reference results on the frozen future cohort}" + "\n"
            r"\setlength{\tabcolsep}{4.5pt}" + "\n"
            r"\section{Limitations and open questions}"
        )
        appendix = (
            r"\section{Additional reference results}" + "\n"
            r"\setlength{\tabcolsep}{3.5pt}" + "\n"
            r"\section{Reference implementation details}"
        )
        alignment._reject_hardcoded_result_decimals(body, appendix)

    def test_public_status_docs_reject_pending_claim_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in alignment.PUBLIC_STATUS_DOCS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("current\n", encoding="utf-8")
            (root / "results_v2/CLAIM_LEDGER.md").write_text(
                "| claim | **rerun/regeneration pending** |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "public status documents retain"
            ):
                alignment._reject_stale_public_status_docs(root)

    def test_public_status_docs_accept_completed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in alignment.PUBLIC_STATUS_DOCS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("complete and verified\n", encoding="utf-8")
            alignment._reject_stale_public_status_docs(root)

    @staticmethod
    def _write_chronology_docs(root: Path, text: str = "Independent release gates.\n") -> None:
        for relative in alignment.PUBLIC_CHRONOLOGY_DOCS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def test_compute_complete_then_review_started_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "The results were completed and frozen. Human review then began.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_review_started_after_compute_complete_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "results_v2/README.md").write_text(
                "Human review started after the computations were completed.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_review_deferred_until_sealed_run_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "paper/body.tex").write_text(
                "Human review did not begin until the formal run had been sealed.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_completed_outputs_before_review_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "paper/appendix.tex").write_text(
                "Verified benchmark outputs were available before human review began.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_results_already_complete_when_review_began_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "When human review began, the results had already been completed.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_only_then_review_started_is_a_chronology_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "results_v2/CLAIM_LEDGER.md").write_text(
                "The result set was finalized; only then did human review start.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_independent_gates_and_conditional_rerun_do_not_imply_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(
                root,
                "Release requires complete computations and a canonical review receipt. "
                "The branches are independent and this list asserts no execution order. "
                "If review accepts a construct change, rerun all dependent results.\n",
            )
            alignment._reject_public_chronology_leaks(
                root, review_receipt_verified=False
            )

    def test_negated_compute_before_review_claim_is_not_misread(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "The results were not completed before human review began.\n",
                encoding="utf-8",
            )
            alignment._reject_public_chronology_leaks(
                root, review_receipt_verified=False
            )

    def test_completed_review_claim_requires_verified_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "The human review has been completed.\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "without a verified canonical receipt"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=False
                )
            alignment._reject_public_chronology_leaks(
                root, review_receipt_verified=True
            )

    def test_row_completion_claim_requires_verified_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "All 610 human-review rows have been completed.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                alignment.NumberAlignmentError, "without a verified canonical receipt"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=False
                )

    def test_conditional_review_completion_is_not_a_status_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            (root / "README.md").write_text(
                "If the human review has been completed, verify its canonical receipt.\n",
                encoding="utf-8",
            )
            alignment._reject_public_chronology_leaks(
                root, review_receipt_verified=False
            )

    def test_optional_rebuttal_is_scanned_only_when_release_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_chronology_docs(root)
            rebuttal = root / "paper/REBUTTAL_NOTES.md"
            rebuttal.parent.mkdir(parents=True, exist_ok=True)
            rebuttal.write_text(
                "The formal run was sealed before human review began.\n",
                encoding="utf-8",
            )
            alignment._reject_public_chronology_leaks(
                root, review_receipt_verified=True
            )
            selected = set(alignment.release_manifest.ROOT_RELEASE_FILES) | {
                "paper/REBUTTAL_NOTES.md"
            }
            with mock.patch.object(
                alignment.release_manifest, "ROOT_RELEASE_FILES", selected
            ), self.assertRaisesRegex(
                alignment.NumberAlignmentError, "computation-before-human-review"
            ):
                alignment._reject_public_chronology_leaks(
                    root, review_receipt_verified=True
                )

    def test_current_release_facing_prose_has_no_chronology_leak(self) -> None:
        alignment._reject_public_chronology_leaks(
            ROOT, review_receipt_verified=False, profile="repository"
        )

    def test_chronology_inventory_covers_release_prose_not_protocol_json(self) -> None:
        governed = {path.as_posix() for path in alignment.PUBLIC_CHRONOLOGY_DOCS}
        self.assertTrue(
            {
                "README.md",
                "ARTIFACT.md",
                "DATA_LICENSE.md",
                "BENCHMARK_V2_SPEC.md",
                "results_v2/README.md",
                "results_v2/CLAIM_LEDGER.md",
                "docs/V2_RELEASE_WORKFLOW.md",
                "benchmark/upgrade-bench-v2/README.md",
                "benchmark/upgrade-bench-v2/DATASHEET.md",
                "paper/body.tex",
                "paper/appendix.tex",
            }.issubset(governed)
        )
        self.assertNotIn(
            "chains/evidence/registry_curation_protocol.json", governed
        )


class StrictJSONTests(unittest.TestCase):
    def test_tex_integer_parser_accepts_canonical_grouping(self) -> None:
        self.assertEqual(alignment._tex_int("302{,}406", "fixture"), 302406)

    def test_tex_integer_parser_rejects_noncanonical_grouping(self) -> None:
        with self.assertRaisesRegex(alignment.NumberAlignmentError, "canonical TeX integer"):
            alignment._tex_int("302,406", "fixture")

    def test_alignment_gate_and_tests_are_publicly_allowlisted(self) -> None:
        self.assertIn(
            "tools/verify_v2_number_alignment.py",
            alignment.public_policy.PUBLIC_CURRENT_TOOL_ALLOWLIST,
        )
        self.assertIn(
            "tests/test_verify_v2_number_alignment.py",
            alignment.public_policy.PUBLIC_CURRENT_TEST_ALLOWLIST,
        )

    def test_repository_release_ci_runs_alignment_gate(self) -> None:
        workflow = (ROOT / ".github/workflows/release-artifact.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "python tools/verify_v2_number_alignment.py --mode release --profile repository",
            workflow,
        )


class OrchestrationTests(unittest.TestCase):
    def _run(
        self, profile: str
    ) -> tuple[mock.Mock, mock.Mock, mock.Mock, mock.Mock, mock.Mock]:
        audit = {"summary": {"included_codes": 1}}
        profile_payload = {"chains": []}

        def load(path: Path, role: str, **_: object) -> dict:
            if path.name == "registry_audit.json":
                return audit
            if path.name == "registry_evidence.json":
                return {"chains": {}}
            if path.name == "registry_curation_protocol.json":
                return {"quality_controls": {}}
            if path.name == "v2_benchmark_profile.json":
                return profile_payload
            raise AssertionError((path, role))

        review = mock.Mock(return_value={})
        public_receipt = mock.Mock(return_value={})
        resolved = mock.Mock(return_value={})
        paper_verify = mock.Mock()
        ultra_verify = mock.Mock()
        with (
            mock.patch.object(alignment.loco_results, "verify_outputs"),
            mock.patch.object(
                alignment.ultra_results, "verify_outputs", ultra_verify
            ),
            mock.patch.object(alignment.benchmark_profile, "verify_outputs"),
            mock.patch.object(alignment.contemporary_results, "verify_outputs"),
            mock.patch.object(
                alignment.contemporary_results,
                "parse_tex_macros",
                return_value={},
            ),
            mock.patch.object(
                alignment.audit_chain_registry,
                "verify_outputs",
                return_value=audit,
            ),
            mock.patch.object(alignment, "_load_json", side_effect=load),
            mock.patch.object(
                alignment,
                "_paper_interface",
                return_value=({"sources": {}}, {}),
            ),
            mock.patch.object(alignment, "_profile_paper_crosscheck"),
            mock.patch.object(alignment, "_lint_manuscript"),
            mock.patch.object(alignment, "_reject_stale_public_status_docs"),
            mock.patch.object(alignment, "_reject_public_chronology_leaks"),
            mock.patch.object(
                alignment.registry_human_review_receipt,
                "verify_release_gate",
                review,
            ),
            mock.patch.object(
                alignment.resolve_v2_invalidation,
                "verify_public_receipt",
                public_receipt,
            ),
            mock.patch.object(
                alignment.resolve_v2_invalidation,
                "verify_resolved",
                resolved,
            ),
            mock.patch.object(
                alignment.paper_numbers, "verify_outputs", paper_verify
            ),
        ):
            alignment.verify_alignment(mode="release", profile=profile)
        return review, public_receipt, resolved, paper_verify, ultra_verify

    def test_repository_release_requires_public_receipt_and_human_review(self) -> None:
        review, public_receipt, resolved, paper_verify, ultra_verify = self._run(
            "repository"
        )
        review.assert_called_once_with(alignment.ROOT)
        public_receipt.assert_called_once_with(alignment.ROOT, profile="repository")
        resolved.assert_not_called()
        paper_verify.assert_not_called()
        ultra_verify.assert_not_called()

    def test_full_release_adds_private_recompute_gate(self) -> None:
        review, public_receipt, resolved, paper_verify, ultra_verify = self._run("full")
        review.assert_called_once_with(alignment.ROOT)
        public_receipt.assert_called_once_with(alignment.ROOT, profile="full")
        resolved.assert_called_once_with(alignment.ROOT)
        paper_verify.assert_called_once()
        ultra_verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
