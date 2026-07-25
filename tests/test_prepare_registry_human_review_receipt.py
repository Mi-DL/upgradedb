import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_registry_human_review_receipt as prepare  # noqa: E402
import registry_human_review_receipt as public  # noqa: E402


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"


def q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def canonical(value: object) -> bytes:
    return prepare._canonical_json_bytes(value)


def column_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def sheet_xml(
    rows: dict[int, dict[int, tuple[str, str | None, str | None]]],
    sheet_name: str,
) -> bytes:
    root = ET.Element(q(MAIN, "worksheet"))
    data = ET.SubElement(root, q(MAIN, "sheetData"))
    for row_number in sorted(rows):
        row = ET.SubElement(data, q(MAIN, "row"), {"r": str(row_number)})
        for column in sorted(rows[row_number]):
            value, formula, cached = rows[row_number][column]
            reference = f"{column_name(column)}{row_number}"
            if formula is None:
                cell = ET.SubElement(row, q(MAIN, "c"), {"r": reference, "t": "inlineStr"})
                inline = ET.SubElement(cell, q(MAIN, "is"))
                text = ET.SubElement(inline, q(MAIN, "t"))
                if value.startswith(" ") or value.endswith(" "):
                    text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                text.text = value
            else:
                cell = ET.SubElement(row, q(MAIN, "c"), {"r": reference, "t": "str"})
                ET.SubElement(cell, q(MAIN, "f")).text = formula
                ET.SubElement(cell, q(MAIN, "v")).text = cached or value
    validations: list[tuple[str, str]] = []
    if sheet_name == "Review":
        stages = ",".join(row[2] for row in prepare._expected_stage_rows(ROOT))
        validations = [
            (f"N2:N{prepare.SAMPLED_REVIEW_LAST_ROW}", '"Yes,No,Uncertain"'),
            (f"O2:O{prepare.SAMPLED_REVIEW_LAST_ROW}", '"include,exclude,out_of_stage"'),
            (f"P2:P{prepare.SAMPLED_REVIEW_LAST_ROW}", f'"{stages}"'),
            (f"R2:R{prepare.SAMPLED_REVIEW_LAST_ROW}", '"Yes,No"'),
        ]
    elif sheet_name == "Stage definitions":
        validations = [
            (f"K2:K{prepare.STAGE_REVIEW_LAST_ROW}", '"Yes,No,Uncertain"'),
            (f"O2:O{prepare.STAGE_REVIEW_LAST_ROW}", '"Yes,No"'),
        ]
    if validations:
        container = ET.SubElement(root, q(MAIN, "dataValidations"), {"count": str(len(validations))})
        for reference, formula in validations:
            validation = ET.SubElement(
                container, q(MAIN, "dataValidation"), {"type": "list", "sqref": reference}
            )
            ET.SubElement(validation, q(MAIN, "formula1")).text = formula
    if sheet_name == "Review":
        formatting = ET.SubElement(
            root,
            q(MAIN, "conditionalFormatting"),
            {"sqref": f"S2:S{prepare.SAMPLED_REVIEW_LAST_ROW}"},
        )
        for dxf, priority, text in (("0", "1", "Complete"), ("1", "2", "Incomplete"), ("2", "3", "Not started")):
            ET.SubElement(
                formatting,
                q(MAIN, "cfRule"),
                {
                    "type": "containsText", "dxfId": dxf, "priority": priority,
                    "operator": "containsText", "text": text,
                },
            )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def table_xml(table_id: int, name: str, reference: str, headers: tuple[str, ...]) -> bytes:
    root = ET.Element(
        q(MAIN, "table"),
        {
            "id": str(table_id), "name": name, "displayName": name, "ref": reference,
            "headerRowCount": "1", "totalsRowCount": "0", "totalsRowShown": "0",
        },
    )
    columns = ET.SubElement(root, q(MAIN, "tableColumns"), {"count": str(len(headers))})
    for index, header in enumerate(headers, start=1):
        ET.SubElement(columns, q(MAIN, "tableColumn"), {"id": str(index), "name": header})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def relationships(items: list[tuple[str, str, str, str | None]]) -> bytes:
    root = ET.Element(q(REL_PKG, "Relationships"))
    for identity, rel_type, target, target_mode in items:
        attributes = {"Id": identity, "Type": rel_type, "Target": target}
        if target_mode:
            attributes["TargetMode"] = target_mode
        ET.SubElement(root, q(REL_PKG, "Relationship"), attributes)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_workbook(
    path: Path,
    *,
    incomplete_first: bool = False,
    no_record: str | None = None,
    no_has_correction: bool = True,
    no_has_note: bool = True,
    uncertain_record: str | None = None,
    uncertain_has_note: bool = True,
    no_stage_record: str | None = None,
    no_stage_has_correction: bool = True,
    no_stage_has_note: bool = True,
    uncertain_stage_record: str | None = None,
    uncertain_stage_has_note: bool = True,
    legacy_verdict: str | None = None,
    legacy_stage_verdict: str | None = None,
) -> None:
    ledger = prepare._load_ledger(ROOT)
    expected_decisions = prepare._expected_review_rows(ROOT, ledger)
    expected_stages = prepare._expected_stage_rows(ROOT)
    formulas = prepare._expected_formula_maps()
    freeze = json.loads(public.DEFAULT_FREEZE.read_text(encoding="utf-8"))
    sample = json.loads(public.DEFAULT_SAMPLE.read_text(encoding="utf-8"))

    sheets: dict[int, dict[int, dict[int, tuple[str, str | None, str | None]]]] = {
        index: {} for index in range(1, 7)
    }

    def put(sheet: int, row: int, column: int, value: str, formula: str | None = None, cached: str | None = None) -> None:
        sheets[sheet].setdefault(row, {})[column] = (value, formula, cached)

    for reference, value in prepare.README_CELLS.items():
        column, row = prepare._cell_coordinate(reference)
        put(1, row, column, value)

    for column, header in enumerate(prepare.REVIEW_HEADERS, start=1):
        put(2, 1, column, header)
    for row_number, expected in enumerate(expected_decisions, start=2):
        values = list(expected)
        values[2] = str(int(values[2]))
        reviewer = "" if incomplete_first and row_number == 2 else "Reviewer A"
        review_values = values + [reviewer, "2026-07-19", "Yes", "", "", "", "Yes"]
        if row_number == 2 and legacy_verdict is not None:
            review_values[13] = legacy_verdict
        if no_record == expected[0]:
            review_values[13] = "No"
            if no_has_correction:
                review_values[14] = "include"
                review_values[15] = "exp_live"
            if no_has_note:
                review_values[16] = "Adjudicate the proposed correction."
        if uncertain_record == expected[0]:
            review_values[13] = "Uncertain"
            if uncertain_has_note:
                review_values[16] = "The official description requires adjudication."
        for column, value in enumerate(review_values, start=1):
            if value:
                put(2, row_number, column, value)
        put(
            2, row_number, 19, "Complete", formulas["Review"][f"S{row_number}"],
            "Complete",  # deliberately untrusted, including for the incomplete row
        )

    for column, header in enumerate(prepare.STAGE_HEADERS, start=1):
        put(3, 1, column, header)
    for row_number, expected in enumerate(expected_stages, start=2):
        stage_values = list(expected) + [
            "Reviewer A", "2026-07-19", "Yes", "", "", "", "Yes",
        ]
        if row_number == 2 and legacy_stage_verdict is not None:
            stage_values[10] = legacy_stage_verdict
        if no_stage_record == expected[0]:
            stage_values[10] = "No"
            if no_stage_has_correction:
                stage_values[11] = f"{expected[3]} [corrected]"
                stage_values[12] = f"{expected[5]} [corrected]"
            if no_stage_has_note:
                stage_values[13] = "Adjudicate the proposed stage-definition correction."
        if uncertain_stage_record == expected[0]:
            stage_values[10] = "Uncertain"
            if uncertain_stage_has_note:
                stage_values[13] = "The stage definition requires adjudication."
        for column, value in enumerate(stage_values, start=1):
            if value:
                put(3, row_number, column, value)
        put(3, row_number, 16, "Complete", formulas["Stage definitions"][f"P{row_number}"], "Complete")

    for reference, value in prepare.SUMMARY_LABELS.items():
        column, row = prepare._cell_coordinate(reference)
        put(4, row, column, value)
    for reference, formula in prepare.SUMMARY_FORMULAS.items():
        column, row = prepare._cell_coordinate(reference)
        put(4, row, column, "forged-cache", formula, "forged-cache")

    audit_values = {
        "audit_id": freeze["audit_id"],
        "workbook_generated_date": "2026-07-19",
        "registry_evidence_schema": "upgrade-bench/hs92-registry-evidence/3",
        "registry_evidence_sha256": freeze["review_inputs_sha256"]["chains/evidence/registry_evidence.json"],
        "full_ledger_sha256": freeze["review_inputs_sha256"]["chains/evidence/registry_full_audit_ledger.csv"],
        "review_codebook_sha256": freeze["review_inputs_sha256"]["docs/REGISTRY_REVIEW_CODEBOOK.md"],
        "recall_rule_sha256": freeze["review_inputs_sha256"]["chains/evidence/registry_candidate_recall_rule.json"],
        "source_dictionary_member_sha256": freeze["benchmark_identity"]["source_dictionary_member_sha256"],
        "selected_metadata_sha256": freeze["review_inputs_sha256"]["chains/evidence/hs92_selected_product_codes.csv"],
        "human_validation_sample_sha256": hashlib.sha256(
            public.DEFAULT_SAMPLE.read_bytes()
        ).hexdigest(),
        "sample_plan_id": sample["plan_id"],
        "decision_records": str(prepare.FRAME_DECISION_RECORDS),
        "sampled_decision_records": str(prepare.SAMPLED_DECISION_RECORDS),
        "unique_hs6": "588", "human_reviews_completed_at_generation": "0",
        "primary_reviewer": "Reviewer A", "review_start_date": "2026-07-19",
        "review_end_date": "2026-07-19", "outcome_access_declaration": prepare.OUTCOME_ACCESS_DECLARATION,
        "final_audit_owner": "Owner A", "audit_notes": "",
    }
    for column, header in enumerate(prepare.AUDIT_HEADERS, start=1):
        put(5, 1, column, header)
    for row_number, field in enumerate(prepare.AUDIT_FIELDS, start=2):
        values = [
            field, audit_values[field], "Yes" if prepare.AUDIT_REQUIRED[field] else "No",
            prepare.AUDIT_DESCRIPTIONS[field],
        ]
        for column, value in enumerate(values, start=1):
            if value:
                put(5, row_number, column, value)

    ledger_map = {(row["chain_id"], row["code"]): row for row in ledger}
    for column, header in enumerate(prepare.BOUNDARY_HEADERS, start=1):
        put(6, 1, column, header)
    for row_number, identity in enumerate(prepare.BOUNDARY_IDENTITIES, start=2):
        source = ledger_map[identity]
        values = [identity[0], identity[1], source["description"], source["decision"], source["stage"], source["rationale"]]
        for column, value in enumerate(values, start=1):
            if value:
                put(6, row_number, column, value)

    workbook = ET.Element(q(MAIN, "workbook"))
    sheet_nodes = ET.SubElement(workbook, q(MAIN, "sheets"))
    for index, name in enumerate(prepare.SHEET_NAMES, start=1):
        ET.SubElement(
            sheet_nodes, q(MAIN, "sheet"),
            {"name": name, "sheetId": str(index), q(REL_DOC, "id"): f"rId{index}"},
        )

    content_types = ET.Element(q(CONTENT, "Types"))
    ET.SubElement(content_types, q(CONTENT, "Default"), {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, q(CONTENT, "Default"), {"Extension": "xml", "ContentType": "application/xml"})

    worksheet_rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    table_rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/table"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        archive.writestr(
            "_rels/.rels",
            relationships([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "xl/workbook.xml", None)]),
        )
        archive.writestr("xl/workbook.xml", ET.tostring(workbook, encoding="utf-8", xml_declaration=True))
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            relationships([(f"rId{i}", worksheet_rel_type, f"worksheets/sheet{i}.xml", None) for i in range(1, 7)]),
        )
        for index in range(1, 7):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                sheet_xml(sheets[index], prepare.SHEET_NAMES[index - 1]),
            )
        table_specs = {
            2: (
                1,
                "RegistryReviewTable",
                f"A1:S{prepare.SAMPLED_REVIEW_LAST_ROW}",
                prepare.REVIEW_HEADERS,
            ),
            3: (
                2,
                "StageReviewTable",
                f"A1:P{prepare.STAGE_REVIEW_LAST_ROW}",
                prepare.STAGE_HEADERS,
            ),
            5: (
                3,
                "AuditSettingsTable",
                f"A1:D{len(prepare.AUDIT_FIELDS) + 1}",
                prepare.AUDIT_HEADERS,
            ),
            6: (4, "BoundaryCasesTable", "A1:F9", prepare.BOUNDARY_HEADERS),
        }
        for sheet_index, (table_id, name, reference, headers) in table_specs.items():
            archive.writestr(
                f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels",
                relationships([("rId1", table_rel_type, f"/xl/tables/table{table_id}.xml", None)]),
            )
            archive.writestr(
                f"xl/tables/table{table_id}.xml",
                table_xml(table_id, name, reference, headers),
            )


def rewrite_member(path: Path, name: str, transform) -> None:
    replacement = path.with_suffix(".replacement.xlsx")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(replacement, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            raw = source.read(info.filename)
            target.writestr(info.filename, transform(raw) if info.filename == name else raw)
    os.replace(replacement, path)


class PrivateHumanReviewPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        # The implementation deliberately requires output below ROOT/private.
        # Create that ignored boundary explicitly so the test is hermetic in a
        # clean checkout without weakening or bypassing the production guard.
        private_root = ROOT / "private"
        private_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=private_root)
        self.temp = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def workbook(self, **kwargs) -> Path:
        path = self.temp / "completed.xlsx"
        make_workbook(path, **kwargs)
        return path

    def output(self, name: str = "out") -> Path:
        path = self.temp / name
        path.mkdir()
        return path

    def completion(self, **workbook_kwargs):
        workbook = prepare.Workbook(self.workbook(**workbook_kwargs))
        prepare._validate_formulas(workbook)
        ledger = prepare._load_ledger(ROOT)
        prepare._verify_static_sheets(workbook, ledger)
        return prepare._review_completion(
            workbook,
            ROOT,
            ledger,
            {
                "primary_reviewer": "Reviewer A",
                "review_start_date": "2026-07-19",
                "review_end_date": "2026-07-19",
            },
        )

    def test_no_change_raw_workbook_produces_private_record_and_valid_draft(self) -> None:
        workbook = self.workbook()
        private_path, draft_path = prepare.prepare(
            workbook, self.output(), receipt_issued_date="2026-07-19", root=ROOT
        )
        payload = public.validate_receipt(draft_path, root=ROOT, require_release_eligible=True)
        self.assertEqual(payload["construct"]["accepted_change_count"], 0)
        self.assertEqual(
            payload["completion"]["sampled_decision_records_complete"],
            prepare.SAMPLED_DECISION_RECORDS,
        )
        self.assertEqual(hashlib.sha256(private_path.read_bytes()).hexdigest(), payload["private_record"]["normalized_review_sha256"])
        public_text = draft_path.read_text(encoding="utf-8")
        self.assertNotIn("Reviewer A", public_text)
        self.assertNotIn("Owner A", public_text)
        self.assertNotIn(str(workbook), public_text)

    def test_forged_cached_complete_does_not_complete_missing_raw_cells(self) -> None:
        with self.assertRaisesRegex(prepare.PrivateReviewError, "reviewer_id"):
            prepare.prepare(
                self.workbook(incomplete_first=True), self.output(),
                receipt_issued_date="2026-07-19", root=ROOT,
            )

    def test_no_requires_exact_adjudication_partition_and_can_block_release(self) -> None:
        record_id = "CODE-sheep-010420"
        workbook = self.workbook(no_record=record_id)
        with self.assertRaisesRegex(prepare.PrivateReviewError, "requires.*adjudication"):
            prepare.prepare(
                workbook, self.output("missing"), receipt_issued_date="2026-07-19", root=ROOT
            )
        adjudication = {
            "schema_version": prepare.ADJUDICATION_SCHEMA,
            "audit_id": json.loads(public.DEFAULT_FREEZE.read_text(encoding="utf-8"))["audit_id"],
            "records": [{
                "record_id": record_id, "kind": "code_decision", "reviewer_verdict": "No",
                "resolution": "accept_reviewer_change", "final": {"decision": "include", "stage": "exp_live"},
                "adjudicator_id": "Owner A", "adjudication_date": "2026-07-19",
                "note": "The correction is accepted for the versioned registry branch.",
            }],
        }
        adjudication_path = self.temp / "adjudication.json"
        adjudication_path.write_bytes(canonical(adjudication))
        _, draft = prepare.prepare(
            workbook, self.output("accepted"), adjudication_path=adjudication_path,
            receipt_issued_date="2026-07-19", new_registry_id="registry-r4",
            new_benchmark_version="2.2-dev", root=ROOT,
        )
        payload = public.validate_receipt(draft, root=ROOT)
        self.assertEqual(payload["disposition"]["kind"], public.CONSTRUCT_CHANGE_REQUIRED)
        self.assertEqual(payload["completion"]["reviewer_change_count"], 1)
        self.assertEqual(
            payload["construct"]["accepted_change_count"]
            + payload["completion"]["documentation_only_resolution_count"],
            payload["completion"]["adjudication_required_count"],
        )
        with self.assertRaises(public.HumanReviewReleaseBlocked):
            public.validate_receipt(draft, root=ROOT, require_release_eligible=True)

    def test_yes_completes_and_no_requires_correction_fields_and_note(self) -> None:
        decisions, stages, reviewers, required = self.completion()
        self.assertEqual(len(decisions), prepare.SAMPLED_DECISION_RECORDS)
        self.assertEqual(len(stages), prepare.STAGE_DEFINITION_RECORDS)
        self.assertEqual(reviewers, {"Reviewer A"})
        self.assertEqual(required, {})
        self.assertTrue(all(row["verdict"] == "Yes" for row in decisions + stages))

        record_id = "CODE-sheep-010420"
        decisions, _, _, required = self.completion(no_record=record_id)
        reviewed = next(row for row in decisions if row["record_id"] == record_id)
        self.assertEqual(reviewed["verdict"], "No")
        self.assertEqual(
            reviewed["correction"], {"decision": "include", "stage": "exp_live"}
        )
        self.assertTrue(reviewed["reviewer_note"])
        self.assertEqual(set(required), {(record_id, "code_decision")})
        for suffix, workbook in (
            (
                "missing-correction",
                self.workbook(no_record=record_id, no_has_correction=False),
            ),
            (
                "missing-note",
                self.workbook(no_record=record_id, no_has_note=False),
            ),
        ):
            with self.subTest(case=suffix), self.assertRaisesRegex(
                prepare.PrivateReviewError, "No row|correction|reviewer note"
            ):
                parsed = prepare.Workbook(workbook)
                prepare._validate_formulas(parsed)
                ledger = prepare._load_ledger(ROOT)
                prepare._review_completion(
                    parsed,
                    ROOT,
                    ledger,
                    {
                        "primary_reviewer": "Reviewer A",
                        "review_start_date": "2026-07-19",
                        "review_end_date": "2026-07-19",
                    },
                )

        stage_record_id = prepare._expected_stage_rows(ROOT)[0][0]
        _, stages, _, required = self.completion(no_stage_record=stage_record_id)
        reviewed_stage = next(
            row for row in stages if row["record_id"] == stage_record_id
        )
        self.assertEqual(reviewed_stage["verdict"], "No")
        self.assertIsNotNone(reviewed_stage["correction"])
        self.assertTrue(reviewed_stage["reviewer_note"])
        self.assertEqual(set(required), {(stage_record_id, "stage_definition")})
        for suffix, workbook in (
            (
                "stage-missing-correction",
                self.workbook(
                    no_stage_record=stage_record_id,
                    no_stage_has_correction=False,
                ),
            ),
            (
                "stage-missing-note",
                self.workbook(
                    no_stage_record=stage_record_id,
                    no_stage_has_note=False,
                ),
            ),
        ):
            with self.subTest(case=suffix), self.assertRaisesRegex(
                prepare.PrivateReviewError, "No stage row|corrected_|reviewer note"
            ):
                parsed = prepare.Workbook(workbook)
                prepare._validate_formulas(parsed)
                ledger = prepare._load_ledger(ROOT)
                prepare._review_completion(
                    parsed,
                    ROOT,
                    ledger,
                    {
                        "primary_reviewer": "Reviewer A",
                        "review_start_date": "2026-07-19",
                        "review_end_date": "2026-07-19",
                    },
                )

    def test_uncertain_requires_note_and_then_requires_adjudication(self) -> None:
        record_id = "CODE-sheep-010420"
        with self.assertRaisesRegex(prepare.PrivateReviewError, "Uncertain row"):
            self.completion(
                uncertain_record=record_id,
                uncertain_has_note=False,
            )

        _, _, _, required = self.completion(uncertain_record=record_id)
        self.assertEqual(set(required), {(record_id, "code_decision")})
        self.assertEqual(required[(record_id, "code_decision")]["verdict"], "Uncertain")

        stage_record_id = prepare._expected_stage_rows(ROOT)[0][0]
        with self.assertRaisesRegex(prepare.PrivateReviewError, "Uncertain stage row"):
            self.completion(
                uncertain_stage_record=stage_record_id,
                uncertain_stage_has_note=False,
            )
        _, _, _, required = self.completion(uncertain_stage_record=stage_record_id)
        self.assertEqual(set(required), {(stage_record_id, "stage_definition")})
        self.assertEqual(
            required[(stage_record_id, "stage_definition")]["verdict"], "Uncertain"
        )

    def test_legacy_correct_and_change_verdicts_are_rejected(self) -> None:
        for verdict in ("Correct", "Change"):
            with self.subTest(sheet="Review", verdict=verdict), self.assertRaisesRegex(
                prepare.PrivateReviewError, "verdict.*invalid|invalid.*verdict"
            ):
                self.completion(legacy_verdict=verdict)
            with self.subTest(
                sheet="Stage definitions", verdict=verdict
            ), self.assertRaisesRegex(
                prepare.PrivateReviewError, "verdict.*invalid|invalid.*verdict"
            ):
                self.completion(legacy_stage_verdict=verdict)

    def test_formula_tampering_duplicate_extra_hidden_macro_and_external_fail(self) -> None:
        cases = []
        formula = self.workbook()
        def tamper_formula(raw: bytes) -> bytes:
            root = ET.fromstring(raw)
            first = root.find(f".//{q(MAIN, 'f')}")
            assert first is not None
            first.text = "1"
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
        rewrite_member(
            formula, "xl/worksheets/sheet2.xml",
            tamper_formula,
        )
        cases.append((formula, "formula"))

        hidden = self.temp / "hidden.xlsx"
        make_workbook(hidden)
        rewrite_member(
            hidden, "xl/workbook.xml",
            lambda raw: raw.replace(b'name="Review"', b'name="Review" state="hidden"', 1),
        )
        cases.append((hidden, "hidden"))

        external = self.temp / "external.xlsx"
        make_workbook(external)
        rewrite_member(
            external, "_rels/.rels",
            lambda raw: raw.replace(b'</ns0:Relationships>', b'<ns0:Relationship Id="x" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.test" TargetMode="External" /></ns0:Relationships>'),
        )
        cases.append((external, "external"))

        macro = self.temp / "macro.xlsx"
        make_workbook(macro)
        with zipfile.ZipFile(macro, "a", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("xl/vbaProject.bin", b"not-a-real-macro")
        cases.append((macro, "macro"))

        extra = self.temp / "extra.xlsx"
        make_workbook(extra)
        extra_row = prepare.SAMPLED_REVIEW_LAST_ROW + 1
        rewrite_member(
            extra, "xl/worksheets/sheet2.xml",
            lambda raw: raw.replace(
                b'</ns0:sheetData>',
                (
                    f'<ns0:row r="{extra_row}"><ns0:c r="A{extra_row}" t="inlineStr">'
                    '<ns0:is><ns0:t>EXTRA</ns0:t></ns0:is></ns0:c></ns0:row>'
                    '</ns0:sheetData>'
                ).encode("utf-8"),
            ),
        )
        cases.append((extra, "extra"))

        duplicate = self.temp / "duplicate.xlsx"
        make_workbook(duplicate)
        rewrite_member(
            duplicate, "xl/worksheets/sheet2.xml",
            lambda raw: raw.replace(b'CODE-sheep-010420', b'CODE-sheep-010410', 1),
        )
        cases.append((duplicate, "immutable"))

        for index, (workbook, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(prepare.PrivateReviewError, message):
                prepare.prepare(
                    workbook, self.output(f"bad-{index}"), receipt_issued_date="2026-07-19", root=ROOT
                )

    def test_private_path_boundary_blocks_public_or_escaped_output(self) -> None:
        workbook = self.workbook()
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(prepare.PrivateReviewError, "ROOT/private"):
                prepare.prepare(
                    workbook, Path(outside), receipt_issued_date="2026-07-19", root=ROOT
                )


if __name__ == "__main__":
    unittest.main()
