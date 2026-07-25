#!/usr/bin/env python3
"""Build the frozen registry-lexicon negative-control evidence.

This diagnostic applies a fixed set of lexical challenge regular expressions
to every BACI HS92 product-description row.  For every challenge hit it asks a
single question: did the chain's *main* frozen recall regular expression also
match that same official description?

The result is evidence about the variants that were actually tested.  It is
not a proof that the main lexicons contain every possible synonym, and it does
not adjudicate inclusion, exclusion, stage fit, or the 610-row registry ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "BACI_HS92_V202401b.zip"
DEFAULT_RULE = ROOT / "chains" / "evidence" / "registry_candidate_recall_rule.json"
DEFAULT_OUTPUT = ROOT / "chains" / "evidence" / "registry_lexicon_negative_control.json"


class NegativeControlError(RuntimeError):
    """Raised when source, rule, or committed evidence fails closed."""


# This is a deliberately finite, targeted challenge list.  It contains
# observed and zero-hit probes; retaining the latter makes the scope of the
# test visible.  These terms are lexical diagnostics only and are never used
# as evidence for an include or stage decision.
CHALLENGES: dict[str, tuple[dict[str, str], ...]] = {
    "sheep": (
        {
            "id": "ovine_word",
            "class": "lexical_alternative",
            "regex": r"\bovine\b",
            "note": "Tests the standalone adjective without matching the substring in 'bovine'.",
        },
        {
            "id": "mutton_word",
            "class": "product_term",
            "regex": r"\bmutton\b",
            "note": "Tests an alternative sheep-meat term.",
        },
        {
            "id": "sheep_inflection",
            "class": "morphology",
            "regex": r"\bsheeps?\b",
            "note": "Tests singular and plural morphology of sheep.",
        },
        {
            "id": "lamb_inflection",
            "class": "morphology",
            "regex": r"\blambs?\b",
            "note": "Tests singular and plural morphology of lamb.",
        },
        {
            "id": "wool_inflection",
            "class": "morphology",
            "regex": r"\bwools?\b",
            "note": "Tests singular and plural morphology of wool.",
        },
        {
            "id": "woollen_woolen",
            "class": "spelling_derivative",
            "regex": r"\bwool(?:len|en)\b",
            "note": "Tests British and US adjectival spellings.",
        },
        {
            "id": "sheepskin_lambskin_compounds",
            "class": "compound_spelling",
            "regex": r"\b(?:sheep|lamb)(?:[ -]?skins?)\b",
            "note": "Tests closed, spaced, and hyphenated skin compounds.",
        },
        {
            "id": "fleece_word",
            "class": "associated_product_term",
            "regex": r"\bfleeces?\b",
            "note": "Tests a wool-associated product term as a possible lexical surrogate.",
        },
        {
            "id": "lanolin_word",
            "class": "associated_product_term",
            "regex": r"\blanolin\b",
            "note": "Tests a wool-grease product term as a possible lexical surrogate.",
        },
        {
            "id": "ewe_ram_inflections",
            "class": "species_term",
            "regex": r"\b(?:ewes?|rams?)\b",
            "note": "Tests sex-specific sheep terms and their plurals.",
        },
    ),
    "cotton": (
        {
            "id": "cotton_inflection",
            "class": "morphology",
            "regex": r"\bcottons?\b",
            "note": "Tests singular and plural morphology of cotton.",
        },
        {
            "id": "cottonseed_spacing",
            "class": "compound_spelling",
            "regex": r"\bcotton(?:[ -]?seeds?)\b",
            "note": "Tests closed, spaced, and hyphenated cottonseed forms.",
        },
        {
            "id": "cotton_linter_spacing",
            "class": "compound_spelling",
            "regex": r"\bcotton[ -]?linters?\b",
            "note": "Tests closed, spaced, and hyphenated cotton-linter forms.",
        },
        {
            "id": "gossypium_word",
            "class": "botanical_term",
            "regex": r"\bgossypium\b",
            "note": "Tests the botanical genus as a possible lexical surrogate.",
        },
        {
            "id": "denim_word",
            "class": "associated_product_term",
            "regex": r"\bdenim\b",
            "note": "Tests a fabric term that could otherwise occur without the material token.",
        },
        {
            "id": "calico_muslin_gingham_words",
            "class": "associated_product_term",
            "regex": r"\b(?:calico|muslin|gingham)\b",
            "note": "Tests three fabric terms as possible lexical surrogates.",
        },
    ),
    "aluminium": (
        {
            "id": "aluminium_british_spelling",
            "class": "spelling",
            "regex": r"\baluminium\b",
            "note": "Tests the British spelling used by the classification.",
        },
        {
            "id": "aluminum_us_spelling",
            "class": "spelling",
            "regex": r"\baluminum\b",
            "note": "Tests the US spelling as an alternative.",
        },
        {
            "id": "alumina_word",
            "class": "morphology",
            "regex": r"\balumina\b",
            "note": "Tests the alumina noun separately from aluminium.",
        },
        {
            "id": "aluminous_word",
            "class": "morphology",
            "regex": r"\baluminous\b",
            "note": "Tests the aluminous adjective.",
        },
        {
            "id": "aluminate_compounds",
            "class": "morphology_and_compound",
            "regex": r"\b(?:[a-z]+)?aluminates?\b",
            "note": "Tests aluminate inflection and prefixed fluoroaluminate compounds.",
        },
        {
            "id": "bauxite_inflection",
            "class": "morphology",
            "regex": r"\bbauxites?\b",
            "note": "Tests singular and plural morphology of bauxite.",
        },
        {
            "id": "corundum_word",
            "class": "lexical_alternative",
            "regex": r"\bcorundum\b",
            "note": "Tests corundum as explicitly frozen in the main lexicon.",
        },
    ),
    "nickel": (
        {
            "id": "nickel_inflection",
            "class": "morphology",
            "regex": r"\bnickels?\b",
            "note": "Tests singular and plural morphology of nickel.",
        },
        {
            "id": "cupronickel_spacing",
            "class": "compound_spelling",
            "regex": r"\bcupro[ -]?nickel\b",
            "note": "Tests closed, spaced, and hyphenated cupro-nickel forms.",
        },
        {
            "id": "nickel_silver_spacing",
            "class": "compound_spelling",
            "regex": r"\bnickel[ -]?silver\b",
            "note": "Tests spaced and hyphenated nickel-silver forms.",
        },
        {
            "id": "nickelic_nickelous_words",
            "class": "morphology",
            "regex": r"\b(?:nickelic|nickelous)\b",
            "note": "Tests two chemical adjectival forms.",
        },
    ),
    "cocoa": (
        {
            "id": "cocoa_inflection",
            "class": "morphology",
            "regex": r"\bcocoas?\b",
            "note": "Tests singular and plural morphology of cocoa.",
        },
        {
            "id": "cacao_word",
            "class": "spelling_or_lexical_alternative",
            "regex": r"\bcacao\b",
            "note": "Tests cacao as an alternative focal token.",
        },
        {
            "id": "chocolate_inflection",
            "class": "morphology",
            "regex": r"\bchocolates?\b",
            "note": "Tests singular and plural morphology of chocolate.",
        },
        {
            "id": "cocoa_butter_phrase",
            "class": "punctuation_and_phrase",
            "regex": r"\bcocoa\b.{0,15}\bbutter\b",
            "note": "Tests the cocoa-butter phrase across intervening classification punctuation.",
        },
        {
            "id": "theobroma_word",
            "class": "botanical_term",
            "regex": r"\btheobroma\b",
            "note": "Tests the botanical genus as a possible lexical surrogate.",
        },
    ),
    "oilseed-soy": (
        {
            "id": "soy_inflection",
            "class": "morphology",
            "regex": r"\bsoys?\b",
            "note": "Tests singular and plural morphology of soy.",
        },
        {
            "id": "soya_spelling",
            "class": "spelling",
            "regex": r"\bsoya\b",
            "note": "Tests the soya spelling used by the classification.",
        },
        {
            "id": "soybean_closed_inflection",
            "class": "compound_spelling",
            "regex": r"\bsoybeans?\b",
            "note": "Tests closed soybean forms and number inflection.",
        },
        {
            "id": "soy_bean_spacing",
            "class": "compound_spelling",
            "regex": r"\bsoy[ -]?beans?\b",
            "note": "Tests closed, spaced, and hyphenated soy-bean forms.",
        },
        {
            "id": "soya_bean_spacing",
            "class": "compound_spelling",
            "regex": r"\bsoya[ -]?beans?\b",
            "note": "Tests closed, spaced, and hyphenated soya-bean forms.",
        },
        {
            "id": "soja_word",
            "class": "spelling_or_lexical_alternative",
            "regex": r"\bsoja\b",
            "note": "Tests soja as an alternative spelling.",
        },
        {
            "id": "glycine_max_phrase",
            "class": "botanical_term",
            "regex": r"\bglycine\s+max\b",
            "note": "Tests the botanical species name as a possible lexical surrogate.",
        },
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _records_hash(records: Sequence[Mapping[str, str]]) -> str:
    snapshot = [
        {"code": str(row["code"]), "description": str(row["description"])}
        for row in records
    ]
    return _sha256(_canonical_bytes(snapshot))


def _challenge_definition() -> list[dict[str, str]]:
    return [
        {"chain_id": chain_id, **dict(challenge)}
        for chain_id, challenges in CHALLENGES.items()
        for challenge in challenges
    ]


def read_source_rows(archive_path: Path, member_name: str) -> tuple[bytes, list[dict[str, str]]]:
    if not archive_path.is_file():
        raise NegativeControlError(f"BACI archive not found: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.read(member_name)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise NegativeControlError(
            f"cannot read {member_name!r} from {archive_path}: {exc}"
        ) from exc

    try:
        reader = csv.DictReader(io.StringIO(member.decode("utf-8-sig"), newline=""))
        rows = [
            {
                "code": (row.get("code") or "").strip(),
                "description": row.get("description") or "",
            }
            for row in reader
        ]
    except UnicodeDecodeError as exc:
        raise NegativeControlError(f"source member is not UTF-8: {exc}") from exc

    if reader.fieldnames != ["code", "description"]:
        raise NegativeControlError(
            f"unexpected source columns {reader.fieldnames!r}; expected ['code', 'description']"
        )
    if any(not row["code"] or not row["description"] for row in rows):
        raise NegativeControlError("source contains an empty code or description")
    if len({row["code"] for row in rows}) != len(rows):
        raise NegativeControlError("source contains duplicate product codes")
    return member, rows


def evaluate_challenges(
    rows: Sequence[Mapping[str, str]],
    lexicons: Mapping[str, Mapping[str, Any]],
    challenges: Mapping[str, Iterable[Mapping[str, str]]] = CHALLENGES,
) -> dict[str, Any]:
    """Evaluate challenge hits and main-lexicon coverage on official descriptions."""

    unknown = sorted(set(challenges) - set(lexicons))
    missing = sorted(set(lexicons) - set(challenges))
    if unknown or missing:
        raise NegativeControlError(
            f"challenge/lexicon chain mismatch; unknown={unknown}, missing={missing}"
        )

    chain_results: dict[str, Any] = {}
    all_pair_hits: set[tuple[str, str]] = set()
    all_unique_codes: set[str] = set()
    total_hit_events = 0
    total_unrecalled_events = 0
    challenges_with_hits = 0
    challenges_without_hits = 0

    for chain_id, chain_challenges in challenges.items():
        main_regex = str(lexicons[chain_id]["regex"])
        try:
            main_pattern = re.compile(main_regex, re.IGNORECASE)
        except re.error as exc:
            raise NegativeControlError(f"invalid main regex for {chain_id}: {exc}") from exc

        main_hits = [row for row in rows if main_pattern.search(str(row["description"]))]
        chain_pair_hits: set[str] = set()
        chain_unrecalled_pairs: set[str] = set()
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw_challenge in chain_challenges:
            challenge = dict(raw_challenge)
            challenge_id = challenge["id"]
            if challenge_id in seen_ids:
                raise NegativeControlError(f"duplicate challenge id for {chain_id}: {challenge_id}")
            seen_ids.add(challenge_id)
            try:
                pattern = re.compile(challenge["regex"], re.IGNORECASE)
            except re.error as exc:
                raise NegativeControlError(
                    f"invalid challenge regex for {chain_id}/{challenge_id}: {exc}"
                ) from exc

            matched = [row for row in rows if pattern.search(str(row["description"]))]
            recalled = [row for row in matched if main_pattern.search(str(row["description"]))]
            unrecalled = [row for row in matched if not main_pattern.search(str(row["description"]))]
            if matched:
                challenges_with_hits += 1
            else:
                challenges_without_hits += 1
            total_hit_events += len(matched)
            total_unrecalled_events += len(unrecalled)
            chain_pair_hits.update(str(row["code"]) for row in matched)
            chain_unrecalled_pairs.update(str(row["code"]) for row in unrecalled)
            all_unique_codes.update(str(row["code"]) for row in matched)
            all_pair_hits.update((chain_id, str(row["code"])) for row in matched)

            if not matched:
                recall_status = "no_source_hits"
            elif not unrecalled:
                recall_status = "all_source_hits_recalled"
            else:
                recall_status = "unrecalled_hits_present"
            results.append(
                {
                    **challenge,
                    "case_insensitive": True,
                    "full_dictionary_hits": len(matched),
                    "main_lexicon_recalled_hits": len(recalled),
                    "newly_unrecalled_hits": len(unrecalled),
                    "main_lexicon_recall_status": recall_status,
                    "matched_records_sha256": _records_hash(matched),
                    "newly_unrecalled_records": [
                        {
                            "code": str(row["code"]),
                            "description": str(row["description"]),
                        }
                        for row in unrecalled
                    ],
                }
            )

        chain_results[chain_id] = {
            "main_lexicon_regex": main_regex,
            "main_lexicon_candidate_records": len(main_hits),
            "challenge_count": len(results),
            "challenge_hit_events_with_overlap": sum(
                int(result["full_dictionary_hits"]) for result in results
            ),
            "unique_source_rows_hit_by_any_challenge": len(chain_pair_hits),
            "unique_source_rows_not_recalled_by_main_lexicon": len(chain_unrecalled_pairs),
            "challenges": results,
        }

    return {
        "chains": chain_results,
        "summary": {
            "challenge_count": sum(len(value) for value in challenges.values()),
            "challenges_with_source_hits": challenges_with_hits,
            "challenges_with_zero_source_hits": challenges_without_hits,
            "challenge_hit_events_with_overlap": total_hit_events,
            "unique_chain_hs6_records_hit_by_any_challenge": len(all_pair_hits),
            "unique_hs6_hit_by_any_challenge": len(all_unique_codes),
            "newly_unrecalled_hit_events": total_unrecalled_events,
            "newly_unrecalled_unique_chain_hs6_records": sum(
                value["unique_source_rows_not_recalled_by_main_lexicon"]
                for value in chain_results.values()
            ),
        },
    }


def build_artifact(archive_path: Path = DEFAULT_ARCHIVE, rule_path: Path = DEFAULT_RULE) -> dict[str, Any]:
    rule_bytes = rule_path.read_bytes()
    try:
        rule = json.loads(rule_bytes)
    except json.JSONDecodeError as exc:
        raise NegativeControlError(f"invalid recall-rule JSON: {exc}") from exc

    source = rule["source_universe"]
    member_name = str(source["member"])
    member, rows = read_source_rows(archive_path, member_name)
    actual_member_hash = _sha256(member)
    expected_member_hash = str(source["member_sha256"])
    if actual_member_hash != expected_member_hash:
        raise NegativeControlError(
            f"source member hash mismatch: expected {expected_member_hash}, got {actual_member_hash}"
        )
    if len(rows) != int(source["rows_scanned"]):
        raise NegativeControlError(
            f"source row-count mismatch: expected {source['rows_scanned']}, got {len(rows)}"
        )
    numeric_rows = sum(len(row["code"]) == 6 and row["code"].isdigit() for row in rows)
    if numeric_rows != int(source["numeric_hs6_rows"]):
        raise NegativeControlError(
            f"numeric HS6 row-count mismatch: expected {source['numeric_hs6_rows']}, got {numeric_rows}"
        )
    non_numeric = sorted(row["code"] for row in rows if not row["code"].isdigit())
    if non_numeric != sorted(source["legacy_non_wco_rows"]):
        raise NegativeControlError(
            f"legacy non-WCO rows mismatch: expected {source['legacy_non_wco_rows']}, got {non_numeric}"
        )

    lexicons = rule["recall_rule"]["lexicons"]
    evaluated = evaluate_challenges(rows, lexicons)
    for chain_id, chain in evaluated["chains"].items():
        expected = int(lexicons[chain_id]["candidate_records"])
        actual = int(chain["main_lexicon_candidate_records"])
        if actual != expected:
            raise NegativeControlError(
                f"main-lexicon count drift for {chain_id}: expected {expected}, got {actual}"
            )

    summary = evaluated["summary"]
    status = (
        "PASS_TESTED_VARIANTS_ONLY"
        if summary["newly_unrecalled_unique_chain_hs6_records"] == 0
        else "FAIL_UNRECALLED_TESTED_VARIANTS"
    )
    challenge_definition = _challenge_definition()
    return {
        "schema_version": "upgrade-bench/registry-lexicon-negative-control/1",
        "evidence_id": "observable-attribution-lexicon-negative-control-v1",
        "status": status,
        "frozen_date": rule["frozen_date"],
        "generated_from": {
            "recall_rule": "chains/evidence/registry_candidate_recall_rule.json",
            "recall_rule_sha256": _sha256(rule_bytes),
            "generator": "tools/build_registry_lexicon_negative_control.py",
        },
        "source": {
            "dataset": source["dataset"],
            "classification": source["classification"],
            "member": member_name,
            "member_sha256": actual_member_hash,
            "source_rows_automatically_regex_scanned": len(rows),
            "numeric_hs6_rows": numeric_rows,
            "legacy_non_wco_rows": non_numeric,
            "manual_row_adjudications_performed_by_this_artifact": 0,
        },
        "execution_scope": {
            "operation": "Every one of the 5,022 source descriptions receives every applicable frozen challenge regex and the chain's main recall regex.",
            "main_pipeline_boundary": "The main recall pipeline is a 5,022-row automated regex scan yielding 576 observable chain-HS6 candidates for row-level ledger adjudication; it is not 5,022 manual reviews.",
            "ledger_boundary": "This artifact performs no include, exclude, out_of_stage, or stage-fit adjudication. The generated full ledger must separately reproduce all 576 observable-candidate decisions and 34 legacy-only carry-over decisions.",
            "challenge_set_provenance": "This finite targeted challenge set is frozen in the generator and retained with zero-hit probes. It is a deterministic diagnostic, not an independent or exhaustive vocabulary review.",
            "external_knowledge_nonuse": "Lexical alternatives are probes only. They never add a candidate, repair missing observable attribution, or supply an include or stage decision.",
        },
        "claim_boundary": {
            "supported": "For the frozen tested variants, every description hit in the full BACI dictionary is also recalled by that chain's main lexicon; any counterexample is emitted verbatim by code and official description.",
            "not_supported": "This does not prove lexicon completeness, does not test every possible synonym or wording, does not constitute manual review of 5,022 rows, and does not validate registry decisions or stage membership.",
            "zero_hit_interpretation": "A zero-hit probe establishes only that the tested expression is absent from this pinned dictionary member.",
        },
        "challenge_set": {
            "case_insensitive": True,
            "challenge_unit": "chain_id,challenge_id",
            "challenge_definition_sha256": _sha256(_canonical_bytes(challenge_definition)),
            "definitions": challenge_definition,
        },
        "summary": summary,
        "chains": evaluated["chains"],
    }


def render_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write deterministic evidence JSON")
    mode.add_argument("--check", action="store_true", help="verify committed evidence byte-for-byte")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        rendered = render_json(build_artifact(args.archive, args.rule))
        if args.write:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
            action = "wrote"
        else:
            if not args.output.is_file():
                raise NegativeControlError(f"committed evidence not found: {args.output}")
            committed = args.output.read_text(encoding="utf-8")
            if committed != rendered:
                raise NegativeControlError(
                    f"committed evidence is stale: regenerate with --write ({args.output})"
                )
            action = "verified"
    except (KeyError, OSError, NegativeControlError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"{action} {args.output}: {artifact_hash(rendered)}; "
        "PASS_TESTED_VARIANTS_ONLY (not a lexicon-completeness proof)"
    )
    return 0


def artifact_hash(rendered: str) -> str:
    return _sha256(rendered.encode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
