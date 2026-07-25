#!/usr/bin/env python3
"""Build the publishable code- and stage-level HS92 registry evidence artifact.

The selected product descriptions are an exact subset of
``product_codes_HS92_V202401b.csv`` from CEPII BACI V202401b.  This builder
does not read trade-flow data and deliberately contains no raw-data path.

Run::

    python tools/build_registry_evidence.py --write
    python tools/build_registry_evidence.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHAINS_DIR = ROOT / "chains"
METADATA = CHAINS_DIR / "evidence" / "hs92_selected_product_codes.csv"
OUTPUT = CHAINS_DIR / "evidence" / "registry_evidence.json"
CHAIN_IDS = ("sheep", "cotton", "aluminium", "nickel", "cocoa", "oilseed-soy")
SOURCE_ID = "cepii-baci-hs92-v202401b-product-codes"
SOURCE_VERSION = "CEPII BACI V202401b; HS 1988/1992 (HS92)"
SOURCE_METADATA_MEMBER_SHA256 = "51bcc9e8946b5efb0a7c7e9756c42e72c35861f32a7a7059e2d768a0f2f61166"


def _stage(
    specificity: str,
    canonical_definition: str,
    observed_scope: str,
) -> dict[str, str]:
    return {
        "specificity": specificity,
        "canonical_definition": canonical_definition,
        "fit_rationale": (
            f"The official HS92 description identifies {observed_scope}, which is inside the "
            "canonical stage definition. No unobserved product form, feedstock, species, or end use "
            "is inferred from the commodity name alone."
        ),
    }


# Every active stage must be deliberately registered here.  A newly added or renamed
# stage therefore fails evidence generation until its scientific rationale is reviewed.
INCLUDE_STAGE_META: dict[str, dict[str, dict[str, str]]] = {
    "sheep": {
        "exp_live": _stage("sheep-specific", "Live sheep.", "live sheep and no other species"),
        "exp_meat": _stage(
            "sheep-specific",
            "Fresh, chilled, or frozen meat of sheep or lamb.",
            "meat of sheep or lamb",
        ),
        "exp_rawskin": _stage(
            "sheep-specific",
            "Raw sheep or lamb skins, with or without wool, not tanned or further prepared.",
            "raw sheep or lamb skins",
        ),
        "exp_woolraw": _stage(
            "wool-explicit",
            "Wool not carded or combed, including greasy, degreased, or carbonised forms.",
            "wool explicitly described as not carded or combed",
        ),
        "exp_woolgrease": _stage(
            "wool-explicit",
            "Non-crude wool grease and fatty substances derived from wool grease, including "
            "lanolin. This form-limited stage deliberately excludes crude wool grease, which "
            "is recorded out of stage.",
            "non-crude wool grease or fatty substances derived from it",
        ),
        "exp_wooltop": _stage(
            "wool-explicit",
            "Wool tops and other combed wool, in fragments or otherwise; this form-limited "
            "stage excludes carded-wool fibre and yarn.",
            "wool tops or other combed wool, not yarn",
        ),
        "exp_woolyarn": _stage(
            "wool-explicit",
            "Yarn explicitly of carded or combed wool; this stage excludes wool tops/combed-wool fibre.",
            "yarn explicitly of carded or combed wool",
        ),
    },
    "cotton": {
        "exp_cottonraw": _stage(
            "cotton-explicit", "Cotton not carded or combed.", "cotton not carded or combed"
        ),
        "exp_cottonwaste": _stage(
            "cotton-explicit",
            "Cotton yarn/thread waste and other cotton waste.",
            "cotton waste, including yarn or thread waste",
        ),
        "exp_cottonprepared": _stage(
            "cotton-explicit", "Carded or combed cotton fibre.", "cotton explicitly carded or combed"
        ),
        "exp_cottonyarn": _stage(
            "cotton-explicit",
            "Non-sewing yarn explicitly containing or made of cotton, including "
            "cross-material blends.",
            "non-sewing yarn explicitly containing cotton, including cross-material blends",
        ),
        "exp_cottonsewthread": _stage(
            "cotton-explicit", "Cotton sewing thread.", "cotton sewing thread"
        ),
        "exp_cottonfabric": _stage(
            "cotton-explicit",
            "Woven fabric categories that explicitly identify cotton content.",
            "woven fabric explicitly containing or made of cotton",
        ),
        "exp_cottonknitfabric": _stage(
            "cotton-explicit",
            "Knitted or crocheted fabric explicitly of cotton.",
            "knitted or crocheted cotton fabric",
        ),
        "exp_cottonapparel_woven": _stage(
            "cotton-explicit",
            "Selected non-knit cotton apparel: men's or boys' shirts; men's or boys' "
            "trousers, bib and brace overalls, breeches, and shorts; and the corresponding "
            "women's or girls' lower-body garments. This form-limited stage does not represent "
            "all Chapter 62 cotton apparel.",
            "one of the declared non-knit cotton-apparel forms",
        ),
        "exp_cottonapparel_knit": _stage(
            "cotton-explicit",
            "Selected knitted or crocheted cotton apparel: men's or boys' shirts; T-shirts, "
            "singlets, and other vests; men's or boys' and women's or girls' trousers and "
            "related lower-body garments; and jerseys, pullovers, cardigans, waistcoats, and "
            "similar articles. This form-limited stage does not represent all Chapter 61 "
            "cotton apparel.",
            "one of the declared knitted or crocheted cotton-apparel forms",
        ),
        "exp_cottonhomewares": _stage(
            "cotton-explicit",
            "Cotton bed linen and cotton kitchen or toilet linen. This form-limited stage "
            "does not represent all cotton household articles.",
            "bed, kitchen, or toilet linen explicitly of cotton",
        ),
    },
    "aluminium": {
        "exp_aluminium_ore": _stage(
            "aluminium-explicit",
            "Aluminium ores and concentrates; the stage does not claim that every flow is bauxite.",
            "aluminium ores and concentrates",
        ),
        "exp_scrap": _stage("aluminium-explicit", "Aluminium waste and scrap.", "aluminium waste and scrap"),
        "exp_aluminium_hydroxide": _stage(
            "aluminium-explicit", "Aluminium hydroxide.", "aluminium hydroxide"
        ),
        "exp_aluminium_oxide": _stage(
            "aluminium-explicit",
            "Aluminium oxide other than artificial corundum (the alumina stage).",
            "aluminium oxide other than artificial corundum",
        ),
        "exp_corundum": _stage(
            "aluminium-explicit", "Artificial aluminium-oxide corundum.", "artificial aluminium-oxide corundum"
        ),
        "exp_unwrought": _stage("aluminium-explicit", "Unwrought aluminium, alloyed or not.", "unwrought aluminium"),
        "exp_semis_barrod": _stage(
            "aluminium-explicit",
            "Aluminium bars, rods, and profiles, including hollow profiles.",
            "aluminium bars, rods, or profiles",
        ),
        "exp_semis_wire": _stage("aluminium-explicit", "Aluminium wire.", "aluminium wire"),
        "exp_semis_plate": _stage(
            "aluminium-explicit", "Aluminium plates, sheets, and strip.", "aluminium plates, sheets, or strip"
        ),
        "exp_semis_foil": _stage("aluminium-explicit", "Aluminium foil, backed or unbacked.", "aluminium foil"),
        "exp_semis_tube": _stage(
            "aluminium-explicit",
            "Aluminium tubes, pipes, and tube/pipe fittings.",
            "aluminium tubes, pipes, or their fittings",
        ),
    },
    "nickel": {
        "exp_nickel_ore": _stage("nickel-explicit", "Nickel ores and concentrates.", "nickel ores and concentrates"),
        "exp_nickel_matte": _stage("nickel-explicit", "Nickel mattes.", "nickel mattes"),
        "exp_nickel_intermediate": _stage(
            "nickel-explicit",
            "Nickel oxide sinters and other intermediate products of nickel metallurgy, excluding nickel mattes.",
            "oxide sinters or other intermediate products of nickel metallurgy",
        ),
        "exp_ferronickel": _stage("nickel-explicit", "Ferro-nickel.", "ferro-nickel"),
        "exp_unwrought": _stage(
            "nickel-explicit",
            "Nickel-explicit unwrought metal and alloy baskets: unwrought nickel and "
            "explicitly named copper-nickel or copper-nickel-zinc alloys.",
            "unwrought nickel or a nickel-explicit copper-nickel alloy basket",
        ),
        "exp_powder": _stage("nickel-explicit", "Nickel powders and flakes.", "nickel powders or flakes"),
        "exp_bars_wire": _stage(
            "nickel-explicit",
            "Nickel-explicit bars, rods, profiles, and wire: nickel products and explicitly "
            "named copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
            "nickel-explicit bars, rods, profiles, or wire, including copper-nickel alloy baskets",
        ),
        "exp_plates_foil": _stage(
            "nickel-explicit",
            "Nickel-explicit plates, sheets, strip, and foil: nickel products and explicitly "
            "named copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
            "nickel-explicit plates, sheets, strip, or foil, including copper-nickel alloy baskets",
        ),
        "exp_tubes": _stage(
            "nickel-explicit",
            "Nickel-explicit tubes, pipes, and fittings: nickel products and explicitly "
            "named copper-nickel or copper-nickel-zinc alloy baskets in those forms.",
            "nickel-explicit tubes, pipes, or fittings, including copper-nickel alloy baskets",
        ),
        "exp_other": _stage("nickel-explicit", "Other nickel articles not elsewhere specified.", "other nickel articles"),
        "exp_nickel_salts": _stage(
            "nickel-explicit",
            "Nickel sulphates and nickel chlorides; the stage makes no battery end-use claim.",
            "nickel sulphate or nickel chloride without an asserted end use",
        ),
    },
    "cocoa": {
        "exp_cocoabean": _stage("cocoa-explicit", "Whole or broken cocoa beans, raw or roasted.", "cocoa beans"),
        "exp_cocoawaste": _stage("cocoa-explicit", "Cocoa shells, husks, skins, and other cocoa waste.", "cocoa waste"),
        "exp_cocoapaste": _stage("cocoa-explicit", "Cocoa paste, defatted or not.", "cocoa paste"),
        "exp_cocoabutter": _stage(
            "cocoa-explicit", "Cocoa butter, fat, and oil.", "cocoa butter, fat, or oil"
        ),
        "exp_cocoapowder_unsw": _stage(
            "cocoa-explicit", "Cocoa powder without added sugar or sweetener.", "unsweetened cocoa powder"
        ),
        "exp_cocoapowder_sw": _stage(
            "cocoa-explicit", "Cocoa powder containing added sugar or sweetener.", "sweetened cocoa powder"
        ),
        "exp_cocoa_prep_bulk": _stage(
            "cocoa-explicit",
            "Chocolate or other cocoa-containing food preparations in bulk forms or containers over 2 kg.",
            "chocolate or other cocoa-containing food preparations in the specified bulk form",
        ),
        "exp_cocoa_prep_blocks_bars": _stage(
            "cocoa-explicit",
            "Chocolate or other cocoa-containing food preparations in filled or unfilled blocks, slabs, or bars of 2 kg or less.",
            "chocolate or other cocoa-containing preparations in blocks, slabs, or bars",
        ),
        "exp_cocoa_prep_other": _stage(
            "cocoa-explicit",
            "Other chocolate or cocoa-containing food preparations not elsewhere specified in chapter 18.",
            "other chocolate or cocoa-containing food preparations",
        ),
    },
    "oilseed-soy": {
        "exp_soybean": _stage("soy-explicit", "Soya beans, whether broken or not.", "soya beans"),
        "exp_soymeal": _stage(
            "soy-explicit",
            "Oil-cake and other solid residues from extracting soybean oil, including ground or pellet forms.",
            "solid residues from soybean-oil extraction",
        ),
        "exp_soyoil_crude": _stage(
            "soy-explicit", "Crude soybean oil and its fractions.", "crude soybean oil or its fractions"
        ),
        "exp_soyoil_noncrude": _stage(
            "soy-explicit",
            "Soybean oil and fractions other than crude, whether or not refined; the stage does not assert that every flow is refined.",
            "soybean oil explicitly described as other than crude",
        ),
        "exp_soyflour_meal": _stage(
            "soy-explicit",
            "Flours and meals made directly from soya beans, distinct from oil-extraction residues in exp_soymeal.",
            "flours or meals of soya beans",
        ),
    },
}


# Groups enumerate every code removed from the pre-audit registry.  They are part of
# the public evidence and may not be silently discarded.
EXCLUDED_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_live",
        "specificity": "goat-only",
        "rationale": "The official description identifies goats, not sheep.",
        "codes": ("010420",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_meat",
        "specificity": "goat-only",
        "rationale": "The official description identifies goat meat, not sheep meat.",
        "codes": ("020450",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_offal",
        "specificity": "mixed-species",
        "rationale": "The HS92 category combines sheep with goats and equine species, so BACI cannot attribute the flow to sheep.",
        "codes": ("020680", "020690"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolraw",
        "specificity": "fine-animal-hair-not-wool",
        "rationale": "The description identifies fine animal hair and does not establish wool content.",
        "codes": ("510210",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolraw",
        "specificity": "mixed-wool-or-fine-animal-hair",
        "rationale": "The HS92 category combines wool with fine animal hair, so BACI cannot isolate wool.",
        "codes": ("510310", "510320"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_petfood",
        "specificity": "species-unspecified-animal-feed",
        "rationale": "Dog/cat food does not identify sheep, sheep meat, or wool as an input.",
        "codes": ("230910", "230990"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_meal",
        "specificity": "species-unspecified-meat-meal",
        "rationale": "Meat/offal meal is not species-specific and cannot be attributed to sheep.",
        "codes": ("230110",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_casings",
        "specificity": "species-unspecified-animal-product",
        "rationale": "The category covers guts, bladders, and stomachs of animals generally, not sheep specifically.",
        "codes": ("050400",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_fats",
        "specificity": "material-unspecified-fat-or-oil",
        "rationale": "The category combines animal and vegetable fats and oils and does not identify sheep or wool.",
        "codes": ("151800",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_prepared",
        "specificity": "species-unspecified-meat-preparation",
        "rationale": "The category is generic across meat or animal species and cannot be attributed to sheep.",
        "codes": ("021090", "160100", "160290"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_furleather",
        "specificity": "species-unspecified-furskin",
        "rationale": "The residual tanned/dressed furskin category does not identify sheep or lamb.",
        "codes": ("430219",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolyarn",
        "specificity": "mixed-wool-or-fine-animal-hair",
        "rationale": "The HS92 category combines wool yarn with fine-animal-hair yarn, so BACI cannot isolate wool.",
        "codes": ("510910",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolfabric",
        "specificity": "mixed-wool-or-fine-animal-hair",
        "rationale": "The HS92 woven-fabric category combines wool with fine animal hair, so BACI cannot isolate wool.",
        "codes": ("511119", "511190", "511211", "511219", "511230", "511290"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolcarpet",
        "specificity": "mixed-wool-or-fine-animal-hair",
        "rationale": "The carpet category combines wool with fine animal hair, so BACI cannot isolate wool.",
        "codes": ("570110", "570241", "570310"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolcarpet",
        "specificity": "explicitly-non-wool-material",
        "rationale": "The official description explicitly assigns a non-wool material (man-made, polyamide, or another textile material).",
        "codes": ("570190", "570232", "570242", "570259", "570299", "570320", "570330", "570390"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolblanket",
        "specificity": "mixed-wool-or-fine-animal-hair",
        "rationale": "The blanket category combines wool with fine animal hair, so BACI cannot isolate wool.",
        "codes": ("630120",),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolblanket",
        "specificity": "explicitly-non-wool-material",
        "rationale": "The official description identifies cotton or synthetic fibres, not wool.",
        "codes": ("630130", "630140"),
    },
    {
        "chain_id": "sheep",
        "legacy_stage": "exp_woolblanket",
        "specificity": "material-unspecified-blanket",
        "rationale": "The residual blanket category does not identify wool and cannot be assigned to the wool chain.",
        "codes": ("630190",),
    },
    {
        "chain_id": "cotton",
        "legacy_stage": "exp_cottonknitfabric",
        "specificity": "material-unspecified-knitted-fabric",
        "rationale": "The official description specifies width and construction but no cotton content.",
        "codes": ("600220",),
    },
    {
        "chain_id": "cotton",
        "legacy_stage": "exp_cottonapparel_woven",
        "specificity": "material-unspecified-or-non-cotton-specific-apparel",
        "rationale": "The official description does not establish cotton content, so the apparel flow cannot be attributed to cotton.",
        "codes": ("620329", "621210"),
    },
    {
        "chain_id": "cotton",
        "legacy_stage": "exp_cottonhomewares",
        "specificity": "material-unspecified-homeware-or-rag",
        "rationale": "The official description does not establish cotton content, so the flow cannot be attributed to cotton.",
        "codes": ("630419", "631090"),
    },
    {
        "chain_id": "nickel",
        "legacy_stage": "exp_stainless",
        "specificity": "nickel-content-unidentified-stainless-steel",
        "rationale": "The HS92 description identifies stainless steel but no nickel content; BACI therefore cannot attribute the flow to nickel input.",
        "codes": ("721810", "721890", "721911", "721922", "721932", "721990", "722011", "722020", "722090"),
    },
    {
        "chain_id": "oilseed-soy",
        "legacy_stage": "exp_lecithin",
        "specificity": "feedstock-unspecified-lecithin",
        "rationale": "The category covers lecithins and other phosphoaminolipids without identifying soy as the feedstock.",
        "codes": ("292320",),
    },
)


PREVIOUS_STAGE = {
    ("sheep", "150590"): "exp_fats",
    ("sheep", "510521"): "exp_woolyarn",
    ("sheep", "510529"): "exp_woolyarn",
    ("cotton", "520210"): "exp_cottonraw",
    ("cotton", "520299"): "exp_cottonraw",
    ("cotton", "520300"): "exp_cottonraw",
    ("aluminium", "260600"): "exp_bauxite",
    ("aluminium", "281820"): "exp_alumina",
    ("aluminium", "281830"): "exp_alumina",
    ("nickel", "750110"): "exp_matte",
    ("nickel", "750120"): "exp_matte",
    ("nickel", "282735"): "exp_battery_salts",
    ("nickel", "283324"): "exp_battery_salts",
    ("cocoa", "180620"): "exp_chocolate_bulk",
    ("cocoa", "180631"): "exp_chocolate_bar",
    ("cocoa", "180632"): "exp_chocolate_bar",
    ("cocoa", "180690"): "exp_chocolate_other",
    ("oilseed-soy", "150790"): "exp_soyoil_refined",
    ("oilseed-soy", "120810"): "exp_soyflour",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_descriptions() -> dict[str, str]:
    with METADATA.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, str] = {}
    for row in rows:
        code = row.get("code", "")
        description = row.get("description", "").strip()
        if len(code) != 6 or not code.isdigit() or not description:
            raise ValueError(f"invalid selected product metadata row: {row!r}")
        if code in out:
            raise ValueError(f"duplicate selected product code: {code}")
        out[code] = description
    return out


def _load_chain(chain_id: str) -> dict[str, Any]:
    data = json.loads((CHAINS_DIR / f"{chain_id}.json").read_text(encoding="utf-8"))
    if data.get("id") != chain_id:
        raise ValueError(f"chain id mismatch in {chain_id}.json")
    return data


def _excluded_by_chain() -> dict[str, dict[str, dict[str, str]]]:
    result = {chain_id: {} for chain_id in CHAIN_IDS}
    for group in EXCLUDED_GROUPS:
        chain_id = group["chain_id"]
        if chain_id not in result:
            raise ValueError(f"unknown exclusion chain: {chain_id}")
        for code in group["codes"]:
            if code in result[chain_id]:
                raise ValueError(f"duplicate excluded decision: {chain_id}/{code}")
            result[chain_id][code] = {
                "legacy_stage": group["legacy_stage"],
                "specificity": group["specificity"],
                "rationale": group["rationale"],
            }
    return result


def build() -> dict[str, Any]:
    descriptions = _load_descriptions()
    excluded = _excluded_by_chain()
    chains_out: dict[str, Any] = {}
    all_decided: set[str] = set()

    for chain_id in CHAIN_IDS:
        chain = _load_chain(chain_id)
        stages = chain["stages"]
        stage_meta = INCLUDE_STAGE_META[chain_id]
        if set(stage_meta) != set(stages):
            raise ValueError(
                f"{chain_id}: stage evidence mismatch; missing={sorted(set(stages)-set(stage_meta))}, "
                f"stale={sorted(set(stage_meta)-set(stages))}"
            )
        active_codes = {code for codes in stages.values() for code in codes}
        overlap = active_codes & set(excluded[chain_id])
        if overlap:
            raise ValueError(f"{chain_id}: code is both active and excluded: {sorted(overlap)}")

        records: list[dict[str, Any]] = []
        for stage, codes in stages.items():
            for code in codes:
                if code not in descriptions:
                    raise ValueError(f"{chain_id}/{stage}: missing selected metadata for {code}")
                record: dict[str, Any] = {
                    "code": code,
                    "decision": "include",
                    "stage": stage,
                    "description": descriptions[code],
                    "source_id": SOURCE_ID,
                    "source_version": SOURCE_VERSION,
                    "specificity": stage_meta[stage]["specificity"],
                    "rationale": stage_meta[stage]["fit_rationale"],
                    "stage_fit": {
                        "status": "supported",
                        "canonical_definition": stage_meta[stage]["canonical_definition"],
                        "evidence": descriptions[code],
                        "rationale": stage_meta[stage]["fit_rationale"],
                    },
                }
                previous = PREVIOUS_STAGE.get((chain_id, code))
                if previous:
                    record["previous_stage"] = previous
                records.append(record)
                all_decided.add(code)

        for code, meta in sorted(
            excluded[chain_id].items(), key=lambda item: (item[1]["legacy_stage"], item[0])
        ):
            if code not in descriptions:
                raise ValueError(f"{chain_id}: missing selected metadata for excluded {code}")
            records.append(
                {
                    "code": code,
                    "decision": "exclude",
                    "stage": None,
                    "legacy_stage": meta["legacy_stage"],
                    "description": descriptions[code],
                    "source_id": SOURCE_ID,
                    "source_version": SOURCE_VERSION,
                    "specificity": meta["specificity"],
                    "rationale": meta["rationale"],
                    "stage_fit": {
                        "status": "unsupported",
                        "canonical_definition": (
                            f"No active stage assignment; reviewed against legacy stage "
                            f"{meta['legacy_stage']}."
                        ),
                        "evidence": descriptions[code],
                        "rationale": meta["rationale"],
                    },
                }
            )
            all_decided.add(code)

        chains_out[chain_id] = {
            "display_description": chain.get("description", ""),
            "stage_definitions": {
                stage: {
                    "canonical_definition": meta["canonical_definition"],
                    "specificity": meta["specificity"],
                    "fit_rule": meta["fit_rationale"],
                }
                for stage, meta in stage_meta.items()
            },
            "included_count": len(active_codes),
            "excluded_count": len(excluded[chain_id]),
            "decisions": records,
        }

    if all_decided != set(descriptions):
        raise ValueError(
            "selected metadata/decision coverage mismatch; "
            f"undecided={sorted(set(descriptions)-all_decided)}, "
            f"missing_metadata={sorted(all_decided-set(descriptions))}"
        )

    return {
        "schema_version": "upgrade-bench/hs92-registry-evidence/2",
        "review_date": "2026-07-13",
        "decision_policy": (
            "Strict observable attribution: include an HS6 only when its official HS92 description "
            "itself identifies the registered species, material, or commodity. Do not infer an input "
            "from a broad downstream category or rename a broad code as if it were specific."
        ),
        "stage_policy": (
            "Each active stage has an explicit canonical definition. Every included HS6 must have a "
            "supported stage_fit whose evidence is its official description and whose rationale shows "
            "that the description falls inside that definition. A commodity name alone may not imply "
            "an unobserved processing form, feedstock, species, or end use; distinct processing tiers "
            "are split when one honest umbrella would change the benchmark unit."
        ),
        "source": {
            "id": SOURCE_ID,
            "publisher": "CEPII",
            "dataset": "BACI",
            "release_version": "V202401b",
            "classification": "HS 1988/1992 (HS92), six-digit subheadings",
            "metadata_member": "product_codes_HS92_V202401b.csv",
            "source_metadata_member_sha256": SOURCE_METADATA_MEMBER_SHA256,
            "description_authority": "United Nations Statistics Division / UN Comtrade HS 1988/1992 descriptions, as distributed in CEPII BACI V202401b",
            "citation": "Gaulier and Zignago (2010), CEPII Working Paper 2010-23",
            "license": "Etalab Open Licence 2.0, as stated by CEPII for BACI",
            "license_url": "https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf",
            "cepii_url": "https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html",
            "unsd_url": "https://unstats.un.org/unsd/classifications/econ",
            "unsd_hs1992_json_url": "https://comtradeapi.un.org/files/v1/app/reference/H0.json",
            "unsd_class_code": "H0",
            "unsd_class_name": "HS1992",
            "unsd_selected_code_membership": "184/184 selected codes verified on 2026-07-12",
            "selected_metadata_file": "chains/evidence/hs92_selected_product_codes.csv",
            "selected_metadata_sha256": _sha256(METADATA),
        },
        "summary": {
            "chain_count": len(CHAIN_IDS),
            "active_stages": sum(len(value["stage_definitions"]) for value in chains_out.values()),
            "included_codes": sum(value["included_count"] for value in chains_out.values()),
            "excluded_codes": sum(value["excluded_count"] for value in chains_out.values()),
            "reviewed_codes": len(descriptions),
            "reassigned_included_codes": sum(
                1
                for value in chains_out.values()
                for decision in value["decisions"]
                if decision["decision"] == "include" and "previous_stage" in decision
            ),
        },
        "chains": chains_out,
    }


def _render(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    # The historical builder remains importable because the prospective
    # full-dictionary generator reuses its frozen stage definitions.  Once the
    # revision decision specification exists, however, the public CLI must not
    # regenerate the obsolete 184-proposal artifact over the 610-record ledger.
    if (CHAINS_DIR / "evidence" / "registry_revision_decision_spec.json").is_file():
        from build_registry_revision import main as revision_main

        return revision_main()

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the canonical evidence JSON")
    mode.add_argument("--check", action="store_true", help="fail if the evidence JSON is stale")
    args = parser.parse_args()

    rendered = _render(build())
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")
        return 0

    if not OUTPUT.exists():
        raise SystemExit(f"missing generated evidence: {OUTPUT.relative_to(ROOT)}; run --write")
    current = OUTPUT.read_text(encoding="utf-8")
    if current != rendered:
        raise SystemExit(f"stale generated evidence: {OUTPUT.relative_to(ROOT)}; run --write")
    print(f"evidence current: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
