# Registry Review Codebook

This codebook documents the operational interpretation for reviewing and
reporting the current UpgradeBench HS92 registry. Frozen chain lexicons are
applied automatically to all 5,022 rows of the pinned BACI product dictionary.
They yield 576 observable `(chain_id, hs6)` records; 34 additional legacy-only
provenance records bring the complete audit ledger to 610 records covering 588
unique HS6 codes. It clarifies the observable-attribution and stage-fit
policies; it does not change any recorded decision, stage assignment, upstream
relation, candidate table, label, or experimental result.

The 576 observable records split into 283 `include`, 194 `exclude`, and 99
`out_of_stage` decisions. After adding the legacy-only provenance records, the
full-ledger split is 283/228/99. These are chain--HS6 decision-record counts:
the 228 excluded records correspond to 207 globally unique excluded HS6 codes.
These machine-ledger totals do not, by themselves, establish sampled human-validation
completion; that status is reported only by the curation protocol and its
hash-bound sampled-validation receipt.

The codebook is a current construct definition and does not itself establish
validation status. The hash-bound curation sidecar and sampled-validation
receipt are the authoritative status records.

The full-dictionary recall rule was frozen before the current membership
revision in `chains/evidence/registry_candidate_recall_rule.json` and is
documented in `docs/REGISTRY_CANDIDATE_RECALL_RULE.md`. The 5,022-row operation
is an automated regex scan, not manual adjudication of every source row. Recall
is conditional on the frozen lexicons: the negative-control artifact exercises
declared synonyms and spelling variants, but cannot prove that every possible
description wording has been anticipated. The historical 184-proposal set is
not the current candidate-universe denominator; the 34 records found only in
that historical set are nevertheless retained as provenance within the current
610-record audit ledger.

The generated evidence/audit files retain `reviewed_codes` and `PASS` as legacy
compatibility labels. `reviewed_codes=610` means that 610 chain--HS6 decision
records passed the automated source, coverage, and mapping checks; it is not a
unique-code count and does not mean that 610 human row reviews were completed.
The curation sidecar and sampled-validation receipt, rather than this codebook, record the
current human-validation status.
For the same legacy schema, `review_date` denotes the generated-ledger audit
date rather than completion of human review, and `excluded_codes=228` denotes
exclusion decision records rather than unique HS6 codes (the latter count is
207).

## 1. Code inclusion

Reviewers decide from the official HS92 description shown in the evidence
record. External knowledge may be used to understand terminology, but it may
not supply a commodity, material, species, processing form, or end use that is
absent from that description.

- **Include** a code when the official description explicitly identifies the
  focal commodity, material, or species and the described product form is
  inside a declared stage.
- A commodity-explicit blend remains eligible. For example, a description that
  explicitly says wool is not made material-ambiguous merely because wool is
  less than 85 percent of the product.
- A commodity-explicit residual or `n.e.s.` category may remain eligible. Its
  inclusion supports inference only about the published HS6 basket, not the
  basket's internal composition.
- **Exclude** a generic, multi-commodity, mixed-species, or material-ambiguous
  code when the description does not isolate the focal commodity or material.
  A legacy stage label or likely commercial use cannot repair that ambiguity.
- Record **out of stage** when the description observably attributes the traded
  product to the focal commodity or material, but its product form lies outside
  every frozen stage. A negated, optional, or unresolved focal reference remains
  `exclude`, not `out_of_stage`.

All trade value for an included code is the value of the complete published
HS6 basket. UpgradeBench does not estimate the focal-material share within a
blend or residual category.

Illustrative boundary cases already present in the registry are:

- `292320` (lecithins and other phosphoaminolipids) is excluded from the soy
  chain because the description does not identify soy.
- `510910` (yarn of wool or fine animal hair) is excluded because the
  description does not isolate wool from fine animal hair.
- `510620` and `510720` are eligible because their descriptions explicitly
  identify wool, even though wool is less than 85 percent by weight.
- `750800` and `180690` are retained as nickel-explicit and cocoa-explicit
  residual baskets, respectively; neither is interpreted as a homogeneous
  product or a within-code composition measure.
- `740323`, `740722`, `740822`, `740940`, and `741122` are retained as
  nickel-explicit copper-nickel or copper-nickel-zinc alloy baskets. Their
  placement in nickel analytical stages does not relabel them as nickel-only
  or necessarily nickel-base products.
- `550953`, `550962`, `550992`, and `551030` are cotton-explicit non-sewing
  yarn baskets because their descriptions say they are mixed mainly or solely
  with cotton. They are not interpreted as pure or cotton-primary yarn.

In the sheep-chain description, therefore, "mixed-material" means a heading
that leaves the focal material unresolved (such as "wool or fine animal
hair"), not every product that explicitly identifies wool and also contains
another material.

## 2. Stage assignment

A stage is a registry-defined **analytical product family**. It need not map
one-to-one to a unique physical operation or to every distinction made by the
HS nomenclature.

- Closely related HS6 product forms may be pooled when that family is the
  intended benchmark unit, such as nickel bars/rods/profiles with wire or
  metal tubes/pipes with their fittings.
- Distinct forms must be split when pooling would change the intended
  benchmark unit or make the entry/incumbency condition misleading.
- When forms are pooled, Track A incumbency and Track B entry are evaluated at
  the family level. The benchmark may therefore mask first entry into one
  constituent HS6 form by prior activity in another form in the same family.
- The sheep wool families deliberately cover wool not carded or combed,
  non-crude wool grease and derivatives, wool tops and other combed wool, and
  wool yarn. Crude wool grease (`150510`) and carded-wool fibre (`510510`) are
  observable wool products but remain `out_of_stage`; this is a declared
  analytical-family limit, not a claim that they are unrelated to wool.
- Cotton apparel and homewares use form-limited families. Knitted apparel is
  limited to the five declared basic-garment forms, non-knit apparel to the
  three declared shirt/lower-body forms, and homewares to bed and kitchen or
  toilet linen. These are precision-oriented scope choices, not claims to
  exhaust Chapters 61--63 or all cotton downstream products.
- `120810` is supported by the observable phrase "flours and meals of soya
  beans." Any wording used to distinguish this family from oil-extraction
  residues is a registry distinction, not evidence of an observed production
  route.

The registry's `derived_from` and `upstream_map` links remain eligibility
assumptions. They are not measured input-output coefficients or evidence that
a particular shipment was produced through the stated route.

## 3. Sampled human-validation record

The machine-assigned population is the frozen 610-record ledger. Human
validation applies to a frozen 212-record stratified sample (34.8%), while all
53 stage definitions are designated for a separate census review. The remaining 398
decision records are not claimed as individually human-verified. Each sampled
review record shows the
identifier, chain, HS6 code, official description, proposed decision, proposed
stage, candidate source (`observable_regex` or `legacy_only`), and stage
definition. `proposal_decision` is the repository proposal (`Include`,
`Exclude`, or `Out of stage`), not the reviewer's answer. The reviewer-entered
verdict is `Yes`, `No`, or `Uncertain`: `Yes` means the machine proposal is
correct, `No` means it requires correction, and `Uncertain` means the curator
cannot determine the answer from the permitted evidence. A `No` verdict
requires the corrected decision/stage and a short note; an `Uncertain` verdict
requires a short note.

The sample combines 74 certainty units with 138 probability-selected units.
Certainty units include every `legacy_only` record, every stratum containing at
most five records, every stage-reassignment record, and the eight published
boundary cases (with overlaps counted once). The remaining slots are allocated
across strata by Hamilton largest remainder in proportion to each random-pool
size; records within a stratum are selected by a frozen SHA-256 rank.
Exact inclusion probabilities are stored in
`chains/evidence/registry_human_validation_sample.json`. Any full-frame
discrepancy estimate must therefore use the stored design weights, such as a
Horvitz--Thompson estimator. Simple-random-sample margins or rule-of-three
bounds are not valid for this unequal-probability design. No fixed fraction,
including 25%, is an academic sufficiency criterion; adequacy depends on the
sampling design and intended estimand. The 212-row target is a prespecified,
design-specific validation scope.

The review instrument must be generated from the current 610-record sampling
frame and contain exactly the frozen 212 decision records plus all 53 stage
definitions. It must record the current evidence, ledger, and sample-plan
hashes. Earlier 184-row or 610-row workbooks
are historical templates only; they cannot support any completed-review claim
for the current registry.

The Review column `canonical_stage_definition` is the read-only definition
against which a proposed or corrected stage is judged.
`allowed_stages_for_chain` supplies the dropdown values and is not independent
evidence for a decision.

The review is outcome-blind. The reviewer receives no trade values, labels,
cohort impacts, model scores, or downstream result summaries; review is limited
to the frozen chain/HS6 description, stage-definition evidence, and proposal.
The review therefore evaluates application of the declared ontology; it does
not select candidates or rules in response to benchmark outcomes.

The author team defines this ontology; the curator applies it. Completion
covers the frozen 212-record sample and 53-definition census, not all 610
decision records. It does not independently search all possible lexicons or
manually review all 5,022 source descriptions for omitted candidates. The
current protocol is pending, specifies one curator, and provides no independent
second review or inter-annotator agreement. Any completed-validation coverage,
second-review coverage, or agreement statistic may be reported only after the
corresponding records exist; the curation protocol and hash-bound receipt are
the authoritative status records.

## 4. LLM assistance and provenance

LLM assistance was used to develop supporting code, the frozen lexicons,
initial decision proposals, and partitioned rechecks of the generated ledger.
Those rechecks corrected six boundary records but were neither human review nor
independent replication. Automated full-source regex application, finite
lexical negative controls, source checks, relation checks, and semantic
regression tests can detect specified omissions or later drift, but they are
not human judgments, do not prove lexicon completeness, and do not constitute
inter-reviewer agreement. Sampled human-validation scope and status are recorded separately
in `chains/evidence/registry_curation_protocol.json` and the hash-bound sampled-validation
receipt.

## 5. Change boundary

A later accepted change to a code decision, stage membership or assignment, stage definition, or
semantic `upstream`, `upstream_map`, `derived_from`, `derived_from_hs`, `produces`, `form_of`, or
`named_sources` field changes the benchmark construct, creates a new registry version, and
requires a full rebuild of registry-dependent cohorts, results, summaries, paper interfaces, and
release seals. A completed sampled validation that accepts none of those changes adds only its hash-bound
receipt and refreshed documentation/manifests; it does not require a computational rerun. The
same no-rerun rule applies to a documentation-only clarification that leaves all of those construct
fields unchanged.
