# Data license and third-party source notice

This notice covers the dataset, metadata, score, and result artifacts shipped
with UPGRADE-BENCH. The root `LICENSE` applies to software code only.

## Project-authored material

To the extent that the UPGRADE-BENCH contributors own copyright or database
rights in the following material, they license those contributions under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- the value-chain registry and project-authored schema/metadata in `chains/`;
- the selection, organization, and project-authored annotations in the
  benchmark metadata files; and
- project-authored documentation of the derived benchmark tables.

That grant does **not** relicense, replace, or expand rights in any underlying
third-party data. The derived candidate, covariate, score, and result files are
composite artifacts and remain subject to the source terms below. Reusers must
preserve source attribution and this notice.

## Source ledger

| Source | Frozen input used here | Project outputs that depend on it | Official terms and redistribution note |
|---|---|---|---|
| CEPII BACI | HS92, release `V202401b` (minor update documented 8 April 2024); official BACI page and `V202401b` release notes accessed 26 July 2026; annual bilateral flows through 2022. The benchmark windows use 1998–2002, 2008–2012, and 2018–2022. Recorded raw-archive SHA-256: `1dafcfd5b26b2b2c88a69ca11ed67b7067f5c38c5a12c2e1766cf28df159909a`. | audited HS6/stage provenance, candidate lane tables and labels, `size`, `lateval`, graph facts, coverage/label audits, and downstream scores/metrics | CEPII identifies BACI as licensed under [Etalab Open Licence 2.0](https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf). That license permits copying, adapting, and redistribution, including commercially, with attribution to CEPII/BACI and the most recent update date of the reused information. Cite CEPII and Gaulier & Zignago (2010). [Official BACI page](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html); [`V202401b` release notes](https://cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202401.pdf). |
| CEPII Gravity | release `V202211`; year-2010 covariates for the main fold | the PPML-gravity feature `grav` and analyses using it | CEPII identifies Gravity as licensed under Etalab Open Licence 2.0. Preserve CEPII attribution and the dataset version. Cite Conte, Cotterlaz & Mayer (2022). [Official Gravity page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8). |
| FAOSTAT QCL | `Production_Crops_Livestock` normalized bulk archive documented in `requirements/DATA.md` | optional production/stocks graph edges and audits | FAO statistical databases are generally CC BY 4.0, supplemented by FAO's database terms and possible third-party exceptions. Redistribution requires FAO attribution and compliance with those additional terms. The raw archive is not shipped. [FAO database terms](https://www.fao.org/contact-us/terms/db-terms-of-use/en). |

The exact raw filenames, download locations, and construction commands are in
`requirements/DATA.md`. The raw BACI, Gravity, FAOSTAT, and TRAINS archives are
excluded from version control and from the release manifests.

For the audited registry, `docs/registry_audit.json` and
`chains/evidence/registry_evidence.json` are the project-authored decision
ledger. Both use schema version 3 and identify the CEPII-distributed HS92
product-code metadata member by SHA-256. Frozen regexes were applied
automatically to all 5,022 source rows, yielding 576 observable chain--HS6
records; adding 34 legacy-only provenance records gives 610 ledger records
covering 588 unique HS6 codes. The full-ledger decisions are 283 included, 228
excluded, and 99 out of stage across 53 active stages. This is exhaustive
application of the frozen regexes, not proof that their lexicons cover every
possible wording and not 5,022 completed human reviews. The retained negative
controls test declared variants only.
Supporting code, frozen lexicons, and initial decision proposals used LLM
assistance. Release-valid human-review status is established only by the
canonical hash-bound receipt, independently of the computation receipts. This
audit trail documents how source descriptions were interpreted; it does not
alter CEPII's rights or terms.

The private filtered BACI cache is an internal reproducibility acceleration,
not a release artifact or a new data source. Neither its annual extracts nor
machine-specific raw paths may enter public bundles. Public derived tables bind
their own bytes through the data-artifact index, while audit reports retain
source and registry hashes needed to detect a mismatched rebuild.

## Distribution bundles

Derived release payloads are catalogued by raw-byte SHA-256 in
`release/DATA_ARTIFACT_INDEX.json` and packaged according to
`docs/DATA_DISTRIBUTION.md`. Each generated data archive includes this notice;
moving a file from Git to a release asset does not change its source terms. The
bundle index is an integrity and provenance aid, not a grant of additional
rights in third-party inputs or derived extracts.

Permission-gated exploratory inputs and all superseded output trees are excluded
from both public Git and public bundles. Their presence in private staging does
not authorize redistribution or make them part of this benchmark release.

## Required attribution

Publications or redistributed artifacts should, at minimum:

1. cite UPGRADE-BENCH and identify the benchmark release/version;
2. cite every source used by the redistributed file, including the frozen
   source version where one is recorded above;
3. retain this notice and any source-provided notices or disclaimers; and
4. avoid implying endorsement by CEPII, FAO, the World Bank, UNCTAD, the WTO,
   the United Nations, or any source institution.

For BACI `V202401b`, the official update date and the documentation access date
are recorded above. A raw-archive download date is not inferred from local file
timestamps; source update dates, documentation access dates, and archive
download dates are distinct provenance fields.

## Privacy, attestations, and use boundary

The released benchmark units are country, HS6-derived stage, exporter, importer,
and time-window aggregates. They are not intended to contain names, contact
details, account identifiers, transaction-level commercial records, or other
personal information. Historical country/reporting-entity codes should not be
treated as current political or legal determinations.

Protocol attestations distributed with the evaluator bind a main-run result to
the exact benchmark CSV, score CSV, and frozen selection configuration. They
are schema-checked **self-attestations**: they do not independently verify model
development history, create a blind leaderboard, grant data rights, or replace
the source-specific attribution and redistribution obligations above.

The benchmark is for aggregate research and method comparison. Export-emergence
scores are not evidence of domestic processing, firm conduct, or individual
behavior and should not be used as automated sanctions, credit, eligibility, or
investment decisions.

## No warranty or legal advice

The artifacts are provided as-is for research and benchmarking. This notice is
an attribution and provenance ledger, not legal advice. A downstream user is
responsible for confirming that their use and redistribution comply with the
terms applicable in their jurisdiction.
