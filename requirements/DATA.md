# Raw data: acquisition and release boundary

Raw third-party archives are not committed. Keep them in a private directory
and point `VCU_RAW` to that directory. The same commodity-agnostic archives
serve all six chains.

```text
$VCU_RAW/
├── BACI_HS92_V202401b.zip           2.45 GB  CEPII BACI, HS92
├── Gravity_csv_V202211.zip          207 MB   CEPII Gravity
└── Production_Crops_Livestock.zip    34 MB   FAOSTAT QCL (optional)
```

Never copy these raw archives into a public repository, release bundle, or
benchmark data archive. Consult `DATA_LICENSE.md` before redistribution.

## 1. CEPII BACI

- Purpose: bilateral trade, candidate construction, `size`, and `lateval`.
- Source: <http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37>
- Version: HS92, V202401b.
- Required filename: `$VCU_RAW/BACI_HS92_V202401b.zip`.
- Expected inner files include `BACI_HS92_Y{year}_V202401b.csv`,
  `country_codes_V202401b.csv`, and `product_codes_HS92_V202401b.csv`.
- Citation: Gaulier and Zignago (2010); use subject to CEPII terms.
- The public registry evidence contains only the 588 unique HS6 metadata rows
  represented in the 610-record chain--HS6 audit ledger
  (`chains/evidence/hs92_selected_product_codes.csv`), never the raw trade
  archive. The full 5,022-row source operation is an automated frozen-regex
  scan, not a public copy of the dictionary or completed human review. Holders
  of the private archive can run `python tools/audit_chain_registry.py --check
  --baci-zip $VCU_RAW/BACI_HS92_V202401b.zip` for a read-only dictionary
  SHA/row diff-check.

## 2. CEPII Gravity

- Purpose: ex-ante gravity covariates.
- Source: <http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8>
- Version: V202211.
- Required filename: `$VCU_RAW/Gravity_csv_V202211.zip`.
- The v2 main window uses the prespecified ex-ante gravity vintage documented by
  the pipeline; the historical fold uses its corresponding earlier vintage.
- Citation: Conte, Cotterlaz, and Mayer (2022).

## 3. FAOSTAT QCL (optional)

- Purpose: optional production/stocks `produces` edges, not core v2 labels.
- Source: <https://www.fao.org/faostat/en/#data/QCL>, bulk download
  *Production_Crops_Livestock_E_All_Data (Normalized)*.
- Required filename: `$VCU_RAW/Production_Crops_Livestock.zip`.
- Expected inner file:
  `Production_Crops_Livestock_E_All_Data_(Normalized).csv`.
- Country codes are mapped from FAOSTAT M49 to ISO3 with `pycountry`, not with
  BACI numeric identifiers.
- License/citation: CC BY 4.0; cite FAOSTAT.

## What can run without raw archives?

The canonical v2 tables will be distributed in the main and historical release bundles, not as
ordinary Git objects. After extraction they are mounted under `data/processed_v2/` and are
sufficient for:

- `tools/validate_v2.py`;
- `tools/v2_rolling_cpu_baselines.py --verify-output`;
- the standalone `benchmark/upgrade-bench-v2/` loader and scorer; and
- unit and exporter-stage diagnostic-split invariants.

During the active hold, `results_v2/` contains the invalidation marker and documentation only.
Authorized lightweight metrics and audit reports enter that surface after resolution.

Raw BACI/Gravity are required for candidate rebuilds, KGE/path-GNN feature
construction, and the independent raw-label audit
(`tools/audit_v2.py` without `--verify-output`). The optional FAOSTAT archive is
not required by the released candidate, scorer, or reference-result contracts.

## Maintainer workspace boundary

Exploratory output trees may be retained in maintainer staging for provenance;
public selectors exclude them by prefix.
