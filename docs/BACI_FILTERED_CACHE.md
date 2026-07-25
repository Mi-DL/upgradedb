# Private BACI filtered cache

The corrected v2 cohort rebuild needs the same small subset of BACI repeatedly:
the union of active HS6 codes in the six audited chain registries. The private
filtered cache reads each required raw annual CSV once and retains every matching
trade row—not a candidate subset—with exactly these columns:

```text
i,j,k,year,v
```

The fixed cache years are 1998–2002, 2008–2012, and 2018–2022. This covers the
historical selection fold and the main target fold. Other folds fail closed when
this cache is selected; direct ZIP reading remains available.

## Build once in a private workspace

The output must be new and must live under an explicit `private`, `.private`,
`tmp`, or `temp` path component. Supply all machine-specific locations through
environment variables; the values below are placeholders:

```bash
export REPO_ROOT=/path/to/upgrade-bench
export PYTHON=/path/to/audited-python
export VCU_RAW=/path/to/private/raw
export CACHE=/path/to/private/baci-filtered-cache-v202401b
cd "$REPO_ROOT"

"$PYTHON" tools/build_baci_filtered_cache.py \
  --baci-zip "$VCU_RAW/BACI_HS92_V202401b.zip" \
  --output "$CACHE" \
  --chunk-rows 500000
```

The builder never replaces an existing directory. It writes into a sibling
partial directory and atomically renames it only after all years and the final
manifest are complete. A failed build leaves no usable target cache.

The manifest is `<cache>/manifest.json`. It records:

- the complete raw archive byte size and SHA-256 (not a timestamp/size proxy);
- the country-code member hash and source ZIP member metadata;
- whole-file hashes of `docs/registry_audit.json`, registry evidence, and all
  six exact registry JSONs;
- the exact 283-code active audited union and exact year list; and
- every cache file's year, path, row count, byte size, SHA-256, and observed
  HS6 set.

No absolute source, user-home, or host path is persisted in the manifest.

## Use in cohort construction

Direct ZIP reads remain the default. Enable the cache explicitly either on the
command line or through the environment:

```bash
export VCU_BACI_CACHE="$CACHE"
VCU_FOLD=fold2 "$PYTHON" src/temporal_backtest.py \
  --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir /path/to/private/processed-v2-scratch

# Equivalent explicit form:
VCU_FOLD=fold2 "$PYTHON" src/temporal_backtest.py \
  --baci-cache "$CACHE" --chain cocoa --upgrade --enum-only \
  --aggregation calendar_mean --output-dir /path/to/private/processed-v2-scratch
```

Before returning rows, the reader checks the current audit/evidence/registry
snapshot, exact 15-year inventory, every compressed file's size and SHA-256,
and the requested files' schema, row count, year column, numeric values, and
observed code set. It also verifies the country-code member read from the raw
ZIP. The full 12 GB archive SHA-256 is computed and recorded during the one-time
build; it is intentionally not recomputed for every chain invocation because
the cache files are self-contained and individually hashed.

The raw filtered rows are reusable by strict audits and diagnostics:

```python
import os
from pathlib import Path
from baci_filtered_cache import BaciFilteredCache

cache = BaciFilteredCache(
    Path(os.environ["VCU_BACI_CACHE"]),
    requested_years=range(2008, 2013),
)
rows_2010 = cache.read_year(2010)  # complete audited-union i,j,k,year,v rows
```

## Expected acceleration and privacy boundary

A six-chain, two-task, two-fold cohort rebuild invokes the current loader 24
times. Direct mode decompresses ten full annual BACI members per invocation
(240 annual decompressions). The cache builder decompresses the 15 distinct
required members once, a 16× reduction in full-year decompressions; subsequent
invocations parse only the 283-code filtered payload. BACI-loading time should
therefore improve by roughly one order of magnitude, although end-to-end speed
also depends on gravity fitting and candidate enumeration. A formal full build
must be timed separately; the synthetic equivalence tests are not a performance
claim.

The cache contains raw-derived BACI rows and is private. It is ignored by Git
under conventional private/cache names and independently denied by the public
release selector. Do not place it in `data/processed_v2`, `results_v2`, a
release bundle, or a public artifact store.
