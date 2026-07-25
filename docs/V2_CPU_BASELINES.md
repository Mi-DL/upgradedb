# UPGRADE-BENCH rolling CPU reference protocol

> **Verified artifact:** `results_v2/metrics/rolling_cpu_baselines.{json,csv}`
> uses schema version 2 and is generated from the audited registry and frozen
> historical/main cohorts. The public release receipt and manifest bind the
> exact artifact bytes; `--verify-output` fails on schema or provenance drift.

`tools/v2_rolling_cpu_baselines.py` implements a strict select--freeze--evaluate
boundary. It opens every requested `fold2` table and freezes every model before
opening any main-window table. Main labels are never used for preprocessing,
hyperparameter selection, calibration, model selection, or imputation.

## Historical selection objectives

| task | CV grouping | objective for logistic `C` | exact hyperparameter tie-break |
|---|---|---|---|
| Track A: destination extension | exporter | mean validation average precision | maximize mean objective, then smaller `C` |
| Track B1: processed-export stage entry | exporter | mean validation average precision | maximize mean objective, then smaller `C` |
| Track B2: conditional destination ranking | exporter-stage entry | per-positive-entry macro recall@3 | maximize mean objective, then smaller `C` |

Track B2 selects `C` with the task-aligned conditional ranking objective. In each
group-safe validation fold, destinations are ranked separately inside every
positive exporter-stage entry, recall@3 is computed for each entry, and those
recalls are averaged without entry-size weights. Exact score ties use ascending
destination ISO. Training and validation entry sets are asserted disjoint. The
fold means are weighted by their validation-entry counts, which is exactly the
unweighted macro mean over all out-of-fold positive entries. The final
preprocessing pipeline and selected classifier are then refit on all historical
conditional lanes.

Track A uses exporter groups for both historical CV and main average-precision
uncertainty. This keeps all stages belonging to one exporter together and
covers their cross-stage dependence. Track B1 likewise uses exporters. Track
B2 resamples complete exporter-stage entry groups.

## Main metrics, value points, and uncertainty

The main-cohort headline metrics remain Track A average precision, Track B1
average precision, and Track B2 per-positive-entry macro recall@3. Row-level
cluster-bootstrap intervals are allowed for those task metrics:

- Track A: exporter clusters;
- Track B1: exporter clusters;
- Track B2: exporter-stage entry clusters.

The macro summary also fixes one realized-value reporting point before main
evaluation:

| task | fixed reporting point | realized-value metric |
|---|---|---|
| Track A | global top-500 destination lanes within each chain | observed late-value capture among all positive late value in that chain |
| Track B1 | global top-50 exporter-stage entries within each chain | observed late-value capture among all positive late value in that chain |
| Track B2 | top-3 destinations inside each actual positive entry | within-entry value capture, then an unweighted macro mean over entries |

For every model, schema 2 records the headline and realized-value macro mean
and every chain's underlying value. The global budgets use
`effective_k=min(requested_k, target_rows)`; the B2 budget analogously truncates
inside short entries.

All unordered pairs from the protocol-fixed model order are reported as
`left_model - right_model` for both headline and realized-value metrics. The
output includes each chain's paired delta, descriptive mean/median, and
win/tie/loss counts. It deliberately sets `chain_level_ci95` to null: six
registered chains do not justify a precise super-population interval. These
exhaustive pairwise rows are descriptive and never choose a main-window
champion.

## Schema and verification

The JSON schema identifier is
`upgrade-bench-v2-rolling-cpu-baselines-2`. Its contract includes:

1. selection candidates with `fold_objective_values`,
   `fold_objective_units`, `mean_objective`, and `std_objective` fields;
2. every selection record names and defines its objective, group unit,
   ranking tie-break, hyperparameter tie-break, and group-overlap check;
3. Track A selection and AP uncertainty use exporter clusters;
4. `macro_summary` records a fixed `budget_definition`, per-model
   `realized_value`, and protocol-fixed `pairwise_deltas`;
5. the flat CSV carries headline, realized-value, reporting-budget, and selection-
   objective columns;
6. `--verify-output` checks schema, inputs, JSON/CSV agreement, and hashes.

The deterministic rerun and verification order is:

```powershell
& '.\.venv\Scripts\python.exe' tools\audit_v2.py
& '.\.venv\Scripts\python.exe' tools\v2_rolling_cpu_baselines.py --bootstrap 200
& '.\.venv\Scripts\python.exe' tools\v2_rolling_cpu_baselines.py --verify-output
```

The implementation also provides synthetic/unit checks:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_v2_rolling_cpu_baselines -v
& '.\.venv\Scripts\python.exe' tools\v2_rolling_cpu_baselines.py --self-test
```
