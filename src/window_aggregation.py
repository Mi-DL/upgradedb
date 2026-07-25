"""Canonical aggregation of sparse trade rows over a fixed calendar window.

BACI stores positive trade rows sparsely: an absent ``(lane, HS6, year)`` row must
therefore contribute zero to a fixed-window annual mean.  The canonical operation is:

1. sum all HS6 rows within each ``(lane/stage, calendar year)``;
2. materialize every requested calendar year, filling absent years with zero; and
3. divide the resulting window total by ``len(years)``.

The legacy benchmark instead averaged each HS6 over only the years in which that HS6
was present, then summed those conditional means.  It is retained only as the explicit
``legacy_present_hs6_mean`` mode for migration diagnostics; it is never the default.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


CALENDAR_MEAN = "calendar_mean"
LEGACY_PRESENT_HS6_MEAN = "legacy_present_hs6_mean"
VALID_MODES = frozenset({CALENDAR_MEAN, LEGACY_PRESENT_HS6_MEAN})


def _normalise_years(years: Sequence[int]) -> list[int]:
    out = [int(year) for year in years]
    if not out:
        raise ValueError("years must contain at least one calendar year")
    if len(set(out)) != len(out):
        raise ValueError(f"years must be unique, got {out}")
    return out


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def stage_year_totals(
    raw: pd.DataFrame,
    years: Sequence[int],
    group_cols: Sequence[str],
    *,
    year_col: str = "year",
    value_col: str = "v",
    expected_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a complete stage-by-calendar-year panel with absent years set to zero.

    ``group_cols`` identifies the output unit, normally
    ``[i_iso, j_iso, stage]``. Multiple HS6 rows in the same stage-year are summed
    *before* the calendar window is completed. ``expected_keys`` is optional; when
    supplied it also materializes groups with no raw rows anywhere in the window.
    Observed keys are retained as well, so diagnostics cannot silently discard data.
    """
    years = _normalise_years(years)
    group_cols = list(group_cols)
    if not group_cols:
        raise ValueError("group_cols must contain at least one key column")
    _require_columns(raw, group_cols + [year_col, value_col], "raw")

    if expected_keys is not None:
        _require_columns(expected_keys, group_cols, "expected_keys")

    work = raw[group_cols + [year_col, value_col]].copy()
    if not work.empty:
        work[year_col] = pd.to_numeric(work[year_col], errors="raise").astype(int)
        outside = sorted(set(work[year_col]) - set(years))
        if outside:
            raise ValueError(f"raw contains years outside the requested window: {outside}")
        work[value_col] = pd.to_numeric(work[value_col], errors="raise")

    observed_keys = work[group_cols].drop_duplicates()
    if expected_keys is None:
        keys = observed_keys
    else:
        keys = pd.concat(
            [observed_keys, expected_keys[group_cols]], ignore_index=True
        ).drop_duplicates()

    columns = group_cols + [year_col, value_col]
    if keys.empty:
        return pd.DataFrame(columns=columns)

    annual = (
        work.groupby(group_cols + [year_col], as_index=False, dropna=False)[value_col]
        .sum()
    )
    keys = keys.copy()
    keys["_window_cross_key"] = 1
    calendar = pd.DataFrame({year_col: years, "_window_cross_key": 1})
    panel = keys.merge(calendar, on="_window_cross_key", how="inner").drop(
        columns="_window_cross_key"
    )
    panel = panel.merge(annual, on=group_cols + [year_col], how="left")
    panel[value_col] = panel[value_col].fillna(0.0)
    return panel[columns].sort_values(group_cols + [year_col], kind="stable").reset_index(drop=True)


def aggregate_trade_window(
    raw: pd.DataFrame,
    years: Sequence[int],
    group_cols: Sequence[str],
    *,
    mode: str = CALENDAR_MEAN,
    year_col: str = "year",
    value_col: str = "v",
    hs6_col: str = "k",
    output_col: str | None = None,
    expected_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate sparse trade rows to one value per group over ``years``.

    The default ``calendar_mean`` is ``stage-year sum -> window total / number of
    requested years``. ``legacy_present_hs6_mean`` exactly names the old behavior:
    average each HS6 over its present years and then sum those conditional means.
    """
    years = _normalise_years(years)
    group_cols = list(group_cols)
    output_col = output_col or value_col
    if mode not in VALID_MODES:
        raise ValueError(f"unknown aggregation mode {mode!r}; choose from {sorted(VALID_MODES)}")

    if mode == CALENDAR_MEAN:
        panel = stage_year_totals(
            raw,
            years,
            group_cols,
            year_col=year_col,
            value_col=value_col,
            expected_keys=expected_keys,
        )
        if panel.empty:
            return pd.DataFrame(columns=group_cols + [output_col])
        result = panel.groupby(group_cols, as_index=False, dropna=False)[value_col].sum()
        result[value_col] = result[value_col] / float(len(years))
        return result.rename(columns={value_col: output_col})

    _require_columns(raw, group_cols + [year_col, value_col, hs6_col], "raw")
    work = raw[group_cols + [column for column in [hs6_col, year_col, value_col]
                             if column not in group_cols]].copy()
    if not work.empty:
        work[year_col] = pd.to_numeric(work[year_col], errors="raise").astype(int)
        outside = sorted(set(work[year_col]) - set(years))
        if outside:
            raise ValueError(f"raw contains years outside the requested window: {outside}")
        work[value_col] = pd.to_numeric(work[value_col], errors="raise")

    atomic_cols = group_cols + ([hs6_col] if hs6_col not in group_cols else [])
    if work.empty:
        legacy = pd.DataFrame(columns=group_cols + [output_col])
    else:
        # Sum duplicate source rows within an HS6-year, then average only over years
        # in which that HS6 was present: this is the old, intentionally non-default mode.
        hs6_year = (
            work.groupby(atomic_cols + [year_col], as_index=False, dropna=False)[value_col]
            .sum()
        )
        conditional = (
            hs6_year.groupby(atomic_cols, as_index=False, dropna=False)[value_col]
            .mean()
        )
        legacy = conditional.groupby(group_cols, as_index=False, dropna=False)[value_col].sum()
        legacy = legacy.rename(columns={value_col: output_col})

    if expected_keys is not None:
        _require_columns(expected_keys, group_cols, "expected_keys")
        keys = pd.concat(
            [legacy[group_cols], expected_keys[group_cols]], ignore_index=True
        ).drop_duplicates()
        legacy = keys.merge(legacy, on=group_cols, how="left")
        legacy[output_col] = legacy[output_col].fillna(0.0)
    return legacy[group_cols + [output_col]]


__all__ = [
    "CALENDAR_MEAN",
    "LEGACY_PRESENT_HS6_MEAN",
    "VALID_MODES",
    "aggregate_trade_window",
    "stage_year_totals",
]
