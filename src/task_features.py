"""Ex-ante feature construction shared by UPGRADE-BENCH tasks."""
from __future__ import annotations

import numpy as np


def build_size_lookups(early, upgrade_stages, upstream):
    """Build exporter/importer capacity lookups for destination and entry tasks."""
    processed_out = early.groupby(["i_iso", "stage"]).v.sum().to_dict()
    processed_in = early.groupby(["j_iso", "stage"]).v.sum().to_dict()
    upstream_out = {}
    for downstream_stage in upgrade_stages:
        upstream_stages = list(upstream.get(downstream_stage, []))
        upstream_rows = early[early.stage.isin(upstream_stages)]
        upstream_out[downstream_stage] = upstream_rows.groupby("i_iso").v.sum().to_dict()
    return processed_out, processed_in, upstream_out


def candidate_size_components(
    i,
    j,
    stage,
    *,
    first_time,
    processed_out,
    processed_in,
    upstream_out,
):
    """Return log exporter capacity, log importer demand, and their additive prior.

    Track A uses established processed-stage exporter capacity. Track B uses capacity
    across the raw and/or intermediate stages registered in ``upstream_map[stage]``
    because processed-stage volume is zero by construction.
    """
    if first_time:
        exporter = np.log1p(upstream_out.get(stage, {}).get(i, 0.0))
    else:
        exporter = np.log1p(processed_out.get((i, stage), 0.0))
    importer = np.log1p(processed_in.get((j, stage), 0.0))
    return float(exporter), float(importer), float(exporter + importer)


__all__ = ["build_size_lookups", "candidate_size_components"]
