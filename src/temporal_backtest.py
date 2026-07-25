"""
O2 temporal back-test + O1(a) value-chain enrichment, with size-controlled metrics.

Train on EARLY window (2008-2012), predict which (early-exporter, stage, dest) pairs become
new links by the LATE window (2018-2022). Beyond raw AUC vs a market-size baseline, we add:
  - within-size-decile GNN-AUC : reasoning value AFTER controlling for size (the hard test)
  - residual logistic           : does GNN add predictive power on top of size?
--enrich densifies the value chain: HS6-level supplies/demands + transitive derived_from
+ form_of depth (fresh<->frozen, carcass->cuts), to test whether making the value chain
richer lets the GNN beat the size prior on the non-obvious gaps.
"""
import io, zipfile, sys, argparse, os
from pathlib import Path
import torch, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("NBFNET_PATH", str(ROOT.parent / "third_party" / "NBFNet-PyG")))
import universe as U
from baci_filtered_cache import BaciFilteredCache, read_trade_year
from task_features import build_size_lookups, candidate_size_components
from window_aggregation import CALENDAR_MEAN, VALID_MODES, aggregate_trade_window
from split import split_test_mask
# NBFNet / gap_discovery are imported lazily (only when the GNN is trained/scored) so that
# --enum-only runs without torch_scatter; the GNN feature is computed on the cluster, not here.
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RAWB = Path(os.environ.get("VCU_RAW", str(ROOT.parent / "data" / "raw"))); BACI_ZIP = RAWB / "BACI_HS92_V202401b.zip"
GRAV_ZIP = RAWB / "Gravity_csv_V202211.zip"
GCONT = ["lgdp_o", "lgdp_d", "ldist"]; GBIN = ["contig", "comlang_off", "fta_wto"]
# Temporal fold (B3 robustness). VCU_FOLD selects the (EARLY, LATE, gravity-year) triple; the module-level
# values propagate to everything that imports them (benchmark.setup KG build, hp_tune frontier). Both folds
# are non-overlapping 5yr windows with a 5-6yr gap; fold2's LATE (2008-12) is entirely PRE-COVID, so
# reproducing the findings on fold2 doubles as a COVID-sensitivity check. Outputs are fold-tagged (below).
_FOLDS = {
    "main":  ([2008, 2009, 2010, 2011, 2012], [2018, 2019, 2020, 2021, 2022], 2010),
    "fold2": ([1998, 1999, 2000, 2001, 2002], [2008, 2009, 2010, 2011, 2012], 2000),
    # shift (E4): same MODERN era as main but LATE = 2015-2019 has ZERO COVID years -> isolates the
    # pandemic (fold2 is a different era + GFC; shift keeps the era fixed and only removes COVID).
    "shift": ([2005, 2006, 2007, 2008, 2009], [2015, 2016, 2017, 2018, 2019], 2007),
}


def get_fold_spec(fold):
    """Return copies of ``(early_years, late_years, gravity_year)`` for ``fold``.

    New v2 orchestration code passes the fold explicitly instead of relying on the
    import-time ``VCU_FOLD`` switch.  Returning copies prevents callers from
    mutating the module-level protocol definition.
    """
    if fold not in _FOLDS:
        raise ValueError(f"unknown fold={fold!r}; choose from {sorted(_FOLDS)}")
    early, late, gravity_year = _FOLDS[fold]
    return list(early), list(late), int(gravity_year)


FOLD = os.environ.get("VCU_FOLD", "main")
if FOLD not in _FOLDS:
    raise SystemExit(f"unknown VCU_FOLD={FOLD!r}; choose from {list(_FOLDS)}")
EARLY, LATE, GYEAR = _FOLDS[FOLD]
FOLD_SUFFIX = "" if FOLD == "main" else f"_{FOLD}"     # tags candidate/frontier output files
THRESH = 100.0
# Canonical default: aggregate stage-year totals and divide by every calendar year
# in the window. The old present-HS6-year conditional mean remains available only
# as the explicit VCU_WINDOW_AGG=legacy_present_hs6_mean migration mode.
WINDOW_AGG = os.environ.get("VCU_WINDOW_AGG", CALENDAR_MEAN)
if WINDOW_AGG not in VALID_MODES:
    raise ValueError(f"unknown VCU_WINDOW_AGG={WINDOW_AGG!r}; choose from {sorted(VALID_MODES)}")
REL = {r: i for i, r in enumerate(U.EXPORT_RELATIONS + U.BG_RELATIONS)}; NUM_FWD = len(REL)
FRESH = {"020410": "020430", "020421": "020441", "020422": "020442", "020423": "020443"}

# UPGRADE task (single estimand): for each downstream/processed stage, the registered upstream
# stages an exporter must already export to count as an upgrade candidate. These may be raw or
# intermediate stages; the exact set comes from the chain registry
# (universe.UPSTREAM for the active chain); rebound per --chain inside main().
UPSTREAM = U.UPSTREAM
UPGRADE_STAGES = U.UPGRADE_STAGES


def member(zf, n): return zf.open(n).read()


def to(data, dev):   # local copy of gap_discovery.to (avoids importing NBFNet for --enum-only)
    for k in ("edge_index", "edge_type", "target_edge_index", "target_edge_type"):
        if getattr(data, k, None) is not None:
            setattr(data, k, getattr(data, k).to(dev))
    return data


# Dissolved / non-sovereign states (N6): a country dissolved before the LATE window cannot form a NEW
# late relationship, so it only contributes necessarily-negative candidates. Filtered out at build time.
# (S19="Other Asia, nes" ~ Taiwan is a LIVE trade entity with real positives -> retained, disclosed.)
# The v1.6 frozen tables predate this filter and still contain these rows (0 positives, ceiling Δ<=0.0004,
# <=0.68%/chain -- see results/metrics/iso_hygiene.json); a rebuild with this code drops them.
BAD_ISO = {"ANT", "SCG", "YUG", "SUN", "CSK", "DDR"}


def load_window(zf, iso, years, aggregation=WINDOW_AGG):
    uni = set(U.ALL_HS); fr = []
    for y in years:
        df = read_trade_year(zf, y)
        df["k"] = df["k"].str.zfill(6); df = df[df["k"].isin(uni)]
        df["v"] = pd.to_numeric(df["v"], errors="coerce"); df["year"] = y
        fr.append(df[["i", "j", "k", "year", "v"]])
    t = pd.concat(fr, ignore_index=True)
    t["i_iso"] = t["i"].map(iso); t["j_iso"] = t["j"].map(iso)
    t = t.dropna(subset=["i_iso", "j_iso"])
    t = t[~t.i_iso.isin(BAD_ISO) & ~t.j_iso.isin(BAD_ISO)]        # N6: drop dissolved-state rows
    t["stage"] = t["k"].map(U.stage_of)
    t = t.dropna(subset=["stage"])
    stage = aggregate_trade_window(
        t, years, ["i_iso", "j_iso", "stage"], mode=aggregation, hs6_col="k"
    )
    hs6 = aggregate_trade_window(
        t, years, ["i_iso", "j_iso", "k", "stage"], mode=aggregation, hs6_col="k"
    )
    return stage[stage.v > THRESH], hs6[hs6.v > 50]


def auc_ap(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    if y.sum() == 0 or (y == 0).sum() == 0: return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def load_gravity_early(gravity_year=None):
    """Load gravity covariates for an explicit early-window reference year.

    ``None`` preserves the historical import-time behavior.  Strict rolling v2
    callers pass a year from :func:`get_fold_spec`, which avoids an accidental
    main/fold2 mismatch when more than one phase is orchestrated.
    """
    gravity_year = GYEAR if gravity_year is None else int(gravity_year)
    zf = zipfile.ZipFile(GRAV_ZIP)
    cols = ["year", "iso3_o", "iso3_d", "dist", "contig", "comlang_off", "gdp_o", "gdp_d", "fta_wto"]
    keep = []
    for ch in pd.read_csv(io.BytesIO(member(zf, "Gravity_V202211.csv")), usecols=cols, chunksize=400000):
        keep.append(ch[ch.year == gravity_year])
    g = pd.concat(keep)
    g = g[g.iso3_o != g.iso3_d].dropna(subset=["dist", "gdp_o", "gdp_d", "contig", "comlang_off", "fta_wto"])
    g = g[(g.gdp_o > 0) & (g.gdp_d > 0) & (g.dist > 0)]
    g["lgdp_o"] = np.log(g.gdp_o); g["lgdp_d"] = np.log(g.gdp_d); g["ldist"] = np.log(g.dist)
    return g[["iso3_o", "iso3_d"] + GCONT + GBIN]


def fit_gravity_per_stage(grav, early):
    """PPML gravity per stage on EARLY window. Returns dict stage->(model, scaler) + covariate index."""
    gidx = {(o, d): i for i, (o, d) in enumerate(zip(grav.iso3_o, grav.iso3_d))}
    Xall = grav[GCONT + GBIN].to_numpy(float)
    models = {}
    for st in U.EXPORT_RELATIONS:
        obs = early[early.stage == st][["i_iso", "j_iso", "v"]]
        panel = grav.merge(obs, left_on=["iso3_o", "iso3_d"], right_on=["i_iso", "j_iso"], how="left")
        y = (panel.v.fillna(0.0) / 1000.0).to_numpy()           # $M
        X = panel[GCONT + GBIN].to_numpy(float)
        sc = StandardScaler().fit(X[:, :len(GCONT)])
        Xz = np.column_stack([sc.transform(X[:, :len(GCONT)]), X[:, len(GCONT):]])
        pr = PoissonRegressor(alpha=1e-4, max_iter=400).fit(Xz, y)
        models[st] = (pr, sc)
    return models, gidx, Xall


def within_decile_auc(s, y, size, nb=10):
    s, y, size = map(np.asarray, (s, y, size))
    q = np.quantile(size, np.linspace(0, 1, nb + 1)); aucs = []; w = []
    for b in range(nb):
        m = (size >= q[b]) & (size <= q[b+1]) if b == nb-1 else (size >= q[b]) & (size < q[b+1])
        if y[m].sum() >= 3 and (y[m] == 0).sum() >= 3:
            aucs.append(roc_auc_score(y[m], s[m])); w.append(m.sum())
    return (np.average(aucs, weights=w) if aucs else float("nan")), aucs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--upgrade", action="store_true",
                    help="restrict candidates to UPGRADE-type (single coherent estimand)")
    ap.add_argument("--first-time", dest="first_time", action="store_true",
                    help="B2: stage-entry subset -- exporter has a REGISTERED UPSTREAM stage but does "
                         "NOT already export the processed stage s (ups - early_exp[st]); the default "
                         "--upgrade task is destination extension for incumbent processors (early_exp[st] & ups)")
    ap.add_argument("--chain", default=os.environ.get("VCU_CHAIN", "sheep"),
                    help="value-chain id from the registry (chains/*.json)")
    ap.add_argument("--enum-only", dest="enum_only", action="store_true",
                    help="enumerate candidates, save the table, print counts, and exit (no GNN)")
    ap.add_argument(
        "--aggregation",
        choices=sorted(VALID_MODES),
        default=WINDOW_AGG,
        help=("fixed-window trade aggregation; calendar_mean is the v2 default, while "
              "legacy_present_hs6_mean is retained only for v1 migration diagnostics"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("candidate output directory; defaults to data/processed_v2 for calendar_mean and "
              "data/processed for the explicit legacy aggregation"),
    )
    ap.add_argument("--benchmark-version", default="2.1-dev",
                    help="version string embedded in newly written candidate tables")
    cache_default = os.environ.get("VCU_BACI_CACHE")
    ap.add_argument(
        "--baci-cache",
        type=Path,
        default=Path(cache_default) if cache_default else None,
        help=("optional strict private filtered cache built by "
              "tools/build_baci_filtered_cache.py (or set VCU_BACI_CACHE); "
              "the default remains direct BACI ZIP reads"),
    )
    args = ap.parse_args()
    if args.first_time and not args.upgrade:
        ap.error("--first-time requires --upgrade (it flips the upgrade candidate filter)")
    global REL, NUM_FWD, UPSTREAM, UPGRADE_STAGES
    U.set_active_chain(args.chain)
    REL = {r: i for i, r in enumerate(U.EXPORT_RELATIONS + U.BG_RELATIONS)}; NUM_FWD = len(REL)
    UPSTREAM = U.UPSTREAM; UPGRADE_STAGES = U.UPGRADE_STAGES
    output_dir = args.output_dir
    if output_dir is None:
        dirname = "processed_v2" if args.aggregation == CALENDAR_MEAN else "processed"
        output_dir = ROOT.parent / "data" / dirname
    elif not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    print(f"[chain={U.ACTIVE_CHAIN}] {len(U.EXPORT_RELATIONS)} stages, {len(U.ALL_HS)} HS6  "
          f"window_agg={args.aggregation} output_dir={output_dir}")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache = None
    if args.baci_cache is not None:
        cache = BaciFilteredCache(
            args.baci_cache,
            requested_years=sorted(set(EARLY + LATE)),
            chains_dir=U.CHAINS_DIR,
        )
        print("BACI input=verified private filtered cache")
    else:
        print("BACI input=direct raw ZIP")
    with zipfile.ZipFile(BACI_ZIP) as zf:
        country_payload = (
            cache.country_codes_bytes(zf, archive_path=BACI_ZIP)
            if cache is not None
            else member(zf, "country_codes_V202401b.csv")
        )
        cc = pd.read_csv(io.BytesIO(country_payload))
        iso = dict(zip(cc.country_code, cc.country_iso3))
        trade_source = cache if cache is not None else zf
        early, early_hs6 = load_window(
            trade_source, iso, EARLY, aggregation=args.aggregation
        )
        late, _ = load_window(trade_source, iso, LATE, aggregation=args.aggregation)
    print(f"[enrich={args.enrich}] early links={len(early)} late links={len(late)}")

    countries = set(early.i_iso) | set(early.j_iso) | set(late.i_iso) | set(late.j_iso)
    products = set(U.ALL_HS) | {h[:4] for h in U.ALL_HS} | {h[:2] for h in U.ALL_HS}
    ents = sorted(countries) + sorted(products); eid = {e: i for i, e in enumerate(ents)}
    n_nodes = len(ents)

    tr = torch.tensor([early.i_iso.map(eid).to_numpy(), early.j_iso.map(eid).to_numpy(),
                       early.stage.map(REL).to_numpy()], dtype=torch.long)
    bh, bt, br = [], [], []
    def add(s, d, r): bh.extend(s); bt.extend(d); br.extend([REL[r]] * len(s))

    # form_of: chain form_of pairs (sheep: live -> meat) + sheep-only enrich (fresh<->frozen, carcass->cuts)
    fs, fd = [], []
    for sa, sb in U.FORM_OF:
        for lv in U.STAGES[sa]:
            for mt in U.STAGES[sb]: fs.append(eid[lv]); fd.append(eid[mt])
    if args.enrich and U.ACTIVE_CHAIN == "sheep":
        for a, b in FRESH.items():
            if a in eid and b in eid: fs += [eid[a]]; fd += [eid[b]]
        carc = ["020410", "020421", "020430", "020441"]; cuts = ["020422", "020423", "020442", "020443"]
        for c in carc:
            for k in cuts:
                if c in eid and k in eid: fs += [eid[c]]; fd += [eid[k]]
    add(fs, fd, "form_of")

    # derived_from (+ enrich: transitive source->ALL downstream HS)
    ds, dd = [], []
    for st, srcs in U.DERIVED_FROM.items():
        for hs in U.STAGES[st]:
            for sc in srcs: ds.append(eid[sc]); dd.append(eid[hs])
    for hs, srcs in U.DERIVED_FROM_HS.items():
        for sc in srcs: ds.append(eid[sc]); dd.append(eid[hs])
    # The strict sheep registry has no extra transitive downstream baskets beyond
    # the explicit DERIVED_FROM map.  In particular, optional enrichment must not
    # resurrect the removed generic food/fat/furskin/fabric/carpet/blanket stages.
    add(ds, dd, "derived_from")

    hc, hp = [], []
    for h in U.ALL_HS: hc += [eid[h], eid[h[:4]]]; hp += [eid[h[:4]], eid[h[:2]]]
    add(hc, hp, "hs_parent")

    # supplies/demands: stage-rep (baseline) OR HS6-level (enrich)
    if args.enrich:
        sup = early_hs6[["i_iso", "k"]].drop_duplicates(); dem = early_hs6[["j_iso", "k"]].drop_duplicates()
        sup = sup[sup.i_iso.isin(eid) & sup.k.isin(eid)]; dem = dem[dem.j_iso.isin(eid) & dem.k.isin(eid)]
        add([eid[c] for c in sup.i_iso], [eid[p] for p in sup.k], "supplies")
        add([eid[c] for c in dem.j_iso], [eid[p] for p in dem.k], "demands")
    else:
        rep = {st: eid[U.STAGES[st][0]] for st in U.EXPORT_RELATIONS}
        sup = early[["i_iso", "stage"]].drop_duplicates(); dem = early[["j_iso", "stage"]].drop_duplicates()
        add([eid[c] for c in sup.i_iso], [rep[s] for s in sup.stage], "supplies")
        add([eid[c] for c in dem.j_iso], [rep[s] for s in dem.stage], "demands")

    bg = torch.tensor([bh, bt, br], dtype=torch.long)
    fwd = torch.cat([tr, bg], 1)
    ei = torch.cat([fwd[:2], torch.stack([fwd[1], fwd[0]])], 1)
    ety = torch.cat([fwd[2], fwd[2] + NUM_FWD])
    train_data = to(Data(edge_index=ei, edge_type=ety, num_nodes=n_nodes,
                         target_edge_index=tr[:2], target_edge_type=tr[2],
                         num_relations=2 * NUM_FWD), dev)
    print(f"bg edges (fwd)={bg.shape[1]}  | EARLY graph ready")

    early_set = set(map(tuple, early[["i_iso", "j_iso", "stage"]].to_numpy()))
    late_set = set(map(tuple, late[["i_iso", "j_iso", "stage"]].to_numpy()))
    late_val = {(i, j, s): v for i, j, s, v in late[["i_iso", "j_iso", "stage", "v"]].to_numpy()}
    early_imp = {st: set(early[early.stage == st].j_iso) for st in U.EXPORT_RELATIONS}
    early_exp = {st: set(early[early.stage == st].i_iso) for st in U.EXPORT_RELATIONS}
    out_tot, in_tot, upstream_out_tot = build_size_lookups(early, UPGRADE_STAGES, UPSTREAM)

    # full gravity model (PPML per stage, early window) — the formal baseline
    print("fitting PPML gravity per stage (early)...")
    grav = load_gravity_early(); gmodels, gidx, GX = fit_gravity_per_stage(grav, early)

    # ---- enumerate candidates ONCE (labels / size / gravity / late-value are seed-independent) ----
    stages_use = UPGRADE_STAGES if args.upgrade else U.EXPORT_RELATIONS
    groups = []      # (i_eid, r, [(j_eid, label, size, grav_or_nan, has, lateval)])
    cand_meta = []   # parallel to flat: (i_iso, j_iso, stage)
    exporter_capacity_meta = []
    importer_demand_meta = []
    for st in stages_use:
        r = REL[st]; pr, scl = gmodels[st]
        ups = set().union(*[early_exp.get(u, set()) for u in UPSTREAM.get(st, [])]) if args.upgrade else None
        if not args.upgrade:
            pool = early_exp[st]
        elif args.first_time:                # has a registered upstream stage, does NOT yet export s
            pool = ups - early_exp[st]
        else:                                # incumbent: already exports s AND a registered upstream stage
            pool = early_exp[st] & ups
        # Sets are intentionally sorted before enumeration: candidate-table row order is
        # part of the release artifact and must not depend on Python's hash randomization.
        for i in sorted(pool):
            cand = sorted(j for j in early_imp[st] if (i, j, st) not in early_set and j != i)
            if not cand: continue
            items = []
            for j in cand:
                lab = 1 if (i, j, st) in late_set else 0
                ot, importer_demand, size = candidate_size_components(
                    i, j, st, first_time=args.first_time, processed_out=out_tot,
                    processed_in=in_tot, upstream_out=upstream_out_tot,
                )
                lv = float(late_val.get((i, j, st), 0.0)) if lab else 0.0
                gi = gidx.get((i, j))
                if gi is None:
                    items.append((eid[j], lab, size, np.nan, False, lv))
                else:
                    x = GX[gi:gi + 1]
                    xz = np.column_stack([scl.transform(x[:, :len(GCONT)]), x[:, len(GCONT):]])
                    items.append((eid[j], lab, size, float(pr.predict(xz)[0]), True, lv))
                cand_meta.append((i, j, st))
                exporter_capacity_meta.append(ot)
                importer_demand_meta.append(importer_demand)
            groups.append((eid[i], r, items))
    flat = [it for _, _, items in groups for it in items]
    y_all = np.array([t[1] for t in flat]); b_all = np.array([t[2] for t in flat])
    g_all = np.array([t[3] for t in flat]); has_all = np.array([t[4] for t in flat])
    lv_all = np.array([t[5] for t in flat])   # realized LATE value if materialized ($k)
    task_tag = "first-time-upgrader" if args.first_time else ("destination-extension" if args.upgrade else "all-export")
    print(f"[fold={FOLD} EARLY={EARLY[0]}-{EARLY[-1]} LATE={LATE[0]}-{LATE[-1]} upgrade={args.upgrade} "
          f"task={task_tag}] candidates={len(y_all)} materialized={int(y_all.sum())} "
          f"base={y_all.mean():.3f}  realizable value=${lv_all.sum()/1000:,.0f}M")

    if args.enum_only:   # GNN-free: labels/size/grav/lateval only (gnn pending) -> per-chain table + counts
        if args.upgrade:
            ct = pd.DataFrame(cand_meta, columns=["i_iso", "j_iso", "stage"])
            ct["y"] = y_all; ct["size"] = b_all
            ct["log_exporter_capacity"] = exporter_capacity_meta
            ct["log_importer_demand"] = importer_demand_meta
            ct["size_basis"] = ("registered_upstream_exporter_plus_processed_importer" if args.first_time
                                else "processed_exporter_plus_processed_importer")
            ct["grav"] = g_all; ct["gnn"] = np.nan; ct["lateval"] = lv_all
            ct["benchmark_version"] = args.benchmark_version
            ct["aggregation"] = args.aggregation
            ct["early_window"] = f"{EARLY[0]}-{EARLY[-1]}"
            ct["late_window"] = f"{LATE[0]}-{LATE[-1]}"
            stem = f"candidates_firsttime_{U.ACTIVE_CHAIN}" if args.first_time else f"candidates_{U.ACTIVE_CHAIN}"
            out = output_dir / f"{stem}{FOLD_SUFFIX}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            ct.to_csv(out, index=False); print(f"[enum-only] saved -> {out} ({len(ct)} rows)")
        print(f"[enum-only] DONE chain={U.ACTIVE_CHAIN}: candidates={len(y_all)} materialized={int(y_all.sum())}")
        return

    @torch.no_grad()
    def score_groups(model):
        from nbfnet import tasks
        model.eval(); out = np.empty(len(y_all)); k = 0
        for h, r, items in groups:
            pos = torch.tensor([[h, h, r]], device=dev)
            tb, _ = tasks.all_negative(train_data, pos)
            sc = model(train_data, tb).squeeze(0).cpu().numpy()
            for it in items:
                out[k] = sc[it[0]]; k += 1
        return out

    def z(x): return (x - x.mean()) / (x.std() + 1e-9)

    def metrics(s):
        identities = np.asarray(cand_meta, dtype=str)
        te = split_test_mask(
            U.ACTIVE_CHAIN,
            identities[:, 0],
            identities[:, 2],
            identities[:, 1],
        )
        m = ~te
        d = {}
        d["gnn_auc"], d["gnn_ap"] = auc_ap(s, y_all)
        d["wd_size"], _ = within_decile_auc(s, y_all, b_all)
        Xs = z(b_all).reshape(-1, 1); Xb = np.column_stack([z(b_all), z(s)])
        rs = LogisticRegression(max_iter=200).fit(Xs[m], y_all[m])
        rb = LogisticRegression(max_iter=200).fit(Xb[m], y_all[m])
        d["res_size"] = (roc_auc_score(y_all[~m], rb.predict_proba(Xb[~m])[:, 1])
                         - roc_auc_score(y_all[~m], rs.predict_proba(Xs[~m])[:, 1]))
        h = has_all & np.isfinite(g_all); sy, yy, gg = s[h], y_all[h], g_all[h]
        d["gnn_auc_sub"], _ = auc_ap(sy, yy); d["grav_auc"], _ = auc_ap(gg, yy)
        d["wd_grav"], _ = within_decile_auc(sy, yy, gg)
        mm = m[h]
        Xg = z(gg).reshape(-1, 1); Xgg = np.column_stack([z(gg), z(sy)])
        rgv = LogisticRegression(max_iter=200).fit(Xg[mm], yy[mm])
        rgg = LogisticRegression(max_iter=200).fit(Xgg[mm], yy[mm])
        d["res_grav"] = (roc_auc_score(yy[~mm], rgg.predict_proba(Xgg[~mm])[:, 1])
                         - roc_auc_score(yy[~mm], rgv.predict_proba(Xg[~mm])[:, 1]))
        return d

    # ---- multi-seed: retrain GNN per seed, score same candidates -> mean ± std ----
    seeds = [int(s) for s in args.seeds.split(",")]
    base = y_all.mean()
    hv = has_all & np.isfinite(g_all)
    grav_auc_ref = auc_ap(g_all[hv], y_all[hv])[0]
    runs = []; score_sum = np.zeros(len(y_all))
    from gap_discovery import train_model
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        print(f"-- seed {sd}: training {args.epochs}ep ({args.layers} layers) --")
        model = train_model(train_data, dev, epochs=args.epochs, num_rel=2 * NUM_FWD, layers=args.layers)
        s = score_groups(model); runs.append(metrics(s)); score_sum += s
    score_mean = score_sum / len(seeds)   # mean GNN score -> single ranking for top-k value metrics

    if args.upgrade:   # save candidate-level table for the institutional-residual test (① WITS)
        ct = pd.DataFrame(cand_meta, columns=["i_iso", "j_iso", "stage"])
        ct["y"] = y_all; ct["size"] = b_all
        ct["log_exporter_capacity"] = exporter_capacity_meta
        ct["log_importer_demand"] = importer_demand_meta
        ct["size_basis"] = ("registered_upstream_exporter_plus_processed_importer" if args.first_time
                            else "processed_exporter_plus_processed_importer")
        ct["grav"] = g_all
        ct["gnn"] = score_mean; ct["lateval"] = lv_all
        ct["benchmark_version"] = args.benchmark_version
        ct["aggregation"] = args.aggregation
        ct["early_window"] = f"{EARLY[0]}-{EARLY[-1]}"
        ct["late_window"] = f"{LATE[0]}-{LATE[-1]}"
        # sheep keeps its legacy path (downstream institutional scripts read it); other chains
        # write candidates_<chain>.csv (full table WITH the trained gnn column).
        if args.first_time:
            fname = f"candidates_firsttime_{U.ACTIVE_CHAIN}{FOLD_SUFFIX}.csv"
        elif (FOLD == "main" and U.ACTIVE_CHAIN == "sheep"
              and args.aggregation != CALENDAR_MEAN and output_dir.name == "processed"):
            fname = "upgrade_candidates.csv"   # legacy path (downstream sheep institutional scripts read it)
        else:
            fname = f"candidates_{U.ACTIVE_CHAIN}{FOLD_SUFFIX}.csv"
        out = output_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        ct.to_csv(out, index=False); print(f"saved candidate table -> {out} ({len(ct)} rows)")

    def agg(k):
        v = np.array([r[k] for r in runs]); return v.mean(), v.std()
    print(f"\n{'='*74}\nMULTI-SEED BACK-TEST (enrich={args.enrich}) seeds={seeds} "
          f"#cand={len(y_all)} #new={int(y_all.sum())} base={base:.3f}\n{'='*74}")
    rows = [("gnn_auc", "overall GNN AUC"), ("gnn_ap", "overall GNN AP"),
            ("wd_size", "HARD within-size-decile GNN-AUC"),
            ("res_size", "residual Δ AUC (size -> size+GNN)"),
            ("gnn_auc_sub", "[vs gravity] GNN AUC"), ("grav_auc", "[vs gravity] gravity AUC"),
            ("wd_grav", "[vs gravity] within-gravity-decile GNN-AUC"),
            ("res_grav", "[vs gravity] residual Δ AUC (grav -> grav+GNN)")]
    for k, label in rows:
        mu, sd = agg(k); print(f"  {label:46s} {mu:.3f} ± {sd:.3f}")
    print(f"  (gravity AUC deterministic ref = {grav_auc_ref:.3f}; size within-decile ~0.55 by constr.)")

    # ---- value-weighted top-k (policy metric): how much realizable upgrade value does the
    #      model's top-k shortlist capture, vs size/gravity baselines and a random shortlist ----
    tot_val = lv_all.sum(); tot_new = y_all.sum()
    def topk(score, k):
        idx = np.argsort(score)[::-1][:k]
        return y_all[idx].mean(), lv_all[idx].sum() / max(tot_val, 1)   # precision@k, value-captured@k
    print(f"\n{'-'*74}\nVALUE-WEIGHTED TOP-K (rank by mean-GNN; realizable value=${tot_val/1000:,.0f}M, "
          f"materialized={int(tot_new)})\n{'-'*74}")
    print(f"  (each cell = precision@k / value-captured@k)")
    print(f"{'k':>5s} {'GNN':>14s} {'gravity':>14s} {'size':>14s} {'random':>14s}")
    grav_rank = np.where(np.isfinite(g_all), g_all, np.nanmin(g_all[hv]))
    for k in (50, 100, 250, 500):
        gp, gv = topk(score_mean, k); rp, rv = topk(grav_rank, k); sp, sv = topk(b_all, k)
        print(f"{k:>5d} {gp:>6.2f}/{gv:<6.2f} {rp:>6.2f}/{rv:<6.2f} {sp:>6.2f}/{sv:<6.2f} "
              f"{base:>6.2f}/{k/len(y_all):<6.2f}")


if __name__ == "__main__":
    main()
