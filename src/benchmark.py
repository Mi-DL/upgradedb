"""
ERA-style multi-method benchmark on the SAME scorable task as the temporal back-test:
predict which (exporter, stage, importer) gaps MATERIALIZE (train EARLY 2008-12 -> LATE 2018-22).

Methods scored on the identical candidate set:
  size           log(exp vol)+log(imp vol)            [deterministic]
  gravity PPML   E[v]=exp(b.[lgdp,ldist,contig,...])  [deterministic]
  heuristics     Adamic-Adar / common-neighbours / pref-attach on early country graph  [det.]
  KGC embeddings TransE / RotatE / DistMult / ComplEx  (pykeen, multi-seed)
  GNN KGC        R-GCN / CompGCN                        (pykeen, multi-seed)
Reports same-window diagnostic metrics for the methods actually run. A historical v1 NBFNet
number is deliberately not injected: v2 scores must come from a fresh calendar-mean,
rolling-protocol run.
"""
import sys, io, zipfile, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import universe as U
from split import split_test_mask
from task_features import build_size_lookups, candidate_size_components
from temporal_backtest import (load_window, load_gravity_early, fit_gravity_per_stage,
                               auc_ap, within_decile_auc, member, UPSTREAM, UPGRADE_STAGES,
                               BACI_ZIP, EARLY, LATE, REL, NUM_FWD, GCONT, GBIN,
                               FOLD, WINDOW_AGG, get_fold_spec)
from sklearn.linear_model import LogisticRegression

EMB = ["TransE", "RotatE", "DistMult", "ComplEx"]
GNN = ["RGCN", "CompGCN"]


def _load_early(fold=None, aggregation=None):
    """Load only the early graph window for an explicit fold.

    This helper is deliberately label-blind: it never opens the late BACI files.
    The strict v2 GPU runner uses it to refit representations before target labels
    are read.
    """
    fold = FOLD if fold is None else fold
    aggregation = WINDOW_AGG if aggregation is None else aggregation
    early_years, _, _ = get_fold_spec(fold)
    zf = zipfile.ZipFile(BACI_ZIP)
    cc = pd.read_csv(io.BytesIO(member(zf, "country_codes_V202401b.csv")))
    iso = dict(zip(cc.country_code, cc.country_iso3))
    early, early_hs6 = load_window(zf, iso, early_years, aggregation=aggregation)
    return early, early_hs6


def _build_labeled_triples(early, early_hs6):
    """Build the label-free KGE/path-model graph from an early window."""
    countries = set(early.i_iso) | set(early.j_iso)
    products = set(U.ALL_HS) | {h[:4] for h in U.ALL_HS} | {h[:2] for h in U.ALL_HS}
    ents = sorted(countries) + sorted(products); eid = {e: i for i, e in enumerate(ents)}

    trip = early[["i_iso", "stage", "j_iso"]].to_numpy().tolist()
    for hs in U.ALL_HS:                              # hs_parent
        trip.append([hs, "hs_parent", hs[:4]]); trip.append([hs[:4], "hs_parent", hs[:2]])
    for source_stage, target_stage in U.FORM_OF:      # cross-form chain semantics
        for source_hs in U.STAGES[source_stage]:
            for target_hs in U.STAGES[target_stage]:
                trip.append([source_hs, "form_of", target_hs])
    for st, srcs in U.DERIVED_FROM.items():          # derived_from
        for hs in U.STAGES[st]:
            for s in srcs: trip.append([s, "derived_from", hs])
    for hs, srcs in U.DERIVED_FROM_HS.items():       # HS-specific derived_from overrides
        for source_hs in srcs:
            trip.append([source_hs, "derived_from", hs])
    sup = early_hs6[["i_iso", "k"]].drop_duplicates()
    dem = early_hs6[["j_iso", "k"]].drop_duplicates()
    for c, k in sup.to_numpy():
        if c in eid and k in eid: trip.append([c, "supplies", k])
    for c, k in dem.to_numpy():
        if c in eid and k in eid: trip.append([c, "demands", k])
    return np.array([[str(a), str(b), str(c)] for a, b, c in trip], dtype=str)


def setup_early_graph(fold=None, aggregation=None):
    """Return ``(triples, early_table)`` without opening any late-window file."""
    early, early_hs6 = _load_early(fold=fold, aggregation=aggregation)
    return _build_labeled_triples(early, early_hs6), early


def setup(upgrade=False, first_time=False, fold=None, aggregation=None):
    """Build the historical benchmark problem for an explicit temporal fold.

    Existing callers retain their import-time defaults.  New rolling callers use
    :func:`setup_early_graph` and released candidate identities so representation
    fitting stays label-blind.
    """
    fold = FOLD if fold is None else fold
    aggregation = WINDOW_AGG if aggregation is None else aggregation
    early_years, late_years, gravity_year = get_fold_spec(fold)
    zf = zipfile.ZipFile(BACI_ZIP)
    cc = pd.read_csv(io.BytesIO(member(zf, "country_codes_V202401b.csv")))
    iso = dict(zip(cc.country_code, cc.country_iso3))
    early, early_hs6 = load_window(zf, iso, early_years, aggregation=aggregation)
    late, _ = load_window(zf, iso, late_years, aggregation=aggregation)

    early_set = set(map(tuple, early[["i_iso", "j_iso", "stage"]].to_numpy()))
    late_set = set(map(tuple, late[["i_iso", "j_iso", "stage"]].to_numpy()))
    early_imp = {st: set(early[early.stage == st].j_iso) for st in U.EXPORT_RELATIONS}
    early_exp = {st: set(early[early.stage == st].i_iso) for st in U.EXPORT_RELATIONS}
    out_tot, in_tot, upstream_out_tot = build_size_lookups(
        early, U.UPGRADE_STAGES, U.UPSTREAM
    )

    grav = load_gravity_early(gravity_year); gmodels, gidx, GX = fit_gravity_per_stage(grav, early)

    # candidates: (i_iso, stage, j_iso, label, size, gravity) — UPGRADE-restricted if requested
    stages_use = U.UPGRADE_STAGES if upgrade else U.EXPORT_RELATIONS   # active chain (not import-bound sheep)
    rows = []
    for st in stages_use:
        pr, scl = gmodels[st]
        ups = set().union(*[early_exp.get(u, set()) for u in U.UPSTREAM.get(st, [])]) if upgrade else None
        if not upgrade:
            exps = early_exp[st]
        elif first_time:                       # E1/C2: stage-entry — has upstream raw, does NOT yet export s
            exps = ups - early_exp[st]
        else:                                  # incumbent destination extension
            exps = early_exp[st] & ups
        # Keep candidate identity/order stable across processes and PYTHONHASHSEED values.
        for i in sorted(exps):
            for j in sorted(early_imp[st]):
                if (i, j, st) in early_set or j == i: continue
                _, _, size = candidate_size_components(
                    i,
                    j,
                    st,
                    first_time=bool(upgrade and first_time),
                    processed_out=out_tot,
                    processed_in=in_tot,
                    upstream_out=upstream_out_tot,
                )
                gi = gidx.get((i, j))
                if gi is None:
                    gv = np.nan
                else:
                    x = GX[gi:gi + 1]
                    xz = np.column_stack([scl.transform(x[:, :len(GCONT)]), x[:, len(GCONT):]])
                    gv = float(pr.predict(xz)[0])
                rows.append((i, st, j, 1 if (i, j, st) in late_set else 0, size, gv))
    cand = pd.DataFrame(rows, columns=["i", "st", "j", "y", "size", "grav"])

    # early triples for KGC (stage edges + value-chain background), as label arrays
    trip = _build_labeled_triples(early, early_hs6)
    return cand, trip, early


def heuristics(cand, early):
    # undirected country-country graph from early trade (union over stages)
    from collections import defaultdict
    nbr = defaultdict(set)
    for i, j in early[["i_iso", "j_iso"]].drop_duplicates().to_numpy():
        nbr[i].add(j); nbr[j].add(i)
    deg = {k: len(v) for k, v in nbr.items()}
    aa, cn, pa = [], [], []
    for i, j in cand[["i", "j"]].to_numpy():
        common = nbr.get(i, set()) & nbr.get(j, set())
        cn.append(len(common))
        aa.append(sum(1.0 / np.log(deg[c]) for c in common if deg.get(c, 0) > 1))
        pa.append(deg.get(i, 0) * deg.get(j, 0))
    return dict(adamic_adar=np.array(aa), common_neighbors=np.array(cn),
               pref_attach=np.array(pa, float))


def kgc_score(model_name, trip, cand, seed, dev, epochs):
    # Keep PyKEEN optional for the NBFNet-only strict rolling worker, which uses
    # setup_early_graph() but does not need the embedding stack.
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    # CompGCN wants explicit inverses; RGCN creates them internally (must be False)
    tf = TriplesFactory.from_labeled_triples(trip, create_inverse_triples=(model_name == "CompGCN"))
    res = pipeline(training=tf, testing=tf, model=model_name,
                   model_kwargs=dict(embedding_dim=64),
                   training_kwargs=dict(num_epochs=epochs, batch_size=2048, use_tqdm=False),
                   random_seed=seed, device=dev)
    model = res.model
    e2i = tf.entity_to_id; r2i = tf.relation_to_id
    triples = torch.tensor([[e2i[i], r2i[st], e2i[j]]
                            for i, st, j in cand[["i", "st", "j"]].to_numpy()],
                           dtype=torch.long, device=dev)
    with torch.no_grad():
        sc = model.score_hrt(triples).detach().cpu().numpy().flatten()
    return sc


def metric_row(name, s, y, size):
    auc, ap = auc_ap(s, y)
    wd, _ = within_decile_auc(s, y, size)
    return name, auc, ap, wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=EMB + GNN)
    ap.add_argument("--upgrade", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    cand, trip, early = setup(upgrade=args.upgrade)
    y = cand.y.to_numpy(); size = cand["size"].to_numpy(); grav = cand.grav.to_numpy()
    print(f"candidates={len(cand)}  materialized={int(y.sum())}  base={y.mean():.3f}  triples={len(trip)}")

    table = []  # (name, auc_mean, auc_std, wd_mean, wd_std)
    scores = {}  # name -> score array (seed-0 for stochastic), for the recombination
    # deterministic baselines
    _, a, _, w = metric_row("size", size, y, size); table.append(("size", a, 0.0, w, 0.0))
    scores["size"] = size
    hv = np.isfinite(grav)
    ga, _ = auc_ap(grav[hv], y[hv]); gw, _ = within_decile_auc(grav[hv], y[hv], size[hv])
    table.append(("gravity_PPML", ga, 0.0, gw, 0.0)); scores["gravity"] = grav
    for nm, s in heuristics(cand, early).items():
        _, a, _, w = metric_row(nm, s, y, size); table.append((nm, a, 0.0, w, 0.0)); scores[nm] = s
    # stochastic KGC models (multi-seed)
    for nm in args.models:
        aucs, wds = [], []
        for sd in seeds:
            print(f"-- {nm} seed {sd} --")
            try:
                s = kgc_score(nm, trip, cand, sd, dev, args.epochs)
                a, _ = auc_ap(s, y); w, _ = within_decile_auc(s, y, size)
                aucs.append(a); wds.append(w)
                if sd == seeds[0]: scores[nm] = s
            except Exception as e:
                print(f"   {nm} seed {sd} FAILED: {e}")
        if aucs:
            table.append((nm, float(np.mean(aucs)), float(np.std(aucs)),
                          float(np.mean(wds)), float(np.std(wds))))

    table.sort(key=lambda r: -r[3])  # sort by within-size-decile AUC (the honest metric)

    # ---- ERA-style RECOMBINATION: does an ensemble of complementary signals beat the best single? ----
    def z(x): x = np.asarray(x, float); return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    gfill = np.where(np.isfinite(grav), grav, np.nanmedian(grav[hv]))
    combos = {
        "ens[TransE+grav+AA]": ["TransE", "_grav", "adamic_adar"],
        "ens[TransE+grav+AA+size]": ["TransE", "_grav", "adamic_adar", "size"],
        "ens[grav+AA]": ["_grav", "adamic_adar"],
    }
    te = split_test_mask(U.ACTIVE_CHAIN, cand.i, cand.st, cand.j); m = ~te
    print(f"\n{'-'*78}\nRECOMBINATION (logistic, 50/50 split; vs best single on SAME test half)\n{'-'*78}")
    best_single = max(table, key=lambda r: r[3])
    bs = best_single[0].replace("gravity_PPML", "gravity")
    bs_arr = gfill if bs == "gravity" else scores.get(bs)
    if bs_arr is not None:
        bw, _ = within_decile_auc(bs_arr[~m], y[~m], size[~m]); ba, _ = auc_ap(bs_arr[~m], y[~m])
        print(f"  best single = {best_single[0]:20s} test-half AUC={ba:.3f} within-decile={bw:.3f}")
    for nm, feats in combos.items():
        cols = [gfill if f == "_grav" else scores[f] for f in feats if (f == "_grav" or f in scores)]
        if len(cols) < 2: continue
        X = np.column_stack([z(c) for c in cols])
        lr = LogisticRegression(max_iter=300).fit(X[m], y[m])
        p = lr.predict_proba(X[~m])[:, 1]
        ea, _ = auc_ap(p, y[~m]); ew, _ = within_decile_auc(p, y[~m], size[~m])
        print(f"  {nm:28s} test-half AUC={ea:.3f} within-decile={ew:.3f}")
    np.savez(str(ROOT.parent / "data" / "processed" / "benchmark_scores.npz"),
             _y=y, _grav=grav, **{k: v for k, v in scores.items()})
    print(f"\n{'='*78}\nMULTI-METHOD FRONTIER  (seeds={seeds}, sorted by within-size-decile AUC)\n{'='*78}")
    print(f"{'method':26s} {'overall AUC':>16s} {'within-size-decile':>20s}")
    for nm, a, asd, w, wsd in table:
        print(f"{nm:26s} {a:.3f} ± {asd:.3f}     {w:.3f} ± {wsd:.3f}")


if __name__ == "__main__":
    main()
