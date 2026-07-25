"""
Gap discovery + path explanation on the trained sheep KG.

For a chosen origin country and relation, score ALL candidate country tails,
drop the ones that already exist (the filter set) -> ranked GAPS: links the model
predicts should exist but don't. Then beam-search the supporting multi-hop paths
(NBFNet model.visualize) for the top gaps and decode them into readable chains.

Headline for the live-export-ban thesis:
  - exp_live gaps        = live-sheep markets AU is structurally fit for but absent
  - exp_meat_frozen gaps = processed-meat markets that SUBSTITUTE the live channel
"""
import sys, argparse, os
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("NBFNET_PATH", str(Path(__file__).resolve().parents[1] / "third_party" / "NBFNet-PyG")))
from nbfnet.models import NBFNet
from nbfnet import tasks

ROOT = Path(__file__).resolve().parent.parent
KG = ROOT / "data" / "processed" / "kg.pt"
CKPT = ROOT / "data" / "processed" / "model.pt"


def to(data, dev):
    for k in ("edge_index", "edge_type", "target_edge_index", "target_edge_type"):
        if getattr(data, k, None) is not None:
            setattr(data, k, getattr(data, k).to(dev))
    return data


def loss_fn(pred, temp=0.5):
    pos, neg = pred[:, 0], pred[:, 1:]
    return -F.logsigmoid(pos).mean() - (
        F.softmax(neg / temp, dim=-1).detach() * F.logsigmoid(-neg)).sum(-1).mean()


def train_model(train, dev, epochs, bs=64, neg=32, lr=5e-3, num_rel=20, g1_ids=None, layers=6):
    model = NBFNet(input_dim=32, hidden_dims=[32]*layers, num_relation=num_rel,
                   message_func="distmult", aggregate_func="pna",
                   short_cut=True, layer_norm=True, dependent=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # single-task G1/G2: supervise only G1 relations (G2 stays in MP graph as background)
    if g1_ids is not None:
        mask = torch.tensor([t in g1_ids for t in train.target_edge_type.tolist()], device=dev)
        pool = mask.nonzero().squeeze(1)
    else:
        pool = torch.arange(train.target_edge_index.shape[1], device=dev)
    for ep in range(1, epochs + 1):
        model.train(); perm = pool[torch.randperm(len(pool), device=dev)]; tot = 0; nb = 0
        for s in range(0, len(pool), bs):
            idx = perm[s:s+bs]
            if len(idx) < 2: continue
            pos = torch.stack([train.target_edge_index[0, idx],
                               train.target_edge_index[1, idx],
                               train.target_edge_type[idx]], dim=-1)
            batch = tasks.negative_sampling(train, pos, neg, strict=True)
            loss = loss_fn(model(train, batch))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == epochs:
            print(f"  ep{ep:02d} loss={tot/nb:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="AUS")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--explain_stage", default=None)
    ap.add_argument("--g1", nargs="+", default=None, help="single-task G1 relation names")
    ap.add_argument("--tag", default="multi")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    blob = torch.load(KG, weights_only=False)
    meta = blob["meta"]; ents = meta["ents"]; n_country = meta["n_country"]
    NUM_FWD = meta["num_fwd"]
    rel = meta["rel"]; inv_rel = {v: k for k, v in rel.items()}
    eid = {e: i for i, e in enumerate(ents)}
    train = to(blob["train"], dev); filt = to(blob["filter"], dev)

    # known positive (h,r)->set(t) from all exp splits
    known = {}
    for sp in ("train", "valid", "test"):
        d = blob[sp]
        ti, tt, tr = d.target_edge_index[0], d.target_edge_index[1], d.target_edge_type
        for h, t, r in zip(ti.tolist(), tt.tolist(), tr.tolist()):
            known.setdefault((h, r), set()).add(t)

    ckpt = CKPT.parent / f"model_{args.tag}.pt"
    g1_ids = {rel[n] for n in args.g1} if args.g1 else None
    if g1_ids is not None:
        print(f"SINGLE-TASK G1 = {args.g1}  (everything else = G2 background)")
    if ckpt.exists() and not args.retrain:
        model = NBFNet(input_dim=32, hidden_dims=[32]*6, num_relation=2*NUM_FWD,
                       message_func="distmult", aggregate_func="pna",
                       short_cut=True, layer_norm=True, dependent=True).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev)); print("loaded checkpoint")
    else:
        print(f"training {args.epochs} epochs...")
        model = train_model(train, dev, args.epochs, num_rel=2*NUM_FWD, g1_ids=g1_ids)
        torch.save(model.state_dict(), ckpt); print(f"saved -> {ckpt}")

    def name(i): return ents[i]
    def rlabel(r): return inv_rel[r] if r < NUM_FWD else "INV_" + inv_rel[r - NUM_FWD]
    glob_imp = {int(k): set(v) for k, v in meta["global_importers"].items()}

    @torch.no_grad()
    def score_tails(h, r):
        model.eval()
        pos = torch.tensor([[h, h, r]], device=dev)
        t_batch, _ = tasks.all_negative(filt, pos)
        pred = model(filt, t_batch).squeeze(0)            # (num_nodes,)
        # candidates = real importers of this stage (kills tiny-entity noise)
        cand = torch.full((pred.numel(),), False, dtype=torch.bool, device=dev)
        cand[list(glob_imp.get(r, set()))] = True
        pred[~cand] = float("-inf"); pred[h] = float("-inf")
        return pred

    h = eid[args.origin]
    # rank stages by AUS saturation (ascending) -> least-developed first
    report_ids = g1_ids if g1_ids is not None else set(meta["supervised"])
    stages = [(rn, ri) for rn, ri in rel.items() if ri in report_ids]
    sat = {}
    for rn, ri in stages:
        have = known.get((h, ri), set()); g = glob_imp.get(ri, set())
        sat[rn] = (len(have & g) / max(len(g), 1), len(have & g), len(g))
    ordered = sorted(stages, key=lambda x: sat[x[0]][0])

    print(f"\n{'='*70}\nSATURATION-AWARE GAP DISCOVERY — origin = {args.origin}\n{'='*70}")
    print(f"{'stage':16s} {'sat':>5s} {'AUS/glob':>9s}   top undeveloped markets (score)")
    for rn, ri in ordered:
        pred = score_tails(h, ri); have = known.get((h, ri), set())
        order = pred.argsort(descending=True)
        gaps = [(i.item(), pred[i].item()) for i in order
                if i.item() not in have and pred[i].item() > float("-inf")][:args.topk]
        s, a, gl = sat[rn]
        tops = "  ".join(f"{name(t)}({sc:.2f})" for t, sc in gaps[:6])
        print(f"{rn:16s} {s:5.0%} {a:4d}/{gl:<4d}   {tops}")

    # path explanation for top gap of the least-saturated reported stage
    pick = args.explain_stage or ordered[0][0]
    r = rel[pick]; pred = score_tails(h, r); have = known.get((h, r), set())
    order = pred.argsort(descending=True)
    top_gap = next(i.item() for i in order if i.item() not in have and pred[i] > float("-inf"))
    model.num_beam = 12; model.path_topk = 6
    batch = torch.tensor([[h, top_gap, r]], device=dev)
    paths, weights = model.visualize(filt, batch)
    print(f"\n{'='*70}\nPATH EXPLANATION — why {args.origin} -[{pick}]-> {name(top_gap)}\n{'='*70}")
    for p, w in zip(paths, weights):
        chain = name(p[0][0])
        for hh, tt, rr in p:
            chain += f"  -[{rlabel(rr)}]->  {name(tt)}"
        print(f"  w={w:.4f}  {chain}")


if __name__ == "__main__":
    main()
