"""Behavioural driver-verification baselines on OUR sessions, per generator, next to the proxy-iris distance.

    python tools/behavioural_on_sessions.py     # -> outputs/behavioural_on_sessions.json, paper_gen/tab_behav.tex

Embeddings (one per released video, L2-normalised) come from outputs/behavioural/<model>/emb.npz (tools/behav_*.py): trustfa = inter-frame feature differencing,
graph = motion GCN, dino, clip; fusion follows outputs/behavioural/fusion_spec.json. Pair protocol, within one generator: every pair of sessions; genuine = same driver,
impostor = different driver; three subsets: any source photo, different source photos, same source photo (appearance matched, the hard case).
The proxy-iris row scores the same sessions as a verification of the true driver against the other enrolled proxies (genuine = the driver's proxy, impostor = every other proxy).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
BEH = ROOT / "outputs" / "behavioural"
MODELS = [("trustfa", "Inter-frame differencing"), ("graph", "Motion GCN"), ("dino", "DINOv2"), ("clip", "CLIP"), ("fusion", "Fusion")]
GENS = [("liveportrait", "LivePortrait"), ("personalive", "PersonaLive"), ("liax", "LIA-X"), ("edtalk", "EDTalk")]


def roc(gen, imp):
    from sklearn.metrics import roc_auc_score, roc_curve
    gen, imp = np.asarray(gen, float), np.asarray(imp, float)
    gen, imp = gen[np.isfinite(gen)], imp[np.isfinite(imp)]
    if len(gen) < 2 or len(imp) < 2:
        return None
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    s = np.r_[gen, imp]
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs((1 - tpr) - fpr)))
    return dict(auc=round(100 * float(roc_auc_score(y, s)), 1), eer=round(100 * float((fpr[i] + 1 - tpr[i]) / 2), 1), tar1=round(100 * float(np.interp(1e-2, fpr, tpr)), 1),
                n_genuine=int(len(gen)), n_impostor=int(len(imp)))


def pair_scores(S, meta, idx):
    """S: similarity matrix over the videos of one generator; returns {subset: (genuine, impostor)}."""
    drv = np.array([meta[i]["driver_actor"] for i in idx])
    src = np.array([meta[i]["source_actor"] for i in idx])
    iu = np.triu_indices(len(idx), 1)
    s = S[iu]
    same_d = (drv[:, None] == drv[None, :])[iu]
    same_s = (src[:, None] == src[None, :])[iu]
    out = {}
    for name, mask in (("any_source", np.ones_like(s, bool)), ("different_source", ~same_s), ("same_source", same_s)):
        out[name] = (s[mask & same_d], s[mask & ~same_d])
    return out


def main():
    vids = json.loads((BEH / "videos.json").read_text())
    meta = {v["id"]: v for v in vids}
    sims, embs = {}, {}
    for key, _ in MODELS:
        f = BEH / key / "emb.npz"
        if key != "fusion" and f.exists():
            d = np.load(f, allow_pickle=True)
            ids = [str(x) for x in d["ids"]]
            embs[key] = (ids, np.asarray(d["emb"], np.float32))
    ids_all = [v["id"] for v in vids]
    pos_all = {i: k for k, i in enumerate(ids_all)}
    S_full = {}
    for key, (ids, E) in embs.items():
        X = E[[ {i: k for k, i in enumerate(ids)}[i] for i in ids_all]]
        S_full[key] = X @ X.T
    members = [k for k in ("graph", "dino", "clip") if k in S_full]
    res = {"protocols": {"pairs": "pairs of sessions within one generator; genuine = same driver (subsets: any / different / same source photo)",
                         "reference": "a session is scored against the LivePortrait sessions of every candidate driver (references that share the session's source photo are excluded); "
                                      "genuine = the true driver, impostor = every other driver; same structure as the proxy-iris verification"},
           "models": {}}
    for key, label in MODELS:
        res["models"][key] = dict(label=label, per_generator={})

    def fuse(arrs):
        zs = [(a - np.nanmean(a)) / np.nanstd(a, ddof=1) for a in arrs]
        return np.nanmean(np.vstack(zs), axis=0)
    lp = [v for v in vids if v["generator"] == "liveportrait"]
    drivers = sorted({v["driver_actor"] for v in lp})
    for g, gname in GENS:
        idx = [v["id"] for v in vids if v["generator"] == g]
        sel = [pos_all[i] for i in idx]
        # ---- protocol 1: pairs within the generator
        iu = np.triu_indices(len(idx), 1)
        sub = {k: S_full[k][np.ix_(sel, sel)] for k in S_full}
        if members:
            sub["fusion"] = None
            vecs = [sub[k][iu] for k in members]
            fused = fuse(vecs)
            M = np.zeros((len(idx), len(idx)))
            M[iu] = fused
            sub["fusion"] = M
        for key in sub:
            res["models"][key]["per_generator"].setdefault(g, {})
            res["models"][key]["per_generator"][g].update({name: roc(*gi) for name, gi in pair_scores(sub[key], meta, idx).items()})
        # ---- protocol 2: against LivePortrait references
        rows_g = []
        for i in idx:
            v = meta[i]
            for D in drivers:
                refs = [pos_all[r["id"]] for r in lp if r["driver_actor"] == D and r["source_actor"] != v["source_actor"] and r["id"] != i]
                rows_g.append((pos_all[i], D == v["driver_actor"], refs))
        score = {}
        for k in S_full:
            score[k] = np.array([np.mean(S_full[k][a, refs]) if refs else np.nan for a, _, refs in rows_g])
        if members:
            score["fusion"] = fuse([score[k] for k in members])
        lab = np.array([r[1] for r in rows_g])
        for k, sc in score.items():
            res["models"][k]["per_generator"].setdefault(g, {})["reference"] = roc(sc[lab], sc[~lab])
    # ---- the proxy-iris distance on the same sessions
    try:
        import proxy_ledger as PL
        import proxy_verify as V
        plan = json.loads(V.PLAN.read_text())
        enrol = plan["enrolment"]
        G = V.Gallery(PL.Ledger.load(V.LEDGER))
        didx = {n[0]: i for i, n in enumerate(G.names)}
        ours = {}
        for g, _ in GENS:
            gen_, imp_ = [], []
            for v in vids:
                if v["generator"] != g:
                    continue
                c = V.load_cache(v["id"])
                if c is None:
                    continue
                d = G.vectors(c)[0]["arciris"]
                k = didx[enrol[v["driver_actor"]]["did"]]
                gen_.append(-d[k])
                imp_ += [-x for x in np.delete(d, k) if np.isfinite(x)]
            ours[g] = {"proxy_distance": roc(gen_, imp_)}
        res["models"]["proxy_iris"] = dict(label="Proxy-iris distance (ours, ArcIris)", per_generator=ours)
    except Exception as e:                                  # the embeddings are the point of this script
        print("proxy-iris row skipped:", e)
    out = ROOT / "outputs" / "behavioural_on_sessions.json"
    out.write_text(json.dumps(res, indent=1))
    # ---- table for the paper: AUC / EER on 'any source' and 'same source' per generator
    lines = []
    for key, m in res["models"].items():
        def cell(g, sub):
            x = m["per_generator"].get(g, {})
            a = x.get(sub) or (x.get("proxy_distance") if sub == "reference" else None)
            return ("%.1f / %.1f" % (a["auc"], a["eer"])) if a else "--"
        lines.append(m["label"] + " & " + " & ".join([cell("liveportrait", "reference"), cell("liveportrait", "same_source"), cell("personalive", "reference"), cell("liax", "reference"), cell("edtalk", "reference")]) + r" \\")
    (ROOT / "paper_gen").mkdir(exist_ok=True)
    (ROOT / "paper_gen" / "tab_behav.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    print(json.dumps({k: {g: (v.get("reference") or v.get("proxy_distance")) for g, v in m["per_generator"].items()} for k, m in res["models"].items()}, indent=1)[:3500])


if __name__ == "__main__":
    main()
