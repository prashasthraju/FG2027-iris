"""Behavioural baselines on the IMITATION subset: can a behavioural verifier tell the attacker X from the imitated person D?

    python tools/behav_imitation_score.py        # -> outputs/behavioural_imitation/results.json (+ prints the tables)

Inputs: outputs/behavioural/videos.json + <model>/emb.npz (630 honest sessions), outputs/behavioural/videos_imitation.json +
outputs/behavioural_imitation/<model>/emb.npz (48 imitation sessions + their 48 original cross sessions; identical source photo and driving clip per pair).
Models: trustfa, graph, dino, clip, fusion (= tools/behavioural_on_sessions.py recipe: mean of per-model z-scores of graph, dino, clip; here the z mean/std are
those of the HONEST LivePortrait scores of the same kind (pairs for (a), reference rows for (b)), i.e. label-free and fixed before the imitation scores are seen).

(a) PAIRS. cosine(imitation video, its original) (same motion, same photo, different proxy and driver) vs honest LivePortrait pairs:
    same driver / different driver (any source, same source photo, different photo) and same driving clip with a different photo.
(b) CLAIM AS D, 'reference' protocol of behavioural_on_sessions.py: a video is scored against the honest LivePortrait sessions of every candidate driver
    (references sharing the video's source photo, and the video itself, excluded); score = mean cosine. Genuine = the claimed person.
    imitation video: claimed person D (true driver X is an impostor row). original session: claimed person = its true driver D.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
BEH, IMI = ROOT / "outputs/behavioural", ROOT / "outputs/behavioural_imitation"
SESS = ROOT / "outputs/proxy_attr/sessions"
MODELS = [("trustfa", "Inter-frame differencing"), ("graph", "Motion GCN"), ("dino", "DINOv2"), ("clip", "CLIP"), ("fusion", "Fusion (graph+dino+clip)")]
MEMBERS = ("graph", "dino", "clip")
rng = np.random.default_rng(0)


def auc_eer(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return None
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs((1 - tpr) - fpr)))
    return dict(auc=round(100 * float(roc_auc_score(y, s)), 1), eer=round(100 * float((fpr[i] + 1 - tpr[i]) / 2), 1), n_pos=int(len(pos)), n_neg=int(len(neg)))


def boot_auc(pos, neg, n=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    out = []
    for _ in range(n):
        p, q = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        out.append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    return [round(100 * float(np.percentile(out, 2.5)), 1), round(100 * float(np.percentile(out, 97.5)), 1)]


def load(path):
    d = np.load(path, allow_pickle=True)
    return {str(i): e for i, e in zip(d["ids"], np.asarray(d["emb"], np.float32))}


def main():
    H = json.loads((BEH / "videos.json").read_text())
    HM = {v["id"]: v for v in H}
    X = json.loads((BEH / "videos_imitation.json").read_text())
    XM = {v["id"]: v for v in X}
    imi = [v for v in X if v["role"] == "imitation"]
    ori = {v["id"]: v for v in X if v["role"] == "original"}
    lp = [v for v in H if v["generator"] == "liveportrait"]
    lp_ids = [v["id"] for v in lp]
    drivers = sorted({v["driver_actor"] for v in lp})
    drv_of = {v["id"]: v["driver_actor"] for v in lp}
    src_of = {v["id"]: v["source_actor"] for v in lp}
    # driving clip of every honest LivePortrait session (from its session record)
    clip = {}
    for i in lp_ids:
        f = SESS / f"{i}.json"
        clip[i] = json.loads(f.read_text()).get("driver") if f.exists() else None
    # embeddings: honest (630) and imitation list; check the 48 originals reproduce
    E_h, E_i, repro = {}, {}, {}
    for m, _ in MODELS:
        if m == "fusion":
            continue
        E_h[m], E_i[m] = load(BEH / m / "emb.npz"), load(IMI / m / "emb.npz")
        repro[m] = round(float(min(float(E_h[m][i] @ E_i[m][i]) for i in ori)), 6)
    imi_ids = [v["id"] for v in imi]
    ori_of = {v["id"]: v["orig_session"] for v in imi}
    n = len(lp_ids)

    def z_stats(arr):
        return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))

    # ---------------- (a) pair scores
    iu = np.triu_indices(n, 1)
    drv = np.array([drv_of[i] for i in lp_ids])
    src = np.array([src_of[i] for i in lp_ids])
    clp = np.array([str(clip[i]) for i in lp_ids])
    sd, ss = (drv[:, None] == drv[None, :])[iu], (src[:, None] == src[None, :])[iu]
    sc = (clp[:, None] == clp[None, :])[iu] & ((clp != "None")[:, None] & (clp != "None")[None, :])[iu]
    masks = {"same_driver_any_photo": sd, "same_driver_same_photo": sd & ss, "same_driver_diff_photo": sd & ~ss,
             "diff_driver_any_photo": ~sd, "diff_driver_same_photo": ~sd & ss, "diff_driver_diff_photo": ~sd & ~ss,
             "same_clip_diff_photo (identical motion)": sc & ~ss}
    pair_h, pair_x = {}, {}
    Mh = {m: np.stack([E_h[m][i] for i in lp_ids]) for m in E_h}
    for m in E_h:
        pair_h[m] = (Mh[m] @ Mh[m].T)[iu]
        pair_x[m] = np.array([float(E_i[m][a] @ E_i[m][ori_of[a]]) for a in imi_ids])
    st = {m: z_stats(pair_h[m]) for m in MEMBERS}
    pair_h["fusion"] = np.mean([(pair_h[m] - st[m][0]) / st[m][1] for m in MEMBERS], 0)
    pair_x["fusion"] = np.mean([(pair_x[m] - st[m][0]) / st[m][1] for m in MEMBERS], 0)
    res = {"n": dict(imitation=len(imi), originals=len(ori), honest_liveportrait=n, candidate_drivers=len(drivers)),
           "originals_reproduce_min_cos_with_honest_emb": repro, "models": {}}
    for m, label in MODELS:
        a = {}
        h, x = pair_h[m], pair_x[m]
        a["imitation_vs_original"] = dict(mean=round(float(x.mean()), 4), sd=round(float(x.std()), 4), min=round(float(x.min()), 4), max=round(float(x.max()), 4))
        a["honest"] = {k: dict(mean=round(float(h[mk].mean()), 4), sd=round(float(h[mk].std()), 4), n=int(mk.sum())) for k, mk in masks.items()}
        # operating point: EER threshold of honest same/different-driver verification (any photo, and same photo)
        for tag, sub in (("any_photo", np.ones_like(sd)), ("same_photo", ss)):
            pos, neg = h[sd & sub], h[~sd & sub]
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            s = np.r_[pos, neg]
            fpr, tpr, thr = roc_curve(y, s)
            k = int(np.argmin(np.abs((1 - tpr) - fpr)))
            a[f"imitation_pairs_accepted_as_same_driver_at_honest_EER_threshold_{tag}"] = dict(threshold=round(float(thr[k]), 4), rate=round(float((x >= thr[k]).mean()), 3),
                                                                                       honest_pairs_verification=auc_eer(pos, neg))
        # how far above the honest distributions: fraction of imitation-original pairs whose score exceeds each honest pair
        a["percentile_of_imitation_pair_in_honest_same_driver_same_photo"] = round(float(100 * np.mean([(h[masks["same_driver_same_photo"]] <= v).mean() for v in x])), 1)
        a["percentile_of_imitation_pair_in_honest_diff_driver_diff_photo"] = round(float(100 * np.mean([(h[masks["diff_driver_diff_photo"]] <= v).mean() for v in x])), 1)
        res["models"][m] = dict(label=label, a_pairs=a)

    # ---------------- (b) reference protocol
    pos_h = {i: k for k, i in enumerate(lp_ids)}
    refs_h = {D: [i for i in lp_ids if drv_of[i] == D] for D in drivers}

    def rows_for(vid_id, vemb, source):
        """score of one video vs every candidate driver: list over drivers of {model: mean cos} (nan if no reference)."""
        out = {m: [] for m in E_h}
        for D in drivers:
            refs = [r for r in refs_h[D] if src_of[r] != source and r != vid_id]
            for m in E_h:
                out[m].append(float(np.mean([vemb[m] @ E_h[m][r] for r in refs])) if refs else np.nan)
        return {m: np.array(v) for m, v in out.items()}

    # honest rows (z statistics + honest reference AUC for comparison)
    hon = {}
    for i in lp_ids:
        hon[i] = rows_for(i, {m: E_h[m][i] for m in E_h}, src_of[i])
    stats_ref = {m: z_stats(np.concatenate([hon[i][m] for i in lp_ids])) for m in MEMBERS}

    def fuse_row(r):
        return np.mean([(r[m] - stats_ref[m][0]) / stats_ref[m][1] for m in MEMBERS], 0)

    for i in lp_ids:
        hon[i]["fusion"] = fuse_row(hon[i])
    didx = {D: k for k, D in enumerate(drivers)}
    for m, label in MODELS:
        g, im_ = [], []
        for i in lp_ids:
            k = didx[drv_of[i]]
            s = hon[i][m]
            g.append(s[k])
            im_ += [x for j, x in enumerate(s) if j != k]
        res["models"][m]["b_honest_reference_protocol"] = auc_eer(g, im_)

    claims = {}
    for v in imi:
        vemb = {m: E_i[m][v["id"]] for m in E_h}
        r = rows_for(v["id"], vemb, v["source_actor"])
        r["fusion"] = fuse_row(r)
        o = ori[v["orig_session"]]
        ro = rows_for(o["id"], {m: E_i[m][o["id"]] for m in E_h}, o["source_actor"])
        ro["fusion"] = fuse_row(ro)
        claims[v["id"]] = (v, r, o, ro)
    for m, label in MODELS:
        gi, ii, go, io = [], [], [], []
        rank_D, rank_X, rank_Do, d_gt_x, n_cand = [], [], [], [], []
        for vid, (v, r, o, ro) in claims.items():
            s, so = r[m], ro[m]
            D, Xa = v["imitates"], v["driver_actor"]
            kD, kX = didx[D], didx[Xa]
            gi.append(s[kD])
            ii += [x for j, x in enumerate(s) if j != kD]
            go.append(so[kD])
            io += [x for j, x in enumerate(so) if j != kD]
            ok = np.isfinite(s)
            n_cand.append(int(ok.sum()))
            rank = lambda arr, k: int(1 + np.sum(arr[np.isfinite(arr)] > arr[k])) if np.isfinite(arr[k]) else None
            rank_D.append(rank(s, kD))
            rank_X.append(rank(s, kX))
            rank_Do.append(rank(so, kD))
            d_gt_x.append(bool(s[kD] > s[kX]) if np.isfinite(s[kD]) and np.isfinite(s[kX]) else None)
        gi, go = np.array(gi), np.array(go)
        ok = np.isfinite(gi) & np.isfinite(go)

        def rk(a):
            a = [x for x in a if x is not None]
            return dict(mean=round(float(np.mean(a)), 2), median=float(np.median(a)), top1=round(float(np.mean([x == 1 for x in a])), 3), top5=round(float(np.mean([x <= 5 for x in a])), 3), n=len(a))
        dg = [x for x in d_gt_x if x is not None]
        diffs = gi[ok] - go[ok]
        b = dict(imitation_claims_as_D=auc_eer(gi, ii), original_sessions_claim_by_D=auc_eer(go, io),
                 rank_of_D_for_imitation_video=rk(rank_D), rank_of_attacker_X_for_imitation_video=rk(rank_X), rank_of_D_for_original_video=rk(rank_Do),
                 mean_candidates_with_references=round(float(np.mean(n_cand)), 1), chance_mean_rank=round(float((np.mean(n_cand) + 1) / 2), 1),
                 score_of_D_exceeds_score_of_X=round(float(np.mean(dg)), 3),
                 imitation_vs_original_by_claim_score=dict(auc=auc_eer(gi[ok], go[ok])["auc"], auc_ci95_bootstrap=boot_auc(gi[ok], go[ok]), n=int(ok.sum()),
                                                           mean_score_imitation=round(float(gi[ok].mean()), 4), mean_score_original=round(float(go[ok].mean()), 4),
                                                           paired_mean_diff=round(float(diffs.mean()), 4), paired_wilcoxon_p=round(float(wilcoxon(diffs).pvalue), 3)))
        res["models"][m]["b_claim_as_D"] = b
    # ---------------- driving-clip reuse: does D own a reference session driven by the very same clip (identical motion, other photo)?
    flag = {}
    for vid, (v, r, o, ro) in claims.items():
        c = json.loads((SESS / f"{vid}.json").read_text())["driver"]
        flag[vid] = any(clip[x] == c and src_of[x] != v["source_actor"] for x in refs_h[v["imitates"]])
    res["imitations_where_D_has_a_same_clip_reference_session"] = int(sum(flag.values()))
    for m, _ in MODELS:
        out = {}
        for tag, want in (("D_has_same_clip_reference", True), ("D_has_no_same_clip_reference", False)):
            ks = [vid for vid in claims if flag[vid] == want]
            rk_ = [1 + int(np.sum(claims[vid][1][m][np.isfinite(claims[vid][1][m])] > claims[vid][1][m][didx[claims[vid][0]["imitates"]]])) for vid in ks]
            out[tag] = dict(n=len(ks), mean_rank_of_D=round(float(np.mean(rk_)), 2) if rk_ else None, top1=round(float(np.mean([x == 1 for x in rk_])), 3) if rk_ else None)
        res["models"][m]["b_claim_as_D"]["by_same_clip_reference"] = out
    # ---------------- record-check contrast (from the main session's numbers)
    pn = json.loads((ROOT / "outputs/proxy_attr_paper_numbers.json").read_text())["imitation"]
    res["our_protocol_for_contrast"] = dict(source="outputs/proxy_attr_paper_numbers.json['imitation']", n=pn["n"],
                                            imitated_person_claims_outcomes=pn["imitated_person_claim_outcomes"], imitated_person_accepted=pn["imitated_person_accepted"],
                                            attacker_attributed_correctly=pn["attacker_attributed"], original_sessions_outcomes=pn["original_sessions_outcomes"])
    (IMI / "results.json").write_text(json.dumps(res, indent=1))
    # ---------------- print
    print("min cos of re-embedded originals with the honest embeddings:", repro)
    print("\n(a) mean cosine  | imit-vs-orig | same-drv same-photo | diff-drv same-photo | same-drv diff-photo | diff-drv diff-photo | same-clip diff-photo")
    for m, label in MODELS:
        a = res["models"][m]["a_pairs"]
        h = a["honest"]
        print(f"{label:28s} {a['imitation_vs_original']['mean']:.3f} | {h['same_driver_same_photo']['mean']:.3f} | {h['diff_driver_same_photo']['mean']:.3f} | {h['same_driver_diff_photo']['mean']:.3f} | {h['diff_driver_diff_photo']['mean']:.3f} | {h['same_clip_diff_photo (identical motion)']['mean']:.3f}  "
              f"accepted@EER any/same-photo: {a['imitation_pairs_accepted_as_same_driver_at_honest_EER_threshold_any_photo']['rate']}/{a['imitation_pairs_accepted_as_same_driver_at_honest_EER_threshold_same_photo']['rate']}")
    print("\n(b) model | honest ref AUC/EER | imit-as-D AUC/EER | orig AUC/EER | rank D mean (chance) top1 | rank X top1 | s(D)>s(X) | imit-vs-orig AUC [CI] p")
    for m, label in MODELS:
        r = res["models"][m]
        b = r["b_claim_as_D"]
        f = lambda t: "%.1f/%.1f" % (t["auc"], t["eer"])
        print(f"{label:28s} {f(r['b_honest_reference_protocol'])} | {f(b['imitation_claims_as_D'])} | {f(b['original_sessions_claim_by_D'])} | {b['rank_of_D_for_imitation_video']['mean']} ({b['chance_mean_rank']}) {b['rank_of_D_for_imitation_video']['top1']} | "
              f"{b['rank_of_attacker_X_for_imitation_video']['top1']} | {b['score_of_D_exceeds_score_of_X']} | {b['imitation_vs_original_by_claim_score']['auc']} {b['imitation_vs_original_by_claim_score']['auc_ci95_bootstrap']} p={b['imitation_vs_original_by_claim_score']['paired_wilcoxon_p']}")
    print("same-clip reference of D exists for", res["imitations_where_D_has_a_same_clip_reference_session"], "of 48 imitation videos")
    for m, _ in MODELS:
        print(m, res["models"][m]["b_claim_as_D"]["by_same_clip_reference"])
    print(res["our_protocol_for_contrast"])


if __name__ == "__main__":
    main()
