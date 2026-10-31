"""Measurements of tools/colour_match.py: colour shift (dE vs the raw twin) and attribution at the frozen T1, donor colour vs target colour, paired per session.
Imported by `colour_match.py measure`. Reads outputs/proxy_attr (never writes there), writes outputs/colour_match/results.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import appearance_change as AC  # noqa: E402
import colour_match as CM  # noqa: E402
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402
from provenance import stamp  # noqa: E402

OUT = CM.OUT


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


def rate(k, n):
    return dict(k=int(k), n=int(n), rate=round(k / n, 3) if n else None, wilson95=wilson(k, n))


def auc_eer(gen, imp):
    """Distances: genuine should be LOW. AUC = P(gen < imp); EER by threshold sweep (percent)."""
    gen, imp = np.asarray(gen, float), np.asarray(imp, float)
    s = np.sort(imp)
    lo = np.searchsorted(s, gen, "left")
    hi = np.searchsorted(s, gen, "right")
    auc = float(np.mean((len(s) - hi) + 0.5 * (hi - lo)) / len(s))
    best = (1.0, 0.0)
    for t in np.unique(np.concatenate([gen, imp])):
        frr, far = float((gen > t).mean()), float((imp <= t).mean())
        if abs(frr - far) < abs(best[0] - best[1]):
            best = (frr, far)
    return auc, 100 * (best[0] + best[1]) / 2


# ------------------------------------------------------------------------------------------------ colour shift
def _dE_task(args):
    """(label, m) -> AC.session_task on a session dict whose video / raw_twin are redirected."""
    label, m = args
    r = AC.session_task(m)
    r["label"] = label
    return r


def outside_task(m):
    """Lockstep check: new embedded clip vs new raw twin differ only around the eyes. Reference box = union of the changed pixels of the donor-colour clip vs its twin."""
    import cv2
    cs = [cv2.VideoCapture(str(ROOT / m[k])) for k in ("video", "raw_twin_new", "orig_video", "orig_raw_twin")]
    box, rows, i = None, [], 0
    frames = []
    while True:
        fr = [c.read() for c in cs]
        if not all(o for o, _ in fr):
            break
        frames.append([f for _, f in fr])
    for emb, raw, dem, draw in frames:
        d = np.abs(dem.astype(np.int16) - draw.astype(np.int16)).max(-1) > 0
        ys, xs = np.where(d)
        if len(ys):
            b = (xs.min(), ys.min(), xs.max(), ys.max())
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - 12, y0 - 12, x1 + 12, y1 + 12
    outs, n_out_changed, inside_changed, twin_diff = [], 0, [], []
    for emb, raw, dem, draw in frames[::4]:
        d = np.abs(emb.astype(np.int16) - raw.astype(np.int16)).max(-1)
        mask = np.ones(d.shape, bool)
        mask[y0:y1, x0:x1] = False
        outs.append(float(d[mask].mean()))
        n_out_changed += int((d[mask] > 0).sum())
        inside_changed.append(float((d[~mask] > 0).mean()))
        twin_diff.append(float(np.abs(raw.astype(np.int16) - draw.astype(np.int16)).mean()))
    return dict(id=m["id"], mean_abs_diff_outside_box=float(np.mean(outs)), n_pixels_changed_outside_box=n_out_changed,
                frac_changed_inside_box=float(np.mean(inside_changed)), new_vs_old_raw_twin_mean_abs=float(np.mean(twin_diff)), box=[int(v) for v in (x0, y0, x1, y1)])


def pair_summary(donor_rows, target_rows):
    sd, st = AC.summarise(donor_rows, "donor"), AC.summarise(target_rows, "target")
    return sd, st


def session_vals(rows):
    out = {}
    for r in rows:
        ch = [e for e in r["eyes"] if e["changed"]]
        if ch:
            out[r["id"]] = dict(dE_mean=float(np.mean([e["dE_mean"] for e in ch])), dE_pix=float(np.mean([e["dE_pix"] for e in ch])), dL=float(np.mean([e["dL"] for e in ch])),
                                n_eyes=len(ch), n_eyes_all=len(r["eyes"]))
        else:
            out[r["id"]] = None
    return out


def compact(s):
    return dict(n_sessions=s["n_sessions"], n_sessions_with_changed_eye=s["n_sessions_with_changed_eye"], n_eyes_detected=s["n_eyes_detected"], n_eyes_changed=s["n_eyes_changed"],
                share_eyes_unchanged=s["share_eyes_unchanged"], dE_mean_session=s["session_dE_mean"], dE_pix_session=s["session_dE_pix"], dL_signed=s["session_dL_signed"],
                dL_abs=s["session_dL_abs"], dh_abs=s["session_dh_abs"], share_dE_mean_gt5=s["share_sessions_dE_mean_gt5"], share_dE_mean_gt10=s["share_sessions_dE_mean_gt10"],
                share_dE_pix_gt5=s["share_sessions_dE_pix_gt5"], share_dE_pix_gt10=s["share_sessions_dE_pix_gt10"], iris_diameter_px=s["iris_diameter_px"])


# ------------------------------------------------------------------------------------------------ attribution
def attribution(G, caches, drv_idx, T1):
    """caches {id: cache}; returns per-session genuine distance, impostor distances, nearest, etc."""
    rows = {}
    for sid, c in caches.items():
        d = G.vectors(c)[0]["arciris"]
        g = d[drv_idx[sid]]
        imp = np.delete(d, drv_idx[sid])
        rows[sid] = dict(genuine=None if not np.isfinite(g) else float(g), impostors=[float(x) for x in imp if np.isfinite(x)],
                         nn_is_driver=bool(np.isfinite(g) and np.nanargmin(d) == drv_idx[sid]))
    return rows


def att_summary(rows, ids, T1):
    n = len(ids)
    gen = [rows[i]["genuine"] for i in ids if rows[i]["genuine"] is not None]
    imp = [x for i in ids if rows[i]["genuine"] is not None for x in rows[i]["impostors"]]
    acc = sum(g < T1 for g in gen)
    auc, eer = auc_eer(gen, imp) if gen and imp else (None, None)
    return dict(n_sessions=n, usable_code=rate(len(gen), n), genuine_accepted_at_T1_of_all=rate(acc, n), genuine_accepted_at_T1_of_usable=rate(acc, len(gen)),
                median_genuine_distance=round(float(np.median(gen)), 4) if gen else None, mean_genuine_distance=round(float(np.mean(gen)), 4) if gen else None,
                p90_genuine_distance=round(float(np.quantile(gen, .9)), 4) if gen else None,
                impostor_accepted_at_T1=rate(sum(x < T1 for x in imp), len(imp)), median_impostor_distance=round(float(np.median(imp)), 4) if imp else None,
                AUC_vs_other_proxies=round(auc, 4) if auc is not None else None, EER_percent_vs_other_proxies=round(eer, 3) if eer is not None else None,
                driver_proxy_is_nearest=rate(sum(rows[i]["nn_is_driver"] for i in ids), n))


def main(a):
    t0 = time.time()
    from multiprocessing import Pool
    ms = CM.new_sessions()
    ids = [m["id"] for m in ms]
    print(len(ids), "generated sessions")
    metas = {i: CM.orig_meta(i) for i in ids}
    # ---- colour shift
    tasks = []
    for m in ms:
        o = metas[m["id"]]
        tasks.append(("donor", dict(o)))                                                           # existing donor-colour clip vs its own raw twin
        tasks.append(("target", dict(o, video=m["video"], raw_twin=m["raw_twin_new"])))            # new clip vs its own (same-run) raw twin
        tasks.append(("target_vs_old_twin", dict(o, video=m["video"], raw_twin=m["orig_raw_twin"])))  # sensitivity: against the existing twin
    with Pool(a.workers, initializer=AC._init) as pool:
        res = list(pool.imap_unordered(_dE_task, tasks, chunksize=1))
        outside = list(pool.imap_unordered(outside_task, ms, chunksize=1))
    print("dE done", round(time.time() - t0), "s")
    by = {lab: [r for r in res if r["label"] == lab] for lab in ("donor", "target", "target_vs_old_twin")}
    summ = {lab: AC.summarise(rows, lab) for lab, rows in by.items()}
    vals = {lab: session_vals(rows) for lab, rows in by.items()}
    paired = [i for i in ids if vals["donor"][i] and vals["target"][i]]
    dd = np.array([vals["donor"][i]["dE_mean"] for i in paired])
    dt = np.array([vals["target"][i]["dE_mean"] for i in paired])
    ddp = np.array([vals["donor"][i]["dE_pix"] for i in paired])
    dtp = np.array([vals["target"][i]["dE_pix"] for i in paired])
    rng = np.random.default_rng(20261031)
    boot = [float(np.mean((dt - dd)[rng.integers(0, len(paired), len(paired))])) for _ in range(5000)]
    ref = json.loads((ROOT / "outputs" / "appearance_change" / "results.json").read_text())["reference_scales"]
    colour = dict(n_paired_sessions=len(paired), donor=compact(summ["donor"]), target=compact(summ["target"]), target_vs_existing_raw_twin=compact(summ["target_vs_old_twin"]),
                  paired_dE_mean_target_minus_donor=dict(mean=round(float((dt - dd).mean()), 3), bootstrap95=[round(float(np.quantile(boot, .025)), 3), round(float(np.quantile(boot, .975)), 3)],
                                                         sessions_target_lower=int((dt < dd).sum())),
                  paired_dE_pix_target_minus_donor_mean=round(float((dtp - ddp).mean()), 3),
                  reference_scales_from_appearance_change=dict(twin_vs_twin_next=ref["twin_vs_twin_next"]["ravdess"], source_vs_twin=ref["source_vs_twin"].get("liveportrait/ravdess"),
                                                               between_people=ref["between_people"]["ravdess"]),
                  per_session={i: dict(donor=vals["donor"][i], target=vals["target"][i]) for i in ids})
    lock = dict(n=len(outside), mean_abs_diff_outside_box_max_over_sessions=max(r["mean_abs_diff_outside_box"] for r in outside),
                total_pixels_changed_outside_box=sum(r["n_pixels_changed_outside_box"] for r in outside),
                new_vs_old_raw_twin_mean_abs_rgb=dict(mean=float(np.mean([r["new_vs_old_raw_twin_mean_abs"] for r in outside])), max=float(max(r["new_vs_old_raw_twin_mean_abs"] for r in outside))),
                note="box = union of the pixels the donor-colour hook changed (existing clip vs its twin) dilated 12 px; outside it the new embedded clip must equal its same-run raw twin",
                per_session=outside)
    # ---- attribution
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    G = V.Gallery(PL.Ledger.load(V.LEDGER))
    didx = {n[0]: i for i, n in enumerate(G.names)}
    drv = {i: didx[enrol[metas[i]["driver_actor"]]["did"]] for i in ids}
    T1 = json.loads((ROOT / "outputs" / "proxy_attr_calibration.json").read_text())["claim"]["arciris"]["T1"]
    c_donor = {i: V.load_cache(i) for i in ids}
    c_target = {i: __import__("pickle").loads((CM.CACHE / f"{i}.pkl").read_bytes()) for i in ids if (CM.CACHE / f"{i}.pkl").exists()}
    ids_a = [i for i in ids if i in c_target and c_donor[i] is not None]
    A_d, A_t = attribution(G, {i: c_donor[i] for i in ids_a}, drv, T1), attribution(G, {i: c_target[i] for i in ids_a}, drv, T1)
    # reproducibility: donor distances recomputed from the existing caches vs outputs/scores_liveportrait.json
    sc = {s["id"]: s for s in json.loads((ROOT / "outputs" / "scores_liveportrait.json").read_text())["sessions"]}
    mism = 0
    for i in ids_a:
        want = sc[i]["distance"][drv[i]]
        got = A_d[i]["genuine"]
        mism += int(not ((want is None and got is None) or (want is not None and got is not None and abs(want - got) < 1e-6)))
    cat = dict(both=0, only_donor=0, only_target=0, neither=0)
    for i in ids_a:
        gd = A_d[i]["genuine"] is not None and A_d[i]["genuine"] < T1
        gt = A_t[i]["genuine"] is not None and A_t[i]["genuine"] < T1
        cat["both" if gd and gt else "only_donor" if gd else "only_target" if gt else "neither"] += 1
    by_kind = {}
    for k in ("self", "cross"):
        sub = [i for i in ids_a if metas[i]["kind"] == k]
        by_kind[k] = dict(donor=att_summary(A_d, sub, T1), target=att_summary(A_t, sub, T1))
    att = dict(T1_frozen=T1, matcher="arciris (mean angular distance over observations, EVERY=2, MIN_OBS=5), genuine = driver's proxy, impostors = the other enrolled proxies",
               n_proxies=len(G.names), n_sessions=len(ids_a), donor=att_summary(A_d, ids_a, T1), target=att_summary(A_t, ids_a, T1),
               paired_acceptance=cat, by_kind=by_kind, donor_distances_match_scores_liveportrait_json=dict(mismatches=mism, of=len(ids_a)),
               per_session={i: dict(kind=metas[i]["kind"], donor=A_d[i]["genuine"], target=A_t[i]["genuine"], n_obs_donor=len(c_donor[i]["obs"]), n_obs_target=len(c_target[i]["obs"])) for i in ids_a})
    res_out = dict(note="colour matching (embed_into_eye color='target') vs donor colour, same sessions, n small", seed=20261031, n_sessions_generated=len(ids),
                   colour_shift=colour, lockstep_check=lock, attribution=att, seconds=round(time.time() - t0))
    (OUT / "results.json").write_text(json.dumps(stamp(res_out, seed=20261031), indent=1, default=float))
    print(json.dumps(dict(colour={k: colour[k] for k in ("donor", "target")}, att={k: att[k] for k in ("donor", "target", "paired_acceptance")}, lock={k: v for k, v in lock.items() if k != "per_session"}),
                     indent=1, default=float))


if __name__ == "__main__":
    pass
