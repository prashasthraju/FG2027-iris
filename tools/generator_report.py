"""Side-by-side report of the generators on the SAME sessions (same photo, driving clip and proxy): LivePortrait (reference), PersonaLive, LIA-X, EDTalk.

    python tools/generator_report.py        # -> outputs/generator_comparison/report.html (self-contained, LOCAL ONLY)

Reads the paired sessions of batch 4 (tools/proxy_batch4.py --extra), the evaluation (outputs/proxy_attr_issuer_results.json) and the extraction caches.
The report contains RAVDESS faces and a UTIRIS-derived iris: licensed data, never upload or publish it (outputs/ is git-ignored).
"""
import base64
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402

OUT = ROOT / "outputs" / "generator_comparison"
GENS = [("liveportrait", "LivePortrait", "R_", "reference"), ("personalive", "PersonaLive", "RP_", "diffusion"), ("liax", "LIA-X", "RL_", "autoencoder"),
        ("edtalk", "EDTalk", "RE_", "GAN, 256 px")]
NATIVE = {"liveportrait": ("256 px crop, pasted back on a 3x canvas", "about 27 px (x3 canvas)"), "personalive": ("512 px whole frame", "about 24 px"),
          "liax": ("512 px", "about 26 px"), "edtalk": ("256 px face pasted onto the photo", "about 12 px")}
N_GROUPS = 8


def jpg64(img, q=88):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def frame(path, i):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, n - 1))
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None


def crop(img, cx, cy, half, out):
    x0, y0, n = int(round(cx - half)), int(round(cy - half)), int(round(2 * half))
    pad = max(0, -x0, -y0, x0 + n - img.shape[1], y0 + n - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return cv2.resize(img[y0:y0 + n, x0:x0 + n], (out, out), interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads((ROOT / "outputs" / "proxy_attr_issuer_results.json").read_text())
    rows = {(x["id"], x["variant"]): x for x in R["rows"]}
    metas = {m["id"]: m for m in V.sessions(all_generators=True)}
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(V.LEDGER)
    G = V.Gallery(led)
    didx = {n[0]: i for i, n in enumerate(G.names)}
    T1 = json.loads(V.CALIB.read_text())["claim"]["arciris"]["T1"]
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.segment import extract_irises

    def distances(sid):
        c = V.load_cache(sid)
        if c is None:
            return None
        m = metas[sid]
        d = G.vectors(c)[0]["arciris"]
        dd, ds = d[didx[enrol[m["driver_actor"]]["did"]]], d[didx[enrol[m["source_actor"]]["did"]]]
        return dict(driver=float(dd), source=float(ds), nearest_other=float(np.nanmin(np.delete(d, didx[enrol[m["driver_actor"]]["did"]]))))
    # ---- groups with all four generators
    groups = []
    for r in plan["rows"]:
        if r.get("generator") != "liveportrait" or r["dataset"] != "ravdess" or r["kind"] not in ("self", "cross"):
            continue
        suf = r["id"][2:]
        ids = {g: (pre + suf) for g, _, pre, _ in GENS}
        if all(i in metas for i in ids.values()):
            groups.append(dict(suffix=suf, ids=ids))
    print(len(groups), "groups with all four generators")
    # ---- per generator statistics over the paired groups
    stats = {}
    for g, name, pre, _ in GENS:
        ids = [x["ids"][g] for x in groups]
        rr = [rows.get((i, "original")) for i in ids]
        tw = [rows.get((i, "raw_twin")) for i in ids]
        ok = lambda x: x and x["outcome"] == ("SELF" if x["kind"] == "self" else "CROSS") and x["driver"] == x["truth_driver"]
        dist = [distances(i) for i in ids]
        dd = [d["driver"] for d in dist if d]
        sec = [metas[i]["seconds"] / max(metas[i]["frames"], 1) for i in ids]
        stats[g] = dict(n=len(ids), correct=sum(bool(ok(x)) for x in rr), wrong=sum(1 for x in rr if x and x["outcome"] in ("SELF", "CROSS") and not ok(x)),
                        refused=sum(1 for x in rr if x and x["outcome"] == "INCONCLUSIVE"), twin=sum(1 for x in tw if x and x["outcome"] in ("SELF", "CROSS")),
                        n_twin=sum(1 for x in tw if x), d_mean=float(np.mean(dd)), d_sd=float(np.std(dd)), under=sum(1 for x in dd if x < T1), n_d=len(dd),
                        hook_ms=float(np.median([metas[i]["embed_ms_median"] for i in ids])), s_per_frame=float(np.median(sec)),
                        gated=sum(metas[i]["gated_eyes"] for i in ids), embedded=sum(metas[i]["embedded_eyes"] for i in ids))
    # ---- example cards
    cards = []
    with FaceEyeDetector() as det:
        for grp in groups[:N_GROUPS]:
            first = metas[grp["ids"]["liveportrait"]]
            card = dict(suffix=grp["suffix"], source_actor=first["source_actor"], driver_actor=first["driver_actor"], kind=first["kind"], cells=[])
            card["source"] = jpg64(cv2.resize(cv2.cvtColor(cv2.imread(first["source"]), cv2.COLOR_BGR2RGB), (240, 240), interpolation=cv2.INTER_AREA))
            drv = frame(first["driver"], 24)
            card["driver"] = jpg64(cv2.resize(drv, (240, 240), interpolation=cv2.INTER_AREA)) if drv is not None else ""
            for g, name, pre, kind in GENS:
                sid = grp["ids"][g]
                m = metas[sid]
                raw, emb = frame(ROOT / m["raw_twin"], 24), frame(ROOT / m["video"], 24)
                segs = {s.side: s for s in extract_irises(raw, det) if s.side in ("image_left", "image_right")}
                cell = dict(gen=g, name=name, id=sid)
                if len(segs) == 2 and raw is not None:
                    span = abs(segs["image_right"].iris[0] - segs["image_left"].iris[0])
                    fx = np.mean([segs[x].iris[0] for x in segs])
                    fy = np.mean([segs[x].iris[1] for x in segs]) + 0.55 * span
                    cell["face_raw"], cell["face_emb"] = jpg64(crop(raw, fx, fy, 1.7 * span, 260)), jpg64(crop(emb, fx, fy, 1.7 * span, 260))
                    s = segs["image_left"]
                    cell["eye_raw"], cell["eye_emb"] = jpg64(crop(raw, s.iris[0], s.iris[1], 2.6 * s.iris[2], 200)), jpg64(crop(emb, s.iris[0], s.iris[1], 2.6 * s.iris[2], 200))
                    cell["iris_px"] = float(np.mean([2 * segs[x].iris[2] for x in segs]))
                cell["frame_px"] = f"{emb.shape[1]}x{emb.shape[0]}" if emb is not None else ""
                r0 = rows.get((sid, "original"))
                cell["outcome"] = r0["outcome"] if r0 else "n/a"
                cell["ok"] = bool(r0 and r0["outcome"] == ("SELF" if r0["kind"] == "self" else "CROSS") and r0["driver"] == r0["truth_driver"])
                cell["dist"] = distances(sid)
                tw = rows.get((sid, "raw_twin"))
                cell["twin"] = tw["outcome"] if tw else "n/a"
                card["cells"].append(cell)
            cards.append(card)
    css = """
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1e2229}
.wrap{max-width:1500px;margin:0 auto;padding:24px 28px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px} .sub{color:#5a616b;font-size:13px;margin-bottom:14px}
table{border-collapse:collapse;background:#fff;border:1px solid #d9dde3;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid #e6e9ee;text-align:right} th:first-child,td:first-child{text-align:left} th{background:#eef1f5}
.card{background:#fff;border:1px solid #d9dde3;border-radius:10px;padding:14px;margin:14px 0}
.hdr{font-weight:600;font-size:14px;margin-bottom:8px}.grid{display:grid;grid-template-columns:200px repeat(4,1fr);gap:12px;align-items:start}
.cell{border:1px solid #e1e5ea;border-radius:8px;padding:8px;background:#fafbfc}.cell h4{margin:0 0 6px;font-size:13px}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:5px}.imgs img{width:100%;border-radius:4px;border:1px solid #cfd5dc}
.lab{font-size:10px;color:#5a616b;text-align:center;margin-top:1px}
.chip{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:12px;font-weight:600}.good{background:#1b8f5a}.bad{background:#c0392b}.grey{background:#888}
.small{font-size:11px;color:#5a616b;margin-top:5px;line-height:1.45} .bar{height:8px;border-radius:4px;background:#0072b2;margin:2px 0}
.note{background:#fff7e0;border:1px solid #e8d48a;border-radius:8px;padding:9px 12px;font-size:12.5px;margin:10px 0}
.left img{width:100%;border-radius:6px;border:1px solid #cfd5dc;margin-bottom:6px}
"""
    h = [f"<!doctype html><meta charset='utf-8'><title>Generator comparison</title><style>{css}</style><div class='wrap'>",
         "<h1>The same sessions on four generators</h1>",
         f"<div class='sub'>{len(groups)} RAVDESS sessions rendered by LivePortrait, PersonaLive, LIA-X and EDTalk with the same source photo, the same driving clip and the same proxy iris; "
         f"verdicts are those of the full verifier at the frozen threshold T<sub>1</sub> = {T1:.4f}. LOCAL ONLY: the images are licensed data (RAVDESS faces, UTIRIS-derived iris).</div>",
         "<div class='sub'>Clips: each example card ends with a video (tools/generator_clips.py, files in clips/ next to this page) that plays the four embedded videos side by side, "
         "face row on top and left-eye row below, aligned frame by frame (25 fps, cut to the shortest of the four); verdict labels as in the chips.</div>",
         "<div class='note'>Small subsets: read the counts, not the percentages. The iris is written by the same hook into frames upscaled to a 60 px iris or more; the low-resolution generators "
         "(EDTalk 256 px, PersonaLive and LIA-X 512 px) have a much softer native eye than the pasted iris.</div>",
         "<h2>Summary over the paired sessions</h2><table><tr><th>Generator</th><th>Native output</th><th>Native iris</th><th>Sessions</th><th>Correct</th><th>Wrong</th><th>Refused</th>"
         "<th>Un-embedded copy attributed</th><th>Distance to driver proxy (mean &plusmn; sd)</th><th>Under T<sub>1</sub></th><th>Hook ms/frame</th><th>Session s/frame</th><th>Eyes gated</th></tr>"]
    for g, name, pre, kind in GENS:
        s = stats[g]
        h.append(f"<tr><td><b>{name}</b> <span class='sub'>({kind})</span></td><td>{NATIVE[g][0]}</td><td>{NATIVE[g][1]}</td><td>{s['n']}</td><td>{s['correct']}/{s['n']}</td><td>{s['wrong']}</td>"
                 f"<td>{s['refused']}</td><td>{s['twin']}/{s['n_twin']}</td><td>{s['d_mean']:.3f} &plusmn; {s['d_sd']:.3f}</td><td>{s['under']}/{s['n_d']}</td><td>{s['hook_ms']:.0f}</td>"
                 f"<td>{s['s_per_frame']:.2f}</td><td>{s['gated']}/{s['gated'] + s['embedded']}</td></tr>")
    h.append("</table><div class='sub'>Timings were measured with several jobs running in parallel on one shared 8 GB GPU and are only indicative.</div>")
    bj = ROOT / "outputs" / "behavioural_on_sessions.json"
    if bj.exists():
        B = json.loads(bj.read_text())
        h.append("<h2>Behavioural baselines on the same sessions (AUC / EER, %)</h2><div class='sub'>Each session is scored against the LivePortrait sessions of every candidate driver "
                 "(references with the same photo excluded). The behavioural models were not retrained and are out of domain for these short clips; the proxy-iris row uses the distance to the true driver's proxy.</div>"
                 "<table><tr><th>Method</th><th>LivePortrait</th><th>LivePortrait, same photo</th><th>PersonaLive</th><th>LIA-X</th><th>EDTalk</th></tr>")
        for key, m in B["models"].items():
            def cell(g, sub):
                x = m["per_generator"].get(g, {})
                r_ = x.get(sub) or (x.get("proxy_distance") if sub == "reference" else None)
                return ("%.1f / %.1f" % (r_["auc"], r_["eer"])) if r_ else "--"
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (m["label"], cell("liveportrait", "reference"), cell("liveportrait", "same_source"),
                                                                                                cell("personalive", "reference"), cell("liax", "reference"), cell("edtalk", "reference")))
        h.append("</table>")
    h.append("<h2>Examples (frame 24 of each clip; left eye)</h2>")
    for c in cards:
        h.append(f"<div class='card'><div class='hdr'>Photo {c['source_actor'].replace('_', ' ')} &middot; driver {c['driver_actor'].replace('_', ' ')} &middot; {c['kind']} &middot; clip {html.escape(c['suffix'].split('_')[-1])}</div><div class='grid'>")
        h.append(f"<div class='left'><img src='{c['source']}'><div class='lab'>source photo</div><img src='{c['driver']}'><div class='lab'>driving frame</div></div>")
        for cell in c["cells"]:
            d = cell["dist"]
            chip = f"<span class='chip {'good' if cell['ok'] else 'grey'}'>{cell['outcome']}</span>"
            h.append(f"<div class='cell'><h4>{cell['name']} {chip}</h4>")
            if "face_raw" in cell:
                h.append(f"<div class='imgs'><div><img src='{cell['face_raw']}'><div class='lab'>no hook</div></div><div><img src='{cell['face_emb']}'><div class='lab'>with proxy</div></div>"
                         f"<div><img src='{cell['eye_raw']}'><div class='lab'>eye before</div></div><div><img src='{cell['eye_emb']}'><div class='lab'>eye after</div></div></div>")
            h.append("<div class='small'>")
            if d:
                w = max(5, min(100, 100 * (1 - d['driver'] / 1.7)))
                h.append(f"distance to driver proxy <b>{d['driver']:.3f}</b> (T<sub>1</sub> {T1:.3f}) &middot; to source proxy {d['source']:.3f}<div class='bar' style='width:{w:.0f}%'></div>")
            h.append(f"frame {cell['frame_px']} &middot; iris in output {cell.get('iris_px', float('nan')):.0f} px &middot; un-embedded copy: {cell['twin']}</div></div>")
        h.append("</div>")
        h.append(f"<video controls loop muted playsinline preload='metadata' poster='clips/{html.escape(c['suffix'])}.jpg' src='clips/{html.escape(c['suffix'])}.mp4' style='width:100%;margin-top:12px;border-radius:6px;background:#000'></video>")
        h.append("</div>")
    h.append("</div>")
    (OUT / "report.html").write_text("\n".join(h))
    print("wrote", OUT / "report.html", {g: (s["correct"], s["n"]) for g, s in stats.items()})


if __name__ == "__main__":
    main()
