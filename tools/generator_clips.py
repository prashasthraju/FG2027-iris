"""Side-by-side video clips of the generators on the SAME sessions (companion of tools/generator_report.py).

    python tools/generator_clips.py [--force] [--only SUFFIX]
        -> outputs/generator_comparison/clips/<group>.mp4 (+ <group>.jpg poster, frame 24)

Layout: 2 rows x 4 columns (LivePortrait, PersonaLive, LIA-X, EDTalk), 360x360 px panels, 28 px label bar (generator + verdict, green if correct else grey).
Row 1 = face crop of the EMBEDDED video, row 2 = zoom on the left eye of the EMBEDDED video. Panels are aligned by frame index and cut to the shortest of the four videos.
The eyes are re-detected on every frame (on the un-embedded twin, which has identical geometry; the pasted iris cannot disturb it) and the crop centre and size are smoothed
over time (exponential moving average run forward and backward, so there is no lag), a failed detection reuses the neighbouring crop.
LOCAL ONLY: the clips contain RAVDESS faces and a UTIRIS-derived iris (licensed data); outputs/ is git-ignored, never upload or publish them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import proxy_verify as V  # noqa: E402

OUT = ROOT / "outputs" / "generator_comparison"
CLIPS = OUT / "clips"
GENS = [("liveportrait", "LivePortrait", "R_"), ("personalive", "PersonaLive", "RP_"), ("liax", "LIA-X", "RL_"), ("edtalk", "EDTalk", "RE_")]
N_GROUPS = 8          # same as generator_report.py
P, BAR, CAP, FPS, POSTER = 360, 28, 20, 25, 24
ALPHA = 0.3           # EMA coefficient of the crop smoothing


def select_groups(metas, plan):
    """Same selection and order as generator_report.py (groups with all four generators)."""
    groups = []
    for r in plan["rows"]:
        if r.get("generator") != "liveportrait" or r["dataset"] != "ravdess" or r["kind"] not in ("self", "cross"):
            continue
        suf = r["id"][2:]
        ids = {g: (pre + suf) for g, _, pre in GENS}
        if all(i in metas for i in ids.values()):
            groups.append(dict(suffix=suf, ids=ids))
    return groups[:N_GROUPS]


def n_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_all(path, limit):
    cap = cv2.VideoCapture(str(path))
    while limit > 0:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        limit -= 1
    cap.release()


def ema_zero_phase(x, a=ALPHA):
    """Exponential moving average forward and backward (average of both): smooth, no lag."""
    def one(v):
        o = np.empty_like(v)
        o[0] = v[0]
        for i in range(1, len(v)):
            o[i] = a * v[i] + (1 - a) * o[i - 1]
        return o
    x = np.asarray(x, float)
    return 0.5 * (one(x) + one(x[::-1])[::-1])


def fill(track):
    """track: list of tuples or None -> array, failed frames take the previous crop (leading failures the first valid one)."""
    first = next((t for t in track if t is not None), None)
    if first is None:
        return None
    out, prev = [], first
    for t in track:
        prev = t if t is not None else prev
        out.append(prev)
    return np.array(out, float)


def detect_track(det, extract_irises, raw_path, emb_path, n):
    """Per frame (face cx, cy, half) and (eye cx, cy, half); None when the detection failed."""
    face, eye = [], []
    for raw in read_all(raw_path, n):
        segs = {s.side: s for s in extract_irises(raw, det) if s.side in ("image_left", "image_right")}
        if len(segs) == 2:
            span = abs(segs["image_right"].iris[0] - segs["image_left"].iris[0])
            fx = np.mean([segs[x].iris[0] for x in segs])
            fy = np.mean([segs[x].iris[1] for x in segs]) + 0.55 * span
            s = segs["image_left"]
            face.append((fx, fy, 1.7 * span))
            eye.append((s.iris[0], s.iris[1], 2.6 * s.iris[2]))
        else:
            face.append(None)
            eye.append(None)
    return face, eye


def smooth(track):
    a = fill(track)
    if a is None:
        return None
    return np.stack([ema_zero_phase(a[:, k]) for k in range(3)], 1)


def crop(img, cx, cy, half, out=P):
    """Sub-pixel square crop centred on (cx, cy), side 2*half, resized to out x out (pre-shrunk with INTER_AREA when the zoom is below 1)."""
    s = out / (2 * half)
    k = min(1.0, 1.5 * s)
    if k < 1.0:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    t = s / k
    M = np.array([[t, 0, -(cx * k - half * k) * t], [0, t, -(cy * k - half * k) * t]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_CUBIC if t > 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def text(img, s, org, scale, color, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel_row_tag(img, s):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 10), (0, 0, 0), -1)
    text(img, s, (6, h + 4), 0.45, (255, 255, 255))


def render_generator(path, ftrack, etrack, n):
    """-> list of (face panel, eye panel) RGB, one per frame."""
    out = []
    for i, emb in enumerate(read_all(path, n)):
        fp = crop(emb, *ftrack[i])
        ep = crop(emb, *etrack[i])
        panel_row_tag(fp, "face")
        panel_row_tag(ep, "left eye")
        out.append((fp, ep))
    return out


def label_bar(name, outcome, ok):
    bar = np.zeros((BAR, P, 3), np.uint8)
    bar[:] = (27, 143, 90) if ok else (136, 136, 136)
    s = f"{name}  {outcome}"
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    text(bar, s, ((P - w) // 2, (BAR + h) // 2), 0.62, (255, 255, 255), 2)
    return bar


def make_clip(grp, metas, rows, det, extract_irises, force):
    mp4 = CLIPS / f"{grp['suffix']}.mp4"
    if mp4.exists() and not force:
        print("skip", mp4.name)
        return
    ms = {g: metas[grp["ids"][g]] for g, _, _ in GENS}
    n = min(n_frames(ROOT / m["video"]) for m in ms.values())
    n = min(n, *[n_frames(ROOT / m["raw_twin"]) for m in ms.values()])
    cols = {}
    for g, name, _ in GENS:
        m = ms[g]
        ft, et = detect_track(det, extract_irises, ROOT / m["raw_twin"], ROOT / m["video"], n)
        nfail = sum(t is None for t in ft)
        fs, es = smooth(ft), smooth(et)
        if fs is None:
            raise RuntimeError(f"no eye detected in any frame of {m['id']}")
        panels = render_generator(ROOT / m["video"], fs, es, n)
        n = min(n, len(panels))
        cols[g] = panels
        print(f"  {name}: {len(panels)} frames, {nfail} failed detections")
    r0 = {g: rows.get((ms[g]["id"], "original")) for g in ms}
    bars = []
    for g, name, _ in GENS:
        x = r0[g]
        ok = bool(x and x["outcome"] == ("SELF" if x["kind"] == "self" else "CROSS") and x["driver"] == x["truth_driver"])
        bars.append(label_bar(name, x["outcome"] if x else "n/a", ok))
    bars = np.hstack(bars)
    W, H = 4 * P, BAR + 2 * P + CAP
    cap = np.full((CAP, W, 3), 20, np.uint8)
    text(cap, "LOCAL ONLY - licensed data", (8, 14), 0.42, (210, 210, 210))
    CLIPS.mkdir(parents=True, exist_ok=True)
    tmp = mp4.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster = None
    for i in range(n):
        top = np.hstack([cols[g][i][0] for g, _, _ in GENS])
        bot = np.hstack([cols[g][i][1] for g, _, _ in GENS])
        fr = np.ascontiguousarray(np.vstack([bars, top, bot, cap]))
        if i == min(POSTER, n - 1):
            poster = fr
        pr.stdin.write(fr.tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    tmp.replace(mp4)
    cv2.imwrite(str(CLIPS / f"{grp['suffix']}.jpg"), cv2.cvtColor(poster, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", mp4, f"{n} frames", f"{mp4.stat().st_size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-make clips that already exist")
    ap.add_argument("--only", help="only this group suffix")
    a = ap.parse_args()
    R = json.loads((ROOT / "outputs" / "proxy_attr_issuer_results.json").read_text())
    rows = {(x["id"], x["variant"]): x for x in R["rows"]}
    metas = {m["id"]: m for m in V.sessions(all_generators=True)}
    plan = json.loads(V.PLAN.read_text())
    groups = select_groups(metas, plan)
    print(len(groups), "groups")
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.segment import extract_irises
    with FaceEyeDetector() as det:
        for grp in groups:
            if a.only and grp["suffix"] != a.only:
                continue
            print(grp["suffix"])
            make_clip(grp, metas, rows, det, extract_irises, a.force)


if __name__ == "__main__":
    main()
