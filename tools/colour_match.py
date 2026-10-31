"""Colour-matched proxy embedding (README.md): does color="target" keep attribution and lower the colour shift?

Re-generates a pre-registered subset of honest LivePortrait RAVDESS test sessions (outputs/colour_match/subset.json) with the proxy embedded in the TARGET's colour
statistics (IRIS_COLOR_MODE=target, texture of the pool iris) instead of the pool iris' own colour. Same photo, driving clip, proxy, frames and scale as the existing
(donor-colour) session. Writes ONLY under outputs/colour_match/; the ledger, sessions, caches and clips of outputs/proxy_attr are read, never written.

    third_party/LivePortrait/.venv_lp/bin/python tools/colour_match.py gen [--limit N]      # GPU, one job at a time
    CUDA_VISIBLE_DEVICES='' python tools/colour_match.py extract --workers 4
    CUDA_VISIBLE_DEVICES='' python tools/colour_match.py measure --workers 4   # dE, attribution, results.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "outputs" / "colour_match"
CLIPS = OUT / "clips"
SESS = OUT / "sessions"
CACHE = OUT / "cache"
LP_PY = ROOT / "third_party" / "LivePortrait" / ".venv_lp" / "bin" / "python"
ORIG_SESS = ROOT / "outputs" / "proxy_attr" / "sessions"


def subset():
    return json.loads((OUT / "subset.json").read_text())["order"]


def orig_meta(i):
    return json.loads((ORIG_SESS / f"{i}.json").read_text())


def gen(a):
    CLIPS.mkdir(parents=True, exist_ok=True)
    SESS.mkdir(parents=True, exist_ok=True)
    done = 0
    for sid in subset():
        m = orig_meta(sid)
        meta_f = SESS / f"{sid}.json"
        if meta_f.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t0 = time.time()
        out = CLIPS / sid
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(LP_PY), str(ROOT / "tools" / "reenact_video.py"), "liveportrait", "--source", m["source"], "--frames", str(m["frames"]), "--scale", str(m["scale"]),
               "--driver", m["driver"], "--embed-entry", m["embed_entry"], "-o", str(out)]
        env = dict(os.environ, IRIS_COLOR_MODE="target")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        stem = Path(m["source"]).stem
        emb = out / f"liveportrait__{stem}__inline_{m['embed_entry']}.mp4"
        raw = out / f"liveportrait__{stem}__inline_raw.mp4"
        info_f = emb.with_suffix(".json")
        if p.returncode != 0 or not emb.exists() or not info_f.exists():
            meta_f.write_text(json.dumps(dict(id=sid, status="gen_failed", stderr_tail=p.stderr[-600:]), indent=1))
            print(sid, "generation failed", flush=True)
            continue
        info = json.loads(info_f.read_text())
        meta = dict(id=sid, status="generated", color_mode="target", video=str(emb.relative_to(ROOT)), raw_twin_new=str(raw.relative_to(ROOT)),
                    orig_video=m["video"], orig_raw_twin=m["raw_twin"], embedded_eyes=info["embedded_eyes"], gated_eyes=info["gated_eyes"],
                    embed_ms_median=info["embed_ms_median"], seconds=round(time.time() - t0, 1), cmd=" ".join(cmd) + "  [IRIS_COLOR_MODE=target]")
        meta_f.write_text(json.dumps(meta, indent=1))
        done += 1
        print(f"{sid} embedded {info['embedded_eyes']}/{info['embedded_eyes'] + info['gated_eyes']} {meta['seconds']:.0f}s", flush=True)


def new_sessions():
    out = []
    for sid in subset():
        f = SESS / f"{sid}.json"
        if f.exists():
            m = json.loads(f.read_text())
            if m.get("status") == "generated":
                out.append(m)
    return out


def _ex_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    cv2.setNumThreads(2)


def _extract_one(m):
    import proxy_verify as V
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    f = CACHE / f"{m['id']}.pkl"
    if f.exists():
        return m["id"], 0
    V.EVERY  # same module constant as the existing caches (EVERY = 2)
    t0 = time.time()
    clip = ROOT / m["video"]
    # same function and arguments as proxy_verify.cmd_extract, with the clip path redirected to the colour-matched video
    V.clip_path = lambda key, _c=clip: _c
    with FaceEyeDetector() as det:
        d = V.extract_key(m["id"], det, ArcIrisJudge())
    f.write_bytes(pickle.dumps(d))
    return m["id"], time.time() - t0


def extract(a):
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    ms = new_sessions()
    with Pool(a.workers, initializer=_ex_init) as pool:
        for sid, dt in pool.imap_unordered(_extract_one, ms):
            print("extracted", sid, f"{dt:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "extract", "measure"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"gen": gen, "extract": extract}.get(a.cmd, lambda a: None)(a)
    if a.cmd == "measure":
        import colour_match_measure as M
        M.main(a)
