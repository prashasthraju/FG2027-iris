"""Carrier study: a robust bit watermark (VideoSeal 1.0, 256 bit) versus the proxy iris, same identifier, same generation / compression / resizing conditions.

A reviewer asked for a DIRECT comparison against a robust bit watermark that carries the same identifier. Identifier = the SESSION ID (the signed record on the
ledger is found by it). Two carriers on the same frames of a stratified subset of the released sessions:

  watermark  tag = first 32 bits of SHA-256(session_id) repeated 8 times -> 256 payload bits -> VideoSeal embeds it in the RAW twin (the same generated frames WITHOUT
             the iris), written x264 qp 0 yuv420p (the encoder of the released clips). Decode = frame-mean of the 256 bit logits, sign of the mean over the 8 repeats of
             each tag bit, EXACT lookup of the 32-bit tag among the tags of the subset's session ids. Null: the un-watermarked raw twin through the same conditions.
  iris       the released embedded video (proxy iris, no watermark). ArcIris observations (EVERY=2, as everywhere), mean angular distance to the driver's own proxy
             < T1 (claim.arciris.T1 of outputs/proxy_attr_calibration.json) = genuine acceptance; EER against the other enrolled proxies.
  combined   (optional) VideoSeal on top of the released iris video; clean and CRF 23: does the watermark harm the iris match.

Conditions, byte-identical for all arms (ffmpeg libx264, yuv420p, -an; clean = no re-encode; CRF 23/30/blur are the strings of proxy_verify.FFMPEG):
  crf23 / crf30 : -c:v libx264 -crf N -preset medium
  blur          : -vf gblur=sigma=14, crf 18
  resize05      : -vf scale=trunc(iw/4)*2:trunc(ih/4)*2 (0.5x, even sizes), crf 18
  crop08        : -vf crop=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2,scale=W:H (centre 80 % cut, zoomed back to the original size), crf 18

Commands (WORK = outputs/watermark_study; everything resumable per session):
  python tools/wm_study.py plan                              # subset.json (seeded)
  cd external/Watemark/baseline && .venv_vs/bin/python tools/wm_study.py wm        # GPU, <= 2.5 GB: embed + decode, 6 conditions, null
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python tools/wm_study.py iris --shard K --of N                    # iris-only arm (CPU workers)
  cd .../baseline && .venv_vs/bin/python .../wm_study.py quality --shard K --of N                                            # PSNR/SSIM/LPIPS (CPU)
  cd .../baseline && .venv_vs/bin/python .../wm_study.py comb                                                                # combined arm, embed + decode
  CUDA_VISIBLE_DEVICES= python tools/wm_study.py iris --arm comb --shard K --of N                            # combined arm, iris side
  python tools/wm_study.py summary                           # -> outputs/watermark_study/results.json
The script only imports numpy at top level so that it runs in both venvs. Licensed biometric clips: everything stays under outputs/.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "outputs" / "watermark_study"
SUBSET = WORK / "subset.json"
LOGS = WORK / "logs"
CACHE = WORK / "cache"
TMP = WORK / "tmp"
WMVID = WORK / "videos"             # watermarked clean files (qp 0) and the combined-arm files, kept until the end
SEED = 20261030
REP, TAGBITS, NB = 8, 32, 256
CONDS = ["clean", "crf23", "crf30", "blur", "resize05", "crop08"]
COMB_CONDS = ["clean", "crf23"]
STRATA = [("liveportrait", "ravdess", 35), ("liveportrait", "nir", 35), ("personalive", "ravdess", 16), ("liax", "ravdess", 16), ("edtalk", "ravdess", 18)]
FF_THREADS = "4"
P_CRF = lambda q: ["-c:v", "libx264", "-crf", str(q), "-preset", "medium", "-pix_fmt", "yuv420p", "-an", "-threads", FF_THREADS]
P_Q18 = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-threads", FF_THREADS]
EVERY = 2                           # as tools/proxy_verify.py


def ffargs(cond, w, h):
    return {"crf23": P_CRF(23), "crf30": P_CRF(30),
            "blur": ["-vf", "gblur=sigma=14", *P_Q18],
            "resize05": ["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", *P_Q18],
            "crop08": ["-vf", f"crop=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2,scale={w}:{h}", *P_Q18]}[cond]


# ------------------------------------------------------------------------------------------------ small helpers
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s["nb_read_frames"]), float(a) / float(b)


def probe_fast(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s["nb_frames"]), float(a) / float(b)


def make_condition(src, dst, cond):
    """Re-encode src -> dst with the condition's ffmpeg string; the frame count must be unchanged."""
    w, h, n, _ = probe_fast(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *ffargs(cond, w, h), str(dst)], check=True)
    n2 = probe_fast(dst)[2]
    assert n2 == n, f"{dst.name}: {n2} frames after {cond}, {n} before"


def tag_bits(sid):
    h = hashlib.sha256(sid.encode()).digest()[:4]
    return [int(c) for c in "".join(f"{x:08b}" for x in h)]


def tag_hex(sid):
    return hashlib.sha256(sid.encode()).digest()[:4].hex()


def payload(sid):
    t = tag_bits(sid)
    return [t[j % TAGBITS] for j in range(REP * TAGBITS)]


def decode_logits(logits):
    L = np.asarray(logits, float).reshape(REP, TAGBITS)
    m = L.mean(0)
    return [int(x > 0) for x in m]


def bits_to_hex(bits):
    return f"{int(''.join(map(str, bits)), 2):08x}"


def load_subset():
    """Rows in a stratified round-robin order (position in stratum / stratum size), so that any prefix of the run is itself a stratified sample."""
    sub = json.loads(SUBSET.read_text())
    cnt, pos = {}, {}
    for r in sub["rows"]:
        cnt[r["stratum"]] = cnt.get(r["stratum"], 0) + 1
    keyed = []
    for r in sorted(sub["rows"], key=lambda r: zlib.crc32(r["id"].encode())):          # pseudo-random but fixed order inside a stratum
        k = pos.get(r["stratum"], 0)
        pos[r["stratum"]] = k + 1
        keyed.append(((k + 0.5) / cnt[r["stratum"]], r["id"], r))
    sub["rows"] = [x[2] for x in sorted(keyed, key=lambda t: (t[0], t[1]))]
    return sub


def frame_iter(path):
    w, h, _, _ = probe_fast(path)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.stdout.close()
    p.wait()


def blocks(it, n):
    buf = []
    for f in it:
        buf.append(f)
        if len(buf) == n:
            yield np.stack(buf)
            buf = []
    if buf:
        yield np.stack(buf)


# ------------------------------------------------------------------------------------------------ plan (chain venv)
def cmd_plan(a):
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import proxy_verify as V
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    S = [m for m in V.sessions(all_generators=True) if m["kind"] in ("self", "cross") and not m["tuning"]]
    rng = np.random.default_rng(SEED)
    chosen = []
    for gen, ds, n in STRATA:
        pool = [m for m in S if m.get("generator", "liveportrait") == gen and m["dataset"] == ds]
        pool.sort(key=lambda m: m["id"])
        idx = rng.permutation(len(pool))[:n]
        chosen += [dict(pool[i], stratum=f"{gen}/{ds}") for i in sorted(idx)]
    rows = []
    for m in chosen:
        e = enrol[m["driver_actor"]]
        rows.append(dict(id=m["id"], session_id=m["session_id"], stratum=m["stratum"], generator=m.get("generator", "liveportrait"), dataset=m["dataset"], kind=m["kind"],
                         driver_actor=m["driver_actor"], source_actor=m["source_actor"], driver_did=e["did"], epoch=e["epoch"], video=m["video"], raw_twin=m["raw_twin"],
                         frames=m["frames"], tag=tag_hex(m["session_id"])))
    tags = [r["tag"] for r in rows]
    assert len(set(tags)) == len(tags), "tag collision inside the subset"
    assert len({r["session_id"] for r in rows}) == len(rows)
    WORK.mkdir(parents=True, exist_ok=True)
    SUBSET.write_text(json.dumps(dict(seed=SEED, strata=STRATA, n=len(rows), rows=rows), indent=1))
    from collections import Counter
    print(len(rows), "sessions", Counter((r["stratum"], r["kind"]) for r in rows))


# ------------------------------------------------------------------------------------------------ VideoSeal arm (venv_vs)
def _vs_setup():
    import torch
    sys.path.insert(0, "external/Watemark/baseline/scripts")
    import wm_lib as W
    torch.cuda.set_per_process_memory_fraction(2.5 / 8.0)
    return torch, W, W.load_videoseal()


def vs_embed(torch, model, W, src, msg, dst):
    """RAW (or iris) twin frames -> VideoSeal in blocks of 8 frames (a multiple of the step size 4) -> x264 qp 0 yuv420p (the released clips' encoder)."""
    w, h, _, fps = probe_fast(src)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    sse, n = 0.0, 0
    with torch.no_grad():
        for blk in blocks(frame_iter(src), 8):
            wm = W.to_uint8(model.embed(W.to_tensor(blk), msgs=msg.cuda(), is_video=True)["imgs_w"])
            sse += float(((wm.astype(np.float64) - blk.astype(np.float64)) ** 2).sum())
            n += blk.size
            p.stdin.write(np.ascontiguousarray(wm).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0
    return float(10 * np.log10(255.0 ** 2 / (sse / n)))


def vs_logits(torch, model, W, path):
    tot, n = np.zeros(NB), 0
    with torch.no_grad():
        for blk in blocks(frame_iter(path), 2):
            lg = W.bit_logits(model, blk).numpy()
            tot += lg.sum(0)
            n += len(lg)
    return [round(float(x), 3) for x in tot / n], n


def retry_oom(torch, fn):
    for attempt in range(60):
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("CUDA OOM (shared GPU), retry", attempt + 1, "in 60 s", flush=True)
            time.sleep(60)
    sys.exit("GPU stayed full")


def cond_files(src, tag, conds, d, workers=3):
    """Make the condition files of src in parallel threads; returns {cond: path}."""
    from concurrent.futures import ThreadPoolExecutor
    out = {"clean": src}
    todo = [c for c in conds if c != "clean"]
    for c in todo:
        out[c] = d / f"{tag}_{c}.mp4"
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda c: make_condition(src, out[c], c), [c for c in todo if not out[c].exists()]))
    return out


def cmd_wm(a):
    torch, W, model = _vs_setup()
    rows = load_subset()["rows"]
    (LOGS / "wm").mkdir(parents=True, exist_ok=True)
    WMVID.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for r in rows:
        f = LOGS / "wm" / f"{r['id']}.json"
        if f.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t1 = time.time()
        tmp = TMP / r["id"]
        tmp.mkdir(parents=True, exist_ok=True)
        raw = ROOT / r["raw_twin"]
        wmv = WMVID / f"{r['id']}__wm_clean.mp4"
        msg = torch.tensor([[float(b) for b in payload(r["session_id"])]])
        if wmv.exists():
            wmv.unlink()
        psnr = retry_oom(torch, lambda: vs_embed(torch, model, W, raw, msg, wmv))
        t_embed = time.time() - t1
        wmc = cond_files(wmv, "wm", CONDS, tmp)
        rawc = cond_files(raw, "raw", CONDS, tmp)
        t_cond = time.time() - t1 - t_embed
        res = dict(id=r["id"], session_id=r["session_id"], tag=r["tag"], psnr_embed_db=round(psnr, 2), wm={}, null={}, bytes={})
        for c in CONDS:
            res["wm"][c], n1 = retry_oom(torch, lambda: vs_logits(torch, model, W, wmc[c]))
            res["null"][c], n2 = retry_oom(torch, lambda: vs_logits(torch, model, W, rawc[c]))
            assert n1 == n2, (c, n1, n2)
            res["bytes"][c] = wmc[c].stat().st_size
        res["frames"] = n1
        res["seconds"] = dict(embed=round(t_embed, 1), conditions=round(t_cond, 1), total=round(time.time() - t1, 1))
        res["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        f.write_text(json.dumps(res))
        for g in tmp.glob("*.mp4"):
            g.unlink()
        tmp.rmdir()
        done += 1
        bits = {c: round(float(np.mean([int(x > 0) == b for x, b in zip(res["wm"][c], payload(r["session_id"]))])), 3) for c in CONDS}
        print(f"{r['id']} {r['stratum']}: psnr {psnr:.1f} bit-acc256 {bits} | {res['seconds']} | total {time.time() - t0:.0f}s gpu {res['gpu_peak_gb']} GB", flush=True)


def cmd_comb(a):
    """Combined arm: VideoSeal on top of the released iris video; clean and CRF 23 files are KEPT for the iris extraction, logits stored."""
    torch, W, model = _vs_setup()
    rows = load_subset()["rows"]
    (LOGS / "comb").mkdir(parents=True, exist_ok=True)
    WMVID.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for r in rows:
        f = LOGS / "comb" / f"{r['id']}.json"
        if f.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t1 = time.time()
        src = ROOT / r["video"]
        cv = WMVID / f"{r['id']}__comb_clean.mp4"
        msg = torch.tensor([[float(b) for b in payload(r["session_id"])]])
        if cv.exists():
            cv.unlink()
        psnr = retry_oom(torch, lambda: vs_embed(torch, model, W, src, msg, cv))
        c23 = WMVID / f"{r['id']}__comb_crf23.mp4"
        make_condition(cv, c23, "crf23")
        res = dict(id=r["id"], session_id=r["session_id"], tag=r["tag"], psnr_embed_db=round(psnr, 2), wm={})
        for c, p in (("clean", cv), ("crf23", c23)):
            res["wm"][c], res["frames"] = retry_oom(torch, lambda: vs_logits(torch, model, W, p))
        f.write_text(json.dumps(res))
        done += 1
        bits = {c: round(float(np.mean([int(x > 0) == b for x, b in zip(res["wm"][c], payload(r["session_id"]))])), 3) for c in res["wm"]}
        print(f"{r['id']}: comb bit-acc256 {bits} psnr {psnr:.1f} | {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ iris arm (chain venv)
def cmd_iris(a):
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import proxy_verify as V
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    rows = load_subset()["rows"]
    CACHE.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    arm = a.arm
    conds = CONDS if arm == "iris" else COMB_CONDS
    units = []
    for r in rows:
        for c in conds:
            if arm == "iris" and c == "clean":
                continue                                    # the released file: existing cache of proxy_verify (read only)
            units.append((r, c))
    J = ArcIrisJudge()
    t0, done = time.time(), 0
    with FaceEyeDetector() as det:
        for r, c in units:
            key = f"{r['id']}__{arm}_{c}"
            f = CACHE / f"{key}.pkl"
            if zlib.crc32(key.encode()) % a.of != a.shard or f.exists():
                continue
            if a.limit and done >= a.limit:
                break
            t1 = time.time()
            if arm == "iris":
                tmpf = TMP / f"{key}.mp4"
                make_condition(ROOT / r["video"], tmpf, c)
                path, kill = tmpf, True
            else:
                path, kill = WMVID / f"{r['id']}__comb_{c}.mp4", False
                if not path.exists():
                    continue
            d = extract_path(V, path, det, J)
            d["key"] = key
            f.write_bytes(pickle.dumps(d))
            if kill:
                path.unlink()
            done += 1
            print(f"{key}: {len(d['obs'])} observations, {d['n_frames']} frames, {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)", flush=True)


def extract_path(V, path, det, J):
    """Body of proxy_verify.extract_key for an arbitrary file (same detector, same encode, same judge, EVERY=2)."""
    import proxy_attr_lib as L
    from cancelable.identify import geometry_agrees
    from iris_pipeline.encode import encode
    from iris_pipeline.segment import extract_irises
    frames = L.read_frames(path, every=EVERY)
    obs = []
    for i, rgb in enumerate(frames):
        for seg in extract_irises(rgb, det):
            if seg.side not in L.SIDES:
                continue
            code = encode(seg.strip, seg.strip_mask)
            v, info = J.encode_image(J.eye_crop(rgb, seg.iris))
            rej = bool(info["rejected"])
            obs.append(dict(frame=i, side=seg.side, bits=np.packbits(code.bits.ravel()), mask=np.packbits(code.mask.ravel()),
                            vis=float(np.mean(seg.strip_mask)), arc=None if rej else np.asarray(v, np.float16),
                            geo_ok=False if rej else geometry_agrees(info["iris"])))
    return dict(n_frames=len(frames), obs=obs)


# ------------------------------------------------------------------------------------------------ quality (venv_vs, CPU)
def cmd_quality(a):
    import cv2
    import torch
    import lpips
    from skimage.metrics import structural_similarity
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    rows = load_subset()["rows"]
    (LOGS / "quality").mkdir(parents=True, exist_ok=True)
    STEP = 4

    def lp(x, y):
        def prep(z):
            h, w = z.shape[:2]
            s = 512.0 / min(h, w)
            z = cv2.resize(z, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(z).permute(2, 0, 1)[None].float() / 127.5 - 1
        with torch.no_grad():
            return float(net(prep(x), prep(y)).item())

    def metrics(test, ref):
        """Every STEP-th frame: whole-frame PSNR, luma SSIM, LPIPS (alex, frames area-downscaled to short side 512), fraction of changed pixels,
        PSNR over the bounding box of the changed pixels (the eye band for the iris hook)."""
        ps, ss, ll, fr, pb = [], [], [], [], []
        for i, (x, y) in enumerate(zip(frame_iter(test), frame_iter(ref))):
            if i % STEP:
                continue
            d = x.astype(np.int16) - y.astype(np.int16)
            mse = float((d.astype(np.float64) ** 2).mean())
            ps.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
            ch = np.abs(d).max(2) > 0
            fr.append(float(ch.mean()))
            if ch.any():
                ys, xs = np.nonzero(ch)
                sub = d[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                m2 = float((sub.astype(np.float64) ** 2).mean())
                pb.append(10 * np.log10(255.0 ** 2 / m2) if m2 > 0 else 99.0)
            gx, gy = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)
            ss.append(float(structural_similarity(gx, gy, data_range=255)))
            ll.append(lp(x, y))
        return dict(frames=len(ps), psnr_db=round(float(np.mean(ps)), 2), ssim_luma=round(float(np.mean(ss)), 4), lpips=round(float(np.mean(ll)), 4),
                    changed_pixel_fraction=round(float(np.mean(fr)), 5), psnr_changed_bbox_db=round(float(np.mean(pb)), 2) if pb else None)

    for r in rows:
        key = r["id"]
        f = LOGS / "quality" / f"{key}.json"
        if zlib.crc32(key.encode()) % a.of != a.shard or f.exists():
            continue
        wmv = WMVID / f"{key}__wm_clean.mp4"
        if not wmv.exists():
            continue
        t0 = time.time()
        raw = ROOT / r["raw_twin"]
        out = dict(id=key, watermark_vs_raw=metrics(wmv, raw), iris_vs_raw=metrics(ROOT / r["video"], raw))
        f.write_text(json.dumps(out))
        print(key, out["watermark_vs_raw"]["psnr_db"], out["iris_vs_raw"]["psnr_db"], f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ summary (chain venv)
def rate(k, n):
    """k of n with the Wilson 95 % interval."""
    n = max(int(n), 1)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(k=int(k), n=int(n), rate=round(p, 4), ci95=[round(float(max(c - h, 0)), 3), round(float(min(c + h, 1)), 3)])


def summarize_wm(rows, logs, cond, key="wm"):
    """Per condition: bit accuracies of the tag, exact-lookup outcomes (correct / wrong session / no match), Hamming-tolerant lookup as a supplement."""
    tags = {r["session_id"]: tag_bits(r["session_id"]) for r in rows}
    T = np.array([tags[r["session_id"]] for r in rows])                      # (n, 32)
    sids = [r["session_id"] for r in rows]
    acc32, acc256, outc, ham = [], [], dict(correct=0, wrong_session=0, no_match=0), {h: dict(correct=0, wrong=0, none=0) for h in (0, 2, 4, 6)}
    nul = dict(false_lookup_exact=0, tag_bit_acc=[])
    kk = []
    for r in rows:
        L = np.asarray(logs[r["id"]][key][cond], float)
        bits = np.array(decode_logits(L))
        own = np.array(tag_bits(r["session_id"]))
        acc32.append(float((bits == own).mean()))
        acc256.append(float(np.mean((L > 0) == (np.array(payload(r["session_id"])) == 1))))
        d = (T != bits).sum(1)
        hits = np.where(d == 0)[0]
        if len(hits) == 0:
            outc["no_match"] += 1
        elif sids[hits[0]] == r["session_id"]:
            outc["correct"] += 1
        else:
            outc["wrong_session"] += 1
        j = int(np.argmin(d))
        for h in ham:
            if d[j] <= h:
                ham[h]["correct" if sids[j] == r["session_id"] else "wrong"] += 1
            else:
                ham[h]["none"] += 1
    n = len(rows)
    o = dict(n=n, tag_bit_acc_mean=round(float(np.mean(acc32)), 4), tag_bit_acc_min=round(float(np.min(acc32)), 4), tag_bit_acc_p10=round(float(np.percentile(acc32, 10)), 4),
             bit_acc_256_mean=round(float(np.mean(acc256)), 4), tag_exact_32of32=rate(sum(a_ == 1.0 for a_ in acc32), n),
             session_recovery_exact_lookup=rate(outc["correct"], n), wrong_session=rate(outc["wrong_session"], n), no_match=rate(outc["no_match"], n),
             supplementary_hamming_lookup={f"<= {h} bit errors": dict(correct=rate(v["correct"], n), wrong=rate(v["wrong"], n)) for h, v in ham.items()})
    return o


def cmd_summary(a):
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import proxy_verify as V
    from proxy_paper_numbers import roc_stats
    sub = load_subset()
    planned = sub["rows"]
    complete = lambda r: ((LOGS / "wm" / f"{r['id']}.json").exists() and V.load_cache(r["id"]) is not None
                          and all((CACHE / f"{r['id']}__iris_{c}.pkl").exists() for c in CONDS if c != "clean"))
    rows = [r for r in planned if complete(r)]          # the analysed sessions: every arm finished (the run was shrunk if the machine was too slow)
    plan = json.loads(V.PLAN.read_text())
    T1 = json.loads(V.CALIB.read_text())["claim"]["arciris"]["T1"]
    G = V.Gallery(None)
    names = [n for n in G.names]
    assert len({d for d, _ in names}) == len(names), "a DID has several epochs"
    didx = {d: i for i, (d, _) in enumerate(names)}
    out = dict(subset=dict(seed=sub["seed"], n_planned=len(planned), planned_ids=[r["id"] for r in planned], n=len(rows), strata={}, ids=[r["id"] for r in rows], session_ids=[r["session_id"] for r in rows]), T1=T1,
               conditions=CONDS, ffmpeg={c: ["ffmpeg", "-i", "<in>", *ffargs(c, 'W', 'H')] for c in CONDS if c != "clean"})
    for r in rows:
        k = out["subset"]["strata"].setdefault(r["stratum"], dict(n=0, self=0, cross=0))
        k["n"] += 1
        k[r["kind"]] += 1
    # ---------------- watermark arm
    logs = {r["id"]: json.loads((LOGS / "wm" / f"{r['id']}.json").read_text()) for r in rows if (LOGS / "wm" / f"{r['id']}.json").exists()}
    wrows = [r for r in rows if r["id"] in logs]
    out["watermark_arm"] = dict(n_sessions=len(wrows), by_condition={}, by_stratum={})
    for c in CONDS:
        w = summarize_wm(wrows, logs, c, "wm")
        nl = summarize_wm(wrows, logs, c, "null")
        out["watermark_arm"]["by_condition"][c] = dict(
            watermarked=w,
            unwatermarked_null=dict(tag_bit_acc_mean=nl["tag_bit_acc_mean"], bit_acc_256_mean=nl["bit_acc_256_mean"],
                                    false_lookup_exact=dict(k=nl["session_recovery_exact_lookup"]["k"] + nl["wrong_session"]["k"], n=nl["n"],
                                                            rate=round((nl["session_recovery_exact_lookup"]["k"] + nl["wrong_session"]["k"]) / nl["n"], 4)),
                                    analytic_false_lookup_rate=len(rows) / 2 ** 32,
                                    supplementary_hamming_false_lookup={h: v["correct"]["k"] + v["wrong"]["k"] for h, v in nl["supplementary_hamming_lookup"].items()}))
    for c in ("clean", "crf23", "crf30", "blur", "resize05", "crop08"):
        for st in sorted({r["stratum"] for r in wrows}):
            sel = [r for r in wrows if r["stratum"] == st]
            w = summarize_wm(sel, logs, c, "wm")
            out["watermark_arm"]["by_stratum"].setdefault(st, {})[c] = dict(recovery=w["session_recovery_exact_lookup"], tag_bit_acc_mean=w["tag_bit_acc_mean"])
    out["watermark_arm"]["psnr_embed_db_mean"] = round(float(np.mean([logs[r["id"]]["psnr_embed_db"] for r in wrows])), 2) if wrows else None
    out["watermark_arm"]["seconds_per_session"] = {k: round(float(np.mean([logs[r["id"]]["seconds"][k] for r in wrows])), 1) for k in ("embed", "conditions", "total")} if wrows else None
    # ---------------- iris arm
    def iris_dist(r, key_cache):
        c = V.load_cache(r["id"]) if key_cache == "released" else pickle.loads((CACHE / f"{key_cache}.pkl").read_bytes()) if (CACHE / f"{key_cache}.pkl").exists() else None
        if c is None:
            return None
        return G.vectors(c)[0]["arciris"]

    def iris_cond(sel, keyf):
        gen, imp, acc, n_ok = [], [], 0, 0
        gen_ok_by_kind = {"self": [0, 0], "cross": [0, 0]}
        for r in sel:
            d = iris_dist(r, keyf(r))
            if d is None:
                continue
            n_ok += 1
            g = float(d[didx[r["driver_did"]]])
            ok = bool(np.isfinite(g) and g < T1)
            acc += ok
            gen_ok_by_kind[r["kind"]][0] += ok
            gen_ok_by_kind[r["kind"]][1] += 1
            o = np.delete(d, didx[r["driver_did"]])
            imp += [float(x) for x in o[np.isfinite(o)]]
            gen.append(g if np.isfinite(g) else 9.0)             # an unreadable clip is a reject (worst possible distance) and stays in the denominator
        if not gen:
            return None
        st = roc_stats(gen, imp)
        return dict(n=n_ok, genuine_acceptance_at_T1=rate(acc, n_ok), eer_percent=st["eer"], auc_percent=st["auc"], tar_at_far1_percent=st["tar_far1"],
                    impostor_accepted_at_T1=rate(sum(x < T1 for x in imp), len(imp)), genuine_distance_median=round(float(np.median(gen)), 4),
                    unreadable=int(sum(g >= 9.0 for g in gen)), by_kind={k: rate(*v) for k, v in gen_ok_by_kind.items()})
    out["iris_arm"] = dict(by_condition={}, by_stratum={})
    for c in CONDS:
        kf = (lambda r: "released") if c == "clean" else (lambda r, c=c: f"{r['id']}__iris_{c}")
        out["iris_arm"]["by_condition"][c] = iris_cond(rows, kf)
        for st in sorted({r["stratum"] for r in rows}):
            out["iris_arm"]["by_stratum"].setdefault(st, {})[c] = iris_cond([r for r in rows if r["stratum"] == st], kf)
    # ---------------- combined arm
    clog = {r["id"]: json.loads((LOGS / "comb" / f"{r['id']}.json").read_text()) for r in rows if (LOGS / "comb" / f"{r['id']}.json").exists()}
    crows = [r for r in rows if r["id"] in clog]
    out["combined_arm"] = dict(n_sessions=len(crows), by_condition={})
    for c in COMB_CONDS:
        kf = (lambda r, c=c: f"{r['id']}__comb_{c}")
        ir_c = iris_cond(crows, kf) if crows else None
        ir_o = iris_cond(crows, (lambda r: "released")) if crows else None            # the same sessions without the watermark, same condition
        ir_o_c = iris_cond(crows, (lambda r: "released") if c == "clean" else (lambda r, c=c: f"{r['id']}__iris_{c}")) if crows else None
        wmc = summarize_wm(crows, {k: dict(wm=v["wm"]) for k, v in clog.items()}, c, "wm") if crows else None
        out["combined_arm"]["by_condition"][c] = dict(iris_with_watermark=ir_c, iris_without_watermark_same_sessions=ir_o_c, watermark_on_iris_video=wmc)
    out["combined_arm"]["psnr_embed_db_mean"] = round(float(np.mean([clog[r["id"]]["psnr_embed_db"] for r in crows])), 2) if crows else None
    # ---------------- quality
    q = {r["id"]: json.loads((LOGS / "quality" / f"{r['id']}.json").read_text()) for r in rows if (LOGS / "quality" / f"{r['id']}.json").exists()}
    out["quality"] = dict(n_sessions=len(q), note="every 4th frame; whole-frame PSNR; luma SSIM; LPIPS-alex on frames area-downscaled to short side 512; vs the RAW twin. "
                                                  "iris = released embedded file (qp 0), watermark = VideoSeal output (qp 0)")
    for arm in ("watermark_vs_raw", "iris_vs_raw"):
        if q:
            keys = ["psnr_db", "ssim_luma", "lpips", "changed_pixel_fraction"]
            out["quality"][arm] = {k: dict(mean=round(float(np.mean([v[arm][k] for v in q.values()])), 4), min=round(float(np.min([v[arm][k] for v in q.values()])), 4),
                                           max=round(float(np.max([v[arm][k] for v in q.values()])), 4)) for k in keys}
            pb = [v[arm]["psnr_changed_bbox_db"] for v in q.values() if v[arm]["psnr_changed_bbox_db"] is not None]
            out["quality"][arm]["psnr_changed_bbox_db"] = dict(mean=round(float(np.mean(pb)), 2), min=round(float(np.min(pb)), 2)) if pb else None
    # ---------------- compact table
    tab = {}
    for c in CONDS:
        w = out["watermark_arm"]["by_condition"][c]["watermarked"]
        i = out["iris_arm"]["by_condition"][c]
        tab[c] = dict(wm_session_recovery=w["session_recovery_exact_lookup"]["rate"], wm_session_recovery_ci95=w["session_recovery_exact_lookup"]["ci95"], wm_tag_bit_acc=w["tag_bit_acc_mean"],
                      wm_false_lookup_unwatermarked=out["watermark_arm"]["by_condition"][c]["unwatermarked_null"]["false_lookup_exact"]["rate"],
                      iris_genuine_acceptance=i["genuine_acceptance_at_T1"]["rate"] if i else None, iris_genuine_acceptance_ci95=i["genuine_acceptance_at_T1"]["ci95"] if i else None, iris_eer_percent=i["eer_percent"] if i else None,
                      iris_n=i["n"] if i else 0)
    out["table"] = tab
    (WORK / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(tab, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "wm", "comb", "iris", "quality", "summary"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--arm", default="iris", choices=["iris", "comb"])
    a = ap.parse_args()
    {"plan": cmd_plan, "wm": cmd_wm, "comb": cmd_comb, "iris": cmd_iris, "quality": cmd_quality, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
