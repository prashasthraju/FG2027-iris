"""Parallel generation for the large session (batch 4). Generation workers write clips and a 'generated' session file and never touch the
ledger; ONE recorder process writes the session records (the ledger file is not safe for concurrent writers).

    python tools/proxy_gen_parallel.py gen --shard 0 --of 3 [--generator liveportrait|personalive] [--limit N]
    python tools/proxy_gen_parallel.py record --watch 900        # exits after 900 s without new sessions

Rows come from outputs/proxy_attr_plan4.json (tools/proxy_batch4.py). A row is skipped when its session file exists.
"""
import os
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import proxy_attr as A  # noqa: E402
import proxy_ledger as PL  # noqa: E402

PLAN4 = ROOT / "outputs" / "proxy_attr_plan4.json"
PL_PY = Path(os.environ.get("PERSONALIVE_PYTHON", "python"))
PL_RESULTS = Path(os.environ.get("PERSONALIVE_DIR", "external/PersonaLive")) / "results"
EDTALK_PY = ROOT / "third_party" / "EDTalk" / ".venv_edtalk" / "bin" / "python"
LIAX_PY = ROOT / "third_party" / "LIA-X" / ".venv" / "bin" / "python"


def run_liveportrait(row, out):
    cmd = [str(A.LP_PY), str(ROOT / "tools" / "reenact_video.py"), "liveportrait", "--source", row["source"], "--frames", str(row["frames"]),
           "--scale", str(row["scale"]), "--driver", row["driver"], "--embed-entry", row["embed_entry"], "-o", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stem = Path(row["source"]).stem
    return p, out / f"liveportrait__{stem}__inline_{row['embed_entry']}.mp4", out / f"liveportrait__{stem}__inline_raw.mp4"


def run_personalive(row, out):
    stem = f"{Path(row['source']).stem}_{Path(row['driver']).stem}"
    for f in PL_RESULTS.glob(f"*/*/{stem}*"):                        # inference_offline.py skips a pair whose output already exists
        f.unlink()
    cmd = [str(PL_PY), "-s", str(ROOT / "tools" / "personalive_lowmem.py"), "--tas", "2", "--vae-slicing", "--embed-entry", row["embed_entry"],
           "--target-iris-px", "95", "--out", str(out), "--", "--config", "configs/personalive_rebuilt.yaml", "--reference_image", row["source"],
           "--driving_video", row["driver"], "-L", str(row["frames"]), "-W", "512", "-H", "512"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p, out / f"personalive__inline_{row['embed_entry']}.mp4", out / "personalive__inline_raw.mp4"


def run_other(name, py, script, scale):
    def run(row, out):
        cmd = [str(py), str(ROOT / "tools" / script), "--source", row["source"], "--driver", row["driver"], "--frames", str(row["frames"]), "--scale", str(scale),
               "--embed-entry", row["embed_entry"], "-o", str(out)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        stem = f"{name}__{Path(row['source']).stem}"
        return p, out / f"{stem}__inline_{row['embed_entry']}.mp4", out / f"{stem}__inline_raw.mp4"
    return run


RUNNERS = dict(liveportrait=run_liveportrait, personalive=run_personalive, edtalk=run_other("edtalk", EDTALK_PY, "gen_edtalk.py", 2.5),
               liax=run_other("liax", LIAX_PY, "gen_liax.py", 3))


def gen(a):
    plan = json.loads(PLAN4.read_text())
    A.CLIPS.mkdir(parents=True, exist_ok=True)
    A.SESS.mkdir(parents=True, exist_ok=True)
    done = 0
    rows = list(enumerate(plan["rows"]))
    if a.reverse:
        rows.reverse()
    for i, row in rows:
        if i % a.of != a.shard or row.get("generator", "liveportrait") != a.generator:
            continue
        if a.kinds and row["kind"] not in a.kinds.split(","):
            continue
        if a.reverse and (A.CLIPS / row["id"]).exists():          # another worker has started this row
            continue
        meta_f = A.SESS / f"{row['id']}.json"
        if meta_f.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t0 = time.time()
        out = A.CLIPS / row["id"]
        out.mkdir(parents=True, exist_ok=True)
        p, emb, raw = RUNNERS[a.generator](row, out)
        info_f = emb.with_suffix(".json")
        if p.returncode != 0 or not emb.exists() or not info_f.exists():
            meta_f.write_text(json.dumps(dict(row, status="gen_failed", stderr_tail=p.stderr[-600:]), indent=1))
            print(row["id"], "generation failed", flush=True)
            continue
        info = json.loads(info_f.read_text())
        meta = dict(row, status="generated", video=str(emb.relative_to(ROOT)), raw_twin=str(raw.relative_to(ROOT)), embedded_eyes=info["embedded_eyes"],
                    gated_eyes=info["gated_eyes"], embed_ms_median=info["embed_ms_median"], seconds=round(time.time() - t0, 1))
        meta_f.write_text(json.dumps(meta, indent=1))
        done += 1
        print(f"{row['id']} {a.generator} embedded {info['embedded_eyes']}/{info['embedded_eyes'] + info['gated_eyes']} {meta['seconds']:.0f}s", flush=True)


def record(a):
    plan = json.loads(PLAN4.read_text())
    enrol = plan["enrolment"]
    issuer = PL.account_from(A.ISSUER_SEED, "issuer")
    idle_since = time.time()
    n = 0
    while True:
        todo = [f for f in sorted(A.SESS.glob("*.json")) if json.loads(f.read_text()).get("status") == "generated"]
        if not todo:
            if time.time() - idle_since > a.watch:
                break
            time.sleep(20)
            continue
        led = PL.Ledger.load(A.LEDGER)
        for f in todo:
            m = json.loads(f.read_text())
            import cv2
            c = cv2.VideoCapture(str(ROOT / m["video"]))
            fps = c.get(cv2.CAP_PROP_FPS) or 25.0
            c.release()
            rec = A.make_record(led, m, enrol, ROOT / m["video"], fps, m["frames"])
            h = led.record_session(issuer.address, rec)
            led.save(A.LEDGER)
            m.update(status="ok", ledger_block=h, video_sha256=rec["video"]["sha256"])
            f.write_text(json.dumps(m, indent=1))
            n += 1
            print(f"recorded {m['id']} block {h} ({n})", flush=True)
        idle_since = time.time()
    print("recorder idle; done", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "record"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--generator", default="liveportrait", choices=["liveportrait", "personalive", "edtalk", "liax"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kinds", default="", help="comma list of session kinds to generate (self,cross,imitation)")
    ap.add_argument("--reverse", action="store_true", help="walk the plan backwards and skip rows another worker has started (extra worker next to the shard workers)")
    ap.add_argument("--watch", type=int, default=600)
    a = ap.parse_args()
    {"gen": gen, "record": record}[a.cmd](a)


if __name__ == "__main__":
    main()
