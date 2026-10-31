"""Provenance stamp for result JSONs.

Every result file the paper may cite should say which code, command and environment made it.
File mtimes are useless here (a checkout resets them), so the stamp carries its own UTC time.

    from provenance import stamp
    path.write_text(json.dumps(stamp(result, seed=0), indent=2))

`stamp` adds a top-level "_provenance" key to a dict and leaves any other value unchanged, so
readers that index by name (pool_screen.json["rows"], cancelable_calibration.json) keep working.
"""
import datetime
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBS = ("numpy", "scipy", "torch", "opencv-python", "opencv-contrib-python", "mediapipe", "onnxruntime-gpu", "web3")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def _version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def provenance(seed=None, **extra):
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=_git("rev-parse", "--short", "HEAD") or None,
        git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        argv=sys.argv,
        python=sys.version.split()[0],
        libs={p: v for p in LIBS if (v := _version(p))},
        seed=seed,
        **extra,
    )


def stamp(result, seed=None, **extra):
    if isinstance(result, dict):
        return {**result, "_provenance": provenance(seed, **extra)}
    return result
