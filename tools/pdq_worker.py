"""PDQ perceptual hashes of a video's keyframes. Run with third_party/.venv_pdq/bin/python (pdqhash needs numpy < 2, so it
lives in its own venv; frames are decoded by ffmpeg, not OpenCV).

    third_party/.venv_pdq/bin/python tools/pdq_worker.py VIDEO [STRIDE=10]   ->  JSON {"hashes": [64-hex-char, ...], "n_frames": N}

Every frame is squashed to 384 x 384 first (same for an original and its re-encoded copy), then every STRIDE-th frame is hashed.
"""
import json
import subprocess
import sys

import numpy as np
import pdqhash

S = 384


def main():
    path, stride = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={S}:{S}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, hashes = 0, []
    size = S * S * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        if n % stride == 0:
            h, _q = pdqhash.compute(np.frombuffer(buf, np.uint8).reshape(S, S, 3))
            hashes.append(np.packbits(h.astype(np.uint8)).tobytes().hex())
        n += 1
    p.wait()
    print(json.dumps(dict(hashes=hashes, n_frames=n)))


if __name__ == "__main__":
    main()
