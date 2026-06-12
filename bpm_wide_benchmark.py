"""Wide BPM benchmark: essentia vs ONNX backends over the MIK-tagged library.

Ground truth = BPM from filenames (Mixed In Key tags). Stratified sample
across 10-BPM buckets; octave-aware exact-integer scoring (detected D
matches tag T if D == round(T * m) for m in {1, 2, 0.5, 4, 0.25}).

Usage:
    python bpm_wide_benchmark.py [--per-bucket 24] [--seed 7] [--out results.csv]
"""

import argparse
import csv
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from keypipe.utils import AUDIO_EXTENSIONS

ROOT = Path("/home/hq/Music/01 Tracks")
TAG = re.compile(r"-\s*(\d{2,3}(?:\.\d{1,2})?)\s*$")


def harvest():
    out = []
    for p in ROOT.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file():
            m = TAG.search(p.stem)
            if m and 50 <= float(m.group(1)) <= 220:
                out.append((p, float(m.group(1))))
    return out


def stratify(corpus, per_bucket, seed):
    rng = random.Random(seed)
    buckets = {}
    for item in corpus:
        buckets.setdefault(int(item[1] // 10), []).append(item)
    sample = []
    for items in buckets.values():
        rng.shuffle(items)
        sample.extend(items[:per_bucket])
    return sample


def octave_match(detected, tag):
    return any(detected == round(tag * m) for m in (1, 2, 0.5, 4, 0.25))


def near_octave_match(detected, tag):
    return any(abs(detected - tag * m) <= 1.01 for m in (1, 2, 0.5, 4, 0.25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="wide_benchmark_results.csv")
    args = ap.parse_args()

    from keypipe.inference import BPMDetector
    from keypipe.inference_onnx import OnnxBPMDetector

    sample = stratify(harvest(), args.per_bucket, args.seed)
    print(f"benchmarking {len(sample)} tracks", flush=True)

    ess = BPMDetector()
    onnx = OnnxBPMDetector()

    def work(item):
        path, tag = item
        try:
            audio = ess._load_mono(path)
            raw, _ = onnx._predict_bpm_raw(audio)  # identical probs either way
            e_onset = ess._onset_bpm(audio)
            o_onset = onnx._onset_bpm(audio)
            e = ess._correct_with_onset(raw, e_onset)
            o = onnx._correct_with_onset(raw, o_onset)
            return (str(path), tag, raw, e_onset, o_onset, e, o, None)
        except Exception as exc:
            return (str(path), tag, None, None, None, None, None, str(exc))

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, item) for item in sample]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                print(f"{i}/{len(sample)} ({rate:.1f} tracks/s)", flush=True)

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "tag", "cnn_raw", "essentia_onset", "librosa_onset",
                         "essentia_bpm", "onnx_bpm", "error"])
        writer.writerows(rows)

    ok = [r for r in rows if r[7] is None]
    print(f"\nscored {len(ok)} / {len(rows)} (errors: {len(rows) - len(ok)})")
    for label, idx in (("essentia", 5), ("onnx    ", 6)):
        exact = sum(octave_match(r[idx], r[1]) for r in ok)
        near = sum(near_octave_match(r[idx], r[1]) for r in ok)
        print(f"{label}: exact(octave-aware) {exact}/{len(ok)} "
              f"({exact/len(ok):.1%})   within +/-1: {near}/{len(ok)} ({near/len(ok):.1%})")
    both = sum(r[5] == r[6] for r in ok)
    print(f"backend agreement: {both}/{len(ok)} ({both/len(ok):.1%})")


if __name__ == "__main__":
    main()
