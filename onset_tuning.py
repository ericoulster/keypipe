"""Tune librosa onset peak-picking for the ONNX backend's BPM tiebreaker.

Stage 1 (slow, once): cache onset envelopes for the wide-benchmark sample.
Stage 2 (fast): grid-search peak-picking params on the TRAIN half only;
report top configs evaluated on the held-out TEST half.

The CNN raw BPM and essentia's results are reused from the benchmark CSV;
only the librosa onset -> autocorrelation -> rounding-tiebreak path varies.
"""

import csv
import json
import sys
from itertools import product
from pathlib import Path

import librosa
import numpy as np

from keypipe.inference import BPMDetector

CSV_IN = "/tmp/wide_benchmark_results.csv"
CACHE = Path("/tmp/onset_envelopes.npz")
SR = 44100
HOP = 512


def rows():
    out = [r for r in csv.DictReader(open(CSV_IN)) if not r["error"]]
    out.sort(key=lambda r: r["path"])
    return out


def cache_envelopes():
    envs = {}
    for i, r in enumerate(rows(), 1):
        y, _ = librosa.load(r["path"], sr=SR, mono=True)
        envs[r["path"]] = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        if i % 25 == 0:
            print(f"envelopes {i}", flush=True)
    np.savez_compressed(CACHE, **envs)
    print(f"cached {len(envs)} envelopes", flush=True)


def onsets_to_bpm(onsets, det):
    """Mirror of BPMDetector._onset_bpm from detected onset times."""
    if len(onsets) < 8:
        return None
    fs = det._AUTOCORR_FS
    signal = np.zeros(int((onsets[-1] + 1.0) * fs))
    idx = (onsets * fs).astype(int)
    signal[idx[(idx >= 0) & (idx < len(signal))]] = 1.0
    min_lag = int(60.0 / det._max_bpm * fs)
    max_lag = min(int(60.0 / det._min_bpm * fs), len(signal) - 1)
    if max_lag <= min_lag:
        return None
    # identical to np.correlate(...)[mid:] over this lag window, but only
    # computes the ~110 lags we actually search (grid-tuning hot path)
    corr = np.array([signal[: len(signal) - lag] @ signal[lag:] for lag in range(min_lag, max_lag + 1)])
    return 60.0 * fs / (min_lag + int(np.argmax(corr)))


def octave_match(d, t):
    return any(d == round(t * m) for m in (1, 2, 0.5, 4, 0.25))


def evaluate(subset, envs, det, params):
    hits = 0
    for r in subset:
        env = envs[r["path"]]
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=SR, hop_length=HOP, units="time", **params
        )
        bpm = det._correct_with_onset(float(r["cnn_raw"]), onsets_to_bpm(np.asarray(frames), det))
        hits += octave_match(bpm, float(r["tag"]))
    return hits


def tune():
    det = BPMDetector.__new__(BPMDetector)  # no essentia init needed
    det._min_bpm, det._max_bpm = 55, 215

    data = rows()
    envs = dict(np.load(CACHE))
    rng = np.random.default_rng(11)
    order = rng.permutation(len(data))
    train = [data[i] for i in order[: len(data) // 2]]
    test = [data[i] for i in order[len(data) // 2:]]

    ess_train = sum(octave_match(int(r["essentia_bpm"]), float(r["tag"])) for r in train)
    ess_test = sum(octave_match(int(r["essentia_bpm"]), float(r["tag"])) for r in test)
    print(f"essentia reference: train {ess_train}/{len(train)}, test {ess_test}/{len(test)}", flush=True)

    grid = {
        "pre_max": [3, 10, 20],
        "post_max": [3, 10, 20],
        "pre_avg": [30, 100],
        "post_avg": [30, 100],
        "delta": [0.0, 0.05, 0.1, 0.2],
        "wait": [0, 10, 20],
    }
    results = []
    combos = list(product(*grid.values()))
    print(f"searching {len(combos)} configs on {len(train)} train tracks", flush=True)
    for i, values in enumerate(combos):
        params = dict(zip(grid.keys(), values))
        score = evaluate(train, envs, det, params)
        results.append((score, params))
        if i % 50 == 0:
            print(f"{i}/{len(combos)} best so far {max(r[0] for r in results)}", flush=True)

    results.sort(key=lambda x: -x[0])
    print("\ntop 5 on train:")
    for score, params in results[:5]:
        test_score = evaluate(test, envs, det, params)
        print(f"  train {score}/{len(train)}  test {test_score}/{len(test)}  {json.dumps(params)}", flush=True)

    # librosa defaults as baseline
    default_train = evaluate(train, envs, det, {})
    default_test = evaluate(test, envs, det, {})
    print(f"\nlibrosa defaults: train {default_train}/{len(train)}, test {default_test}/{len(test)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["cache"]:
        cache_envelopes()
    else:
        tune()
