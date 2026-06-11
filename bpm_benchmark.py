#!/usr/bin/env python3
"""
BPM detection benchmark script.

Compares multiple BPM detection algorithms against ground truth values.
Reports per-track accuracy and timing for each method.

Usage:
    # Provide ground truth as a CSV (path,bpm) or directory + known BPMs in filenames:
    python bpm_benchmark.py /path/to/audio/ --ground-truth ground_truth.csv
    python bpm_benchmark.py /path/to/audio/  # extracts BPM from filenames

    # Restrict BPM range:
    python bpm_benchmark.py /path/to/audio/ --min-bpm 165 --max-bpm 220

Ground truth CSV format:
    path,bpm
    /path/to/song.mp3,172
    /path/to/other.flac,190
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import librosa

SAMPLE_RATE = 44100
TEMPOCNN_SR = 11025


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------

def _clamp(tempo, lo, hi):
    """Map tempo into valid range via harmonic multiples."""
    if lo <= tempo <= hi:
        return tempo
    for mult in [2.0, 0.5, 4.0, 0.25]:
        adjusted = tempo * mult
        if lo <= adjusted <= hi:
            return adjusted
    return tempo


def detect_librosa(audio_44k, sr, lo, hi):
    """Librosa HPSS + median-voted tempo."""
    _, y_perc = librosa.effects.hpss(audio_44k)
    onset_env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, aggregate=np.median
    )
    tempos = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=(lo + hi) / 2,
        aggregate=None,
    )
    return int(round(_clamp(float(np.median(tempos)), lo, hi)))


def detect_percival(audio_44k, lo, hi):
    """Essentia PercivalBpmEstimator."""
    from essentia.standard import PercivalBpmEstimator
    estimator = PercivalBpmEstimator(
        minBPM=lo, maxBPM=hi, sampleRate=SAMPLE_RATE
    )
    tempo = float(estimator(audio_44k.astype(np.float32)))
    return int(round(_clamp(tempo, lo, hi)))


def detect_rhythm2013(audio_44k, lo, hi):
    """Essentia RhythmExtractor2013."""
    from essentia.standard import RhythmExtractor2013
    extractor = RhythmExtractor2013(minTempo=lo, maxTempo=hi)
    bpm, _, _, _, _ = extractor(audio_44k.astype(np.float32))
    return int(round(_clamp(float(bpm), lo, hi)))


def detect_ensemble(audio_44k, sr, lo, hi):
    """Majority vote across Percival, RhythmExtractor2013, and librosa."""
    candidates = [
        detect_percival(audio_44k, lo, hi),
        detect_rhythm2013(audio_44k, lo, hi),
        detect_librosa(audio_44k, sr, lo, hi),
    ]
    tol = 3
    best_group = []
    for i, a in enumerate(candidates):
        group = [a]
        for j, b in enumerate(candidates):
            if i != j and abs(a - b) <= tol:
                group.append(b)
        if len(group) > len(best_group):
            best_group = group
    if len(best_group) >= 2:
        return int(round(float(np.mean(best_group))))
    return candidates[0]


def _make_tempocnn_predictor(model_path):
    """Lazy-load TempoCNN model."""
    from essentia.standard import TensorflowPredictTempoCNN
    return TensorflowPredictTempoCNN(graphFilename=str(model_path))


def detect_madmom(audio_path, lo, hi):
    """madmom DBNBeatTracker → linear regression on beat positions → BPM."""
    import scipy.stats
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    rnn = RNNBeatProcessor()
    dbn = DBNBeatTrackingProcessor(min_bpm=lo, max_bpm=hi)
    beats = dbn(rnn(str(audio_path)))

    if len(beats) < 4:
        return 0

    slope, _, _, _, _ = scipy.stats.linregress(np.arange(len(beats)), beats)
    bpm = 60.0 / slope
    return int(round(_clamp(bpm, lo, hi)))


def detect_tempocnn(audio_44k, lo, hi, predictor):
    """TempoCNN with weighted-peak (window +-1)."""
    audio_11k = librosa.resample(audio_44k, orig_sr=SAMPLE_RATE, target_sr=TEMPOCNN_SR)
    audio_11k = audio_11k.astype(np.float32)

    predictions = np.array(predictor(audio_11k))
    if predictions.size == 0:
        return 0

    bins = np.linspace(30, 286, 256)
    avg = np.mean(predictions, axis=0)

    # Mask outside range
    mask = (bins >= lo) & (bins <= hi)
    masked = avg.copy()
    masked[~mask] = 0.0

    if masked.sum() == 0:
        # Use unmasked peak + harmonic correction
        best_idx = int(np.argmax(avg))
        wlo = max(0, best_idx - 1)
        whi = min(len(avg), best_idx + 2)
        wp, wb = avg[wlo:whi], bins[wlo:whi]
        tempo = float(np.dot(wp, wb) / wp.sum()) if wp.sum() > 0 else float(bins[best_idx])
        return int(round(_clamp(tempo, lo, hi)))

    best_idx = int(np.argmax(masked))
    wlo = max(0, best_idx - 1)
    whi = min(len(masked), best_idx + 2)
    wp, wb = masked[wlo:whi], bins[wlo:whi]
    tempo = float(np.dot(wp, wb) / wp.sum()) if wp.sum() > 0 else float(bins[best_idx])
    return int(round(tempo))


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
                    '.wma', '.aiff', '.aif', '.opus', '.webm'}


def extract_bpm_from_filename(name):
    """Try to extract a BPM from the filename stem."""
    stem = os.path.splitext(name)[0]
    # " - 172", " 172 bpm", "(172 bpm)", trailing bare number
    m = re.search(r'(\d{2,3})\s*[Bb][Pp][Mm]', stem)
    if m:
        return int(m.group(1))
    m = re.search(r'[\s\-]+(\d{2,3})(?:\.\d+)?\s*$', stem)
    if m:
        return int(m.group(1))
    return None


def load_ground_truth(audio_dir, csv_path=None):
    """Return list of (filepath, true_bpm) tuples."""
    if csv_path:
        entries = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Path(row['path'])
                if not p.is_absolute():
                    p = Path(audio_dir) / p
                entries.append((str(p), int(row['bpm'])))
        return entries

    # Scan directory and extract from filenames
    entries = []
    audio_dir = Path(audio_dir)
    for root, _, files in os.walk(audio_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            bpm = extract_bpm_from_filename(f)
            if bpm is not None:
                entries.append((os.path.join(root, f), bpm))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ['librosa', 'percival', 'rhythm2013', 'ensemble', 'tempocnn', 'madmom']


def run_benchmark(entries, methods, lo, hi, tempocnn_model=None):
    """Run all methods on all files, return results dict."""
    # Pre-load TempoCNN if needed
    predictor = None
    if 'tempocnn' in methods:
        if tempocnn_model is None:
            candidates = [
                Path(__file__).parent / 'models' / 'deepsquare-k16-3.pb',
                Path.home() / '.keypipe' / 'deepsquare-k16-3.pb',
            ]
            for c in candidates:
                if c.exists():
                    tempocnn_model = str(c)
                    break
        if tempocnn_model:
            predictor = _make_tempocnn_predictor(tempocnn_model)
        else:
            print("  Warning: TempoCNN model not found, skipping tempocnn method")
            methods = [m for m in methods if m != 'tempocnn']

    results = {m: {'detections': [], 'times': [], 'errors': []} for m in methods}

    for i, (filepath, true_bpm) in enumerate(entries):
        fname = os.path.basename(filepath)
        print(f"  [{i+1}/{len(entries)}] {fname}")

        # Load audio once at 44100
        try:
            audio_44k, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"    ERROR loading: {e}")
            continue

        for method in methods:
            t0 = time.perf_counter()
            try:
                if method == 'librosa':
                    detected = detect_librosa(audio_44k, sr, lo, hi)
                elif method == 'percival':
                    detected = detect_percival(audio_44k, lo, hi)
                elif method == 'rhythm2013':
                    detected = detect_rhythm2013(audio_44k, lo, hi)
                elif method == 'ensemble':
                    detected = detect_ensemble(audio_44k, sr, lo, hi)
                elif method == 'tempocnn':
                    detected = detect_tempocnn(audio_44k, lo, hi, predictor)
                elif method == 'madmom':
                    detected = detect_madmom(filepath, lo, hi)
                else:
                    continue
            except Exception as e:
                print(f"    {method} ERROR: {e}")
                continue
            elapsed = time.perf_counter() - t0

            error = detected - true_bpm
            results[method]['detections'].append(detected)
            results[method]['times'].append(elapsed)
            results[method]['errors'].append(error)

    return results


def print_detail_table(entries, results, methods):
    """Print per-file results table."""
    header = f"  {'File':<45}"
    header += f"  {'True':>5}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (filepath, true_bpm) in enumerate(entries):
        fname = os.path.basename(filepath)[:44]
        row = f"  {fname:<45}  {true_bpm:>5}"
        for m in methods:
            if i < len(results[m]['detections']):
                det = results[m]['detections'][i]
                err = results[m]['errors'][i]
                marker = " " if err == 0 else "X"
                row += f"  {det:>4}({err:>+3}){marker}"
            else:
                row += f"  {'err':>12}"
        print(row)


def print_summary(results, methods):
    """Print aggregate stats per method."""
    print(f"\n{'='*70}")
    print(f"  {'Method':<15} {'Accuracy':>9} {'Correct':>8} {'Total':>6} {'MAE':>6} {'Avg Time':>9}")
    print(f"  {'-'*15} {'-'*9} {'-'*8} {'-'*6} {'-'*6} {'-'*9}")

    for m in methods:
        errors = results[m]['errors']
        times = results[m]['times']
        if not errors:
            print(f"  {m:<15}  (no results)")
            continue

        abs_errors = [abs(e) for e in errors]
        n = len(errors)
        mae = sum(abs_errors) / n
        correct = sum(1 for e in abs_errors if e == 0)
        accuracy = correct / n * 100
        avg_time = sum(times) / n

        print(f"  {m:<15} {accuracy:>8.1f}% {correct:>7}/{n:<4} {mae:>6.2f} {avg_time:>8.2f}s")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark BPM detection algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', type=Path, help='Audio directory to scan')
    parser.add_argument('--ground-truth', '-g', type=Path, default=None,
                        help='CSV file with ground truth (path,bpm)')
    parser.add_argument('--min-bpm', type=int, default=55,
                        help='Minimum BPM (default: 55)')
    parser.add_argument('--max-bpm', type=int, default=215,
                        help='Maximum BPM (default: 215)')
    parser.add_argument('--methods', type=str, default=','.join(METHODS),
                        help=f'Comma-separated methods (default: {",".join(METHODS)})')
    parser.add_argument('--tempocnn-model', type=Path, default=None,
                        help='Path to TempoCNN .pb model file')
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(',')]
    for m in methods:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {', '.join(METHODS)}")
            sys.exit(1)

    print(f"Scanning: {args.path}")
    entries = load_ground_truth(args.path, args.ground_truth)
    if not entries:
        print("No files with ground truth BPM found.")
        sys.exit(1)

    print(f"Found {len(entries)} files with ground truth")
    print(f"BPM range: {args.min_bpm}-{args.max_bpm}")
    print(f"Methods: {', '.join(methods)}")
    print()

    results = run_benchmark(
        entries, methods, args.min_bpm, args.max_bpm,
        tempocnn_model=str(args.tempocnn_model) if args.tempocnn_model else None
    )

    print(f"\n{'='*70}")
    print("  DETAIL")
    print(f"{'='*70}")
    print_detail_table(entries, results, methods)
    print_summary(results, methods)


if __name__ == '__main__':
    main()
