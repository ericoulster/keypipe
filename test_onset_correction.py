#!/usr/bin/env python3
"""
Test Approach 5: Onset-assisted correction for TempoCNN BPM detection.

Strategy: Use onset positions to derive an independent BPM estimate via
autocorrelation of the onset train. When TempoCNN's raw float estimate
rounds to a different integer than the onset-derived estimate, prefer
the one closer to an integer (i.e., more "confident" rounding).

This targets the 5 tracks where TempoCNN overshoots by +0.5-0.6 BPM.
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import librosa

SAMPLE_RATE = 44100
TEMPOCNN_SR = 11025
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
                    '.wma', '.aiff', '.aif', '.opus', '.webm'}


def extract_bpm_from_filename(name):
    stem = os.path.splitext(name)[0]
    # Match "- 172" or "- 175.00" or "- 87.50" at end
    m = re.search(r'[\s\-]+(\d{2,3}(?:\.\d+)?)\s*$', stem)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d{2,3})\s*[Bb][Pp][Mm]', stem)
    if m:
        return float(m.group(1))
    return None


def load_ground_truth(audio_dir):
    entries = []
    audio_dir = Path(audio_dir)
    for f in sorted(os.listdir(audio_dir)):
        if Path(f).suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        bpm = extract_bpm_from_filename(f)
        if bpm is not None:
            # Double half-time
            if bpm < 100:
                bpm *= 2
            entries.append((str(audio_dir / f), int(bpm)))
    return entries


def _clamp(tempo, lo, hi):
    """Map tempo into valid range via harmonic multiples."""
    if lo <= tempo <= hi:
        return tempo
    for mult in [2.0, 0.5, 4.0, 0.25]:
        adjusted = tempo * mult
        if lo <= adjusted <= hi:
            return adjusted
    return tempo


def get_tempocnn_raw(audio_44k, predictor, lo=55, hi=215):
    """Run TempoCNN, return raw float BPM (with harmonic correction, not rounded)."""
    audio_11k = librosa.resample(audio_44k, orig_sr=SAMPLE_RATE, target_sr=TEMPOCNN_SR)
    audio_11k = audio_11k.astype(np.float32)

    predictions = np.array(predictor(audio_11k))
    if predictions.size == 0:
        return 0.0

    bins = np.linspace(30, 286, 256)
    avg = np.mean(predictions, axis=0)

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
        return _clamp(tempo, lo, hi)

    best_idx = int(np.argmax(masked))

    wlo = max(0, best_idx - 1)
    whi = min(len(masked), best_idx + 2)
    wp, wb = masked[wlo:whi], bins[wlo:whi]
    tempo = float(np.dot(wp, wb) / wp.sum()) if wp.sum() > 0 else float(bins[best_idx])
    return tempo


def onset_bpm(audio_44k):
    """Derive BPM from onset positions using autocorrelation of onset train."""
    from essentia.standard import OnsetRate

    detector = OnsetRate()
    onsets, rate = detector(audio_44k.astype(np.float32))

    if len(onsets) < 8:
        return None

    # Compute inter-onset intervals
    iois = np.diff(onsets)
    iois = iois[(iois > 0.15) & (iois < 1.5)]  # Filter obvious non-beat intervals

    if len(iois) < 4:
        return None

    # Build a histogram of IOIs to find the dominant period
    # Use fine bins for precision
    hist_bins = np.arange(0.15, 1.5, 0.005)
    hist, edges = np.histogram(iois, bins=hist_bins)

    # Smooth the histogram
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

    # Find the peak
    peak_idx = np.argmax(hist_smooth)
    peak_period = (edges[peak_idx] + edges[peak_idx + 1]) / 2

    bpm = 60.0 / peak_period
    return bpm


def onset_bpm_autocorr(audio_44k):
    """Derive BPM from onset positions using autocorrelation."""
    from essentia.standard import OnsetRate

    detector = OnsetRate()
    onsets, rate = detector(audio_44k.astype(np.float32))

    if len(onsets) < 8:
        return None

    # Create an onset signal (impulse train) at 100 Hz resolution
    fs = 100  # samples per second
    duration = onsets[-1] + 1.0
    signal = np.zeros(int(duration * fs))
    for t in onsets:
        idx = int(t * fs)
        if 0 <= idx < len(signal):
            signal[idx] = 1.0

    # Autocorrelation
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # positive lags only

    # Search for peaks in the BPM range 55-215
    min_lag = int(60.0 / 215 * fs)  # ~28 samples
    max_lag = int(60.0 / 55 * fs)   # ~109 samples

    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    search = corr[min_lag:max_lag+1]
    if len(search) == 0:
        return None

    best_lag = min_lag + np.argmax(search)
    bpm = 60.0 * fs / best_lag
    return bpm


def correct_with_onset(tempocnn_raw, onset_est):
    """
    Use onset estimate to correct TempoCNN when it's near an integer boundary.

    Logic: If TempoCNN rounds to X but the onset estimate suggests X-1 or X+1
    is more likely, check which rounding is closer to an integer.

    Skip correction for half-time detections (raw < 100) — onset estimates
    are unreliable at half the actual tempo.
    """
    if onset_est is None or tempocnn_raw < 55:  # skip when below min_bpm (half-time)
        return int(round(tempocnn_raw))

    tcnn_rounded = int(round(tempocnn_raw))

    # Harmonically align onset estimate to TempoCNN's range
    onset_aligned = onset_est
    for mult in [1.0, 2.0, 0.5, 4.0, 0.25]:
        candidate = onset_est * mult
        if abs(candidate - tempocnn_raw) < 10:
            onset_aligned = candidate
            break

    onset_rounded = int(round(onset_aligned))

    # If they agree, trust TempoCNN
    if tcnn_rounded == onset_rounded:
        return tcnn_rounded

    # They disagree — TempoCNN says X, onset says Y
    # Check: how far is TempoCNN's raw value from its rounded integer?
    tcnn_frac = abs(tempocnn_raw - tcnn_rounded)
    # If TempoCNN is very close to its integer (< 0.3), trust it
    if tcnn_frac < 0.3:
        return tcnn_rounded

    # TempoCNN is borderline (0.3-0.7 from integer) — use onset as tiebreaker
    # Round TempoCNN toward the onset estimate's integer
    if onset_rounded < tcnn_rounded:
        return int(np.floor(tempocnn_raw))
    else:
        return int(np.ceil(tempocnn_raw))


def main():
    audio_dir = "/home/hq/Music/03. Set Planning/Kawaii Karnival/Kawaii Karnival Set"
    entries = load_ground_truth(audio_dir)
    print(f"Found {len(entries)} tracks with ground truth\n")

    from essentia.standard import TensorflowPredictTempoCNN
    model_path = str(Path(__file__).parent / 'models' / 'deepsquare-k16-3.pb')
    predictor = TensorflowPredictTempoCNN(graphFilename=model_path)

    results_baseline = []
    results_hist = []
    results_autocorr = []

    print(f"{'Track':<55} {'True':>4} {'Raw':>6} {'Base':>4} {'Hist':>4} {'HstE':>6} {'Auto':>4} {'AutE':>6}")
    print("-" * 110)

    for filepath, true_bpm in entries:
        fname = os.path.basename(filepath)[:54]
        audio_44k, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

        # TempoCNN raw
        raw = get_tempocnn_raw(audio_44k, predictor)
        baseline = int(round(raw))

        # Onset estimates
        onset_hist = onset_bpm(audio_44k)
        onset_auto = onset_bpm_autocorr(audio_44k)

        # Corrected results
        corrected_hist = correct_with_onset(raw, onset_hist)
        corrected_auto = correct_with_onset(raw, onset_auto)

        def matches(detected_int, raw_float, true):
            """Exact match, allowing half-time via raw float doubling."""
            if detected_int == true:
                return True
            # Half-time: use raw float * 2 for better precision
            if int(round(raw_float * 2)) == true:
                return True
            return False

        baseline_ok = matches(baseline, raw, true_bpm)
        results_baseline.append(baseline_ok)
        results_hist.append(matches(corrected_hist, raw, true_bpm))
        results_autocorr.append(matches(corrected_auto, raw, true_bpm))

        hist_str = f"{onset_hist:.1f}" if onset_hist else "None"
        auto_str = f"{onset_auto:.1f}" if onset_auto else "None"

        mark_b = " " if baseline_ok else "X"
        mark_h = " " if matches(corrected_hist, raw, true_bpm) else "X"
        mark_a = " " if matches(corrected_auto, raw, true_bpm) else "X"

        print(f"{fname:<55} {true_bpm:>4} {raw:>6.1f} {baseline:>3}{mark_b} {corrected_hist:>3}{mark_h} {hist_str:>6} {corrected_auto:>3}{mark_a} {auto_str:>6}")

    n = len(entries)
    b = sum(results_baseline)
    h = sum(results_hist)
    a = sum(results_autocorr)
    print(f"\n{'='*110}")
    print(f"Baseline (TempoCNN):          {b}/{n} ({100*b/n:.1f}%)")
    print(f"Onset-corrected (histogram):  {h}/{n} ({100*h/n:.1f}%)")
    print(f"Onset-corrected (autocorr):   {a}/{n} ({100*a/n:.1f}%)")


if __name__ == '__main__':
    main()
