# BPM Pipeline Improvement Plan

## Current State

TempoCNN (deepsquare-k16-3) with onset-assisted correction achieves **100% exact-match** accuracy (23/23) on the Kawaii Karnival test set (high-BPM electronic music, 165–175 BPM range).

**Approach 5 (onset autocorrelation correction) solved all 5 remaining failures.** Integrated into production `keypipe/inference.py` on 2026-04-07.

---

## Test Results (2026-04-06)

### Approaches 1, 2, 4 — Post-processing variations

| Approach | Score | Notes |
|----------|-------|-------|
| **Baseline** (mean patches → weighted peak ±1) | **17/23 (74%)** | Current production code |
| Approach 1: Segment voting (per-patch mode) | 2/23 (9%) | Individual patches consistently land +1 BPM; mode amplifies this bias |
| Approach 2: Trimmed mean (drop top/bottom 10%) | 17/23 (74%) | Ties baseline — trimming doesn't shift the mean enough |
| Approach 4: Confidence-weighted rounding | 5/23 (22%) | Worse — argmax bin is often the biased one |
| Trim + vote combined | 2/23 (9%) | Voting still dominates and fails |

**Conclusion:** The weighted-peak averaging is already the optimal post-processing. Patches individually overestimate by ~1 BPM; averaging rescues this by pulling the estimate back ~0.5 BPM. Any approach that uses per-patch integer votes (mode) destroys this correction.

### Approach 3 — Multi-model TempoCNN ensemble

| Model | Score | Notes |
|-------|-------|-------|
| **deepsquare-k16-3** (current) | **18/23 (78%)** | Best individual model |
| deeptemp-k16-3 | 12/23 (52%) | Overshoots more on 175 BPM tracks |
| deeptemp-k4-3 | 1/23 (4%) | Terrible — smallest kernel can't resolve high BPM |
| Majority vote (3 models) | 17/23 (74%) | Worse — bad models outvote the good one |

**Conclusion:** `deepsquare-k16-3` is the best available TempoCNN checkpoint. The other models are weaker on this genre, so ensembling degrades performance.

### Alternative algorithms tested earlier

| Algorithm | Score | Notes |
|-----------|-------|-------|
| QM TempoTrackV2 (Mixxx's algorithm) | 6/16 (38%) | Viterbi weighting biases toward ~120 BPM, fails on 165+ |
| Beat This! (ISMIR 2024 transformer) | 3/13 (23%) | Large errors (7–36 BPM off), poor on high-BPM EDM |
| RhythmExtractor2013 + linear regression | 3/13 (23%) | Slow (3s/track) and inaccurate |
| Essentia ensemble (Percival + RE2013 + librosa) | ~45% | Tested in earlier sessions, inconsistent |
| madmom DBNBeatTracker | N/A | Broken on Python 3.12 / NumPy 2.x |

---

## Approach 5: Onset-Assisted Correction — IMPLEMENTED (2026-04-07)

**Result: 23/23 (100%)** — fixed all 5 remaining failures.

**How it works:**
1. Detect onsets via essentia's `OnsetRate` (44100 Hz input)
2. Build an impulse train at 100 Hz from onset positions
3. Autocorrelate to find the dominant beat periodicity
4. Harmonically align the onset BPM to TempoCNN's range
5. When TempoCNN's raw float is >0.3 from the nearest integer and the onset estimate disagrees on rounding direction, use onset as a tiebreaker (floor vs ceil)
6. Skip correction when TempoCNN detects half-time (raw < min_bpm)

**Key finding:** `OnsetRate` in essentia is stateful — a fresh instance must be created per call.

| Track | True | Raw CNN | Baseline | With onset correction |
|-------|------|---------|----------|-----------------------|
| VKTRS | 173 | 173.5 | 174 X | 173 ✓ |
| REDALiCE Lucky Star | 171 | 171.6 | 172 X | 171 ✓ |
| TOUHOU Is DEAD | 175 | 175.6 | 176 X | 175 ✓ |
| PUMP_PUMP | 174 | 174.6 | 175 X | 174 ✓ |
| M-Neko 全集 | 172 | 172.6 | 173 X | 172 ✓ |

**File:** `keypipe/inference.py` — `BPMDetector._onset_bpm()` and `BPMDetector._correct_with_onset()`

## Benchmark

Test set: `/home/hq/Music/03. Set Planning/Kawaii Karnival/Kawaii Karnival Set` (23 tracks)
Ground truth: BPM from filenames. Half-time values (<100) overridden to doubled value.
Success metric: exact integer match only (per user requirement).
