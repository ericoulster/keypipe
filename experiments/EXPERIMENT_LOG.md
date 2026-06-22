# keypipe experiment log

## 2026-06-19 — Key-change detection validation

**Who:** Eric + Claude.
**Question:** Does windowed key-segment detection (KeyDetector.detect_segments)
work well enough to auto-flag modulating tracks in keydup?

**Ground truth:** the library itself. 415 tracks the user/source tagged
"Xa or Xb" in the filename = human-judged ambiguous/modulating (POSITIVES);
single-camelot-tag, no "or" = single-key (NEGATIVES). Deduped across
format/disc variants; sampled 30 + 30 (seed 7).
Harnesses: experiments/key_change_eval.py, key_gate_analysis.py.

**Results:**
- Segmentation, when run, finds the right keys: detected segments overlap the
  human-tagged key(s) **22/23** triggered positives. Sample segmentations are
  correct (4A/3A→[3A,4A], 6B/6A→[6A,6B], As One 10A→10B→10A).
- BUT the confidence gate does not work: positives mean conf 0.43, negatives
  0.47. **AUC 0.55** (≈ random).
- Tried cheaper/other gates on the full-track softmax: top1−top2 gap **AUC
  0.54**, entropy **AUC 0.54**. None separate. The model is broadly
  low-confidence on this library regardless of modulation, so no cheap
  full-track signal can decide which tracks to run the windowed pass on.
- False alarms: **50%** of single-key tracks produced spurious multi-key
  segments. Most are relative/adjacent flicker (1A↔1B, 8A↔7A); the worst are
  very-low-conf noise or actual minimixes (Gammer minimix → 13 keys @ 0.07,
  which is genuinely many songs).

**Conclusion:** The *segmentation* is sound, but **auto-detection is not
viable** — there is no signal to gate on, so "auto on low confidence" fires on
~70% of all tracks and mislabels half the single-key library. Two honest paths:
(A) on-demand detection (user invokes per track; they already know which tracks
are ambiguous — they tag them), plus collapse smoothing to kill neighbor
flicker; or (B) always-run the windowed pass on the whole library with strict
acceptance criteria (cost: hours per large library; FP-reduction unproven).
Recommended: **A**. The earlier "auto on low confidence" decision rested on an
n=3 spike (0.49/0.49 vs 0.73) that did not hold at n=60.

**Status:** keydup key-change commits are LOCAL only (not pushed); revise
before shipping. Next-phase cohort + measurement plan in
`key_change_validation_plan.md` (blocked on Eric adding an SSD; pending datasets:
GiantSteps Key, Isophonics, scaled-split library tags, synthetic splices, SWD).

## 2026-06-19 (later) — Neighbour-aware collapse: tried, MEASURED, insufficient

Live library had ~126 tracks auto-flagged multi-key; precision ~12% (only the
dual-tagged ones plausibly real). Added `are_harmonic_neighbors` + rewrote
`_collapse_windows` so neighbour wobble (relative / +-1) collapses and a key
change needs sustained evidence (distant: min_run=2 windows; neighbour:
min_run_neighbor=4). Synthetic unit cases all pass (wobble collapses, sustained
+1 outro survives, distant survives).

**But on REAL audio it barely helped: only 12/120 false alarms cleared, 108
still flagged.** Root cause (diagnostic, not fixable by collapse): KeyNet
produces DIFFERENT keys for different SECTIONS of a *single-key* track —
sustained blocks, not rapid wobble — because sections emphasise different
chords. Of 114 still-flagged: 34 neighbour-only, **80 contain a distant
transition** (e.g. a 5A-tagged track segmented 8B+5B — just the detector being
wrong differently across the track). Dropping neighbour modulations entirely
would clear only 34 more and would discard genuine sustained neighbour
modulations, which are **indistinguishable** from single-key section-colouring
with this detector.

**Conclusion:** local key segmentation is bottlenecked on PER-WINDOW DETECTOR
QUALITY, not collapse heuristics. No smoothing rescues it. This is the concrete
motivation for the better-detector research (deep-research running 2026-06-19).
Recommend: disable auto key-change flagging in keydup (it's ~90% wrong), keep
detect_segments for on-demand/future, revisit when a higher-accuracy /
better-calibrated per-window key detector is available.
