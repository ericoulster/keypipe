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
