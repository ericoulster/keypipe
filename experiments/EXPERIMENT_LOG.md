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

## 2026-06-19 (later) — S-KEY empirical eval on the user's library: NEGATIVE

Branch `skey-experiment`. Ran Deezer's S-KEY (MIT, ChromaNet, github.com/
deezer/skey) vs our KeyNet on 80 filename-tagged tracks from the live library
(40 dual-tag "X or Y" ambiguous + 40 single-tag), KeyNet preds pulled from the
keydup DB, S-KEY run fresh in an isolated env. Harness: experiments/skey_eval.py.

**Accuracy (prediction matches a human-tagged key):**
- ambiguous: KeyNet 35/40, S-KEY 34/40 (tied — and these MIK-style "X or Y"
  tags are independent of KeyNet, so this is the clean comparison)
- single-key: KeyNet 36/40, S-KEY 31/40 (KeyNet ahead, but single tags MAY be
  keypipe-generated → partly circular for KeyNet; discount this)
- overall: KeyNet 71/80 (89%), S-KEY 65/80 (81%)
→ S-KEY is parity-to-slightly-worse on OUR library. (Benchmark parity ~72-73%
  holds; the high % here is the lenient "matches either tagged key" metric.)

**Confidence (the real question — does S-KEY fix the dead gate?): NO.**
- KeyNet conf AUC 0.68 on this sample (better than the prior 0.54 — weak signal,
  not nothing), S-KEY conf AUC **0.58** (worse). S-KEY softmax peak ≈ 0.07
  (uniform = 0.042) — its distribution is NEARLY FLAT. Expected: S-KEY's
  self-supervised CPSD objective optimises transposition-equivariance, NOT
  calibrated key probabilities, so its "confidence" is meaningless.

**Verdict:** S-KEY out-of-the-box is NOT worth adopting — no accuracy gain, and
its confidence is worse-than-useless. Confirms the research framing: S-KEY's only
value is as a label-free RETRAINING substrate for genre adaptation (untested
here, a real project). The naive "swap in a newer model" option is now closed
with data. KeyNet stays.

## 2026-06-20 — KeyMyna (Myna-Vertical + MLP head) on the user's library

Branch `skey-experiment`. Myna base = MIT, ViT-S ~22M, 122MB weights (gdrive).
Ran LOCALLY (no transformers/trust_remote_code) via the authors' SimpleViT +
the downloadable MLP head, on the SAME 80 filename-tagged tracks as S-KEY (seed
7). Harness: experiments/keymyna_eval.py.

**CONFOUND (critical):** the only downloadable head is the **Billboard (pop)**
head — the README's GiantSteps(EDM) and Billboard links are the SAME gdrive id
(1pgBB...), and inference.py names it keymyna-bb.pth. The EDM head (KeyMyna-GS)
is NOT obtainable. So this is a POP head on doujin/EDM = an unfair accuracy test.

- Accuracy (matches tag): KeyNet 71/80 (89%), KeyMyna **52/80 (65%)**. Worse,
  but EXPECTED for a pop head on EDM — INCONCLUSIVE for KeyMyna-GS.
- Confidence: KeyMyna AUC **0.68** (= KeyNet), peaky (ambiguous mean 0.52,
  single 0.67), low-conf correlates with errors. **Meaningful** — unlike S-KEY's
  flat-useless 0.07. So Myna+head gives a usable probability distribution.

**The decisive untested experiment:** extract Myna embeddings for our tagged
tracks, train the (tiny 384->2048->24) head on a train split, evaluate on a
holdout. This tests whether Myna's self-supervised FEATURES are better for OUR
library than KeyNet — the genre-adaptation question — independent of which head
shipped. The head is trivial; cost is embedding extraction (ViT on CPU).

**Net:** can't crown KeyMyna a replacement (no EDM head, pop-head test poor on
doujin), but its features + a retrained head remain the most promising path,
and its confidence is real. KeyNet stays for now.

## 2026-06-20 — KeyMyna head RETRAINED on our library: KeyNet still wins (decisive)

Trained our own 384->2048->24 head on Myna-Vertical embeddings of 400 single-tag
tracks (18,499 chunk-embeddings, all 24 keys present, 60 epochs, GPU), evaluated
on a single-tag holdout AND on 192 INDEPENDENT dual-tag MIK tracks. Harness:
experiments/keymyna_retrain.py.

- single-tag holdout (exact): retrained KeyMyna **45/79 (57%)** vs KeyNet 63/79 (80%)
- dual-tag, independent MIK truth: retrained KeyMyna **130/192 (68%)** vs KeyNet **163/192 (85%)**

**KeyNet wins by ~17-23 points even with a fair, library-trained head.** Eric's
point stands and is validated: doujin/j-core is NOT exotic — KeyNet handles it
fine (85% dual-tag agreement). The gap isn't distribution shift; Myna's FROZEN
self-supervised features just don't encode key as well as KeyNet's task-specific
end-to-end features on electronic music. Consistent with the research finding
that frozen representation models + a probe (MERT/MULE/Jukebox = 62-67%)
underperform dedicated key CNNs (74.3%); KeyMyna only reaches 75.91% with its
own pretraining + a GiantSteps-trained head, which doesn't transfer here.
Confidence: KeyMyna AUC 0.68 = KeyNet 0.68 — no edge there either.

## CONCLUSION OF THE DETECTOR-REPLACEMENT INVESTIGATION

No replacement beats KeyNet for this use case, proven with data on the user's
own library:
- No drop-in is more accurate (research: ~74-76% ceiling; KeyMyna +1.3 on
  benchmark only).
- Ensembles won't help (no gating signal, AUC 0.54-0.68; matches Eric's prior
  TempoCNN-ensemble failure).
- S-KEY out-of-the-box: parity-to-worse accuracy, useless confidence.
- KeyMyna/Myna even with a fair retrained head: ~17pt WORSE than KeyNet here.
**KeyNet stays.** The real levers are product-side (on-demand key changes, no
auto-flag — already shipped) and accepting the ~74% electronic-key ceiling +
genuine harmonic ambiguity (which the user's own "X or Y" tags confirm is real
in the music, not a model defect).
