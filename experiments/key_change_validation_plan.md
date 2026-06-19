# Key-change detection — validation plan (for later implementation)

Status: **research / planning only.** Do not download or build cohorts yet
(Eric is adding an SSD first). This captures what we learned, the cohorts to
build, their disk cost, and the measurement passes to run once storage exists.

## TL;DR

Windowed key-segment detection (`KeyDetector.detect_segments`) produces the
right keys when a track genuinely modulates, but we **cannot** reliably decide
*which* tracks to run it on, and it over-segments single-key tracks. Before
shipping any auto-detection we need cohorts with **clean single-key labels** and
**time-aligned key-change ground truth** — the user's filename tags give
neither cleanly. This plan picks those cohorts and defines the measurements.

## What we already established (2026-06-19, n=30+30 from the library)

- Segmentation is sound when invoked: detected keys overlap the human-tagged
  key(s) **22/23** triggered positives; As One 10A→10B→10A reproduced exactly
  (the file's own metadata agreed).
- **The confidence gate is dead.** top1 conf AUC 0.55, top1−top2 gap 0.54,
  entropy 0.54 — all ≈ random. KeyNet is uniformly low-confidence on dense
  electronic material whether or not a track modulates. No cheap full-track
  signal separates ambiguous from single-key. A broader cohort will NOT fix
  this for an electronic library; treat auto-gating as unavailable.
- **~50% false-alarm rate** on single-key tracks (spurious multi-key), mostly
  relative/adjacent flicker (1A↔1B, 8A↔7A).
- Cost (measured, GPU 4070 Ti): single-key 0.86 s/track; windowed 1.44 s/track
  (+0.58 s). Always-run on a 14k library ≈ +0.6 h (4 workers). Cost is NOT the
  blocker; reliability is.

## The product decision this unblocks

Three paths (Eric to decide once measured):
1. **On-demand + smoothing** — right-click "Detect key changes"; no auto gate.
   Eliminates false alarms by construction. Needs the collapse smoothing below.
2. **Always-run + strict acceptance** — windowed pass on every track, flag a
   modulation only when segmentation is robust. Affordable (+0.6 h) but the
   false-alarm fix is unproven.
3. **Drop** — single key per track (DJ-standard). Discards working segmentation.

The measurements below decide whether (2) is achievable (can strict acceptance
get false-alarm < ~5-10% while keeping change recall?), which is the only open
question between (1) and (2).

## Why a better cohort is needed (the label-quality failures)

- Positives ("X or Y" tags) conflate **relative ambiguity** (10A/10B — maybe no
  timed change) with **real modulation** (10B/7A). Must be split.
- Negatives (single tag) are **dirty** — a single tag ≠ provably single-key
  (the Gammer minimix "negative" really has 13 keys; scored as a false alarm).
- **No boundary truth** — tags give the key set, never *when* it changes, so we
  can't measure boundary accuracy or over-segmentation, the actual failure mode.

A good cohort therefore needs exact, time-aligned ground truth and clean
single-key negatives.

## Candidate datasets (researched 2026-06-19)

| Dataset | Genre | Tracks | Time-aligned key changes? | Audio access | License | ~Disk | Best for |
|---|---|---|---|---|---|---|---|
| **GiantSteps Key** | EDM | 604 × 2 min | No (single key) | Beatport DL script / Zenodo via mirdata | CC BY-SA 4.0 | **~0.85 GB** | Clean single-key false-alarm + confidence calibration, **genre-matched** |
| **GiantSteps+ / MTG-Beatport Key** | EDM | ~1.5k (extended) | No (single key; has "modal change" notes) | Zenodo / mirdata `beatport_key` | CC BY-SA 4.0 | ~2 GB | Larger genre-matched single-key set |
| **Isophonics** (Beatles/Queen/Carole King/Zweieck) | pop/rock | 179+20+14+18 | **Yes**, key `.lab` w/ on/offset times — BUT "key changes may be omitted in some files" (use Beatles with care) | annotations free; **audio not distributed** (must own/source) | annotations CC; audio not incl. | annotations <0.1 GB; audio ~3-8 GB if sourced | Real modulation recall + boundaries (caveat: incomplete change labels) |
| **Schubert Winterreise (SWD)** | classical lieder | 24 songs × 9 perf. | **Yes**, local key by 3 annotators, time-aligned, rich modulation | 2 performances CC-included; other 7 commercial | CC BY for the 2; others not | ~1-2 GB (the 2 CC perf.) | Gold-standard time-aligned local key with FREE audio; genre far from EDM |
| **McGill Billboard** | pop/rock '58-'91 | 890 slots / 742 songs (625 w/ key) | tonic only, **no mode**, limited change detail | features free; **audio not distributed** | annotations free | features small; audio if sourced | Secondary pop single-key cross-check (mode-less) |
| **FMA** (Free Music Archive) | 161 genres | 106k | No key labels | **free CC download** | CC (per-track) | medium ~22 GB / large ~93 GB / full ~879 GB | CC audio at scale to build synthetic cohorts / broad confidence sweep |
| **RWC Music DB** | pop/classical/jazz | ~315 | AIST annotations (chords; key partial) | licensed, media shipped (not free DL) | paid research license | n/a | Only if a license is already held |
| **Synthetic splices** (our own) | **our library** | unlimited | **Yes, exact** (we set keys+boundaries) | from existing library | n/a | ~0 (or a few GB if materialized) | Tuning smoothing + boundary accuracy + false alarms, genre-matched, controllable difficulty |

Notes:
- **Time-aligned local key + freely downloadable audio is rare.** Only SWD
  (classical) really qualifies; Isophonics has the labels but not the audio.
  Hence synthetic splices carry most of the boundary-accuracy load.
- Symbolic harmony corpora (DCML, When-in-Rome, TAVERN) have local keys but
  **no audio** → not usable for our audio model. Excluded.
- No EDM time-aligned key-change dataset appears to exist (2023-24 search dry).
  This is the genre gap synthetic splices fill.

## Selected cohorts (Eric's picks + recommended)

1. **GiantSteps Key** — clean, genre-matched single-key → trustworthy
   false-alarm rate + confidence calibration. (~0.85 GB, CC, mirdata one-liner.)
2. **Isophonics** (Beatles/Queen) — real modulations with boundaries (recall +
   boundary error), respecting the "incomplete change labels" caveat. Audio must
   be sourced.
3. **Scaled + split library tags** — all ~415 "X or Y", split relative-vs-
   distinct, stratified by folder; cheap real-world cross-check (coarse, no
   boundaries).
4. **Synthetic splices** (recommended addition) — the only genre-matched source
   of exact boundaries; primary tool for tuning the collapse.
5. **SWD** (optional) — free audio gold-standard for time-aligned local key, to
   sanity-check boundary logic independent of genre.

## Disk budget (for the SSD sizing)

- Minimal (GiantSteps + Isophonics annotations + SWD-CC + synthetic): **~5 GB**.
- + Beatles/Queen audio sourced + GiantSteps+: **~10-15 GB**.
- + FMA-large for CC-audio-at-scale synthetic: **+~93 GB**.
- Recommend provisioning **~30-50 GB** for the validation work excluding FMA;
  add ~100 GB only if we want FMA-scale synthetic/confidence sweeps.

## Tooling

- **mirdata** (`pip install mirdata`) — standardized download + loaders for
  GiantSteps Key, Beatport Key, McGill Billboard, etc. Use it instead of
  hand-rolling Beatport scripts.
- **mir_eval** — `mir_eval.key` (weighted score: exact/fifth/relative/parallel)
  and `mir_eval.segment` (boundary F-measure at tolerance, pairwise) for the
  metrics below. Don't reinvent these.

## Synthetic-splice cohort (design, when built)

- Source: high-confidence single-key tracks from the library (use detected key
  + filter for stable single-segment + agreement with filename tag).
- **Change set**: concatenate two clips of *known different* keys → ground truth
  = key A for [0,t), key B for [t,end). Vary t and the key relationship (fifth,
  relative, distant). Difficulty knob: hard cut vs N-second crossfade.
- **Single-key control**: concatenate two halves of the *same* track → ground
  truth = one key throughout (false-alarm test with certainty).
- Materialize a few hundred; store under a cohort dir on the new SSD.

## Measurement passes (metrics + procedure)

Run per cohort; all reusable from a single harness extending
`experiments/key_change_eval.py`:

1. **False-alarm rate** (GiantSteps, SWD single-key songs, synthetic
   same-track): % of provably single-key tracks reporting >1 segment after
   smoothing. **Target < 5-10%.**
2. **Change recall** (Isophonics, SWD, synthetic splices): % of true key
   changes with a detected boundary within ±T s (T = 10 s, mir_eval.segment).
3. **Boundary accuracy**: median |detected − true| boundary error; F-measure
   at ±10 s.
4. **Segment key accuracy** (mir_eval.key weighted): per-span key correctness,
   reporting exact vs fifth/relative/parallel leniency separately (relative
   confusion is expected and arguably acceptable for DJ use).
5. **Over-segmentation**: mean #detected vs #true segments.
6. **Strict-acceptance sweep**: grid over the acceptance rule (below) measuring
   the false-alarm / recall trade-off → decides whether always-run is viable.

## Collapse / smoothing parameters to tune

Current `detect_segments`: `window_s=30, hop_s=15, conf_floor=0.35,
min_segment_s=45`. Add and tune:
- **Run-length / median filter**: require K consecutive agreeing windows before
  switching key (kills single-window A/B/A flicker). Likely the biggest win.
- **Harmonic-neighbor smoothing**: treat relative/±1 flips as the dominant key
  unless sustained ≥ some span (the 1A↔1B, 8A↔7A false alarms).
- **Acceptance gate** (for always-run): only flag modulation if the secondary
  key occupies ≥ X% of the track AND ≥ K consecutive windows AND its mean
  confidence ≥ floor. Sweep X, K, floor against cohorts 1-3.

## Open questions / risks

- Auto-gating is unavailable for EDM regardless of cohort — confirm we accept
  on-demand for the user's genre even if always-run works for pop/classical.
- Isophonics change labels are incomplete (esp. Beatles) → recall is a lower
  bound there; weight SWD + synthetic for boundary truth.
- Relative-key confusion: decide product policy — is 10A↔10B a "change" worth
  flagging or noise? (For DJ harmonic mixing, relative is compatible, so maybe
  only flag non-neighbor changes.)
- Audio sourcing for Isophonics/McGill is on Eric (copyright); plan around SWD +
  synthetic if that stalls.

## Concrete next steps (once the SSD is in)

1. `pip install mirdata mir_eval` in the keydup dev env.
2. mirdata-download GiantSteps Key (+ Beatport Key) → false-alarm + confidence.
3. Build the synthetic-splice generator + cohort.
4. Extend the harness with mir_eval boundary/key metrics; run passes 1-5.
5. Run the strict-acceptance sweep (pass 6) → decide on-demand vs always-run.
6. Implement collapse smoothing; re-run; then wire the chosen UX in keydup.

## Sources

- GiantSteps Key: https://github.com/GiantSteps/giantsteps-key-dataset ·
  https://zenodo.org/records/4153506 ·
  https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/giantsteps_key.html
- Isophonics reference annotations: http://isophonics.net/content/reference-annotations
- Schubert Winterreise Dataset: https://zenodo.org/records/3968389 ·
  https://www.audiolabs-erlangen.de/resources/MIR/SWD
- McGill Billboard: https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/
- FMA: https://github.com/mdeff/fma
- mirdata: https://mirdata.readthedocs.io · mir_eval: https://craffel.github.io/mir_eval/
