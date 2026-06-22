# Key-detection algorithms vs KeyNet — research findings (2026-06-19)

Deep-research report (101 agents, 19 sources, 21/25 claims adversarially
confirmed). Question: is there a better key detector than our KeyNet CNN for an
EDM/doujin offline DJ tool? Short answer: **not for accuracy — KeyNet is already
near the ceiling. The real wins are confidence output, ensembling, and
genre-specific retraining.**

## The headline: accuracy is a dead end

- KeyNet's true GiantSteps score is **74.3-74.6% MIREX-weighted** (our "73.5%"
  was slightly low). The 2026 SOTA, **KeyMyna** (ViT + masked contrastive
  pre-training), reaches only **75.91%** — a **+1.3 point** gain.
- Five years (2018-2026) of research moved SOTA ~1.6 points. The top scores
  cluster at 74.3-75.9. **~74-76% weighted is a hard ceiling** on EDM, driven by
  genuine key ambiguity + GiantSteps label noise. KeyNet is near SOTA; replacing
  it for accuracy buys ~2 points at best, with real integration risk.

## Candidate scorecard (GiantSteps, MIREX-weighted)

| Model | Weighted | Open? | License | Integrable | Confidence out | Verdict |
|---|---|---|---|---|---|---|
| **KeyMyna** (ViT, masked-contrastive) | **75.91%** | weights on HF | **UNKNOWN** (CC-BY-4.0 claim REFUTED) | pip/ONNX unconfirmed | distribution | Best accuracy, +1.3pt only, license/packaging risk |
| InceptionKeyNet | 75.68% | preprint | unknown | unconfirmed | dist | Marginal; same caveats |
| **AllConv** (K&W 2018) | 74.6% | yes | — | — | softmax | Genre-agnostic KeyNet variant |
| **KeyNet / madmom** (ours) | 74.3% | yes | madmom BSD-ish (non-commercial!) | pip (but madmom BROKEN on py3.12/numpy2) | **24-dim softmax** | madmom = same model, but exposes probability dist |
| **S-KEY / STONE / Semi-TONE** (Deezer, self-supervised) | 72-73% | yes | S-KEY ~MIT (UNCONFIRMED, split vote) | pip (github.com/deezer/skey) | distribution | Parity accuracy, **label-free retrain** on our library |
| Foundation probes (MERT/MULE/Jukebox + linear) | 62-67% | yes | varies | heavy | embeddings | Downgrade for key; not worth it |
| **Essentia edma/edmm** (HPCP profiles) | ~0.68 | yes | **Apache-2.0** (profiles are params, not NC weights) | **pip, already shipped** | strength NOT reliable (REFUTED) | Below KeyNet but COMPLEMENTARY errors → best ensemble partner |
| DeepSquare/DeepTemp (Schreiber) | 58.5% RAW | yes | — | — | — | Downgrade; raw not weighted |
| Mixed In Key (commercial) | ~86-95% claimed | NO | closed | non-integrable | — | Reference only; can't ship |

## What actually matters (the three real levers)

1. **Confidence output.** madmom (= the KeyNet model, same authors) natively
   emits a 24-dim softmax key distribution; STONE/S-KEY/KeyMyna also expose
   distributions. BUT softmax ≠ calibrated on EDM, and the open question is
   whether ANY model escapes the EDM "ambiguity blind spot" — our AUC 0.54 may
   be intrinsic to the genre's harmonic ambiguity, not a KeyNet defect. Must be
   measured, not assumed. (Essentia's 'strength' is NOT a usable confidence —
   refuted.)
2. **Local/time-varying key.** NO model does it natively. The standard route is
   a sliding window over any model — exactly what we already built. KeyNet's own
   authors PROPOSED this and "left it to future work"; **we implemented what the
   literature only proposed.** We're not behind; the bottleneck is per-window
   accuracy, which the ceiling caps.
3. **Our genre is out-of-distribution for EVERYTHING.** Every model here is
   trained on Western/Beatport EDM. Japanese doujin is OOD → real-world accuracy
   is likely WORSE than benchmarks, and our "uniformly low confidence" is exactly
   the symptom of distribution shift. **The single biggest potential win is
   genre-specific retraining/fine-tuning**, for which the self-supervised,
   label-free S-KEY/STONE family is uniquely suited (we have a large unlabeled
   library + 415 filename-tagged tracks for validation).

## Recommended experiments (cheap → expensive)

1. **[~free] KeyNet + Essentia-edma ensemble.** We already ship essentia; edma
   is one `KeyExtractor(profileType='edma')` call, Apache-2.0. CNN vs HPCP
   templates have complementary errors → plausibly beats either alone (report
   says plausible but UNMEASURED). Validate against our 415 filename-tagged
   tracks. Near-zero cost, real upside. **Do this first.**
2. **[cheap] Confidence probe.** Does edma agreement-with-KeyNet, or KeyNet's
   own top1-top2 margin (already tested: AUC 0.54), or a CNN+template-disagreement
   signal, carry the modulation/ambiguity signal we need? Test on the tagged set.
3. **[medium] S-KEY drop-in.** MIT (verify!), pip, parity accuracy, distribution
   output. Test whether its probabilities are better-calibrated on our material.
4. **[big project, parked] Genre-specific self-supervised retraining** (S-KEY/
   STONE CPSD loss on our unlabeled library, validate on filename tags). The only
   path with real accuracy headroom on doujin, but a real undertaking — needs the
   SSD + the validation cohorts already planned.

## Hard caveats (from the adversarial pass)

- **Licenses unverified where it matters:** KeyMyna license REFUTED (unknown);
  S-KEY MIT was a split vote (verify the repo LICENSE); madmom is non-commercial
  (re-check before shipping); Essentia *core* is Apache-2.0 but its pretrained TF
  models are CC-BY-NC-SA — the edma/edmm PROFILES are Apache params, so
  KeyExtractor(edma) is the license-cleanest upgrade.
- **Don't compare raw to weighted** (differ ~6-7 pts). DeepSquare 58.5 is RAW.
- **Benchmark contamination:** GiantSteps train/test both Beatport EDM; edma was
  tuned on overlapping EDM → reported numbers OVERSTATE our doujin performance.
- madmom is attractive (probability output) but **broken on our py3.12/numpy2**
  stack — non-trivial to actually use.

## Bottom line

Stop looking for a more accurate KeyNet — there isn't one worth the integration
risk. Two cheap, high-value moves: (1) ensemble KeyNet+edma and measure it on our
tagged set; (2) check whether any model's probabilities fix the confidence gate.
The genuine long-term win is genre-specific (self-supervised) retraining, which
slots into the already-planned SSD + validation-cohort work.

Sources: KeyNet arXiv:1706.02921 / 1808.05340; KeyMyna arXiv:2604.10021;
MARBLE arXiv:2306.10548; STONE arXiv:2407.07408; S-KEY arXiv:2501.12907 +
github.com/deezer/skey; madmom features/key.py; Essentia KeyExtractor docs;
MIREX 2025 Audio Key Detection.
