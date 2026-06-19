"""Evaluate windowed key-change detection against human-labeled ground truth.

Ground truth comes from the library itself:
  - POSITIVES: filenames the user/source tagged "Xa or Xb" (judged ambiguous
    or modulating).
  - NEGATIVES: filenames with a single camelot tag and no "or".

Measures three things that define "works well":
  1. Trigger: does key confidence < THRESHOLD separate ambiguous from single-key?
  2. Segment correctness: when a positive triggers, do detected segment keys
     match the tagged pair?
  3. False alarms: do single-key tracks that happen to trigger report spurious
     multi-key segments?

Usage: python experiments/key_change_eval.py [n_per_class]
"""

import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from keypipe.inference import KeyDetector, find_model_path

MUSIC = Path("/home/hq/Music")
AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}
THRESHOLD = 0.6
DUAL = re.compile(r"(\d{1,2}[AB])\s+or\s+(\d{1,2}[AB])", re.I)
SINGLE = re.compile(r"[-\s](\d{1,2}[AB])[-\s]")


def norm(stem: str) -> str:
    """Collapse format/disc/track variants of the same track to one key."""
    s = re.sub(r"^\d+[\.\-)\s]+", "", stem)
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())[:40]


def harvest():
    positives, negatives = {}, {}
    for p in MUSIC.rglob("*"):
        if p.suffix.lower() not in AUDIO:
            continue
        m = DUAL.search(p.stem)
        if m:
            positives.setdefault(norm(p.stem), (m.group(1).upper(), m.group(2).upper(), p))
        elif SINGLE.search(p.stem) and " or " not in p.stem.lower():
            negatives.setdefault(norm(p.stem), (SINGLE.search(p.stem).group(1).upper(), None, p))
    return list(positives.values()), list(negatives.values())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = random.Random(7)
    pos, neg = harvest()
    rng.shuffle(pos); rng.shuffle(neg)
    pos, neg = pos[:n], neg[:n]
    print(f"deduped pool -> sampling {len(pos)} ambiguous, {len(neg)} single-key", flush=True)

    det = KeyDetector(find_model_path(), device="cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(items, is_positive):
        rows = []
        for a, b, path in items:
            try:
                key, conf = det.detect_with_confidence(str(path))
            except Exception as exc:
                print("  skip (load error):", Path(path).name[:40], exc)
                continue
            triggered = conf < THRESHOLD
            seg_keys = None
            if triggered:
                segs = det.detect_segments(str(path))
                seg_keys = [s["key"] for s in segs]
            rows.append((a, b, key, conf, triggered, seg_keys, Path(path).name))
        return rows

    print("\n--- ambiguous (positives) ---", flush=True)
    prows = evaluate(pos, True)
    print("\n--- single-key (negatives) ---", flush=True)
    nrows = evaluate(neg, False)

    # metrics
    p_trig = sum(r[4] for r in prows)
    n_trig = sum(r[4] for r in nrows)
    # segment match: detected distinct keys overlap the tagged pair
    matched = 0
    found_both = 0
    for a, b, key, conf, trig, seg_keys, name in prows:
        if not trig or not seg_keys:
            continue
        distinct = set(seg_keys)
        tagpair = {a, b}
        if distinct & tagpair:
            matched += 1
        if len(distinct) > 1 and distinct <= tagpair | {key}:
            found_both += 1
    # false alarms: negatives that triggered AND reported >1 distinct segment key
    fp = sum(1 for r in nrows if r[4] and r[5] and len(set(r[5])) > 1)

    print("\n================ RESULTS ================")
    print(f"POSITIVES (human-tagged ambiguous): n={len(prows)}")
    print(f"  triggered low-conf pass: {p_trig}/{len(prows)} ({p_trig/max(1,len(prows)):.0%})")
    print(f"  segments overlap tagged key(s): {matched}/{p_trig}")
    print(f"  segments span >1 key within tag set: {found_both}/{p_trig}")
    print(f"NEGATIVES (single-key): n={len(nrows)}")
    print(f"  triggered low-conf pass: {n_trig}/{len(nrows)} ({n_trig/max(1,len(nrows)):.0%})")
    print(f"  spurious multi-key (false alarm): {fp}/{len(nrows)}")
    pc = [r[3] for r in prows]; nc = [r[3] for r in nrows]
    print(f"confidence: positives mean {sum(pc)/max(1,len(pc)):.2f}, "
          f"negatives mean {sum(nc)/max(1,len(nc)):.2f}")
    # show a few positive segmentations
    print("\nsample positive segmentations:")
    for a, b, key, conf, trig, seg_keys, name in prows:
        if trig and seg_keys and len(set(seg_keys)) > 1:
            print(f"  tag {a}/{b}  detected {seg_keys}  conf {conf:.2f}  {name[:45]}")

    # characterize the false alarms: are negatives' spurious keys wild or
    # just adjacent wobble? print them.
    print("\nnegative (single-key) spurious segmentations:")
    for a, b, key, conf, trig, seg_keys, name in nrows:
        if trig and seg_keys and len(set(seg_keys)) > 1:
            print(f"  tag {a}  detected {seg_keys}  conf {conf:.2f}  {name[:45]}")


if __name__ == "__main__":
    main()
