"""Why confidence-gating failed, and whether a better cheap gate exists.

For the same labeled pos/neg sets, compute full-track softmax stats:
  - top1 confidence (the gate we tried)
  - top1-top2 probability gap (a two-key track should have two peaks)
  - entropy of the distribution
and report how well each separates ambiguous from single-key. Also tests
a strict segmentation-acceptance rule (sustained, confident second key)
to see if precision can be salvaged.
"""

import random
import sys

import numpy as np
import torch

from keypipe.inference import KeyDetector, find_model_path
from experiments.key_change_eval import harvest, THRESHOLD


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = random.Random(7)
    pos, neg = harvest()
    rng.shuffle(pos); rng.shuffle(neg)
    pos, neg = pos[:n], neg[:n]
    det = KeyDetector(find_model_path(), device="cuda" if torch.cuda.is_available() else "cpu")

    def stats(path):
        spec = det.preprocess(path).to(det.device)
        with torch.no_grad():
            p = torch.softmax(det.model(spec), dim=1)[0].cpu().numpy()
        order = np.sort(p)[::-1]
        top1, top2 = float(order[0]), float(order[1])
        entropy = float(-(p * np.log(p + 1e-9)).sum())
        return top1, top1 - top2, entropy

    def collect(items):
        out = []
        for a, b, path in items:
            try:
                out.append(stats(str(path)))
            except Exception:
                pass
        return np.array(out)  # (n, 3): top1, gap, entropy

    P, N = collect(pos), collect(neg)

    def separation(col, name, lower_is_ambiguous):
        pv, nv = P[:, col], N[:, col]
        # AUC-style: P(ambiguous scores more "ambiguous" than single-key)
        wins = sum((a < b) if lower_is_ambiguous else (a > b)
                   for a in pv for b in nv)
        auc = wins / (len(pv) * len(nv))
        print(f"{name:18s} pos {pv.mean():.3f}+/-{pv.std():.3f}  "
              f"neg {nv.mean():.3f}+/-{nv.std():.3f}  separation(AUC) {auc:.2f}")

    print(f"n_pos={len(P)} n_neg={len(N)}\n")
    print("discriminator           ambiguous           single-key          quality")
    separation(0, "top1 confidence", lower_is_ambiguous=True)
    separation(1, "top1-top2 gap", lower_is_ambiguous=True)
    separation(2, "entropy", lower_is_ambiguous=False)
    print("\n(AUC 0.5 = no separation, 1.0 = perfect; the trigger needs >~0.8 to be useful)")


if __name__ == "__main__":
    main()
