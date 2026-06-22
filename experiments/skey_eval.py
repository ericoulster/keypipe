"""Empirically compare S-KEY vs our KeyNet on the user's own library.

Ground truth = filename tags. KeyNet predictions + confidence come from the
live keydup DB (already computed). S-KEY is run fresh (its logits give a
confidence too). Measures:
  1. Accuracy: does each model's key match the human-tagged key(s)?
  2. Confidence separability: does S-KEY's confidence distinguish
     ambiguous ("X or Y") from single-key tracks better than KeyNet's
     (which was a useless AUC ~0.54)?

Run in the isolated skey-env:
  cd experiments/external/skey
  /tmp/skey-env/bin/python ../../skey_eval.py [n_per_class]
"""

import glob
import os
import re
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "external", "skey"))  # skey pkg
from skey.chromanet import ChromaNet  # noqa: E402
from skey.hcqt import VQT, CropCQT  # noqa: E402
from skey.key_detection import key_map, load_audio, load_checkpoint, load_model_components  # noqa: E402

DB = os.path.expanduser("~/.local/share/keydup/library.db")
MUSIC_TAG = re.compile(r"(\d{1,2}[AB])(?:\s+or\s+(\d{1,2}[AB]))?", re.I)
DUAL = re.compile(r"\d{1,2}[AB]\s+or\s+\d{1,2}[AB]", re.I)

# S-KEY key string -> Camelot
SKEY_TO_CAMELOT = {
    "A Major": "11B", "Bb Major": "6B", "B Major": "1B", "C Major": "8B",
    "C# Major": "3B", "D Major": "10B", "D# Major": "5B", "E Major": "12B",
    "F Major": "7B", "F# Major": "2B", "G Major": "9B", "G# Major": "4B",
    "B minor": "10A", "C minor": "5A", "C# minor": "12A", "D minor": "7A",
    "D# minor": "2A", "E minor": "9A", "F minor": "4A", "F# minor": "11A",
    "G minor": "6A", "G# minor": "1A", "A minor": "8A", "Bb minor": "3A",
}


def tagged_keys(filename):
    """Return the set of camelot keys a filename declares, or None."""
    m = MUSIC_TAG.search(re.sub(r"\d{2,3}\s*(bpm)?\s*$", "", filename, flags=re.I))
    if not m:
        return None
    keys = {m.group(1).upper()}
    if m.group(2):
        keys.add(m.group(2).upper())
    return keys


def infer_skey(hcqt, chromanet, crop_fn, audio, device):
    """Return (camelot_key, confidence) - confidence = softmax peak of the
    frame-averaged logits."""
    batch = audio.unsqueeze(0).to(device)
    with torch.no_grad():
        cropped = crop_fn(hcqt(batch), torch.zeros(1).to(device))
        logits = chromanet(cropped)               # (frames, 24)
        mean = torch.mean(logits, dim=0)
        probs = torch.softmax(mean, dim=0)
        idx = int(probs.argmax())
    return SKEY_TO_CAMELOT[key_map[idx]], float(probs[idx])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(7)

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT path, filename, key_camelot, key_confidence FROM tracks "
        "WHERE status='done' AND key_camelot IS NOT NULL"
    ).fetchall()

    dual, single = [], []
    for r in rows:
        if not os.path.exists(r["path"]):
            continue
        keys = tagged_keys(r["filename"])
        if not keys:
            continue
        (dual if len(keys) > 1 else single).append(dict(r) | {"truth": keys})
    rng.shuffle(dual); rng.shuffle(single)
    dual, single = dual[:n], single[:n]
    print(f"sample: {len(dual)} ambiguous (dual-tag), {len(single)} single-key", flush=True)

    ckpt = load_checkpoint(None)
    sr = ckpt["audio"]["sr"]
    device = torch.device("cpu")
    hcqt, chromanet, crop_fn = load_model_components(ckpt, device)

    def run(items, is_dual):
        kn_hit = sk_hit = 0
        sk_confs, kn_confs = [], []
        for it in items:
            try:
                audio = load_audio(it["path"], sr).to(device)
                sk_key, sk_conf = infer_skey(hcqt, chromanet, crop_fn, audio, device)
            except Exception as e:
                print("  skip:", it["filename"][:40], e)
                continue
            kn_key = it["key_camelot"]
            kn_hit += kn_key in it["truth"]
            sk_hit += sk_key in it["truth"]
            sk_confs.append(sk_conf)
            kn_confs.append(it["key_confidence"] or 0.0)
            if is_dual:
                flag = "" if sk_key in it["truth"] else "  <-MISS"
                print(f"  tag {'/'.join(sorted(it['truth']))}: KeyNet={kn_key} "
                      f"S-KEY={sk_key} (c={sk_conf:.2f}){flag}  {it['filename'][:38]}")
        return kn_hit, sk_hit, sk_confs, kn_confs

    print("\n--- ambiguous (dual-tag) ---", flush=True)
    d_kn, d_sk, d_skc, d_knc = run(dual, True)
    print("\n--- single-key ---", flush=True)
    s_kn, s_sk, s_skc, s_knc = run(single, False)

    def auc(amb, sng):  # P(ambiguous scores lower-confidence than single)
        if not amb or not sng:
            return float("nan")
        return sum(a < b for a in amb for b in sng) / (len(amb) * len(sng))

    nd, ns = len(d_skc), len(s_skc)
    print("\n================ RESULTS ================")
    print(f"ACCURACY (prediction matches a human-tagged key):")
    print(f"  on ambiguous:  KeyNet {d_kn}/{nd}  S-KEY {d_sk}/{nd}")
    print(f"  on single-key: KeyNet {s_kn}/{ns}  S-KEY {s_sk}/{ns}")
    print(f"  overall:       KeyNet {(d_kn+s_kn)}/{nd+ns}  S-KEY {(d_sk+s_sk)}/{nd+ns}")
    print(f"\nCONFIDENCE separability (ambiguous vs single-key, AUC; 0.5=useless):")
    print(f"  KeyNet conf AUC: {auc(d_knc, s_knc):.2f}  (prior measurement ~0.54)")
    print(f"  S-KEY  conf AUC: {auc(d_skc, s_skc):.2f}")
    print(f"  S-KEY conf: ambiguous mean {np.mean(d_skc):.2f}, single mean {np.mean(s_skc):.2f}")
    print(f"  agreement KeyNet==S-KEY: "
          f"{sum(1 for a,b in zip(d_skc+s_skc,[1]*0))} (see per-row)")


if __name__ == "__main__":
    main()
