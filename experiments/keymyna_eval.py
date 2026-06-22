"""KeyMyna (Myna-Vertical + MLP head) vs KeyNet on the user's library.

Local pipeline (no transformers/trust_remote_code): SimpleViT from the Myna
repo + the KeyMyna MLP head. Same DB/sampling (seed 7) as skey_eval.py, so the
80 tracks are identical -> direct KeyNet vs S-KEY vs KeyMyna comparison.

Run in keymyna-env from the Myna repo dir:
  cd experiments/external/myna
  /tmp/keymyna-env/bin/python ../../keymyna_eval.py [n_per_class]
"""

import math
import os
import re
import sqlite3
import sys
from argparse import Namespace

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from nnAudio.features.mel import MelSpectrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/external/myna")
from vit import SimpleViT  # noqa: E402

DB = os.path.expanduser("~/.local/share/keydup/library.db")
MYNA_WEIGHTS = "/tmp/myna-vertical.pth"
HEAD_WEIGHTS = "/tmp/keymyna-head.pth"
MYNA_SR = 16000
N_SAMPLES = 100000  # KeyMyna constant
MUSIC_TAG = re.compile(r"(\d{1,2}[AB])(?:\s+or\s+(\d{1,2}[AB]))?", re.I)

# KeyMyna's key order (inference.py) -> Camelot
KM_KEYS = ['c major','c minor','db major','db minor','d major','d minor',
           'eb major','eb minor','e major','e minor','f major','f minor',
           'gb major','gb minor','g major','g minor','ab major','ab minor',
           'a major','a minor','bb major','bb minor','b major','b minor']
NAME_TO_CAMELOT = {
    'c major':'8B','c minor':'5A','db major':'3B','db minor':'12A','d major':'10B',
    'd minor':'7A','eb major':'5B','eb minor':'2A','e major':'12B','e minor':'9A',
    'f major':'7B','f minor':'4A','gb major':'2B','gb minor':'11A','g major':'9B',
    'g minor':'6A','ab major':'4B','ab minor':'1A','a major':'11B','a minor':'8A',
    'bb major':'6B','bb minor':'3A','b major':'1B','b minor':'10A',
}
KM_TO_CAMELOT = [NAME_TO_CAMELOT[k] for k in KM_KEYS]


def get_n_frames(n_samples, sr, patch_time):
    mel = MelSpectrogram(sr=sr, n_mels=128, verbose=False)
    f = mel(torch.randn(1, 1, n_samples)).shape[-1]
    return math.floor(f / patch_time) * patch_time


def load_myna(n_frames):
    model = SimpleViT(image_size=(128, n_frames), channels=1, patch_size=(128, 2),
                      num_classes=50, dim=384, depth=12, heads=6, mlp_dim=1536)
    ckpt = torch.load(MYNA_WEIGHTS, map_location="cpu")
    filtered = {k: v for k, v in ckpt.items() if not k.startswith("linear_head")}
    model.load_state_dict(filtered, strict=False)
    model.linear_head = torch.nn.Identity()
    model.eval()
    head = torch.nn.Sequential(torch.nn.Linear(384, 2048), torch.nn.ReLU(),
                               torch.nn.Dropout(0.75), torch.nn.Linear(2048, 24))
    head.load_state_dict(torch.load(HEAD_WEIGHTS, map_location="cpu"))
    head.eval()
    return model, head


def tagged_keys(filename):
    m = MUSIC_TAG.search(re.sub(r"\d{2,3}\s*(bpm)?\s*$", "", filename, flags=re.I))
    if not m:
        return None
    keys = {m.group(1).upper()}
    if m.group(2):
        keys.add(m.group(2).upper())
    return keys


_mel = MelSpectrogram(sr=MYNA_SR, n_mels=128, verbose=False)


def predict(model, head, path, n_frames):
    signal, sr = torchaudio.load(path, backend="soundfile")
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != MYNA_SR:
        signal = T.Resample(sr, MYNA_SR)(signal)
    ms = _mel(signal)                                   # (1, 128, frames)
    nchunks = ms.shape[-1] // n_frames
    if nchunks == 0:
        return None, None
    ms = ms[:, :, :nchunks * n_frames]
    batch = torch.stack(torch.chunk(ms, nchunks, dim=2))  # (nchunks,1,128,n_frames)
    with torch.no_grad():
        embeds = model(batch)                           # (nchunks, 384)
        probs = head(embeds).mean(dim=0).softmax(dim=0).numpy()
    idx = int(probs.argmax())
    return KM_TO_CAMELOT[idx], float(probs[idx])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(7)
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT path, filename, key_camelot, key_confidence FROM tracks "
                       "WHERE status='done' AND key_camelot IS NOT NULL").fetchall()
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
    print(f"sample: {len(dual)} ambiguous, {len(single)} single-key", flush=True)

    n_frames = get_n_frames(N_SAMPLES, MYNA_SR, 2)
    model, head = load_myna(n_frames)

    def run(items, is_dual):
        kn_hit = km_hit = 0
        km_confs, kn_confs = [], []
        for it in items:
            try:
                km_key, km_conf = predict(model, head, it["path"], n_frames)
            except Exception as e:
                print("  skip:", it["filename"][:36], e); continue
            if km_key is None:
                continue
            kn_hit += it["key_camelot"] in it["truth"]
            km_hit += km_key in it["truth"]
            km_confs.append(km_conf); kn_confs.append(it["key_confidence"] or 0.0)
            if is_dual:
                flag = "" if km_key in it["truth"] else "  <-MISS"
                print(f"  tag {'/'.join(sorted(it['truth']))}: KeyNet={it['key_camelot']} "
                      f"KeyMyna={km_key} (c={km_conf:.2f}){flag}  {it['filename'][:34]}", flush=True)
        return kn_hit, km_hit, km_confs, kn_confs

    print("\n--- ambiguous ---", flush=True)
    d_kn, d_km, d_kmc, d_knc = run(dual, True)
    print("\n--- single-key ---", flush=True)
    s_kn, s_km, s_kmc, s_knc = run(single, False)

    def auc(a, b):
        return float("nan") if not a or not b else sum(x < y for x in a for y in b)/(len(a)*len(b))

    nd, ns = len(d_kmc), len(s_kmc)
    print("\n================ RESULTS ================")
    print(f"ACCURACY (matches a human-tagged key):")
    print(f"  ambiguous:  KeyNet {d_kn}/{nd}  KeyMyna {d_km}/{nd}")
    print(f"  single-key: KeyNet {s_kn}/{ns}  KeyMyna {s_km}/{ns}")
    print(f"  overall:    KeyNet {d_kn+s_kn}/{nd+ns}  KeyMyna {d_km+s_km}/{nd+ns}")
    print(f"\nCONFIDENCE separability (ambiguous vs single, AUC; 0.5=useless):")
    print(f"  KeyNet  AUC: {auc(d_knc, s_knc):.2f}")
    print(f"  KeyMyna AUC: {auc(d_kmc, s_kmc):.2f}")
    print(f"  KeyMyna conf: ambiguous mean {np.mean(d_kmc):.2f}, single mean {np.mean(s_kmc):.2f}")


if __name__ == "__main__":
    main()
