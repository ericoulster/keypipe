"""Fair test of Myna features: train our OWN head on Myna-Vertical embeddings of
the library's tagged tracks, then evaluate against KeyNet.

Avoids circularity: head is trained on SINGLE-tag tracks; the decisive eval is on
DUAL-tag MIK tracks ("X or Y"), which are independent of KeyNet. Compares
retrained-KeyMyna vs KeyNet (from DB) on the same dual-tags where the public pop
head scored 25/40 and KeyNet 35/40.

Caches Myna embeddings to /tmp so the head is cheap to re-tune.

  cd experiments/external/myna
  /tmp/myna-gpu/bin/python ../../keymyna_retrain.py
"""

import math
import os
import re
import sqlite3
import sys

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from nnAudio.features.mel import MelSpectrogram

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP + "/external/myna")
from vit import SimpleViT  # noqa: E402

DB = os.path.expanduser("~/.local/share/keydup/library.db")
MYNA_WEIGHTS = "/tmp/myna-vertical.pth"
CACHE = "/tmp/myna_embeds.pt"
MYNA_SR, N_SAMPLES = 16000, 100000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_HOLDOUT = 400, 80
MUSIC_TAG = re.compile(r"(\d{1,2}[AB])(?:\s+or\s+(\d{1,2}[AB]))?", re.I)

KM_KEYS = ['c major','c minor','db major','db minor','d major','d minor',
           'eb major','eb minor','e major','e minor','f major','f minor',
           'gb major','gb minor','g major','g minor','ab major','ab minor',
           'a major','a minor','bb major','bb minor','b major','b minor']
NAME_TO_CAMELOT = {
    'c major':'8B','c minor':'5A','db major':'3B','db minor':'12A','d major':'10B',
    'd minor':'7A','eb major':'5B','eb minor':'2A','e major':'12B','e minor':'9A',
    'f major':'7B','f minor':'4A','gb major':'2B','gb minor':'11A','g major':'9B',
    'g minor':'6A','ab major':'4B','ab minor':'1A','a major':'11B','a minor':'8A',
    'bb major':'6B','bb minor':'3A','b major':'1B','b minor':'10A'}
KM_TO_CAMELOT = [NAME_TO_CAMELOT[k] for k in KM_KEYS]
CAMELOT_TO_KM = {c: i for i, c in enumerate(KM_TO_CAMELOT)}


def tagged_keys(fn):
    m = MUSIC_TAG.search(re.sub(r"\d{2,3}\s*(bpm)?\s*$", "", fn, flags=re.I))
    if not m:
        return None
    k = {m.group(1).upper()}
    if m.group(2):
        k.add(m.group(2).upper())
    return k


def get_n_frames():
    mel = MelSpectrogram(sr=MYNA_SR, n_mels=128, verbose=False)
    f = mel(torch.randn(1, 1, N_SAMPLES)).shape[-1]
    return math.floor(f / 2) * 2


def build_myna(n_frames):
    m = SimpleViT(image_size=(128, n_frames), channels=1, patch_size=(128, 2),
                  num_classes=50, dim=384, depth=12, heads=6, mlp_dim=1536)
    ck = torch.load(MYNA_WEIGHTS, map_location="cpu")
    m.load_state_dict({k: v for k, v in ck.items() if not k.startswith("linear_head")}, strict=False)
    m.linear_head = torch.nn.Identity()
    return m.to(DEVICE).eval()


def main():
    n_frames = get_n_frames()
    mel = MelSpectrogram(sr=MYNA_SR, n_mels=128, verbose=False).to(DEVICE)
    myna = build_myna(n_frames)

    @torch.no_grad()
    def embed(path):
        sig, sr = torchaudio.load(path, backend="soundfile")
        if sig.shape[0] > 1:
            sig = sig.mean(0, keepdim=True)
        if sr != MYNA_SR:
            sig = T.Resample(sr, MYNA_SR)(sig)
        ms = mel(sig.to(DEVICE))
        nch = ms.shape[-1] // n_frames
        if nch == 0:
            return None
        ms = ms[:, :, :nch * n_frames]
        batch = torch.stack(torch.chunk(ms, nch, dim=2))
        return myna(batch).cpu()                       # (nchunks, 384)

    # gather tracks
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT path, filename, key_camelot FROM tracks "
                       "WHERE status='done' AND key_camelot IS NOT NULL").fetchall()
    rng = np.random.default_rng(7)
    singles, duals = [], []
    for r in rows:
        if not os.path.exists(r["path"]):
            continue
        keys = tagged_keys(r["filename"])
        if not keys:
            continue
        item = {"path": r["path"], "truth": keys, "kn": r["key_camelot"],
                "label": CAMELOT_TO_KM.get(next(iter(keys)))}
        (duals if len(keys) > 1 else singles).append(item)
    rng.shuffle(singles); rng.shuffle(duals)
    train = singles[:N_TRAIN]
    holdout = singles[N_TRAIN:N_TRAIN + N_HOLDOUT]
    test_dual = duals
    print(f"train {len(train)} single | holdout {len(holdout)} single | "
          f"test {len(test_dual)} dual-tag (independent) | device {DEVICE}", flush=True)

    cache = torch.load(CACHE) if os.path.exists(CACHE) else {}

    def get_embed(path):
        if path not in cache:
            cache[path] = embed(path)
        return cache[path]

    def materialize(items, label):
        for i, it in enumerate(items):
            try:
                it["emb"] = get_embed(it["path"])
            except Exception:
                it["emb"] = None
            if (i + 1) % 50 == 0:
                print(f"  embedded {label} {i+1}/{len(items)}", flush=True)
                torch.save(cache, CACHE)
        return [it for it in items if it.get("emb") is not None]

    print("extracting Myna embeddings...", flush=True)
    train = materialize(train, "train"); holdout = materialize(holdout, "holdout")
    test_dual = materialize(test_dual, "dual"); torch.save(cache, CACHE)

    # build per-chunk training tensors
    X = torch.cat([it["emb"] for it in train if it["label"] is not None])
    y = torch.cat([torch.full((it["emb"].shape[0],), it["label"]) for it in train if it["label"] is not None])
    X, y = X.to(DEVICE), y.to(DEVICE)
    print(f"training head on {X.shape[0]} chunk-embeddings, {len(set(y.tolist()))} keys present", flush=True)

    head = torch.nn.Sequential(torch.nn.Linear(384, 2048), torch.nn.ReLU(),
                               torch.nn.Dropout(0.75), torch.nn.Linear(2048, 24)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    head.train()
    for epoch in range(60):
        perm = torch.randperm(X.shape[0], device=DEVICE)
        for s in range(0, X.shape[0], 256):
            idx = perm[s:s + 256]
            opt.zero_grad(); loss = lossf(head(X[idx]), y[idx]); loss.backward(); opt.step()
    head.eval()

    @torch.no_grad()
    def predict(emb):
        logits = head(emb.to(DEVICE)).mean(0)
        return KM_TO_CAMELOT[int(logits.argmax())]

    # single-holdout: exact match to the single tag
    h_hit = h_kn = 0
    for it in holdout:
        p = predict(it["emb"])
        h_hit += p in it["truth"]; h_kn += it["kn"] in it["truth"]
    # dual-tag (independent): pred in tagged pair
    d_hit = d_kn = 0
    for it in test_dual:
        p = predict(it["emb"])
        d_hit += p in it["truth"]; d_kn += it["kn"] in it["truth"]

    nh, nd = len(holdout), len(test_dual)
    print("\n================ RESULTS ================")
    print("Retrained KeyMyna head (Myna features + our tags) vs KeyNet:")
    print(f"  single-tag holdout (exact): retrained {h_hit}/{nh}  KeyNet {h_kn}/{nh}")
    print(f"  dual-tag (independent MIK): retrained {d_hit}/{nd}  KeyNet {d_kn}/{nd}")
    print(f"  (pop-head KeyMyna scored 25/40 and KeyNet 35/40 on the earlier dual sample)")


if __name__ == "__main__":
    main()
