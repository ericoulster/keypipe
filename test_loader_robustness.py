#!/usr/bin/env python3
"""Loader and short-audio robustness.

Regression tests for three real files that crashed or errored cryptically:
- a valid but ~1 s WAV kick sample -> KeyNet's pooling collapsed the time
  axis to zero and the forward pass raised (now tiled up to a floor);
- a corrupt WMA no decoder accepts -> audioread's NoBackendError stringifies
  to "" (now a clear AudioDecodeError);
- an HTML page saved with an .mp3 extension -> decoded to zero samples and
  librosa.resample raised ZeroDivisionError (now a clear AudioDecodeError).

Run: uv run --no-sync pytest test_loader_robustness.py -q
"""

import numpy as np
import pytest
import soundfile as sf

from keypipe.inference import MIN_TIME_FRAMES, SAMPLE_RATE, KeyDetector
from keypipe.utils import AudioDecodeError, _decode_with_audioread, load_audio_mono
from keypipe.model import load_model  # noqa: F401  (import guard)


def _model_path():
    from pathlib import Path
    import keypipe

    return Path(keypipe.__file__).parent / "checkpoints" / "keynet.pt"


def test_short_clip_gets_a_key_not_a_crash(tmp_path):
    """A ~1 s tone is far below the CNN's minimum window; it must analyze."""
    dur_s = 1.0
    t = np.linspace(0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False)
    wav = tmp_path / "short.wav"
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SAMPLE_RATE)

    kd = KeyDetector(_model_path(), device="cpu")
    key, conf = kd.detect_with_confidence(str(wav))
    assert key and key[-1] in ("A", "B")     # a real Camelot label
    assert 0.0 <= conf <= 1.0


def test_spec_tiled_to_minimum_width():
    kd = KeyDetector(_model_path(), device="cpu")
    tiny = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # 0.25 s
    spec = kd._spec_tensor(tiny)
    assert spec.shape[-1] >= MIN_TIME_FRAMES   # padded, so the CNN can run


def test_empty_decode_raises_clear_error(tmp_path):
    """Non-audio that a backend 'opens' but yields no samples: no ZeroDivision."""
    html = tmp_path / "notreally.mp3"
    html.write_bytes(b"<!DOCTYPE html>\n<html><body>nope</body></html>\n" * 50)
    with pytest.raises(AudioDecodeError):
        load_audio_mono(str(html), SAMPLE_RATE)


def test_zero_rate_stream_guarded(monkeypatch):
    """A corrupt container reporting sr=0 must not reach librosa.resample."""
    class _Fake:
        samplerate, channels = 0, 0
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read_data(self): return [b""]

    import audioread
    monkeypatch.setattr(audioread, "audio_open", lambda p: _Fake())
    with pytest.raises(AudioDecodeError):
        _decode_with_audioread("whatever", SAMPLE_RATE)
