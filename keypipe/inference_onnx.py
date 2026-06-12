"""ONNX BPM backend: TempoCNN without TensorFlow/essentia.

Replaces essentia's TensorflowPredictTempoCNN with a numerically
validated reimplementation (onnxruntime + numpy), and OnsetRate with
librosa onset detection. Everything downstream (weighted peak, harmonic
fold, onset-assisted correction) is inherited from BPMDetector
unchanged.

Input pipeline parity vs essentia (verified 2026-06-12):
- mel frames match TensorflowInputTempoCNN to ~1e-6 (extracted
  filterbank + symmetric hann + magnitude rFFT, zero-centered frames)
- per-patch probabilities match TensorflowPredictTempoCNN to ~3e-6
  (patches of 256 frames, hop 128, z-normalized per patch)
- the .pb -> .onnx conversion itself is exact to ~8e-7
"""

from pathlib import Path

import numpy as np

from keypipe.inference import BPMDetector

FRAME_SIZE = 1024
HOP_SIZE = 512
PATCH_FRAMES = 256
PATCH_HOP = 128


def _find_onnx_assets() -> tuple:
    base_candidates = [
        Path(__file__).parent / 'models',
        Path(__file__).parent.parent / 'models',
        Path.home() / '.keypipe',
    ]
    for base in base_candidates:
        model = base / 'tempocnn-deepsquare.onnx'
        bank = base / 'tempocnn-melbank.npy'
        if model.exists() and bank.exists():
            return model, bank
    raise FileNotFoundError(
        'tempocnn-deepsquare.onnx / tempocnn-melbank.npy not found in '
        'models/ or ~/.keypipe/'
    )


class OnnxTempoPredictor:
    """Drop-in for TensorflowPredictTempoCNN: audio @ 11025 Hz mono in,
    (n_patches, 256) softmax probabilities out."""

    def __init__(self):
        import onnxruntime as ort

        model_path, bank_path = _find_onnx_assets()
        self._session = ort.InferenceSession(
            str(model_path), providers=['CPUExecutionProvider']
        )
        self._bank = np.load(bank_path).astype(np.float32)  # (40, 513)
        self._window = np.hanning(FRAME_SIZE).astype(np.float32)

    def _mel_frames(self, audio_11k: np.ndarray) -> np.ndarray:
        """Zero-centered framing (first frame starts at -hop), matching
        essentia's FrameGenerator(startFromZero=False)."""
        n = len(audio_11k)
        # frame starts: -512 + 512*i while start < n
        n_frames = max(1, (n - 1) // HOP_SIZE + 2)
        pad_front = HOP_SIZE
        pad_back = (n_frames - 1) * HOP_SIZE - HOP_SIZE + FRAME_SIZE - n
        padded = np.concatenate([
            np.zeros(pad_front, np.float32),
            audio_11k.astype(np.float32),
            np.zeros(max(0, pad_back) + HOP_SIZE, np.float32),
        ])
        strided = np.lib.stride_tricks.sliding_window_view(
            padded, FRAME_SIZE
        )[::HOP_SIZE][:n_frames]
        spectra = np.abs(np.fft.rfft(strided * self._window, axis=1))
        return spectra.astype(np.float32) @ self._bank.T  # (T, 40)

    def __call__(self, audio_11k: np.ndarray) -> np.ndarray:
        frames = self._mel_frames(audio_11k)
        if frames.shape[0] < PATCH_FRAMES:
            frames = np.pad(frames, ((0, PATCH_FRAMES - frames.shape[0]), (0, 0)))
        n_patches = (frames.shape[0] - PATCH_FRAMES) // PATCH_HOP + 1
        patches = np.stack([
            frames[i * PATCH_HOP : i * PATCH_HOP + PATCH_FRAMES].T
            for i in range(n_patches)
        ]).astype(np.float32)
        mean = patches.mean(axis=(1, 2), keepdims=True)
        std = patches.std(axis=(1, 2), keepdims=True)
        patches = (patches - mean) / (std + 1e-8)
        return self._session.run(
            ['output:0'], {'input:0': patches[..., None]}
        )[0]


class OnnxBPMDetector(BPMDetector):
    """BPMDetector with onnxruntime instead of essentia/TensorFlow.

    Works on platforms without essentia wheels (Windows) and avoids
    bundling two deep-learning runtimes alongside torch.
    """

    def __init__(self, min_bpm: int = 55, max_bpm: int = 215):
        # deliberately not calling super().__init__ (it imports essentia)
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._predictor = OnnxTempoPredictor()
        self._bpm_bins = np.linspace(self.BPM_MIN_BIN, self.BPM_MAX_BIN, self.NUM_BINS)
        self._available = True

    def _detect_onsets(self, audio_44k: np.ndarray) -> np.ndarray:
        import librosa

        from keypipe.inference import SAMPLE_RATE

        return librosa.onset.onset_detect(
            y=audio_44k, sr=SAMPLE_RATE, units='time'
        )
