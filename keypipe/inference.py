"""
Audio inference pipeline for KeyPipe.

Handles:
- Audio file loading (multi-format support via librosa)
- CQT spectrogram extraction
- Model inference
- Key detection
- BPM detection (via librosa beat tracking)
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import librosa

from .model import KeyNet, load_model
from .utils import index_to_camelot


# Audio preprocessing constants (from MusicalKeyCNN)
SAMPLE_RATE = 44100
N_BINS = 105
HOP_LENGTH = 8820
BINS_PER_OCTAVE = 24
FMIN = 65  # Hz


class KeyDetector:
    """
    Musical key detector using KeyNet CNN.

    Usage:
        detector = KeyDetector('/path/to/keynet.pt', device='cuda')
        key = detector.detect('/path/to/song.mp3')
        print(key)  # e.g., '4A'
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cpu'
    ):
        """
        Initialize the key detector.

        Args:
            model_path: Path to the KeyNet model checkpoint (.pt file)
            device: 'cpu' or 'cuda' for GPU inference
        """
        self.device = torch.device(device)
        self.model = load_model(str(model_path), self.device)
        self.model.eval()

    def preprocess(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess audio file to CQT spectrogram.

        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)

        Returns:
            Tensor of shape (1, 1, freq_bins, time_frames)
        """
        # Load audio with librosa (handles resampling and mono conversion)
        waveform_np, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # Compute CQT spectrogram
        cqt = librosa.cqt(
            waveform_np,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            fmin=FMIN
        )

        # Convert to magnitude and apply log scaling
        spec = np.abs(cqt)
        spec = np.log1p(spec)

        # Remove last two frequency bins (as in original MusicalKeyCNN)
        spec = spec[:, :-2] if spec.shape[1] > 2 else spec

        # Convert to tensor with batch and channel dimensions
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)

        return spec_tensor

    def predict_index(self, spec_tensor: torch.Tensor) -> int:
        """
        Run model inference on preprocessed spectrogram.

        Args:
            spec_tensor: Preprocessed spectrogram tensor

        Returns:
            Predicted class index (0-23)
        """
        spec_tensor = spec_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(spec_tensor)
            pred_idx = int(torch.argmax(outputs, dim=1).cpu().item())

        return pred_idx

    def detect(self, audio_path: Union[str, Path]) -> str:
        """
        Detect the musical key of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Camelot key string (e.g., '4A', '11B')
        """
        spec = self.preprocess(audio_path)
        pred_idx = self.predict_index(spec)
        return index_to_camelot(pred_idx)

    def detect_with_confidence(
        self,
        audio_path: Union[str, Path]
    ) -> tuple:
        """
        Detect key with confidence score.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (camelot_key, confidence)
        """
        spec = self.preprocess(audio_path)
        spec = spec.to(self.device)

        with torch.no_grad():
            outputs = self.model(spec)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(outputs, dim=1).cpu().item())
            confidence = float(probs[0, pred_idx].cpu().item())

        return index_to_camelot(pred_idx), confidence


class BPMDetector:
    """
    BPM detector using TempoCNN with onset-assisted correction.

    Uses essentia's TensorflowPredictTempoCNN with the deepsquare-k16 model.
    The CNN outputs per-patch probability distributions over 256 BPM bins
    (30-286 BPM); we average across patches and take the weighted peak.

    An onset autocorrelation step corrects borderline rounding errors:
    when TempoCNN's raw float is far from the nearest integer (>0.3),
    the onset-derived BPM acts as a tiebreaker for rounding direction.

    Usage:
        detector = BPMDetector()
        bpm = detector.detect('/path/to/song.mp3')
        print(bpm)  # e.g., 128
    """

    TEMPOCNN_SR = 11025  # TempoCNN expects 11025 Hz input
    BPM_MIN_BIN = 30
    BPM_MAX_BIN = 286
    NUM_BINS = 256
    _AUTOCORR_FS = 100  # impulse train sample rate for onset autocorrelation

    def __init__(self, min_bpm: int = 55, max_bpm: int = 215):
        """
        Initialize the BPM detector.

        Args:
            min_bpm: Minimum expected BPM (default: 55)
            max_bpm: Maximum expected BPM (default: 215)
        """
        from essentia.standard import TensorflowPredictTempoCNN
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm

        model_path = self._find_model()
        self._predictor = TensorflowPredictTempoCNN(graphFilename=str(model_path))
        self._bpm_bins = np.linspace(self.BPM_MIN_BIN, self.BPM_MAX_BIN, self.NUM_BINS)
        self._available = True

    @staticmethod
    def _find_model() -> Path:
        """Find the TempoCNN model file."""
        candidates = [
            Path(__file__).parent.parent / 'models' / 'deepsquare-k16-3.pb',
            Path(__file__).parent / 'models' / 'deepsquare-k16-3.pb',
            Path.home() / '.keypipe' / 'deepsquare-k16-3.pb',
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            'TempoCNN model not found. Place deepsquare-k16-3.pb in models/ '
            'or ~/.keypipe/'
        )

    @property
    def available(self) -> bool:
        """Check if BPM detection is available."""
        return self._available

    def _load_mono(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio as mono float32 at 44100 Hz."""
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)

    def _weighted_peak(self, probs: np.ndarray) -> tuple:
        """Find the peak BPM using a weighted average around the argmax.

        This gives sub-bin precision, which matters when the result is
        later doubled/halved to correct for half-time detection.
        """
        best_idx = int(np.argmax(probs))
        # Weighted average over a ±1 bin window around the peak
        lo = max(0, best_idx - 1)
        hi = min(len(probs), best_idx + 2)
        window_probs = probs[lo:hi]
        window_bins = self._bpm_bins[lo:hi]
        total = window_probs.sum()
        if total > 0:
            tempo = float(np.dot(window_probs, window_bins) / total)
        else:
            tempo = float(self._bpm_bins[best_idx])
        confidence = float(probs[best_idx])
        return tempo, confidence

    def _predict_bpm_raw(self, audio_44k: np.ndarray) -> tuple:
        """Run TempoCNN and return (raw_float_bpm, confidence).

        Uses a weighted average around the probability peak for sub-bin
        precision, then applies harmonic correction if outside range.
        """
        audio_11k = librosa.resample(
            audio_44k, orig_sr=SAMPLE_RATE, target_sr=self.TEMPOCNN_SR
        )
        predictions = np.array(self._predictor(audio_11k))
        if predictions.size == 0:
            return 0.0, 0.0

        avg = np.mean(predictions, axis=0)

        mask = (self._bpm_bins >= self._min_bpm) & (self._bpm_bins <= self._max_bpm)
        masked = avg.copy()
        masked[~mask] = 0.0

        if masked.sum() == 0:
            tempo, confidence = self._weighted_peak(avg)
            for mult in [2.0, 0.5, 4.0, 0.25]:
                adjusted = tempo * mult
                if self._min_bpm <= adjusted <= self._max_bpm:
                    return adjusted, confidence
            return tempo, confidence

        return self._weighted_peak(masked)

    def _onset_bpm(self, audio_44k: np.ndarray) -> Optional[float]:
        """Derive BPM from onset positions via autocorrelation.

        Builds an impulse train from detected onsets, autocorrelates it,
        and finds the dominant periodicity in the BPM range.
        Returns None if too few onsets are detected.
        """
        from essentia.standard import OnsetRate
        onsets, _ = OnsetRate()(audio_44k)

        if len(onsets) < 8:
            return None

        fs = self._AUTOCORR_FS
        duration = onsets[-1] + 1.0
        signal = np.zeros(int(duration * fs))
        for t in onsets:
            idx = int(t * fs)
            if 0 <= idx < len(signal):
                signal[idx] = 1.0

        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]

        min_lag = int(60.0 / self._max_bpm * fs)
        max_lag = int(60.0 / self._min_bpm * fs)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1

        search = corr[min_lag:max_lag + 1]
        if len(search) == 0:
            return None

        best_lag = min_lag + int(np.argmax(search))
        return 60.0 * fs / best_lag

    def _correct_with_onset(self, tempocnn_raw: float, onset_est: Optional[float]) -> int:
        """Use onset BPM to correct borderline TempoCNN rounding.

        When TempoCNN's raw float is >0.3 from the nearest integer,
        the onset estimate acts as a tiebreaker for floor vs ceil.
        Skipped for half-time detections (raw < min_bpm) where onset
        estimates are unreliable.
        """
        if onset_est is None or tempocnn_raw < self._min_bpm:
            return int(round(tempocnn_raw))

        tcnn_rounded = int(round(tempocnn_raw))

        # Harmonically align onset estimate to TempoCNN's range
        onset_aligned = onset_est
        for mult in [1.0, 2.0, 0.5, 4.0, 0.25]:
            candidate = onset_est * mult
            if abs(candidate - tempocnn_raw) < 10:
                onset_aligned = candidate
                break

        onset_rounded = int(round(onset_aligned))

        if tcnn_rounded == onset_rounded:
            return tcnn_rounded

        tcnn_frac = abs(tempocnn_raw - tcnn_rounded)
        if tcnn_frac < 0.3:
            return tcnn_rounded

        # TempoCNN is borderline — use onset as tiebreaker
        if onset_rounded < tcnn_rounded:
            return int(np.floor(tempocnn_raw))
        else:
            return int(np.ceil(tempocnn_raw))

    def detect(self, audio_path: Union[str, Path]) -> int:
        """
        Detect the BPM of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            BPM as integer (rounded, with onset correction)
        """
        audio = self._load_mono(audio_path)
        tempo_raw, _ = self._predict_bpm_raw(audio)
        onset_est = self._onset_bpm(audio)
        return self._correct_with_onset(tempo_raw, onset_est)

    def detect_with_confidence(
        self,
        audio_path: Union[str, Path]
    ) -> tuple:
        """
        Detect BPM with confidence score.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (bpm, confidence)
        """
        audio = self._load_mono(audio_path)
        tempo_raw, confidence = self._predict_bpm_raw(audio)
        onset_est = self._onset_bpm(audio)
        return self._correct_with_onset(tempo_raw, onset_est), confidence


def find_model_path(model_arg: Optional[str] = None) -> Path:
    """
    Find the model checkpoint file.

    Search order:
    1. Explicit model_arg if provided
    2. ./checkpoints/keynet.pt
    3. ../MusicalKeyCNN/checkpoints/keynet.pt
    4. ~/.keypipe/keynet.pt

    Args:
        model_arg: Explicitly specified model path

    Returns:
        Path to model file

    Raises:
        FileNotFoundError: If no model found
    """
    if model_arg:
        path = Path(model_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f'Model not found: {model_arg}')

    # Search paths
    search_paths = [
        Path('./checkpoints/keynet.pt'),
        Path('./keynet.pt'),
        Path(__file__).parent / 'checkpoints' / 'keynet.pt',  # Inside package
        Path(__file__).parent.parent / 'MusicalKeyCNN' / 'checkpoints' / 'keynet.pt',
        Path.home() / '.keypipe' / 'keynet.pt',
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        'No model found. Please specify with --model or place keynet.pt in:\n'
        '  - ./checkpoints/keynet.pt\n'
        '  - ./keynet.pt\n'
        '  - ~/.keypipe/keynet.pt\n'
        'Or download from the MusicalKeyCNN repository.'
    )
