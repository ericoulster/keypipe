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
    Ensemble BPM detector using three algorithms with majority voting.

    Runs PercivalBpmEstimator, RhythmExtractor2013, and librosa beat_track
    in parallel, then picks the tempo that at least two algorithms agree on.

    Usage:
        detector = BPMDetector()
        bpm = detector.detect('/path/to/song.mp3')
        print(bpm)  # e.g., 128
    """

    AGREEMENT_TOLERANCE = 3  # BPM — two estimates within this count as agreeing

    def __init__(self, min_bpm: int = 55, max_bpm: int = 215):
        """
        Initialize the BPM detector.

        Args:
            min_bpm: Minimum expected BPM (default: 55)
            max_bpm: Maximum expected BPM (default: 215)
        """
        from essentia.standard import PercivalBpmEstimator, RhythmExtractor2013
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._percival = PercivalBpmEstimator(
            minBPM=min_bpm,
            maxBPM=max_bpm,
            sampleRate=SAMPLE_RATE,
        )
        self._rhythm = RhythmExtractor2013(
            minTempo=min_bpm,
            maxTempo=max_bpm,
        )
        self._available = True

    @property
    def available(self) -> bool:
        """Check if BPM detection is available."""
        return self._available

    def _load_mono(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio as mono float32 at 44100 Hz for essentia."""
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)

    def _clamp(self, tempo: float) -> float:
        """Map tempo into the valid range via harmonic multiples."""
        if self._min_bpm <= tempo <= self._max_bpm:
            return tempo
        for mult in [2.0, 0.5, 4.0, 0.25]:
            adjusted = tempo * mult
            if self._min_bpm <= adjusted <= self._max_bpm:
                return adjusted
        return tempo

    def _run_percival(self, audio: np.ndarray) -> float:
        return self._clamp(float(self._percival(audio)))

    def _run_rhythm(self, audio: np.ndarray) -> float:
        bpm, _, _, _, _ = self._rhythm(audio)
        return self._clamp(float(bpm))

    def _run_librosa(self, audio: np.ndarray) -> float:
        _, y_perc = librosa.effects.hpss(audio)
        onset_env = librosa.onset.onset_strength(
            y=y_perc, sr=SAMPLE_RATE, aggregate=np.median
        )
        tempos = librosa.feature.tempo(
            onset_envelope=onset_env,
            sr=SAMPLE_RATE,
            start_bpm=(self._min_bpm + self._max_bpm) / 2,
            aggregate=None,
        )
        return self._clamp(float(np.median(tempos)))

    def _vote(self, candidates: list) -> float:
        """Pick the tempo that the most algorithms agree on.

        Two values "agree" if they are within AGREEMENT_TOLERANCE BPM.
        If a majority (2+) agree, return their mean.
        Otherwise return the first candidate (Percival).
        """
        tol = self.AGREEMENT_TOLERANCE
        best_group = []
        for i, a in enumerate(candidates):
            group = [a]
            for j, b in enumerate(candidates):
                if i != j and abs(a - b) <= tol:
                    group.append(b)
            if len(group) > len(best_group):
                best_group = group
        if len(best_group) >= 2:
            return float(np.mean(best_group))
        return candidates[0]

    def detect(self, audio_path: Union[str, Path]) -> int:
        """
        Detect the BPM of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            BPM as integer (rounded)
        """
        audio = self._load_mono(audio_path)
        candidates = [
            self._run_percival(audio),
            self._run_rhythm(audio),
            self._run_librosa(audio),
        ]
        tempo = self._vote(candidates)
        return int(round(tempo))

    def detect_with_confidence(
        self,
        audio_path: Union[str, Path]
    ) -> tuple:
        """
        Detect BPM with confidence score.

        Confidence is based on inter-algorithm agreement:
        - 1.0: all three agree within tolerance
        - 0.67: two agree
        - 0.33: no agreement (single-algorithm fallback)

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (bpm, confidence)
        """
        audio = self._load_mono(audio_path)
        candidates = [
            self._run_percival(audio),
            self._run_rhythm(audio),
            self._run_librosa(audio),
        ]
        tempo = self._vote(candidates)
        bpm = int(round(tempo))

        # Count how many algorithms agree with the result
        tol = self.AGREEMENT_TOLERANCE
        agreeing = sum(1 for c in candidates if abs(c - tempo) <= tol)
        confidence = agreeing / len(candidates)

        return bpm, confidence


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
