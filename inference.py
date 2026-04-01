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
    BPM detector using essentia's RhythmExtractor2013.

    Usage:
        detector = BPMDetector()
        bpm = detector.detect('/path/to/song.mp3')
        print(bpm)  # e.g., 128
    """

    def __init__(self, min_bpm: int = 55, max_bpm: int = 215):
        """
        Initialize the BPM detector.

        Args:
            min_bpm: Minimum expected BPM (default: 55)
            max_bpm: Maximum expected BPM (default: 215)
        """
        from essentia.standard import RhythmExtractor2013
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._extractor = RhythmExtractor2013(
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

    def detect(self, audio_path: Union[str, Path]) -> int:
        """
        Detect the BPM of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            BPM as integer (rounded)
        """
        audio = self._load_mono(audio_path)
        bpm, _, _, _, _ = self._extractor(audio)
        tempo = float(bpm)

        # Clamp to valid range
        if tempo < self._min_bpm or tempo > self._max_bpm:
            for mult in [2.0, 0.5, 4.0, 0.25]:
                adjusted = tempo * mult
                if self._min_bpm <= adjusted <= self._max_bpm:
                    tempo = adjusted
                    break

        return int(round(tempo))

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
        bpm, _, beats_confidence, _, _ = self._extractor(audio)
        tempo = float(bpm)

        # essentia returns per-beat confidence; aggregate to a single score
        if len(beats_confidence) > 0:
            confidence = float(np.mean(beats_confidence))
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        # Clamp to valid range
        if tempo < self._min_bpm or tempo > self._max_bpm:
            for mult in [2.0, 0.5, 4.0, 0.25]:
                adjusted = tempo * mult
                if self._min_bpm <= adjusted <= self._max_bpm:
                    tempo = adjusted
                    break

        return int(round(tempo)), confidence


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
