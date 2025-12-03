"""
Audio inference pipeline for KeyPipe.

Handles:
- Audio file loading (multi-format support via librosa)
- CQT spectrogram extraction
- Model inference
- Key detection
- BPM detection (via madmom)
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
    BPM detector using madmom's RNN beat processor.

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
        # Lazy import madmom to avoid import errors if not installed
        try:
            from madmom.features.beats import RNNBeatProcessor
            from madmom.features.tempo import TempoEstimationProcessor
            self._beat_processor = RNNBeatProcessor()
            self._tempo_processor = TempoEstimationProcessor(
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                fps=100
            )
            self._available = True
        except ImportError:
            self._available = False
            self._beat_processor = None
            self._tempo_processor = None

    @property
    def available(self) -> bool:
        """Check if madmom is available."""
        return self._available

    def detect(self, audio_path: Union[str, Path]) -> int:
        """
        Detect the BPM of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            BPM as integer (rounded)

        Raises:
            RuntimeError: If madmom is not installed
        """
        if not self._available:
            raise RuntimeError(
                'madmom is not installed. Install with: pip install madmom'
            )

        # Process audio through RNN beat processor
        beat_activations = self._beat_processor(str(audio_path))

        # Estimate tempo from beat activations
        tempos = self._tempo_processor(beat_activations)

        # Return the strongest tempo estimate (first one)
        if len(tempos) > 0:
            return int(round(tempos[0][0]))

        return 0

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
        if not self._available:
            raise RuntimeError(
                'madmom is not installed. Install with: pip install madmom'
            )

        beat_activations = self._beat_processor(str(audio_path))
        tempos = self._tempo_processor(beat_activations)

        if len(tempos) > 0:
            bpm = int(round(tempos[0][0]))
            confidence = float(tempos[0][1])
            return bpm, confidence

        return 0, 0.0


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
