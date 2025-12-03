"""
Utility functions for KeyPipe.

Includes:
- Audio file discovery (recursive)
- Camelot wheel conversion
- BPM detection helpers
- Filename manipulation (key/BPM insertion, duplicate detection)
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
    '.wma', '.aiff', '.aif', '.opus', '.webm'
)

# Camelot key pattern for detecting already-tagged files
# Matches Mixed In Key format: "- 4A -" or "- 11B -" at end of filename
# Also matches: "- 4A", "[4A]", "(11B)" for compatibility
CAMELOT_PATTERN = re.compile(
    r'[\s\-_]*[\[\(]?(\d{1,2}[ABab])[\]\)]?[\s\-_]*$'
)

# More specific pattern for Mixed In Key format: " - 4A - "
MIK_PATTERN = re.compile(
    r'\s-\s(\d{1,2}[ABab])\s-\s*$'
)

# BPM pattern for detecting already-tagged files
# Matches formats like "(128 bpm)", "(128bpm)", "128 BPM", "[174 bpm]"
BPM_PATTERN = re.compile(
    r'[\s\-_]*[\[\(]?(\d{2,3})\s*[Bb][Pp][Mm][\]\)]?[\s\-_]*'
)


def find_audio_files(
    path: Path,
    recursive: bool = True,
    extensions: Tuple[str, ...] = AUDIO_EXTENSIONS
) -> List[Path]:
    """
    Find all audio files in a directory.

    Args:
        path: Directory or file path
        recursive: Search subdirectories if True
        extensions: Tuple of valid audio extensions

    Returns:
        List of Path objects for audio files
    """
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        return []

    if not path.is_dir():
        return []

    files = []
    pattern = '**/*' if recursive else '*'

    for ext in extensions:
        files.extend(path.glob(f'{pattern}{ext}'))
        # Also match uppercase extensions
        files.extend(path.glob(f'{pattern}{ext.upper()}'))

    # Remove duplicates and sort
    files = sorted(set(files))

    return files


def index_to_camelot(pred_idx: int) -> str:
    """
    Convert model output index (0-23) to Camelot notation.

    Index mapping:
        0-11: Minor keys (1A-12A)
        12-23: Major keys (1B-12B)

    Args:
        pred_idx: Model prediction index (0-23)

    Returns:
        Camelot string (e.g., '4A', '11B')
    """
    idx = (pred_idx % 12) + 1
    mode = 'A' if pred_idx < 12 else 'B'
    return f'{idx}{mode}'


def camelot_to_key_name(camelot: str) -> str:
    """
    Convert Camelot notation to standard key name.

    Args:
        camelot: Camelot string (e.g., '4A', '11B')

    Returns:
        Key name (e.g., 'F minor', 'A major')
    """
    # Camelot to key mapping
    camelot_map = {
        '1A': 'Ab minor', '1B': 'B major',
        '2A': 'Eb minor', '2B': 'F# major',
        '3A': 'Bb minor', '3B': 'Db major',
        '4A': 'F minor', '4B': 'Ab major',
        '5A': 'C minor', '5B': 'Eb major',
        '6A': 'G minor', '6B': 'Bb major',
        '7A': 'D minor', '7B': 'F major',
        '8A': 'A minor', '8B': 'C major',
        '9A': 'E minor', '9B': 'G major',
        '10A': 'B minor', '10B': 'D major',
        '11A': 'F# minor', '11B': 'A major',
        '12A': 'Db minor', '12B': 'E major',
    }
    return camelot_map.get(camelot.upper(), 'Unknown')


def has_camelot_tag(filepath: Path) -> Optional[str]:
    """
    Check if filename already contains a Camelot tag.

    Args:
        filepath: Path to audio file

    Returns:
        Existing Camelot tag if found, None otherwise
    """
    stem = filepath.stem
    match = CAMELOT_PATTERN.search(stem)

    if match:
        return match.group(1).upper()

    return None


def insert_key_in_filename(
    filepath: Path,
    key: str
) -> Path:
    """
    Insert key into filename using Mixed In Key format.

    Example: song.mp3 → song - 4A - .mp3

    Args:
        filepath: Original file path
        key: Camelot key string (e.g., '4A')

    Returns:
        New Path with key inserted
    """
    stem = filepath.stem
    suffix = filepath.suffix
    new_name = f'{stem} - {key} - {suffix}'
    return filepath.parent / new_name


def remove_camelot_tag(filepath: Path) -> Path:
    """
    Remove existing Camelot tag from filename.

    Handles both Mixed In Key format (" - 4A - ") and other formats.

    Args:
        filepath: Path with potential Camelot tag

    Returns:
        Path with tag removed
    """
    stem = filepath.stem
    suffix = filepath.suffix

    # Try Mixed In Key format first (" - 4A - ")
    new_stem = MIK_PATTERN.sub('', stem)

    # If no change, try general pattern
    if new_stem == stem:
        new_stem = CAMELOT_PATTERN.sub('', stem)

    new_stem = new_stem.rstrip(' -_')

    return filepath.parent / f'{new_stem}{suffix}'


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{mins}m {secs}s'
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f'{hours}h {mins}m'


def has_bpm_tag(filepath: Path) -> Optional[int]:
    """
    Check if filename already contains a BPM tag.

    Args:
        filepath: Path to audio file

    Returns:
        Existing BPM as int if found, None otherwise
    """
    stem = filepath.stem
    match = BPM_PATTERN.search(stem)

    if match:
        return int(match.group(1))

    return None


def insert_bpm_in_filename(filepath: Path, bpm: int) -> Path:
    """
    Insert BPM into filename.

    Example: song.mp3 → song (128 bpm).mp3

    Args:
        filepath: Original file path
        bpm: BPM value

    Returns:
        New Path with BPM inserted
    """
    stem = filepath.stem
    suffix = filepath.suffix
    new_name = f'{stem} ({bpm} bpm){suffix}'
    return filepath.parent / new_name


def remove_bpm_tag(filepath: Path) -> Path:
    """
    Remove existing BPM tag from filename.

    Args:
        filepath: Path with potential BPM tag

    Returns:
        Path with tag removed
    """
    stem = filepath.stem
    suffix = filepath.suffix

    new_stem = BPM_PATTERN.sub('', stem)
    new_stem = new_stem.rstrip(' -_')

    return filepath.parent / f'{new_stem}{suffix}'


def insert_key_and_bpm_in_filename(
    filepath: Path,
    key: Optional[str] = None,
    bpm: Optional[int] = None
) -> Path:
    """
    Insert key and/or BPM into filename.

    Format: song - 4A - (128 bpm).mp3

    Args:
        filepath: Original file path
        key: Camelot key string (e.g., '4A'), or None to skip
        bpm: BPM value, or None to skip

    Returns:
        New Path with key and/or BPM inserted
    """
    stem = filepath.stem
    suffix = filepath.suffix

    parts = [stem]

    if key:
        parts.append(f' - {key} -')

    if bpm:
        parts.append(f' ({bpm} bpm)')

    new_name = ''.join(parts) + suffix
    return filepath.parent / new_name
