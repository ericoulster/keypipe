# KeyPipe

A fast CLI tool for batch musical key and BPM detection. Detects keys using KeyNet CNN (Camelot notation) and BPM using librosa beat tracking. Automatically renames files in Mixed In Key format.

## Features

- **Key detection**: CNN-based key detection (~73.5% accuracy)
- **BPM detection**: Beat tracking tempo estimation via librosa
- **Multi-format support**: MP3, WAV, FLAC, OGG, M4A, AIFF, and more
- **Batch processing**: Recursively scan directories
- **Parallel execution**: Process multiple files simultaneously
- **Mixed In Key format**: `song.mp3` → `song - 4A - (128 bpm).mp3`
- **Smart overwrite**: Automatically replaces existing tags rather than double-appending
- **GPU acceleration**: Optional CUDA support for key inference

## Installation

```bash
cd keypipe
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- `keynet.pt` model file (from MusicalKeyCNN)

## Usage

```bash
# Single file (key only, default)
python -m keypipe song.flac

# Directory (recursive)
python -m keypipe /path/to/music/

# BPM detection only
python -m keypipe --bpm /path/to/music/

# Both key AND BPM detection
python -m keypipe --bpm --key /path/to/music/

# Preview without renaming
python -m keypipe --dry-run /path/to/music/

# GPU acceleration (key detection)
python -m keypipe --device cuda /path/to/music/

# Keep existing tags (skip re-detection for already-tagged files)
python -m keypipe --no-overwrite /path/to/music/

# Verbose output
python -m keypipe -v /path/to/music/
```

## Options

| Flag | Description |
|------|-------------|
| `--bpm` | Detect BPM |
| `--key` | Detect key (default if --bpm not specified) |
| `--dry-run`, `-n` | Preview changes without renaming files |
| `--device {cpu,cuda}` | Device for key inference (default: cpu) |
| `--workers N`, `-j N` | Number of parallel workers (default: 4) |
| `--model PATH`, `-m PATH` | Path to key model checkpoint |
| `--min-bpm N` | Minimum BPM for detection (default: 55) |
| `--max-bpm N` | Maximum BPM for detection (default: 215) |
| `--no-overwrite` | Keep existing tags, skip re-detection |
| `--no-skip` | Process files even if already tagged |
| `--verbose`, `-v` | Print each file as processed |
| `--no-recursive` | Don't search subdirectories |

## Model Location

KeyPipe searches for `keynet.pt` in these locations (in order):

1. `--model` argument (if provided)
2. `./checkpoints/keynet.pt`
3. `./keynet.pt`
4. `../MusicalKeyCNN/checkpoints/keynet.pt`
5. `~/.keypipe/keynet.pt`

## Output Format

Files are renamed with key and/or BPM:

```
# Key only (default)
song.mp3 → song - 4A - .mp3

# BPM only
song.mp3 → song (128 bpm).mp3

# Both key and BPM
song.mp3 → song - 4A - (128 bpm).mp3
```

## Camelot Wheel Reference

```
Minor (A)              Major (B)
1A  = Ab minor         1B  = B major
2A  = Eb minor         2B  = F# major
3A  = Bb minor         3B  = Db major
4A  = F minor          4B  = Ab major
5A  = C minor          5B  = Eb major
6A  = G minor          6B  = Bb major
7A  = D minor          7B  = F major
8A  = A minor          8B  = C major
9A  = E minor          9B  = G major
10A = B minor          10B = D major
11A = F# minor         11B = A major
12A = Db minor         12B = E major
```

## Accuracy

### Key Detection (KeyNet)

| Method | Accuracy |
|--------|----------|
| KeyFinder | ~65% |
| KeyNet (this model) | ~73.5% |
| Mixed In Key | ~86% |

### BPM Detection (librosa)

BPM detection uses harmonic/percussive source separation (HPSS) to isolate the percussive signal, then derives per-frame tempo estimates via `librosa.feature.tempo` and takes the median across all frames. This is more robust than a single global estimate, especially on tracks with irregular rhythms or prominent harmonic content.

## License

MIT
