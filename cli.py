"""
Command-line interface for KeyPipe.

Provides batch key and BPM detection with parallel processing,
progress display, and automatic file renaming.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from .inference import KeyDetector, BPMDetector, find_model_path
from .utils import (
    find_audio_files,
    has_camelot_tag,
    has_bpm_tag,
    insert_key_in_filename,
    insert_bpm_in_filename,
    insert_key_and_bpm_in_filename,
    remove_camelot_tag,
    remove_bpm_tag,
    format_duration,
    camelot_to_key_name,
    AUDIO_EXTENSIONS
)


def process_single_file(
    filepath: Path,
    key_detector: Optional[KeyDetector] = None,
    bpm_detector: Optional[BPMDetector] = None,
    dry_run: bool = False,
    skip_tagged: bool = True,
    overwrite: bool = False
) -> Tuple[Path, Optional[str], Optional[int], Optional[Path], Optional[str]]:
    """
    Process a single audio file for key and/or BPM detection.

    Args:
        filepath: Path to audio file
        key_detector: KeyDetector instance (None to skip key detection)
        bpm_detector: BPMDetector instance (None to skip BPM detection)
        dry_run: If True, don't actually rename
        skip_tagged: Skip files that already have tags
        overwrite: Replace existing tags

    Returns:
        Tuple of (original_path, detected_key, detected_bpm, new_path, error_message)
    """
    try:
        key = None
        bpm = None
        working_path = filepath

        # Check for existing key tag
        existing_key = has_camelot_tag(filepath) if key_detector else None
        existing_bpm = has_bpm_tag(filepath) if bpm_detector else None

        # Determine what to detect
        detect_key = key_detector is not None
        detect_bpm = bpm_detector is not None

        # Skip if already tagged (unless overwrite)
        if skip_tagged and not overwrite:
            key_skip = existing_key is not None if detect_key else True
            bpm_skip = existing_bpm is not None if detect_bpm else True
            if key_skip and bpm_skip:
                return (filepath, existing_key, existing_bpm, None, 'skipped (already tagged)')

        # Detect key if requested
        if detect_key and (not existing_key or overwrite):
            key = key_detector.detect(filepath)
        elif existing_key:
            key = existing_key

        # Detect BPM if requested
        if detect_bpm and (not existing_bpm or overwrite):
            bpm = bpm_detector.detect(filepath)
        elif existing_bpm:
            bpm = existing_bpm

        # Build new filename
        if overwrite:
            # Remove existing tags first
            if existing_key:
                working_path = remove_camelot_tag(working_path)
            if existing_bpm:
                working_path = remove_bpm_tag(working_path)

        # Insert new tags
        if key and bpm:
            new_path = insert_key_and_bpm_in_filename(working_path, key, bpm)
        elif key:
            new_path = insert_key_in_filename(working_path, key)
        elif bpm:
            new_path = insert_bpm_in_filename(working_path, bpm)
        else:
            new_path = working_path

        # Rename file if not dry run
        if not dry_run and new_path != filepath:
            if new_path.exists():
                return (filepath, key, bpm, new_path, 'target file exists')
            filepath.rename(new_path)

        return (filepath, key, bpm, new_path, None)

    except Exception as e:
        return (filepath, None, None, None, str(e))


def run_batch(
    paths: List[Path],
    key_detector: Optional[KeyDetector] = None,
    bpm_detector: Optional[BPMDetector] = None,
    workers: int = 4,
    dry_run: bool = False,
    skip_tagged: bool = True,
    overwrite: bool = False,
    verbose: bool = False
) -> dict:
    """
    Process multiple files with parallel execution.

    Args:
        paths: List of audio file paths
        key_detector: KeyDetector instance (None to skip key detection)
        bpm_detector: BPMDetector instance (None to skip BPM detection)
        workers: Number of parallel workers
        dry_run: Preview mode (no renaming)
        skip_tagged: Skip already-tagged files
        overwrite: Replace existing tags
        verbose: Print each file as processed

    Returns:
        Statistics dictionary
    """
    results = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'renamed': 0,
    }

    # Determine task description
    if key_detector and bpm_detector:
        desc = 'Detecting key+BPM'
    elif bpm_detector:
        desc = 'Detecting BPM'
    else:
        desc = 'Detecting keys'

    # Use ThreadPoolExecutor for GPU (shares model across threads)
    # Note: For CPU with heavy preprocessing, ProcessPoolExecutor might be faster
    # but requires serializing the model, which is complex
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                path,
                key_detector,
                bpm_detector,
                dry_run,
                skip_tagged,
                overwrite
            ): path
            for path in paths
        }

        with tqdm(total=len(paths), desc=desc, unit='file') as pbar:
            for future in as_completed(futures):
                original_path, key, bpm, new_path, error = future.result()

                if error:
                    if 'skipped' in error:
                        results['skipped'] += 1
                        if verbose:
                            tag_info = []
                            if key:
                                tag_info.append(key)
                            if bpm:
                                tag_info.append(f'{bpm}bpm')
                            tqdm.write(f'  SKIP: {original_path.name} [{", ".join(tag_info)}]')
                    else:
                        results['errors'] += 1
                        tqdm.write(f'  ERROR: {original_path.name}: {error}')
                else:
                    results['processed'] += 1
                    if new_path and new_path != original_path:
                        results['renamed'] += 1

                    if verbose or dry_run:
                        tag_info = []
                        if key:
                            tag_info.append(key)
                        if bpm:
                            tag_info.append(f'{bpm}bpm')
                        tag_str = ', '.join(tag_info)

                        if dry_run:
                            tqdm.write(f'  {original_path.name} → {tag_str}')
                        else:
                            tqdm.write(f'  {tag_str}: {new_path.name}')

                pbar.update(1)

    return results


def main(args: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='keypipe',
        description='Detect musical keys and BPM, tag audio filenames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  keypipe song.flac                    # Single file (key only)
  keypipe /path/to/music/              # Recursive directory
  keypipe --bpm /path/to/music/        # Detect BPM only
  keypipe --bpm --key /path/to/music/  # Detect both key and BPM
  keypipe --dry-run /path/to/music/    # Preview without renaming
  keypipe --device cuda /path/to/music # GPU acceleration
  keypipe --overwrite /path/to/music/  # Re-tag already tagged files

Supported formats: ''' + ', '.join(AUDIO_EXTENSIONS)
    )

    parser.add_argument(
        'path',
        type=Path,
        help='Audio file or directory to process'
    )

    parser.add_argument(
        '--bpm',
        action='store_true',
        help='Detect BPM (requires madmom)'
    )

    parser.add_argument(
        '--key',
        action='store_true',
        help='Detect key (default if --bpm not specified)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for key inference (default: cpu)'
    )

    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to key model checkpoint (auto-detect if not specified)'
    )

    parser.add_argument(
        '--min-bpm',
        type=int,
        default=55,
        help='Minimum BPM for detection (default: 55)'
    )

    parser.add_argument(
        '--max-bpm',
        type=int,
        default=215,
        help='Maximum BPM for detection (default: 215)'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without renaming files'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing tags in filenames'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Process files even if they already have tags'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print each file as it is processed'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )

    parsed = parser.parse_args(args)

    # Determine what to detect
    detect_key = parsed.key or not parsed.bpm  # Default to key if --bpm not specified
    detect_bpm = parsed.bpm

    # Validate input path
    if not parsed.path.exists():
        print(f'Error: Path not found: {parsed.path}', file=sys.stderr)
        sys.exit(1)

    # Initialize detectors
    key_detector = None
    bpm_detector = None

    if detect_key:
        try:
            model_path = find_model_path(parsed.model)
            print(f'Using key model: {model_path}')
            print(f'Loading key model on {parsed.device}...')
            key_detector = KeyDetector(model_path, device=parsed.device)
        except FileNotFoundError as e:
            print(f'Error: {e}', file=sys.stderr)
            sys.exit(1)

    if detect_bpm:
        print('Initializing BPM detector (madmom)...')
        bpm_detector = BPMDetector(min_bpm=parsed.min_bpm, max_bpm=parsed.max_bpm)
        if not bpm_detector.available:
            print('Error: madmom is not installed. Install with: pip install madmom', file=sys.stderr)
            sys.exit(1)

    # Find audio files
    print(f'Scanning: {parsed.path}')
    audio_files = find_audio_files(
        parsed.path,
        recursive=not parsed.no_recursive
    )

    if not audio_files:
        print('No audio files found.')
        sys.exit(0)

    print(f'Found {len(audio_files)} audio files')

    if parsed.dry_run:
        print('\n[DRY RUN - no files will be renamed]\n')

    # Process files
    start_time = time.time()

    results = run_batch(
        audio_files,
        key_detector=key_detector,
        bpm_detector=bpm_detector,
        workers=parsed.workers,
        dry_run=parsed.dry_run,
        skip_tagged=not parsed.no_skip,
        overwrite=parsed.overwrite,
        verbose=parsed.verbose
    )

    elapsed = time.time() - start_time

    # Print summary
    print(f'\n{"=" * 40}')
    print(f'Completed in {format_duration(elapsed)}')
    print(f'  Processed: {results["processed"]}')
    print(f'  Renamed:   {results["renamed"]}')
    print(f'  Skipped:   {results["skipped"]}')
    print(f'  Errors:    {results["errors"]}')
    print(f'{"=" * 40}')


if __name__ == '__main__':
    main()
