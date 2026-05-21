"""Split long audio into fixed-duration chunks for Whisper."""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMP_PREFIX = "asr_chunks_"


def validate_audio_file(filepath: Path) -> bool:
    if not filepath.exists():
        logger.error("Audio file not found: %s", filepath)
        return False
    if filepath.stat().st_size == 0:
        logger.error("Audio file is empty: %s", filepath)
        return False
    return True


def get_duration_sec(audio_path: Path) -> float:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def should_chunk(audio_path: Path, chunk_seconds: int) -> bool:
    if not validate_audio_file(audio_path):
        return False
    try:
        return get_duration_sec(audio_path) > chunk_seconds
    except Exception as exc:
        logger.warning("Could not read duration (%s) — using single-shot ASR", exc)
        return False


def split_audio(
    audio_path: Path,
    chunk_seconds: int = 30,
    sample_rate: int = 16000,
) -> list[Path]:
    if not validate_audio_file(audio_path):
        raise FileNotFoundError(f"Cannot split — invalid audio file: {audio_path}")

    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    duration_ms = len(audio)
    chunk_ms = max(1, int(chunk_seconds * 1000))

    if duration_ms <= chunk_ms:
        logger.info(
            "Audio is %.1fs ≤ %ss — no split needed",
            duration_ms / 1000,
            chunk_seconds,
        )
        return [audio_path]

    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_PREFIX)
    num_chunks = (duration_ms + chunk_ms - 1) // chunk_ms

    chunk_paths: list[Path] = []
    for i in range(num_chunks):
        start = i * chunk_ms
        end = min(start + chunk_ms, duration_ms)
        chunk = audio[start:end]
        chunk_path = Path(temp_dir) / f"chunk_{i:03d}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunk_paths.append(chunk_path)

    logger.info(
        "Split %.1fs audio into %s chunk(s) of ≤%ss",
        duration_ms / 1000,
        num_chunks,
        chunk_seconds,
    )
    return chunk_paths


def cleanup_chunks(chunk_paths: list[Path]) -> int:
    if not chunk_paths:
        return 0

    removed = 0
    parent_dirs: set[Path] = set()

    for path in chunk_paths:
        path_str = str(path)
        if _TEMP_PREFIX not in path_str or not path.exists():
            continue
        try:
            path.unlink()
            removed += 1
            parent_dirs.add(path.parent)
        except OSError as exc:
            logger.warning("Could not remove chunk %s: %s", path, exc)

    for directory in parent_dirs:
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError:
            pass

    if removed:
        logger.info("Cleaned up %s temporary chunk file(s)", removed)
    return removed
