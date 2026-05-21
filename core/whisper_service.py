"""Cached Whisper loading and transcription (single-shot + chunked)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import whisper

from config import (
    BUFFER_MAX_QUEUE,
    CHUNK_DURATION_SEC,
    CHUNK_SAMPLE_RATE,
    WHISPER_DEVICE,
    WHISPER_FP16,
)
from core.audio_chunker import cleanup_chunks, should_chunk, split_audio
from core.buffer_manager import BufferManager

logger = logging.getLogger(__name__)

_models: dict[tuple[str, str], whisper.Whisper] = {}


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot load a model or transcribe audio."""


@dataclass
class TranscribeResult:
    transcript: str
    asr_meta: dict[str, Any] = field(default_factory=dict)


def resolve_device() -> str:
    if WHISPER_DEVICE:
        return WHISPER_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_fp16(device: str) -> bool:
    if WHISPER_FP16:
        return device.startswith("cuda")
    return False


def get_model(model_name: str, device: str | None = None) -> whisper.Whisper:
    dev = device or resolve_device()
    key = (model_name, dev)
    if key not in _models:
        logger.info("Loading Whisper model '%s' on %s", model_name, dev)
        try:
            _models[key] = whisper.load_model(model_name, device=dev)
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load Whisper model '{model_name}': {exc}"
            ) from exc
    return _models[key]


def clear_model_cache() -> None:
    _models.clear()
    logger.info("Whisper model cache cleared")


def _transcribe_path(
    model: whisper.Whisper,
    audio_path: Path,
    *,
    language: str | None,
    initial_prompt: str | None,
    fp16: bool,
) -> str:
    options: dict[str, Any] = {"fp16": fp16}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt

    try:
        result = model.transcribe(str(audio_path), **options)
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper could not transcribe '{audio_path}': {exc}"
        ) from exc
    return result["text"].strip()


def transcribe_single(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> str:
    device = resolve_device()
    model = get_model(model_name, device)
    return _transcribe_path(
        model,
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
        fp16=resolve_fp16(device),
    )


def transcribe_with_buffer(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    device = resolve_device()
    model = get_model(model_name, device)
    fp16 = resolve_fp16(device)
    buffer = BufferManager(max_queue_size=BUFFER_MAX_QUEUE)

    def _run_chunk(chunk_path: Path) -> str:
        return _transcribe_path(
            model,
            chunk_path,
            language=language,
            initial_prompt=initial_prompt,
            fp16=fp16,
        )

    try:
        chunk_paths = split_audio(
            audio_path,
            chunk_seconds=CHUNK_DURATION_SEC,
            sample_rate=CHUNK_SAMPLE_RATE,
        )
    except Exception as exc:
        logger.warning("Audio split failed (%s) — direct transcription", exc)
        text = transcribe_single(audio_path, model_name, language, initial_prompt)
        return text, {"chunked": False, "chunk_count": 1, "model": model_name}

    is_temp_split = bool(chunk_paths) and chunk_paths[0] != audio_path
    total_chunks = len(chunk_paths)
    max_q = buffer.max_queue_size or max(total_chunks, 1)
    num_batches = (total_chunks + max_q - 1) // max_q

    results: list[str] = []
    try:
        buffer.clear()
        i = 0
        batch_idx = 0
        while i < total_chunks:
            batch_idx += 1
            enqueued = 0
            while i < total_chunks:
                if buffer.add_chunk(str(chunk_paths[i])):
                    enqueued += 1
                    i += 1
                else:
                    break

            if enqueued == 0:
                logger.warning("Could not enqueue chunks — aborting batch loop")
                break

            logger.info(
                "Batch %s/%s: processing %s chunk(s)",
                batch_idx,
                num_batches,
                enqueued,
            )
            results.extend(buffer.process_all(lambda p: _run_chunk(Path(p))))
    finally:
        if is_temp_split:
            cleanup_chunks(chunk_paths)

    if not results:
        logger.warning("No buffer results — falling back to direct transcription")
        text = transcribe_single(audio_path, model_name, language, initial_prompt)
        return text, {"chunked": False, "chunk_count": 1, "model": model_name}

    transcript = " ".join(r for r in results if r).strip()
    meta = {
        "chunked": is_temp_split,
        "chunk_count": total_chunks,
        "model": model_name,
        "device": device,
    }
    return transcript, meta


def transcribe_audio_full(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    initial_prompt: str | None = None,
    chunked: bool | None = None,
) -> TranscribeResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    use_chunks = chunked if chunked is not None else should_chunk(
        audio_path, CHUNK_DURATION_SEC
    )

    if use_chunks:
        transcript, meta = transcribe_with_buffer(
            audio_path, model_name, language, initial_prompt
        )
        meta["chunked"] = True
    else:
        transcript = transcribe_single(
            audio_path, model_name, language, initial_prompt
        )
        meta = {
            "chunked": False,
            "chunk_count": 1,
            "model": model_name,
            "device": resolve_device(),
        }

    logger.info(
        "Transcription complete (%s chars, chunked=%s)",
        len(transcript),
        meta.get("chunked"),
    )
    return TranscribeResult(transcript=transcript, asr_meta=meta)
