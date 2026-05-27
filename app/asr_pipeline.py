# app/asr_pipeline.py
# Whisper-based ASR pipeline for Tamil audio transcription

import os
import sys
import logging
import whisper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_config import ASR_CONFIG, BUFFER_CONFIG
from app.buffer_manager import BufferManager
from app.utils import validate_audio_file, save_transcript, split_audio, cleanup_chunks

logger = logging.getLogger(__name__)


class ASRPipeline:
    def __init__(self):
        self.model_id = ASR_CONFIG["model_id"]
        self.language = ASR_CONFIG["language"]
        self.task = ASR_CONFIG["task"]
        self.device = ASR_CONFIG["device"]
        self.model = None
        self.buffer = BufferManager(max_queue_size=BUFFER_CONFIG["max_queue_size"])
        logger.info(f"ASRPipeline initialized — model: {self.model_id}")

    def load_model(self):
        if self.model is not None:
            logger.info("Model already loaded — skipping")
            return

        logger.info(f"Loading Whisper model: {self.model_id}...")
        model_size = self.model_id.replace("openai/whisper-", "")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info(f"Whisper model loaded on {self.device}")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load_model()

        if not validate_audio_file(audio_path):
            raise FileNotFoundError(f"Invalid audio file: {audio_path}")

        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            fp16=False,
        )
        transcript = result["text"].strip()
        logger.info(f"Transcript: {transcript[:80]}...")
        return transcript

    def transcribe_with_buffer(self, audio_path: str) -> str:
        if not validate_audio_file(audio_path):
            raise FileNotFoundError(f"Invalid audio file: {audio_path}")

        chunk_seconds = BUFFER_CONFIG.get("chunk_duration", 30)
        sample_rate = BUFFER_CONFIG.get("sample_rate", 16000)

        try:
            chunk_paths = split_audio(audio_path, chunk_seconds=chunk_seconds, sample_rate=sample_rate)
        except Exception as e:
            logger.warning(f"Audio split failed ({e}) — falling back to direct transcription")
            return self.transcribe(audio_path)

        is_temp_split = bool(chunk_paths) and chunk_paths[0] != audio_path
        total_chunks = len(chunk_paths)
        max_q = self.buffer.max_queue_size or max(total_chunks, 1)
        num_batches = (total_chunks + max_q - 1) // max_q

        results = []
        try:
            self.buffer.clear()
            i = 0
            batch_idx = 0
            while i < total_chunks:
                batch_idx += 1
                enqueued = 0
                while i < total_chunks:
                    if self.buffer.add_chunk(chunk_paths[i]):
                        enqueued += 1
                        i += 1
                    else:
                        break

                if enqueued == 0:
                    logger.warning("Could not enqueue chunks in this batch — aborting")
                    break

                logger.info(f"Batch {batch_idx}/{num_batches}: processing {enqueued} chunk(s)")
                results.extend(self.buffer.process_all(self.transcribe))
        finally:
            if is_temp_split:
                cleanup_chunks(chunk_paths)

        if not results:
            logger.warning("No results from buffer — falling back to direct transcription")
            return self.transcribe(audio_path)

        return " ".join(r for r in results if r).strip()

    def transcribe_and_save(self, audio_path: str) -> tuple:
        transcript = self.transcribe_with_buffer(audio_path)
        saved_path = save_transcript(transcript)
        return transcript, saved_path
