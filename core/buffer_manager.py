"""Queue-based buffer for bounded batch processing of audio chunks."""

import logging
import queue
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class BufferManager:
    def __init__(self, max_queue_size: int = 10) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self._lock = threading.Lock()
        logger.info("BufferManager initialized — max queue size: %s", max_queue_size)

    def add_chunk(self, audio_chunk: str) -> bool:
        try:
            self.queue.put_nowait(audio_chunk)
            logger.debug("Chunk enqueued — size: %s", self.queue.qsize())
            return True
        except queue.Full:
            logger.warning("Buffer queue full")
            return False

    def get_chunk(self, timeout: float = 5.0) -> str | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_empty(self) -> bool:
        return self.queue.empty()

    def clear(self) -> None:
        with self._lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        logger.debug("Buffer queue cleared")

    def process_all(self, process_fn: Callable[[str], str]) -> list[str]:
        results: list[str] = []
        while not self.is_empty():
            chunk = self.get_chunk(timeout=1.0)
            if chunk is not None:
                results.append(process_fn(chunk))
                self.queue.task_done()
        logger.info("Processed %s chunk(s) from buffer", len(results))
        return results
