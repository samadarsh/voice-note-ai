# app/buffer_manager.py
# Queue-based buffer system for handling audio chunks
# Prevents overflow and supports asynchronous processing

import queue
import logging
import threading

logger = logging.getLogger(__name__)


class BufferManager:
    """
    Queue-based buffer to handle chunked audio processing.
    Prevents overflow and supports async processing for
    real-time interaction as required by the submission guide.
    """

    def __init__(self, max_queue_size: int = 10):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self._lock = threading.Lock()
        logger.info(f"BufferManager initialized — max queue size: {max_queue_size}")

    def add_chunk(self, audio_chunk) -> bool:
        """
        Add an audio chunk to the queue.
        Returns True if added successfully, False if queue is full.
        """
        try:
            self.queue.put_nowait(audio_chunk)
            logger.info(f"Chunk added to buffer — queue size: {self.queue.qsize()}")
            return True
        except queue.Full:
            logger.warning("Buffer queue is full — chunk dropped")
            return False

    def get_chunk(self, timeout: float = 5.0):
        """
        Get the next audio chunk from the queue.
        Returns None if queue is empty after timeout.
        """
        try:
            chunk = self.queue.get(timeout=timeout)
            logger.info(f"Chunk retrieved from buffer — remaining: {self.queue.qsize()}")
            return chunk
        except queue.Empty:
            logger.info("Buffer queue is empty")
            return None

    def is_empty(self) -> bool:
        """Check if the buffer queue is empty"""
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if the buffer queue is full"""
        return self.queue.full()

    def size(self) -> int:
        """Return current number of items in the queue"""
        return self.queue.qsize()

    def clear(self):
        """Clear all items from the queue"""
        with self._lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        logger.info("Buffer queue cleared")

    def process_all(self, process_fn) -> list:
        """
        Process all chunks currently in the queue using process_fn.
        Returns list of results.
        """
        results = []
        while not self.is_empty():
            chunk = self.get_chunk(timeout=1.0)
            if chunk is not None:
                result = process_fn(chunk)
                results.append(result)
                self.queue.task_done()
        logger.info(f"Processed {len(results)} chunks from buffer")
        return results
