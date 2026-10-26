"""Process-wide API backpressure shared by scene and detail worker threads."""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

_state_lock = threading.Lock()
_active = None


@contextmanager
def api_session(max_concurrency=2):
    """Limit all provider calls in one batch; nested/concurrent batches fail clearly."""
    global _active
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 32:
        raise ValueError('api_workers must be an integer between 1 and 32')
    semaphore = threading.BoundedSemaphore(max_concurrency)
    with _state_lock:
        if _active is not None:
            raise RuntimeError('Another model API batch is active in this process')
        _active = semaphore
    try:
        yield
    finally:
        with _state_lock:
            _active = None


@contextmanager
def api_slot(timeout):
    """Queue within the caller's total timeout, including all retries/stream reads."""
    with _state_lock:
        semaphore = _active
    started = time.monotonic()
    acquired = semaphore is None or semaphore.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError('Model API concurrency queue exceeded the request timeout')
    try:
        yield time.monotonic() - started if semaphore is not None else 0
    finally:
        if semaphore is not None:
            semaphore.release()
