"""Process-wide API and Blender backpressure shared by worker threads."""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

_state_lock = threading.Lock()
_active = None
_render_condition = threading.Condition()
_render_session_limit = None
_render_running = 0


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
def api_slot(timeout, check_cancelled=None):
    """Queue interruptibly within the caller's total request timeout."""
    with _state_lock:
        semaphore = _active
    started = time.monotonic()
    acquired = False
    try:
        while True:
            if check_cancelled is not None:
                check_cancelled()
            if semaphore is None:
                break
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Model API concurrency queue exceeded the request timeout')
            acquired = semaphore.acquire(timeout=min(.1, remaining))
            if acquired:
                break
        # Stop can race with slot release. Do not start the next request after
        # acquiring a slot, and release that slot even if this check raises.
        if check_cancelled is not None:
            check_cancelled()
        yield time.monotonic() - started if semaphore is not None else 0
    finally:
        if acquired:
            semaphore.release()


@contextmanager
def render_session(max_concurrency=1):
    """Bound Blender processes in one batch; the safe default remains one."""
    global _render_session_limit
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 8:
        raise ValueError('render_workers must be an integer between 1 and 8')
    with _render_condition:
        if _render_session_limit is not None:
            raise RuntimeError('Another Blender batch is active in this process')
        _render_session_limit = max_concurrency
    try:
        yield
    finally:
        with _render_condition:
            _render_session_limit = None
            _render_condition.notify_all()


@contextmanager
def render_slot(check_cancelled=None):
    """Wait interruptibly, then report queue seconds separately from render time.

    Direct run_job callers still serialize on the fallback semaphore. A running
    Blender keeps its slot until it exits; cancellation only stops queued work.
    """
    global _render_running
    started = time.monotonic()
    acquired = False
    try:
        while not acquired:
            if check_cancelled is not None:
                check_cancelled()
            with _render_condition:
                limit = _render_session_limit if _render_session_limit is not None else 1
                if _render_running < limit:
                    _render_running += 1
                    acquired = True
                else:
                    _render_condition.wait(.1)
        if check_cancelled is not None:
            check_cancelled()
        yield time.monotonic() - started
    finally:
        if acquired:
            with _render_condition:
                _render_running -= 1
                _render_condition.notify_all()
