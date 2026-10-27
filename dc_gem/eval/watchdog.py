"""Watchdog: timeout per generation call."""

from __future__ import annotations
import signal
from contextlib import contextmanager


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def handler(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s")
    # signal-based (main thread only)
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds) if seconds >= 1 else 1)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
