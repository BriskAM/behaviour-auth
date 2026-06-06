"""
Keystroke event collector.
Captures raw key press/release timestamps without storing key content.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pynput import keyboard


@dataclass
class KeyEvent:
    """Raw keystroke event — key identity stripped to category only."""
    press_ts: float          # epoch seconds
    release_ts: Optional[float]
    key_category: str        # 'alphanum' | 'symbol' | 'special'
    key_id: str              # normalised key name for digraph tracking (NOT the char)


# High-frequency English digraphs (by pair key_id)
COMMON_DIGRAPHS: set[tuple[str, str]] = {
    ('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
    ('r', 'e'), ('o', 'n'), ('e', 'n'), ('a', 't'), ('e', 'd'),
    ('h', 'a'), ('t', 'o'), ('o', 'r'), ('i', 't'), ('e', 's'),
    ('s', 't'), ('i', 's'), ('n', 'd'), ('a', 's'), ('a', 'r'),
    ('o', 'u'), ('t', 'e'), ('n', 't'), ('n', 'g'), ('t', 'i'),
}

DIGRAPH_FREQUENCY: dict[tuple[str, str], float] = {d: 1.0 for d in COMMON_DIGRAPHS}


def _categorise(key: keyboard.Key | keyboard.KeyCode) -> tuple[str, str]:
    """Return (key_category, key_id) without storing the actual character."""
    if isinstance(key, keyboard.KeyCode):
        ch = key.char or ''
        if ch.isalnum():
            return 'alphanum', ch.lower()
        return 'symbol', 'sym'
    # Special keys (shift, ctrl, backspace, …)
    return 'special', key.name


class KeystrokeCollector:
    """
    Non-blocking keystroke collector.
    Captures timing events; strips character content for privacy.
    """

    def __init__(self, on_event: Optional[Callable[[KeyEvent], None]] = None):
        self._events: list[KeyEvent] = []
        self._pending: dict[str, float] = {}   # key_id -> press_ts
        self._lock = threading.Lock()
        self._on_event = on_event
        self._listener: Optional[keyboard.Listener] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def drain(self) -> list[KeyEvent]:
        """Return and clear the collected event buffer."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    # ------------------------------------------------------------------ #
    # Internal callbacks
    # ------------------------------------------------------------------ #

    def _on_press(self, key) -> None:
        ts = time.perf_counter()
        cat, kid = _categorise(key)
        with self._lock:
            self._pending[kid] = ts

    def _on_release(self, key) -> None:
        ts = time.perf_counter()
        cat, kid = _categorise(key)
        with self._lock:
            press_ts = self._pending.pop(kid, None)
            if press_ts is None:
                return
            evt = KeyEvent(
                press_ts=press_ts,
                release_ts=ts,
                key_category=cat,
                key_id=kid,
            )
            self._events.append(evt)
            if self._on_event:
                self._on_event(evt)
