"""
Terminal UI for the 5-minute enrollment session.

Segment 1 — Pangram repetition (2 min)   → isolates timing consistency
Segment 2 — Natural typing    (2 min)    → captures real usage patterns
Segment 3 — Validation hold-out (1 min) → sets threshold, not used for training
"""
from __future__ import annotations

import sys
import time
import threading
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

from behavguard.collector.keystroke import KeyEvent, KeystrokeCollector
from behavguard.collector.keystroke import COMMON_DIGRAPHS

console = Console()

# Pangrams for Segment 1 — cover all 26 letters for digraph diversity
PANGRAMS = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "The five boxing wizards jump quickly",
    "Sphinx of black quartz judge my vow",
]

SEGMENT_CONFIG = [
    {
        'name':    'Segment 1 — Pangram Repetition',
        'label':   'seg1',
        'seconds': 120,
        'desc':    'Type the sentences below repeatedly. Focus on consistency, not speed.',
        'prompt':  PANGRAMS,
    },
    {
        'name':    'Segment 2 — Natural Typing',
        'label':   'seg2',
        'seconds': 120,
        'desc':    'Type freely — anything you like. Write a few sentences or describe your day.',
        'prompt':  None,
    },
    {
        'name':    'Segment 3 — Validation',
        'label':   'seg3',
        'seconds': 60,
        'desc':    'Continue typing naturally. This segment calibrates your personal threshold.',
        'prompt':  None,
    },
]


def _coverage(events: list[KeyEvent]) -> float:
    """Fraction of common digraphs seen so far."""
    seen = set()
    for i in range(1, len(events)):
        seen.add((events[i-1].key_id, events[i].key_id))
    return len(seen & COMMON_DIGRAPHS) / len(COMMON_DIGRAPHS)


def _wpm(events: list[KeyEvent]) -> float:
    if len(events) < 2:
        return 0.0
    dur = (events[-1].press_ts - events[0].press_ts) / 60.0
    return (len(events) / 5.0) / max(dur, 1e-6)


def run_segment(
    seg: dict,
    collector: KeystrokeCollector,
    all_events: list[KeyEvent],
) -> list[KeyEvent]:
    """Run a single enrollment segment with live stats."""

    console.print()
    console.print(Panel(
        f"[bold]{seg['name']}[/bold]\n{seg['desc']}",
        border_style='cyan',
    ))

    if seg['prompt']:
        for line in seg['prompt'][:3]:
            console.print(f"  [yellow]» {line}[/yellow]")
        console.print()

    console.print(f"  [dim]Duration: {seg['seconds']} seconds. Press keys to begin …[/dim]\n")

    target_keys = 400 if seg['label'] == 'seg1' else 300

    segment_events: list[KeyEvent] = []

    def on_evt(e: KeyEvent):
        segment_events.append(e)

    old_callback = collector._on_event
    collector._on_event = on_evt

    start = time.monotonic()
    end = start + seg['seconds']

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        timer_task = progress.add_task(f"[cyan]{seg['name']}", total=seg['seconds'])
        key_task = progress.add_task(f"[green]Keystrokes (target {target_keys})", total=target_keys)
        cov_task = progress.add_task("[blue]Digraph coverage", total=100)

        while time.monotonic() < end:
            elapsed = time.monotonic() - start
            progress.update(timer_task, completed=min(elapsed, seg['seconds']))
            progress.update(key_task, completed=min(len(segment_events), target_keys))
            cov_pct = _coverage(all_events + segment_events) * 100
            progress.update(cov_task, completed=min(cov_pct, 100))
            time.sleep(0.25)

    collector._on_event = old_callback

    n = len(segment_events)
    wpm = _wpm(segment_events)
    cov = _coverage(all_events + segment_events) * 100

    console.print(f"\n  ✓ Segment complete — [bold]{n}[/bold] keystrokes, "
                  f"[bold]{wpm:.0f}[/bold] WPM, "
                  f"digraph coverage [bold]{cov:.0f}%[/bold]\n")

    return segment_events


def run_enrollment(subject_id: str) -> dict[str, list[KeyEvent]]:
    """
    Full 5-minute enrollment session.
    Returns {'seg1': [...], 'seg2': [...], 'seg3': [...]}
    """
    console.print(Panel(
        f"[bold cyan]BehaveGuard — Enrollment[/bold cyan]\n"
        f"Subject: [bold]{subject_id}[/bold]\n\n"
        "This session takes approximately [bold]5 minutes[/bold].\n"
        "Your typing [italic]rhythm[/italic] is recorded — not the content.\n"
        "Three segments will build your behavioral fingerprint.",
        title="Welcome",
        border_style="green",
    ))

    collector = KeystrokeCollector()
    collector.start()

    results: dict[str, list[KeyEvent]] = {}
    all_events: list[KeyEvent] = []

    try:
        for seg in SEGMENT_CONFIG:
            input(f"\n  Press [ENTER] to start {seg['name']} …")
            events = run_segment(seg, collector, all_events)
            results[seg['label']] = events
            all_events.extend(events)
    finally:
        collector.stop()

    total = sum(len(v) for v in results.values())
    console.print(Panel(
        f"[bold green]Enrollment Complete![/bold green]\n"
        f"Total keystrokes collected: [bold]{total}[/bold]\n"
        f"Digraph coverage: [bold]{_coverage(all_events)*100:.0f}%[/bold]\n\n"
        "Your profile will now be trained …",
        border_style="green",
    ))

    return results
