"""
Live scoring display for an active typing session.
Shows anomaly score, WPM, and window verdicts in real time.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from behavguard.collector.keystroke import KeyEvent

console = Console()

VERDICT_STYLE = {
    'legitimate': '[green]✓ LEGITIMATE[/green]',
    'uncertain':  '[yellow]⚠ UNCERTAIN[/yellow]',
    'anomaly':    '[red]✗ ANOMALY[/red]',
    'pending':    '[dim]… PENDING[/dim]',
}


class ScoreDisplay:
    """Real-time terminal display for scoring results."""

    def __init__(self, subject_id: str, threshold: float):
        self.subject_id = subject_id
        self.threshold = threshold
        self._window_results: deque = deque(maxlen=20)
        self._current_wpm = 0.0
        self._session_events = 0
        self._start = time.monotonic()

    def update(self, window_result: dict, wpm: float, n_events: int) -> None:
        self._window_results.append(window_result)
        self._current_wpm = wpm
        self._session_events = n_events

    def render(self) -> Table:
        elapsed = time.monotonic() - self._start
        mm, ss = divmod(int(elapsed), 60)

        # Header stats
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", min_width=22)
        grid.add_column(style="bold", min_width=14)
        grid.add_row("Subject", self.subject_id)
        grid.add_row("Session time", f"{mm:02d}:{ss:02d}")
        grid.add_row("Events collected", str(self._session_events))
        grid.add_row("WPM", f"{self._current_wpm:.0f}")
        grid.add_row("Threshold", f"{self.threshold:.4f}")

        # Window history
        history = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        history.add_column("Window", justify="right", min_width=6)
        history.add_column("Score", justify="right", min_width=8)
        history.add_column("Raw", justify="right", min_width=10)
        history.add_column("Verdict", min_width=18)

        for i, r in enumerate(self._window_results, 1):
            score_str = f"{r.get('anomaly_score', 0):.4f}"
            raw_str = f"{r.get('raw_decision', 0):.4f}"
            verdict_str = VERDICT_STYLE.get(r.get('verdict', 'pending'), '[dim]…[/dim]')
            history.add_row(str(i), score_str, raw_str, verdict_str)

        # Anomaly rate bar
        if self._window_results:
            n_anomaly = sum(1 for r in self._window_results if r.get('verdict') == 'anomaly')
            rate = n_anomaly / len(self._window_results)
            bar_len = 30
            filled = int(rate * bar_len)
            color = 'green' if rate < 0.2 else ('yellow' if rate < 0.6 else 'red')
            bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/{color}] {rate*100:.0f}%"
            anomaly_row = Text.assemble("Anomaly rate  ", bar)
        else:
            anomaly_row = Text("[dim]Waiting for first window …[/dim]")

        layout = Table.grid(padding=(0, 0))
        layout.add_row(Panel(grid, title="Session Stats", border_style="cyan"))
        layout.add_row(Panel(history, title="Window History", border_style="blue"))
        layout.add_row(Panel(anomaly_row, title="Session Health", border_style="magenta"))
        return layout


def print_session_summary(session_result: dict) -> None:
    verdict = session_result.get('session_verdict', 'unknown')
    style = {'legitimate': 'green', 'uncertain': 'yellow', 'impostor': 'red'}.get(verdict, 'white')
    console.print(Panel(
        f"[bold {style}]Session Verdict: {verdict.upper()}[/bold {style}]\n"
        f"Anomaly rate: [bold]{session_result.get('anomaly_rate', 0)*100:.1f}%[/bold]\n"
        f"Windows scored: [bold]{session_result.get('n_windows', 0)}[/bold]\n"
        f"Mean anomaly score: [bold]{session_result.get('mean_score', 0):.4f}[/bold]",
        title="Session Complete",
        border_style=style,
    ))
