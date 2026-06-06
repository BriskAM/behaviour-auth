"""
Enrollment and scoring pipeline.

enroll()  — runs 5-min test, extracts features, trains SVM, saves profile
score()   — loads profile, runs live scoring session
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
from rich.console import Console

from behavguard.collector.keystroke import KeyEvent
from behavguard.features.extractor import window_to_aggregate, window_to_sequence, WINDOW_FEATURE_DIM
from behavguard.model.svm import BehaveGuardSVM
from behavguard.model.lstm import BehaveGuardLSTM
from behavguard.storage import store

console = Console()

# Sliding window: 50 events with 25-event stride
WINDOW_SIZE = 50
WINDOW_STRIDE = 25
MIN_WINDOWS_TO_TRAIN = 8


def _slice_windows(events: list[KeyEvent], size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE):
    """Yield overlapping windows of KeyEvents."""
    i = 0
    while i + size <= len(events):
        yield events[i:i + size]
        i += stride


def _compute_user_stats(events: list[KeyEvent]) -> dict:
    """Compute population mean for normalisation."""
    dwells, flights, digraphs = [], [], []
    prev = None
    for e in events:
        if e.release_ts:
            dwells.append((e.release_ts - e.press_ts) * 1000)
        if prev and prev.release_ts and e.release_ts:
            flights.append(max(0, (e.press_ts - prev.release_ts) * 1000))
            digraphs.append(max(0, (e.press_ts - prev.press_ts) * 1000))
        prev = e
    return {
        'mean_dwell':   float(np.mean(dwells))   if dwells   else 80.0,
        'mean_flight':  float(np.mean(flights))  if flights  else 120.0,
        'mean_digraph': float(np.mean(digraphs)) if digraphs else 200.0,
    }


def _events_to_windows(
    events: list[KeyEvent],
    user_stats: dict,
) -> list[np.ndarray]:
    windows = []
    for window_events in _slice_windows(events):
        feat = window_to_aggregate(window_events, user_stats)
        if feat is not None:
            windows.append(feat)
    return windows


def _events_to_sequences(
    events: list[KeyEvent],
    user_stats: dict,
) -> list[np.ndarray]:
    sequences = []
    for window_events in _slice_windows(events):
        seq = window_to_sequence(window_events, user_stats)
        if seq is not None:
            sequences.append(seq)
    return sequences


# ------------------------------------------------------------------ #
# Enrollment
# ------------------------------------------------------------------ #

def enroll(subject_id: str, segment_events: dict[str, list[KeyEvent]], model_type: str = "lstm") -> Any:
    """
    Train a model (One-Class SVM or LSTM Autoencoder) from enrollment segments.

    segment_events keys: 'seg1', 'seg2', 'seg3'
    Training uses seg1 + seg2; seg3 is validation only (threshold calibration).
    """
    console.print("[cyan]→ Computing user statistics …[/cyan]")
    train_events = segment_events.get('seg1', []) + segment_events.get('seg2', [])
    val_events = segment_events.get('seg3', [])

    if len(train_events) < 100:
        raise ValueError(
            f"Too few training events ({len(train_events)}). "
            "Need at least 100 keystrokes for a meaningful model."
        )

    user_stats = _compute_user_stats(train_events)

    console.print("[cyan]→ Extracting window features …[/cyan]")
    if model_type == "lstm":
        train_data = _events_to_sequences(train_events, user_stats)
        val_data = _events_to_sequences(val_events, user_stats) if val_events else []
    else:
        train_data = _events_to_windows(train_events, user_stats)
        val_data = _events_to_windows(val_events, user_stats) if val_events else []

    console.print(f"  Training windows: [bold]{len(train_data)}[/bold]")
    console.print(f"  Validation windows: [bold]{len(val_data)}[/bold]")

    if len(train_data) < MIN_WINDOWS_TO_TRAIN:
        raise ValueError(
            f"Only {len(train_data)} windows extracted — need at least {MIN_WINDOWS_TO_TRAIN}. "
            "Type more during enrollment."
        )

    if model_type == "lstm":
        console.print("[cyan]→ Training LSTM Autoencoder …[/cyan]")
        model = BehaveGuardLSTM()
        profile = model.fit(train_data)

        if val_data:
            val_scores = [model.score_window(w)["raw_decision"] for w in val_data]
            profile.t_anomaly = float(np.percentile(val_scores, 95))
            console.print(
                f"  Threshold calibrated on validation set: [bold]{profile.t_anomaly:.4f}[/bold]"
            )
        else:
            console.print(
                f"  Threshold (train 95th pct): [bold]{profile.t_anomaly:.4f}[/bold]"
            )
    else:
        console.print("[cyan]→ Training One-Class SVM …[/cyan]")
        model = BehaveGuardSVM(nu=0.10)
        profile = model.fit(train_data)

        if val_data:
            val_scores = [-model.profile.svm.decision_function(
                model.profile.scaler.transform(w.reshape(1, -1))
            )[0] for w in val_data]
            profile.t_anomaly = float(np.percentile(val_scores, 95))
            console.print(
                f"  Threshold calibrated on validation set: [bold]{profile.t_anomaly:.4f}[/bold]"
            )
        else:
            console.print(
                f"  Threshold (train 95th pct): [bold]{profile.t_anomaly:.4f}[/bold]"
            )

    # Save
    store.ensure_dirs(subject_id)
    model_path = store.profile_path(subject_id)
    model.save(model_path)
    console.print(f"  Profile saved → [bold]{model_path}[/bold]")

    # Save raw events for future retraining
    for seg_label, events in segment_events.items():
        store.save_enrollment_events(subject_id, seg_label, events)

    console.print(f"\n[bold green]✓ Enrollment complete![/bold green]")
    console.print(f"  Trained on [bold]{len(train_data)}[/bold] windows from "
                  f"[bold]{len(train_events)}[/bold] keystrokes")
    console.print(f"  Anomaly threshold: [bold]{profile.t_anomaly:.4f}[/bold]\n")

    return model


# ------------------------------------------------------------------ #
# Live scoring session
# ------------------------------------------------------------------ #

def score_live(subject_id: str, duration_seconds: int = 300) -> dict:
    """
    Run a live scoring session for `duration_seconds`.
    Accumulates keystroke events, scores each 50-key window, returns session result.
    """
    from behavguard.collector.keystroke import KeystrokeCollector
    from behavguard.ui.scoring import ScoreDisplay, print_session_summary
    from rich.live import Live

    if not store.profile_exists(subject_id):
        raise RuntimeError(f"No profile found for '{subject_id}'. Run enrollment first.")

    console.print(f"[cyan]→ Loading profile for [bold]{subject_id}[/bold] …[/cyan]")
    model_path = store.profile_path(subject_id)
    with open(model_path, 'rb') as f:
        profile_data = pickle.load(f)

    if isinstance(profile_data, dict) and profile_data.get('model_type') == 'lstm':
        model = BehaveGuardLSTM()
    else:
        model = BehaveGuardSVM()

    model.load(model_path)
    threshold = model.profile.t_anomaly

    # Compute user stats from enrollment for normalisation
    seg1 = store.load_enrollment_events(subject_id, 'seg1')
    seg2 = store.load_enrollment_events(subject_id, 'seg2')
    user_stats = _compute_user_stats(seg1 + seg2)

    console.print(f"[green]✓ Profile loaded ({type(model).__name__}). Threshold = {threshold:.4f}[/green]")
    console.print(f"[dim]Type normally for {duration_seconds // 60} minutes. "
                  "Press Ctrl+C to stop early.[/dim]\n")

    collector = KeystrokeCollector()
    collector.start()

    display = ScoreDisplay(subject_id=subject_id, threshold=threshold)
    window_results: list[dict] = []
    buffer: list[KeyEvent] = []

    start = time.monotonic()
    end = start + duration_seconds

    try:
        with Live(display.render(), console=console, refresh_per_second=2) as live:
            while time.monotonic() < end:
                new_events = collector.drain()
                buffer.extend(new_events)

                # Score whenever we have a full window
                if len(buffer) >= WINDOW_SIZE:
                    window_events = buffer[:WINDOW_SIZE]
                    buffer = buffer[WINDOW_STRIDE:]  # advance by stride

                    if isinstance(model, BehaveGuardLSTM):
                        feat = window_to_sequence(window_events, user_stats)
                    else:
                        feat = window_to_aggregate(window_events, user_stats)

                    if feat is not None:
                        result = model.score_window(feat)
                        window_results.append(result)

                        all_events_count = collector.count() + len(buffer)
                        wpm_estimate = _wpm_from_buffer(window_events)
                        display.update(result, wpm_estimate, all_events_count)

                live.update(display.render())
                time.sleep(0.5)

    except KeyboardInterrupt:
        console.print("\n[yellow]Session stopped by user.[/yellow]")
    finally:
        collector.stop()

    session_result = model.score_session(window_results)
    print_session_summary(session_result)

    # Persist session log
    store.append_session_log(subject_id, {
        'timestamp': time.time(),
        'subject_id': subject_id,
        **session_result,
    })

    return session_result


def _wpm_from_buffer(events: list[KeyEvent]) -> float:
    if len(events) < 2:
        return 0.0
    dur = (events[-1].press_ts - events[0].press_ts) / 60.0
    return (len(events) / 5.0) / max(dur, 1e-6)
