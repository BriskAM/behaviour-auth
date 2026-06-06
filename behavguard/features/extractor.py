"""
Feature extraction from raw keystroke events.

Per-event vector (shape 7):
  [dwell_ms, flight_ms, digraph_ms, cat_alphanum, cat_symbol, cat_special, digraph_freq]

Window aggregate features (shape 40+) used by the SVM:
  mean/std/skew/kurtosis of dwell, flight, digraph × key category ratios + time encoding
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from scipy.stats import skew, kurtosis

from behavguard.collector.keystroke import KeyEvent, DIGRAPH_FREQUENCY

# Feature vector dimension for a single event
EVENT_FEATURE_DIM = 7
# Feature vector dimension for a window aggregate (SVM input)
WINDOW_FEATURE_DIM = 43


# ------------------------------------------------------------------ #
# Time encoding
# ------------------------------------------------------------------ #

def encode_time(epoch_ts: float) -> tuple[float, float]:
    """Cyclical encoding of time-of-day. Privacy-safe — no exact timestamp."""
    import datetime
    dt = datetime.datetime.fromtimestamp(epoch_ts)
    fraction = (dt.hour * 60 + dt.minute) / 1440.0
    return math.sin(2 * math.pi * fraction), math.cos(2 * math.pi * fraction)


# ------------------------------------------------------------------ #
# Per-event feature vector
# ------------------------------------------------------------------ #

def event_to_vector(
    evt: KeyEvent,
    prev_evt: Optional[KeyEvent],
    user_mean_dwell: float = 80.0,
    user_mean_flight: float = 120.0,
    user_mean_digraph: float = 200.0,
) -> Optional[np.ndarray]:
    """
    Convert a single KeyEvent + its predecessor into a 7-dim feature vector.
    Returns None if timing data is incomplete.
    """
    if evt.release_ts is None:
        return None

    dwell_ms = (evt.release_ts - evt.press_ts) * 1000.0

    # Flight and digraph require a previous event
    if prev_evt is None or prev_evt.release_ts is None:
        return None

    flight_ms = max(0.0, (evt.press_ts - prev_evt.release_ts) * 1000.0)
    digraph_ms = max(0.0, (evt.press_ts - prev_evt.press_ts) * 1000.0)

    # Normalise by user mean (default=population mean on first pass)
    dwell_norm = dwell_ms / max(user_mean_dwell, 1.0)
    flight_norm = flight_ms / max(user_mean_flight, 1.0)
    digraph_norm = digraph_ms / max(user_mean_digraph, 1.0)

    # Key category one-hot
    cat = evt.key_category
    cat_vec = [
        1.0 if cat == 'alphanum' else 0.0,
        1.0 if cat == 'symbol' else 0.0,
        1.0 if cat == 'special' else 0.0,
    ]

    # Digraph frequency weight (1.0 for known pairs, 0.3 for rare)
    pair = (prev_evt.key_id, evt.key_id)
    freq_weight = DIGRAPH_FREQUENCY.get(pair, 0.3)

    return np.array([
        dwell_norm,
        flight_norm,
        digraph_norm,
        *cat_vec,
        freq_weight,
    ], dtype=np.float32)


# ------------------------------------------------------------------ #
# Window → sequence feature vector (LSTM input)
# ------------------------------------------------------------------ #

def window_to_sequence(
    events: list[KeyEvent],
    user_stats: Optional[dict] = None,
    sequence_length: int = 50,
) -> Optional[np.ndarray]:
    """
    Convert a list of KeyEvents into a (sequence_length, 7) sequence of per-event vectors.
    """
    if len(events) < sequence_length:
        return None

    user_stats = user_stats or {}
    mean_dwell = user_stats.get('mean_dwell', 80.0)
    mean_flight = user_stats.get('mean_flight', 120.0)
    mean_digraph = user_stats.get('mean_digraph', 200.0)

    seq = []
    prev_evt = None
    for evt in events[:sequence_length]:
        vec = event_to_vector(evt, prev_evt, mean_dwell, mean_flight, mean_digraph)
        if vec is None:
            # Fallback for the first event in the sequence which lacks a predecessor
            dwell_ms = (evt.release_ts - evt.press_ts) * 1000.0 if evt.release_ts else mean_dwell
            dwell_norm = dwell_ms / max(mean_dwell, 1.0)
            cat = evt.key_category
            cat_vec = [
                1.0 if cat == 'alphanum' else 0.0,
                1.0 if cat == 'symbol' else 0.0,
                1.0 if cat == 'special' else 0.0,
            ]
            vec = np.array([
                dwell_norm,
                0.0,  # flight_norm
                0.0,  # digraph_norm
                *cat_vec,
                0.3,  # default frequency weight
            ], dtype=np.float32)
        seq.append(vec)
        prev_evt = evt

    return np.stack(seq)



# ------------------------------------------------------------------ #
# Window → aggregate feature vector (SVM input)
# ------------------------------------------------------------------ #

def _safe_stats(arr: np.ndarray) -> list[float]:
    """mean, std, skewness, kurtosis — safe against tiny arrays."""
    if len(arr) < 2:
        return [float(arr.mean()) if len(arr) else 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(skew(arr)),
        float(kurtosis(arr)),
    ]


def window_to_aggregate(
    events: list[KeyEvent],
    user_stats: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Convert a list of KeyEvents into a 43-dim aggregate feature vector
    suitable for the One-Class SVM.

    Layout (4 stats × 9 groups + 7 extra = 43):
      groups 0-3 : dwell / flight / digraph / digraph_weighted
      groups 4-6 : dwell per category (alphanum/symbol/special)
      group  7   : IKI (inter-key interval) raw
      group  8   : flight:dwell ratio
      + time_sin, time_cos, alphanum_ratio, symbol_ratio
    """
    if len(events) < 5:
        return None

    user_stats = user_stats or {}
    mean_dwell = user_stats.get('mean_dwell', 80.0)
    mean_flight = user_stats.get('mean_flight', 120.0)
    mean_digraph = user_stats.get('mean_digraph', 200.0)

    dwells, flights, digraphs, weighted_digraphs = [], [], [], []
    cat_dwells = {'alphanum': [], 'symbol': [], 'special': []}
    ikis = []

    prev = None
    for evt in events:
        if evt.release_ts is None:
            prev = evt
            continue

        dwell = (evt.release_ts - evt.press_ts) * 1000.0
        dwells.append(dwell / mean_dwell)
        cat_dwells.get(evt.key_category, cat_dwells['special']).append(dwell / mean_dwell)

        if prev and prev.release_ts:
            flight = max(0.0, (evt.press_ts - prev.release_ts) * 1000.0)
            dgraph = max(0.0, (evt.press_ts - prev.press_ts) * 1000.0)
            iki = max(0.0, (evt.press_ts - prev.press_ts) * 1000.0)

            flights.append(flight / mean_flight)
            digraphs.append(dgraph / mean_digraph)
            ikis.append(iki / mean_digraph)

            pair = (prev.key_id, evt.key_id)
            w = DIGRAPH_FREQUENCY.get(pair, 0.3)
            weighted_digraphs.append((dgraph / mean_digraph) * w)

        prev = evt

    # Build feature groups
    feature_groups = [
        np.array(dwells),
        np.array(flights) if flights else np.array([0.0]),
        np.array(digraphs) if digraphs else np.array([0.0]),
        np.array(weighted_digraphs) if weighted_digraphs else np.array([0.0]),
        np.array(cat_dwells['alphanum']) if cat_dwells['alphanum'] else np.array([0.0]),
        np.array(cat_dwells['symbol']) if cat_dwells['symbol'] else np.array([0.0]),
        np.array(cat_dwells['special']) if cat_dwells['special'] else np.array([0.0]),
        np.array(ikis) if ikis else np.array([0.0]),
    ]

    feats: list[float] = []
    for arr in feature_groups:
        feats.extend(_safe_stats(arr))   # 8 × 4 = 32 features

    # Flight:dwell ratio stats (5th ratio group → 4 more)
    fd_ratio = (np.array(flights) / (np.array(dwells[:len(flights)]) + 1e-6)
                if flights else np.array([1.0]))
    feats.extend(_safe_stats(fd_ratio))  # → 36 features

    # Cyclical time encoding from first event timestamp
    ts_base = events[0].press_ts + time.time() - time.perf_counter()
    t_sin, t_cos = encode_time(ts_base)

    # Category ratios
    n = len(events)
    alphanum_ratio = sum(1 for e in events if e.key_category == 'alphanum') / n
    symbol_ratio = sum(1 for e in events if e.key_category == 'symbol') / n
    special_ratio = 1.0 - alphanum_ratio - symbol_ratio

    # WPM estimate (rough: keystrokes / 5 / minutes)
    if len(events) > 1:
        duration_min = ((events[-1].press_ts - events[0].press_ts) / 60.0) or 1e-6
        wpm = (len(events) / 5.0) / duration_min
    else:
        wpm = 0.0

    feats.extend([t_sin, t_cos, alphanum_ratio, symbol_ratio, special_ratio, wpm / 100.0])
    # → 36 + 6 = 42 features; add one digraph coverage score → 43

    # Digraph coverage: fraction of pairs in COMMON_DIGRAPHS
    from behavguard.collector.keystroke import COMMON_DIGRAPHS
    seen_pairs = set()
    for i in range(1, len(events)):
        seen_pairs.add((events[i-1].key_id, events[i].key_id))
    coverage = len(seen_pairs & COMMON_DIGRAPHS) / len(COMMON_DIGRAPHS)
    feats.append(coverage)

    vec = np.array(feats, dtype=np.float32)
    assert len(vec) == WINDOW_FEATURE_DIM, f"Expected {WINDOW_FEATURE_DIM}, got {len(vec)}"
    return vec
