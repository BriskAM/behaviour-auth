"""
Simple file-based storage for enrollment profiles and session logs.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional


DATA_DIR = Path.home() / '.behaveguard'


def profile_path(subject_id: str) -> Path:
    return DATA_DIR / subject_id / 'model.pkl'


def session_log_path(subject_id: str) -> Path:
    return DATA_DIR / subject_id / 'sessions.jsonl'


def raw_events_path(subject_id: str, segment: str) -> Path:
    return DATA_DIR / subject_id / f'enrollment_{segment}.pkl'


def ensure_dirs(subject_id: str) -> None:
    (DATA_DIR / subject_id).mkdir(parents=True, exist_ok=True)


def save_enrollment_events(subject_id: str, segment: str, events: list) -> None:
    ensure_dirs(subject_id)
    with open(raw_events_path(subject_id, segment), 'wb') as f:
        pickle.dump(events, f)


def load_enrollment_events(subject_id: str, segment: str) -> list:
    p = raw_events_path(subject_id, segment)
    if not p.exists():
        return []
    with open(p, 'rb') as f:
        return pickle.load(f)


def append_session_log(subject_id: str, session: dict) -> None:
    ensure_dirs(subject_id)
    with open(session_log_path(subject_id), 'a') as f:
        f.write(json.dumps(session) + '\n')


def load_session_logs(subject_id: str) -> list[dict]:
    p = session_log_path(subject_id)
    if not p.exists():
        return []
    sessions = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def profile_exists(subject_id: str) -> bool:
    return profile_path(subject_id).exists()


def list_subjects() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
