"""
scorer.py — continuous scoring engine for One-Class SVM approach
"""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import ks_2samp

from collector import KeyEvent, MouseEvent
from features import (extract_keystroke_pairs, pairs_to_window,
                       extract_mouse_features, mouse_to_window,
                       WindowNormalizer, WINDOW_SIZE)
from model import UserProfile


# ── enums ────────────────────────────────────────────────────────────────────

class WindowVerdict(Enum):
    WARMUP      = "warmup"
    LEGITIMATE  = "legitimate"
    UNCERTAIN   = "uncertain"
    ANOMALY     = "anomaly"


class SessionVerdict(Enum):
    LEGITIMATE  = "legitimate"
    UNCERTAIN   = "uncertain"
    IMPOSTOR    = "impostor"


class FPCategory(Enum):
    NONE            = "none"
    FATIGUE         = "fatigue"
    KEYBOARD_CHANGE = "keyboard_change"
    STRESS          = "stress"
    IMPAIRMENT      = "impairment"
    IMPOSTOR        = "impostor"


class DriftType(Enum):
    STABLE    = "stable"
    TEMPORARY = "temporary"
    PERMANENT = "permanent"


class RecommendedAction(Enum):
    NONE           = "none"
    SOFT_CHALLENGE = "soft_challenge"
    HARD_CHALLENGE = "hard_challenge"
    LOCKOUT        = "lockout"


# ── window result ─────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    index: int
    timestamp: float
    recon_error: float            # SVM anomaly score (inverted decision function)
    latent_distance: float        # unused (0.0 for SVM)
    combined_score: float         # SVM anomaly score
    verdict: WindowVerdict
    threshold_used: float


# ── session result ────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    subject_id: str
    start_ts: float
    end_ts: float
    n_windows: int
    anomaly_rate: float
    mean_score: float
    max_score: float
    verdict: SessionVerdict
    fp_category: FPCategory
    recommended_action: RecommendedAction
    window_results: list[WindowResult] = field(default_factory=list)


# ── dynamic threshold ─────────────────────────────────────────────────────────

class DynamicThreshold:
    """
    Adjusts the session-local threshold based on early windows.
    Bound: threshold cannot shift more than ±20% from base.
    """

    def __init__(self, base: float, enrollment_mean_error: float,
                 max_shift_fraction: float = 0.20):
        self.base = base
        self.enrollment_mean = enrollment_mean_error
        self.max_shift = base * max_shift_fraction
        self._session_errors: list[float] = []
        self._warmup_count = 5

    def update(self, error: float):
        self._session_errors.append(error)

    def current(self) -> float:
        if len(self._session_errors) < self._warmup_count:
            return self.base

        recent = self._session_errors[-10:]
        session_mean = np.mean(recent)
        offset = session_mean - self.enrollment_mean
        bounded = float(np.clip(offset, -self.max_shift, self.max_shift))
        return self.base + bounded

    def reset(self):
        self._session_errors.clear()


# ── false positive classifier ─────────────────────────────────────────────────

def classify_false_positive(window_results: list[WindowResult],
                             enrollment_mean_error: float) -> FPCategory:
    """
    Heuristic classification of why a session was flagged.
    """
    if not window_results:
        return FPCategory.NONE

    errors = [w.recon_error for w in window_results]
    n = len(errors)
    if n < 3:
        return FPCategory.NONE

    # fit linear trend to error over time
    x = np.arange(n, dtype=float)
    slope = np.polyfit(x, errors, 1)[0]
    variance = np.var(errors)
    mean_err = np.mean(errors)

    # fatigue: errors increasing monotonically over session
    if slope > 0.002:
        return FPCategory.FATIGUE

    # stress/caffeine: high variance, mean shifted up
    if variance > (enrollment_mean_error * 0.5) and mean_err > enrollment_mean_error * 1.3:
        return FPCategory.STRESS

    # keyboard change: consistent offset, low variance (shape preserved)
    if mean_err > enrollment_mean_error * 1.5 and variance < (enrollment_mean_error * 0.3):
        return FPCategory.KEYBOARD_CHANGE

    return FPCategory.IMPOSTOR


# ── session scorer ────────────────────────────────────────────────────────────

class SessionScorer:
    """
    Stateful scorer for a single user session using the SVM profile.
    """

    WARMUP_WINDOWS = 3

    def __init__(self, profile: UserProfile):
        self.profile = profile
        self._kb_buffer: list[KeyEvent] = []
        self._mouse_buffer: list[MouseEvent] = []
        self._pairs_buffer: list = []
        self._window_results: list[WindowResult] = []
        self._window_index = 0
        self._session_start = time.time()
        self._dynamic_threshold = DynamicThreshold(
            base=profile.T_anomaly,
            enrollment_mean_error=float(profile.T_anomaly * 0.7),
        )

    def ingest_key_events(self, events: list[KeyEvent]):
        """Feed new keyboard events. Emits window results when buffer fills."""
        self._kb_buffer.extend(events)
        new_pairs = extract_keystroke_pairs(self._kb_buffer[-200:])
        self._pairs_buffer.extend(new_pairs[max(0, len(new_pairs)-len(events)):])

        while len(self._pairs_buffer) >= WINDOW_SIZE:
            window_pairs = self._pairs_buffer[:WINDOW_SIZE]
            self._pairs_buffer = self._pairs_buffer[WINDOW_SIZE // 2:]  # 50% overlap

            raw_window = pairs_to_window(window_pairs)
            if raw_window is None:
                continue

            result = self._score_window(raw_window)
            self._window_results.append(result)
            self._window_index += 1

    def _score_window(self, raw_window: np.ndarray) -> WindowResult:
        scores = self.profile.score_window(raw_window)
        threshold = self._dynamic_threshold.current()
        self._dynamic_threshold.update(scores["recon_error"])

        is_warmup = self._window_index < self.WARMUP_WINDOWS

        if is_warmup:
            verdict = WindowVerdict.WARMUP
        elif scores["combined_score"] <= threshold:
            verdict = WindowVerdict.LEGITIMATE
        elif scores["combined_score"] <= threshold * 1.4:
            verdict = WindowVerdict.UNCERTAIN
        else:
            verdict = WindowVerdict.ANOMALY

        return WindowResult(
            index=self._window_index,
            timestamp=time.time(),
            recon_error=scores["recon_error"],
            latent_distance=scores["latent_distance"],
            combined_score=scores["combined_score"],
            verdict=verdict,
            threshold_used=threshold,
        )

    def get_session_result(self) -> SessionResult:
        """Compute session-level verdict from accumulated window results."""
        non_warmup = [w for w in self._window_results
                      if w.verdict != WindowVerdict.WARMUP]

        if not non_warmup:
            return SessionResult(
                subject_id=self.profile.subject_id,
                start_ts=self._session_start,
                end_ts=time.time(),
                n_windows=0,
                anomaly_rate=0.0,
                mean_score=0.0,
                max_score=0.0,
                verdict=SessionVerdict.LEGITIMATE,
                fp_category=FPCategory.NONE,
                recommended_action=RecommendedAction.NONE,
                window_results=self._window_results,
            )

        scores = [w.combined_score for w in non_warmup]
        n_anomaly = sum(1 for w in non_warmup if w.verdict == WindowVerdict.ANOMALY)
        anomaly_rate = n_anomaly / len(non_warmup)

        if anomaly_rate < 0.20:
            verdict = SessionVerdict.LEGITIMATE
        elif anomaly_rate > 0.60:
            verdict = SessionVerdict.IMPOSTOR
        else:
            verdict = SessionVerdict.UNCERTAIN

        fp_cat = FPCategory.NONE
        if verdict in (SessionVerdict.UNCERTAIN, SessionVerdict.IMPOSTOR):
            fp_cat = classify_false_positive(
                non_warmup,
                enrollment_mean_error=self.profile.T_anomaly * 0.7,
            )

        action = _recommended_action(np.mean(scores), self.profile.T_anomaly)

        return SessionResult(
            subject_id=self.profile.subject_id,
            start_ts=self._session_start,
            end_ts=time.time(),
            n_windows=len(non_warmup),
            anomaly_rate=anomaly_rate,
            mean_score=float(np.mean(scores)),
            max_score=float(np.max(scores)),
            verdict=verdict,
            fp_category=fp_cat,
            recommended_action=action,
            window_results=self._window_results,
        )


def _recommended_action(mean_score: float, T_anomaly: float) -> RecommendedAction:
    # Scale bounds relative to T_anomaly
    if mean_score < T_anomaly * 0.7:
        return RecommendedAction.NONE
    elif mean_score < T_anomaly:
        return RecommendedAction.SOFT_CHALLENGE
    elif mean_score < T_anomaly * 1.4:
        return RecommendedAction.HARD_CHALLENGE
    else:
        return RecommendedAction.LOCKOUT


# ── drift detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Nightly job that decides whether to retrain the model.
    """

    MAX_DAILY_VELOCITY = 0.1
    MAX_BUFFER_SIZE = 50

    def __init__(self, profile: UserProfile):
        self.profile = profile
        self._legitimate_buffer: list[np.ndarray] = []
        self._velocity_history: list[float] = []

    def ingest_session(self, session: SessionResult, session_features: np.ndarray):
        """
        session_features: normalized flat feature vector [43] representing session average
        """
        if session.verdict == SessionVerdict.IMPOSTOR:
            return

        if session.anomaly_rate > 0.15:
            return

        dist = self._mahalanobis_from_enrollment(session_features)
        if dist > self.profile.T_drift:
            return

        self._legitimate_buffer.append(session_features)
        if len(self._legitimate_buffer) > self.MAX_BUFFER_SIZE:
            self._legitimate_buffer.pop(0)

    def _mahalanobis_from_enrollment(self, features: np.ndarray) -> float:
        try:
            diff = features.flatten() - self.profile.enrollment_mean
            cov_inv = np.linalg.inv(self.profile.enrollment_cov)
            return float(np.sqrt(diff @ cov_inv @ diff))
        except Exception:
            return 0.0

    def check_drift(self) -> dict:
        if len(self._legitimate_buffer) < 5:
            return {"drift_detected": False, "reason": "insufficient_data"}

        recent = np.array(self._legitimate_buffer)
        enrollment_flat = self.profile.enrollment_mean.reshape(1, -1)
        if recent.ndim == 1:
            recent = recent.reshape(len(self._legitimate_buffer), -1)

        p_values = []
        n_dims = min(recent.shape[1], enrollment_flat.shape[1])
        for i in range(min(n_dims, 20)):
            ref = np.random.normal(enrollment_flat[0, i],
                                   np.sqrt(self.profile.enrollment_cov[i, i] + 1e-8),
                                   size=100)
            stat, p = ks_2samp(ref, recent[:, i])
            p_values.append(p)

        drift_detected = any(p < 0.05 for p in p_values)
        if not drift_detected:
            return {"drift_detected": False, "drift_type": DriftType.STABLE}

        drift_type = self._classify_drift_type(recent)

        if drift_type == DriftType.PERMANENT:
            poisoning_risk = self._check_poisoning(recent)
            if poisoning_risk:
                return {
                    "drift_detected": True,
                    "drift_type": drift_type,
                    "action": "alert",
                    "reason": "suspicious_drift_velocity_or_inconsistency",
                }

        action = "widen_threshold" if drift_type == DriftType.TEMPORARY else "retrain"
        return {
            "drift_detected": True,
            "drift_type": drift_type,
            "action": action,
            "n_sessions": len(self._legitimate_buffer),
        }

    def _classify_drift_type(self, recent: np.ndarray) -> DriftType:
        dists = [self._mahalanobis_from_enrollment(r) for r in recent]
        if len(dists) < 2:
            return DriftType.STABLE

        variance = float(np.var(dists))
        slope = float(np.polyfit(range(len(dists)), dists, 1)[0])

        if variance > 0.5 and abs(slope) < 0.01:
            return DriftType.TEMPORARY

        if slope > 0.02:
            return DriftType.PERMANENT

        return DriftType.STABLE

    def _check_poisoning(self, recent: np.ndarray) -> bool:
        dists = [self._mahalanobis_from_enrollment(r) for r in recent]
        if len(dists) > 1:
            velocity = (dists[-1] - dists[0]) / len(dists)
            self._velocity_history.append(abs(velocity))
            if len(self._velocity_history) > 7:
                self._velocity_history.pop(0)
            if max(self._velocity_history) > self.MAX_DAILY_VELOCITY:
                return True

        feature_drifts = []
        mean = recent.mean(axis=0)
        for i in range(min(recent.shape[1], 20)):
            drift_i = abs(mean[i] - self.profile.enrollment_mean[i])
            std_i = np.sqrt(self.profile.enrollment_cov[i, i] + 1e-8)
            feature_drifts.append(drift_i / std_i)

        if feature_drifts:
            cv = np.std(feature_drifts) / (np.mean(feature_drifts) + 1e-8)
            if cv > 0.6:
                return True

        return False
