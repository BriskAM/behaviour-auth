"""
One-Class SVM baseline model for BehaveGuard.

Trains on enrollment aggregate feature vectors (genuine user only).
Scores subsequent windows; outputs anomaly score ∈ [0, 1].
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class EnrollmentProfile:
    """Persisted profile produced by training."""
    scaler: StandardScaler
    svm: OneClassSVM
    t_anomaly: float               # 95th-percentile enrollment error
    enrollment_mean: np.ndarray    # per-feature mean for drift detection
    enrollment_std: np.ndarray
    nu: float
    n_windows_trained: int
    eer_estimate: Optional[float] = None


class BehaveGuardSVM:
    """
    One-class SVM wrapper following the BehaveGuard design:
      - Trains on genuine windows only
      - Threshold = 95th percentile of enrollment decision values
      - Scores new windows → {score, verdict}
    """

    def __init__(self, nu: float = 0.05, kernel: str = 'rbf', gamma: str = 'scale'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self._profile: Optional[EnrollmentProfile] = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, windows: list[np.ndarray]) -> EnrollmentProfile:
        """
        Train on enrollment windows (genuine user only).
        windows: list of 43-dim aggregate feature vectors
        """
        X = np.stack(windows)  # (N, 43)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        svm.fit(X_scaled)

        # Decision function: positive = normal, negative = anomaly
        # We negate and normalise to [0, 1] anomaly score
        raw_scores = -svm.decision_function(X_scaled)   # higher = more anomalous
        t_anomaly = float(np.percentile(raw_scores, 95))

        self._profile = EnrollmentProfile(
            scaler=scaler,
            svm=svm,
            t_anomaly=t_anomaly,
            enrollment_mean=np.mean(X, axis=0),
            enrollment_std=np.std(X, axis=0) + 1e-8,
            nu=self.nu,
            n_windows_trained=len(windows),
        )
        return self._profile

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score_window(self, window: np.ndarray) -> dict:
        """
        Score a single aggregate feature vector.

        Returns:
          anomaly_score : float ∈ [0, 1]  (0=legitimate, 1=impostor)
          raw_decision  : float (SVM decision function, negated)
          verdict       : 'legitimate' | 'uncertain' | 'anomaly'
          threshold     : the T_anomaly used
        """
        if self._profile is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        p = self._profile
        X = window.reshape(1, -1)
        X_scaled = p.scaler.transform(X)

        raw = float(-p.svm.decision_function(X_scaled)[0])

        # Normalise: 0.0 = right at threshold, scale by threshold magnitude
        if p.t_anomaly > 0:
            norm_score = raw / (p.t_anomaly * 2.0)
        else:
            norm_score = raw / 2.0
        anomaly_score = float(np.clip(norm_score, 0.0, 1.0))

        verdict = _verdict(raw, p.t_anomaly)

        return {
            'anomaly_score': anomaly_score,
            'raw_decision': raw,
            'verdict': verdict,
            'threshold': p.t_anomaly,
        }

    def score_session(self, window_scores: list[dict]) -> dict:
        """Aggregate multiple window scores into a session verdict."""
        if not window_scores:
            return {'session_verdict': 'unknown', 'anomaly_rate': 0.0}

        t = self._profile.t_anomaly if self._profile else 0.5
        anomaly_rate = sum(
            1 for w in window_scores if w['verdict'] == 'anomaly'
        ) / len(window_scores)

        if anomaly_rate < 0.25:
            verdict = 'legitimate'
        elif anomaly_rate > 0.60:
            verdict = 'impostor'
        else:
            verdict = 'uncertain'

        return {
            'session_verdict': verdict,
            'anomaly_rate': anomaly_rate,
            'n_windows': len(window_scores),
            'mean_score': float(np.mean([w['anomaly_score'] for w in window_scores])),
        }

    # ------------------------------------------------------------------ #
    # Nu sweep for threshold tuning
    # ------------------------------------------------------------------ #

    def tune_nu(
        self,
        train_windows: list[np.ndarray],
        val_genuine: list[np.ndarray],
        val_impostor: list[np.ndarray],
        nu_values: Optional[list[float]] = None,
    ) -> float:
        """
        Grid search over nu to minimise EER on a validation set.
        Returns best nu and sets self.nu.

        Note: impostors are ONLY used in evaluation, never in training.
        """
        nu_values = nu_values or [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        best_nu, best_eer = self.nu, float('inf')

        X_train = np.stack(train_windows)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_genuine_s = scaler.transform(np.stack(val_genuine))
        X_impostor_s = scaler.transform(np.stack(val_impostor))

        for nu in nu_values:
            svm = OneClassSVM(nu=nu, kernel=self.kernel, gamma=self.gamma)
            svm.fit(X_train_s)

            genuine_scores = -svm.decision_function(X_genuine_s)
            impostor_scores = -svm.decision_function(X_impostor_s)
            eer = _compute_eer(genuine_scores, impostor_scores)

            if eer < best_eer:
                best_eer, best_nu = eer, nu

        self.nu = best_nu
        if self._profile:
            self._profile.eer_estimate = best_eer
        return best_nu

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._profile, f)

    def load(self, path: str | Path) -> None:
        with open(path, 'rb') as f:
            self._profile = pickle.load(f)

    @property
    def is_trained(self) -> bool:
        return self._profile is not None

    @property
    def profile(self) -> Optional[EnrollmentProfile]:
        return self._profile


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _verdict(raw: float, threshold: float) -> str:
    lo = threshold * 0.6
    hi = threshold * 1.4
    if raw <= lo:
        return 'legitimate'
    elif raw <= hi:
        return 'uncertain'
    return 'anomaly'


def _compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """
    Equal Error Rate via threshold sweep.
    genuine_scores: scores for real user (we want these LOW)
    impostor_scores: scores for impostors (we want these HIGH)
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)

    best_eer = 1.0
    for t in thresholds:
        frr = np.mean(genuine_scores > t)   # real user rejected
        far = np.mean(impostor_scores <= t)  # impostor accepted
        eer_approx = abs(frr - far)
        if eer_approx < abs(best_eer - 0.5):
            best_eer = (frr + far) / 2.0

    return best_eer
