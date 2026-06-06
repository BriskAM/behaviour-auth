"""
LSTM Autoencoder model for BehaveGuard.

Trains on enrollment sequence windows of shape (N, 50, 7) (genuine user only).
Scores subsequent windows; outputs anomaly score ∈ [0, 1].
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


@dataclass
class LSTMEnrollmentProfile:
    """Persisted profile produced by training."""
    model_state_dict: dict
    scaler: StandardScaler
    t_anomaly: float               # 95th-percentile combined anomaly score
    t_anomaly_raw: float           # 95th-percentile raw reconstruction error
    enrollment_mean: np.ndarray    # mean per feature dimension for drift
    enrollment_std: np.ndarray     # std per feature dimension
    latent_centroid: np.ndarray    # mean latent vector of enrollment
    latent_radius: float           # 95th-percentile distance in latent space
    n_windows_trained: int
    eer_estimate: Optional[float] = None


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder model following the BehaveGuard architecture.
    Inputs: shape [batch_size, sequence_length, feature_dim]
    """

    def __init__(self, sequence_length: int = 50, feature_dim: int = 7, latent_dim: int = 16):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(
            input_size=feature_dim,
            hidden_size=64,
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )
        self.latent_proj = nn.Linear(32, latent_dim)

        # Decoder State Projections (solves vanishing gradients in sequence-to-sequence)
        self.dec_h0_proj = nn.Linear(latent_dim, 32)
        self.dec_c0_proj = nn.Linear(latent_dim, 32)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=32,
            batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )
        self.decoder_dense = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        out1, _ = self.encoder_lstm1(x)
        out2, (hn, _) = self.encoder_lstm2(out1)
        
        # hn shape: [1, batch_size, 32]
        latent = self.latent_proj(hn[-1])  # [batch_size, 16]

        # Repeat latent vector seq_len times
        latent_repeated = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Project latent to decoder initial state (h0, c0)
        h0 = self.dec_h0_proj(latent).unsqueeze(0)  # [1, batch_size, 32]
        c0 = self.dec_c0_proj(latent).unsqueeze(0)  # [1, batch_size, 32]

        # Decoder
        dec_out1, _ = self.decoder_lstm1(latent_repeated, (h0, c0))
        dec_out2, _ = self.decoder_lstm2(dec_out1)
        reconstructed = self.decoder_dense(dec_out2)

        return reconstructed, latent


class BehaveGuardLSTM:
    """
    LSTM Autoencoder wrapper following BehaveGuard design:
      - Trains on genuine user event sequences only
      - Uses a weighted MSE loss function
      - Threshold = 95th percentile of enrollment reconstruction errors + latent distance
      - Scores new windows -> {score, verdict}
    """

    def __init__(
        self,
        sequence_length: int = 50,
        feature_dim: int = 7,
        latent_dim: int = 16,
        epochs: int = 250,   # Increased epochs for stable convergence
        lr: float = 0.01,    # Higher learning rate for better convergence
        batch_size: int = 16,
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        self.model = LSTMAutoencoder(sequence_length, feature_dim, latent_dim)
        self._profile: Optional[LSTMEnrollmentProfile] = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model.to(self.device)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, sequences: list[np.ndarray]) -> LSTMEnrollmentProfile:
        """
        Train on enrollment sequences (genuine user only).
        sequences: list of (50, 7) arrays
        """
        X = np.stack(sequences)  # (N, 50, 7)
        N = len(sequences)

        # Fit StandardScaler on flattened sequence data
        scaler = StandardScaler()
        X_flat = X.reshape(-1, self.feature_dim)
        scaler.fit(X_flat)

        # Transform sequences
        X_scaled = scaler.transform(X_flat).reshape(N, self.sequence_length, self.feature_dim)

        # PyTorch Setup
        self.model.train()
        dataset = torch.tensor(X_scaled, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Weighted MSE: tier weights per design guide
        feature_weights = torch.tensor([1.2, 1.0, 1.4, 0.5, 0.3, 0.3, 0.8], device=self.device)

        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset.size(0))
            for i in range(0, dataset.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch = dataset[indices].to(self.device)

                optimizer.zero_grad()
                recon, latent = self.model(batch)

                # Custom weighted MSE loss
                diff = (recon - batch) ** 2
                recon_loss = (diff * feature_weights).mean()

                # Compactness loss: pull latents into a tight cluster
                latent_mean = latent.mean(dim=0, keepdim=True)
                compactness = ((latent - latent_mean) ** 2).mean()

                loss = recon_loss + 0.2 * compactness

                loss.backward()
                optimizer.step()

        # Calibration
        self.model.eval()
        raw_errors = []
        latents_list = []
        with torch.no_grad():
            for i in range(N):
                seq_scaled = X_scaled[i:i+1]
                batch = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)
                recon, latent = self.model(batch)

                # Compute weighted MSE error
                diff = (recon - batch) ** 2
                err = (diff * feature_weights).mean().item()
                raw_errors.append(err)
                latents_list.append(latent.cpu().numpy()[0])

        t_anomaly_raw = float(np.percentile(raw_errors, 95))
        
        # Latent centroid and radius
        latents = np.stack(latents_list)
        latent_centroid = np.mean(latents, axis=0)
        latent_distances = np.linalg.norm(latents - latent_centroid, axis=1)
        latent_radius = float(np.percentile(latent_distances, 95))

        # Calculate combined scores for initial t_anomaly threshold setting
        combined_scores = []
        for err, lat in zip(raw_errors, latents_list):
            l_dist = np.linalg.norm(lat - latent_centroid)
            norm_recon = err / t_anomaly_raw if t_anomaly_raw > 0 else err
            norm_latent = l_dist / latent_radius if latent_radius > 0 else l_dist
            combined_scores.append(0.5 * norm_recon + 0.5 * norm_latent)
        t_anomaly = float(np.percentile(combined_scores, 95))

        self._profile = LSTMEnrollmentProfile(
            model_state_dict=self.model.state_dict(),
            scaler=scaler,
            t_anomaly=t_anomaly,
            t_anomaly_raw=t_anomaly_raw,
            enrollment_mean=np.mean(X_flat, axis=0),
            enrollment_std=np.std(X_flat, axis=0) + 1e-8,
            latent_centroid=latent_centroid,
            latent_radius=latent_radius,
            n_windows_trained=N,
        )
        return self._profile

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score_window(self, sequence: np.ndarray) -> dict:
        """
        Score a single sequence of shape (50, 7).

        Returns:
          anomaly_score : float ∈ [0, 1]  (0=legitimate, 1=impostor)
          raw_decision  : float (Combined reconstruction + latent distance score)
          verdict       : 'legitimate' | 'uncertain' | 'anomaly'
          threshold     : the T_anomaly used
        """
        if self._profile is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        p = self._profile
        self.model.load_state_dict(p.model_state_dict)
        self.model.eval()

        # Scale sequence
        seq_scaled = p.scaler.transform(sequence)
        
        # PyTorch Scoring
        device = self.device
        feature_weights = torch.tensor([1.2, 1.0, 1.4, 0.5, 0.3, 0.3, 0.8], device=device)
        
        batch = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, latent = self.model(batch)
            diff = (recon - batch) ** 2
            raw_recon = float((diff * feature_weights).mean().item())
            
        latent_vec = latent.cpu().numpy()[0]
        latent_dist = float(np.linalg.norm(latent_vec - p.latent_centroid))

        # Normalize score components relative to training maximums
        norm_recon = raw_recon / p.t_anomaly_raw if p.t_anomaly_raw > 0 else raw_recon
        norm_latent = latent_dist / p.latent_radius if p.latent_radius > 0 else latent_dist

        # Combined score (averaging components)
        combined_score = 0.5 * norm_recon + 0.5 * norm_latent

        # Normalise: 0.0 = right at threshold, scale by threshold magnitude
        if p.t_anomaly > 0:
            norm_score = combined_score / (p.t_anomaly * 2.0)
        else:
            norm_score = combined_score / 2.0
        anomaly_score = float(np.clip(norm_score, 0.0, 1.0))

        verdict = _verdict(combined_score, p.t_anomaly)

        return {
            'anomaly_score': anomaly_score,
            'raw_decision': combined_score,
            'verdict': verdict,
            'threshold': p.t_anomaly,
        }

    def score_session(self, window_scores: list[dict]) -> dict:
        """Aggregate multiple window scores into a session verdict."""
        if not window_scores:
            return {'session_verdict': 'unknown', 'anomaly_rate': 0.0}

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
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert state_dict to cpu before saving to avoid GPU dependency on load
        state_dict_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        profile_data = {
            'model_type': 'lstm',
            'model_state_dict': state_dict_cpu,
            'scaler': self._profile.scaler,
            't_anomaly': self._profile.t_anomaly,
            't_anomaly_raw': self._profile.t_anomaly_raw,
            'enrollment_mean': self._profile.enrollment_mean,
            'enrollment_std': self._profile.enrollment_std,
            'latent_centroid': self._profile.latent_centroid,
            'latent_radius': self._profile.latent_radius,
            'n_windows_trained': self._profile.n_windows_trained,
            'eer_estimate': self._profile.eer_estimate,
        }
        with open(path, 'wb') as f:
            pickle.dump(profile_data, f)

    def load(self, path: str | Path) -> None:
        with open(path, 'rb') as f:
            profile_data = pickle.load(f)
            
        self._profile = LSTMEnrollmentProfile(
            model_state_dict=profile_data['model_state_dict'],
            scaler=profile_data['scaler'],
            t_anomaly=profile_data['t_anomaly'],
            t_anomaly_raw=profile_data.get('t_anomaly_raw', profile_data['t_anomaly'] * 0.7),
            enrollment_mean=profile_data['enrollment_mean'],
            enrollment_std=profile_data['enrollment_std'],
            latent_centroid=profile_data['latent_centroid'],
            latent_radius=profile_data['latent_radius'],
            n_windows_trained=profile_data['n_windows_trained'],
            eer_estimate=profile_data.get('eer_estimate'),
        )
        self.model.load_state_dict(self._profile.model_state_dict)
        self.model.to(self.device)

    @property
    def is_trained(self) -> bool:
        return self._profile is not None

    @property
    def profile(self) -> Optional[LSTMEnrollmentProfile]:
        return self._profile


def _verdict(raw: float, threshold: float) -> str:
    lo = threshold * 0.6
    hi = threshold * 1.4
    if raw <= lo:
        return 'legitimate'
    elif raw <= hi:
        return 'uncertain'
    return 'anomaly'
