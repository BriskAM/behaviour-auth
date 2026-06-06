# BehaveGuard — Complete System Design Guide
> Behavioral Continuous Authentication via Keystroke & Mouse Dynamics

---

## Table of Contents
1. [What We Are Building](#what-we-are-building)
2. [The Core Problem](#the-core-problem)
3. [What We Measure](#what-we-measure)
4. [Feature Design](#feature-design)
5. [Data Requirements](#data-requirements)
6. [Model Architecture](#model-architecture)
7. [Loss Functions](#loss-functions)
8. [Threshold System](#threshold-system)
9. [Continuous Learning & Drift](#continuous-learning--drift)
10. [Poisoning Detection](#poisoning-detection)
11. [False Positive Taxonomy](#false-positive-taxonomy)
12. [Fallback Layer](#fallback-layer)
13. [Mouse Dynamics](#mouse-dynamics)
14. [System Workflow](#system-workflow)
15. [API Layer](#api-layer)
16. [Live Stats & Demo](#live-stats--demo)
17. [Research Experiments](#research-experiments)
18. [Resume & SDE Framing](#resume--sde-framing)

---

## What We Are Building

A **continuous behavioral authentication system** that identifies users by *how* they type and move — not what they know or own. After a 5-minute enrollment session, the system silently scores every interaction window in the background and flags anomalies in real time.

**Project name:** BehaveGuard  
**Tagline:** "Authenticates you by how you move, not what you know"  
**SDE framing:** Real-time behavioral signal processing pipeline with sub-100ms inference, cross-platform deployment, and adaptive ML

---

## The Core Problem

This is **one-class classification**, not binary classification.

```
Binary classification:  "Is this user A or user B?"
One-class (ours):       "Does this behavior match user X or not?"
```

You only have genuine data at training time. You never know who the impostors will be. This means:
- Train exclusively on the real user's enrollment data
- Other users' data is used only at **evaluation time** to measure FAR
- Never let negative samples leak into training — this is the most common mistake in published papers

**Identity verification vs anomaly detection:**
- Anomaly detection: "Is this behavior unusual?"
- Identity verification: "Does this behavior match the stored profile?"

We want identity verification. A tired user is unusual but not an impostor. These are different failure modes and must be handled separately.

---

## What We Measure

### Keystroke Dynamics

```
Key press ──────────────── Key release ── Next key press
           |← dwell time →|             |
           |←───────── digraph time ──────────────────→|
                           |← flight time →|
```

| Feature | Definition | Identity signal |
|---|---|---|
| **Dwell time** | Duration key is held down | Strong — motor memory per key |
| **Flight time** | Gap between release and next press | Strong — transition habits |
| **Digraph time** | Down-to-down between consecutive keys | Strongest — full rhythm |
| **Key category** | alphanum / symbol / special | Medium — usage pattern |

**High-frequency digraphs carry the most identity signal:**
`th`, `he`, `in`, `er`, `an`, `re`, `on`, `en` — appear dozens of times per minute, enough for stable estimation.

Rare pairs (`qz`, `xk`) appear too infrequently to model reliably. Weight them less in the loss function.

### Mouse Dynamics

| Feature | Definition | Identity signal |
|---|---|---|
| **Speed** | px/second instantaneous | Medium |
| **Acceleration** | Rate of speed change | Medium |
| **Curvature** | How sharply path bends | Strong — motor control |
| **Angular velocity** | Direction change rate | Strong |
| **Click dwell** | How long mouse button held | Medium |
| **Inter-click interval** | Time between clicks | Medium |
| **Pause duration** | Stillness before movement | Weak |
| **Trajectory shape** | Arc vs straight line | Strong |

**Do not store raw x/y coordinates long-term.** After feature extraction, discard raw positions. Absolute coordinates allow screen layout inference, which is a real privacy concern.

---

## Feature Design

### Per-event feature vector (keystroke)

```python
event_vector = [
    dwell_ms,           # normalized by user mean
    flight_ms,          # normalized by user mean  
    digraph_ms,         # normalized by user mean
    key_category,       # one-hot: [alphanum, symbol, special]
    time_sin,           # sin(2π * hour/24) — cyclical time encoding
    time_cos,           # cos(2π * hour/24)
    digraph_frequency,  # how common is this pair? (weight signal)
]
# Shape: [7]
```

### Per-event feature vector (mouse)

```python
mouse_vector = [
    speed_px_s,         # normalized
    acceleration,       # normalized
    curvature,          # normalized
    angular_velocity,   # normalized
    action_type,        # one-hot: [move, click, scroll, drag]
    time_sin,
    time_cos,
]
# Shape: [7]
```

### Why cyclical time encoding matters

```python
def encode_time(hour, minute):
    fraction = (hour * 60 + minute) / 1440
    return np.sin(2 * np.pi * fraction), np.cos(2 * np.pi * fraction)
```

- Captures 2am vs 1pm behavioral differences
- Cyclical: 23:59 is close to 00:01 (no discontinuity at midnight)
- Privacy-safe: cannot be reverse-engineered to exact timestamp
- Does NOT store when you typed specific things — only what part of day

### Are features stationary?

**No.** Known sources of non-stationarity:

| Source | Effect on features | Pattern |
|---|---|---|
| Time of day | All timings shift up (late night) | Location shift |
| Fatigue | Progressive slowdown within session | Monotonic trend |
| Stress/caffeine | Higher variance, faster tempo | Scale change |
| Cold hands | Longer dwell times specifically | Selective shift |
| Different keyboard | Consistent offset all features | Uniform shift |
| Illness | Slower, more errors | Location + variance |

**Important:** Non-stationarity is not the same as impostors. A tired user's pattern shifts but preserves its **shape**. An impostor has a completely different shape. This distinction drives the false positive taxonomy.

---

## Data Requirements

| Model | Minimum keystrokes | Good | Plateau |
|---|---|---|---|
| One-class SVM | 400 (~2 min) | 1000 | 2000 |
| LSTM-AE | 1500 (~5 min) | 3000 | 5000 |
| TCN-AE | 1000 (~4 min) | 2500 | 4500 |
| Global backbone + finetune | 500 (~2 min) | 1000 | 2000 |

**Quality beats quantity.** 1500 keystrokes of focused enrollment beats 5000 keystrokes of distracted typing.

### Enrollment structure (5 minutes total)

**Segment 1 — Pangram repetition (2 min)**  
Same sentences typed repeatedly. Strips content variability, isolates pure timing consistency. Trains digraph timing model cleanly.

**Segment 2 — Natural typing (2 min)**  
Free-form text. Captures hesitation, correction, backspace patterns. More representative of real usage.

**Segment 3 — Structured interactions (1 min)**  
Click between fields, scroll, fill form. Captures mouse dynamics naturally.

Train on segments 1+2, validate threshold on segment 3. Clean train/val split with no leakage.

---

## Model Architecture

### Build order (do this in sequence)

```
Week 1-2:  One-class SVM — baseline EER, verify pipeline
Week 3-4:  LSTM-AE — primary model
Week 5:    TCN-AE — compare against LSTM-AE
Week 6:    VAE upgrade — better calibration
Week 7:    Fusion layer — headline EER
```

### One-class SVM (baseline)

Trains on 40-50 statistical features per window (mean, std, skew, kurtosis of dwell/flight/digraph).  
`nu` hyperparameter directly controls FAR/FRR tradeoff.  
Expected EER: 10-15% on keystroke-only data.

### LSTM Autoencoder (primary)

```
Input sequence [50, 7]
        ↓
Encoder LSTM layer 1: 64 units
Encoder LSTM layer 2: 32 units
Take last hidden state
        ↓
Latent vector [16]  ← user's behavioral fingerprint
        ↓
Repeat 50 times → [50, 16]
Decoder LSTM layer 1: 32 units
Decoder LSTM layer 2: 64 units
Dense → [50, 7]
        ↓
Reconstructed sequence
        ↓
Reconstruction error = MSE(input, reconstruction)
                     = anomaly score
```

Trains on genuine sessions only. Low error = matches profile. High error = anomaly.

### TCN Autoencoder (recommended primary)

Uses dilated causal convolutions instead of recurrent units.

```
Dilation rates: 1, 2, 4, 8, 16
Each layer sees context of 2^n events back
Processes entire sequence in parallel (not sequential)
```

**TCN vs LSTM comparison:**

| Property | LSTM-AE | TCN-AE |
|---|---|---|
| Training speed (CPU) | Slow | Fast |
| Inference speed (CPU) | ~8ms | ~3ms |
| Small dataset performance | Good | Better |
| Vanishing gradient | Yes | No |
| Parallelizable | No | Yes |

**Verdict:** TCN-AE for production. LSTM-AE as comparison baseline in paper.

### Global backbone + personal fine-tuning (meta-learning)

```
Phase 1: Pretrain global model on all users (CMU + collected data)
         Learns universal keystroke rhythm representations
         Encoder learns: "what makes typing patterns in general"

Phase 2: Per-user fine-tuning
         Freeze encoder weights
         Fine-tune only decoder (20 epochs, lr=1e-4)
         Decoder learns: "how to reconstruct THIS user specifically"
```

**Why this matters:** New user with 500 keystrokes outperforms independent model trained on 2000 keystrokes. The global backbone does the heavy lifting; the fine-tune layer personalizes it.

This is a publishable contribution. Meta-learning for behavioral biometric authentication has not been done cleanly in the literature.

### VAE upgrade

Adds variational component to the autoencoder. Learns a **distribution** over latent space instead of a single point. KL divergence term makes anomaly scores better calibrated.

More sensitive to hyperparameter tuning. Implement after plain AE is working well. Use β-VAE with tunable β to balance reconstruction vs KL term.

### Fusion layer

```python
# Each model outputs score ∈ [0,1]
svm_score    = svm.decision_function(features)
lstm_score   = lstm_ae.reconstruction_error(sequence)
tcn_score    = tcn_ae.reconstruction_error(sequence)

# Stacking meta-classifier (trained on validation set)
final_score  = logistic_regression([svm_score, lstm_score, tcn_score])
```

Stacking outperforms fixed weighted average because it learns non-linear interactions: "if SVM uncertain but TCN confident, trust TCN."

---

## Loss Functions

### Level 1 — MSE (baseline, not optimal)
```python
loss = F.mse_loss(reconstruction, input)
```

### Level 2 — Weighted MSE (better)
```python
# Tier 1 features weighted higher
feature_weights = torch.tensor([1.2, 1.0, 1.4, 0.5, 0.3, 0.3, 0.8])
loss = (F.mse_loss(reconstruction, input, reduction='none') 
        * feature_weights).mean()
```

### Level 3 — Reconstruction + Compactness
```python
recon_loss = weighted_mse(reconstruction, input)
# Pull enrollment latents into tight cluster
latent_mean = latents.mean(dim=0)
compactness = ((latents - latent_mean) ** 2).mean()
loss = recon_loss + beta * compactness
```

### Level 4 — Contrastive (best, for global pretraining)
```python
def contrastive_loss(genuine_latent, impostor_latent,
                     genuine_error, impostor_error, margin=1.0):
    # Latent space: genuine and impostor should be far apart
    latent_similarity = F.cosine_similarity(genuine_latent, impostor_latent)
    # Error space: impostor should reconstruct worse
    error_gap = genuine_error - impostor_error  # should be negative
    separation = F.relu(margin + error_gap)
    return latent_similarity + separation

total = recon_loss + gamma * contrastive_loss(...)
```

**Recommendation:** Level 2 for per-user fine-tuning. Level 4 for global pretraining (where you have multiple users available).

---

## Threshold System

### Two thresholds operating simultaneously

**T_anomaly** — detects impostors  
**T_drift** — detects suspicious behavioral change

```
                    Error ≤ T_anomaly    Error > T_anomaly
                  ┌──────────────────┬───────────────────┐
Drift ≤ T_drift   │   LEGITIMATE     │    UNCERTAIN      │
                  │   add to buffer  │  soft challenge   │
                  ├──────────────────┼───────────────────┤
Drift > T_drift   │   LEGITIMATE*    │    IMPOSTOR       │
                  │   flag for review│  lockout + alert  │
                  └──────────────────┴───────────────────┘
* Possibly new keyboard, injury — needs review, not lockout
```

### Setting T_anomaly
```python
# 95th percentile of enrollment reconstruction errors
enrollment_errors = [model.error(w) for w in enrollment_windows]
T_anomaly = np.percentile(enrollment_errors, 95)
```

### Setting T_drift
```python
# 3x the natural variance seen during enrollment
enrollment_distances = [mahalanobis(s, mean, cov) for s in enrollment]
T_drift = np.percentile(enrollment_distances, 95) * 3.0
# Multiplier is tunable — lower = tighter security
```

### Dynamic threshold (session-local adjustment)

```python
class DynamicThreshold:
    def current_threshold(self, session_errors):
        if len(session_errors) < 5:
            return self.base  # warmup period
        
        recent_mean = np.mean(session_errors[-10:])
        session_offset = recent_mean - self.enrollment_mean_error
        
        # Bound the shift — prevent manipulation
        bounded = np.clip(session_offset, -0.2 * self.base, +0.2 * self.base)
        return self.base + bounded
```

The ±20% bound is critical — without it, a slow impostor could inflate the threshold by starting carefully.

### Session-level scoring (not window-level)

```python
def score_session(window_scores, T):
    anomaly_rate = sum(s > T for s in window_scores) / len(window_scores)
    
    if anomaly_rate < 0.20:   return "LEGITIMATE"
    elif anomaly_rate > 0.60: return "IMPOSTOR"
    else:                      return "UNCERTAIN"
```

Don't flag on a single window — genuine users sneeze, stretch, get distracted. The session-level verdict is what matters.

---

## Continuous Learning & Drift

### Permanent vs temporary drift

| Type | Cause | Duration | Pattern | Response |
|---|---|---|---|---|
| **Temporary** | Tired, sick, stressed, caffeine | Hours–days | All features shift, returns to baseline | Widen threshold temporarily, do NOT retrain |
| **Permanent** | New keyboard, injury, aging | Weeks–months | Sustained trend, new baseline | Retrain on rolling window |

### Classifying drift type

```python
def classify_drift(recent_sessions, enrollment_baseline):
    distances = [mahalanobis(s, enrollment_baseline.mean, 
                             enrollment_baseline.cov) for s in recent_sessions]
    
    drift_variance = np.var(distances)
    drift_trend = np.polyfit(range(len(distances)), distances, 1)[0]
    
    if drift_variance > threshold_var and abs(drift_trend) < 0.01:
        return "TEMPORARY"   # bouncing around, not trending
    elif drift_trend > 0.02:
        return "PERMANENT"   # consistently moving away
    else:
        return "STABLE"
```

### Retraining logic

```python
def retrain(user_profile, recent_sessions):
    # 30% enrollment + 70% recent — never fully forget origin
    training = sample(enrollment, weight=0.3) + sample(recent, weight=0.7)
    
    new_model = train(training)
    new_threshold = calibrate(new_model, held_out_slice)
    
    # Atomic swap — never half-updated
    with scoring_lock:
        os.rename("model_pending.pt", "model_current.pt")
        profile.threshold = new_threshold
        profile.last_retrain = time.time()
    
    profile.log_retrain(reason="permanent_drift_detected")
```

The 30/70 split anchors the model to the original enrollment. It cannot drift arbitrarily far even after many retraining cycles.

### Nightly drift check

```python
def nightly_job(user_profile):
    recent = user_profile.legitimate_buffer
    if len(recent) < 5: return
    
    # KS test per feature dimension
    p_values = [ks_2samp(enrollment[:, i], recent[:, i]).pvalue 
                for i in range(n_features)]
    
    if not any(p < 0.05 for p in p_values): return  # no drift
    
    drift_type = classify_drift(recent, enrollment_baseline)
    
    if drift_type == "TEMPORARY":
        user_profile.widen_threshold(factor=1.5, duration_hours=24)
    elif drift_type == "PERMANENT":
        if mean_drift_distance < T_drift:
            retrain(user_profile, recent)
        else:
            alert("Large drift — possible poisoning")
```

---

## Poisoning Detection

Poisoning is when an attacker feeds the model clean-looking sessions to gradually shift the threshold or retrain buffer.

### Layer 1 — Rate limiting
```python
MAX_UPDATES_PER_DAY = 1
MAX_DRIFT_PER_UPDATE = 0.15  # 15% of enrollment variance
```

### Layer 2 — Consistency check
Genuine drift is correlated across features. Poisoning shifts only controllable features.

```python
def check_consistency(new_session, enrollment):
    feature_drifts = [
        abs(new_session[:, i].mean() - enrollment.mean[i]) / enrollment.std[i]
        for i in range(n_features)
    ]
    cv = np.std(feature_drifts) / np.mean(feature_drifts)
    return "SUSPICIOUS" if cv > 0.5 else "CONSISTENT"
```

### Layer 3 — Velocity check
Genuine drift is slow. Poisoning is fast.

```python
if max(daily_velocity_over_7_days) > MAX_DAILY_VELOCITY:
    freeze_updates()
    alert("Unusual profile velocity — possible poisoning")
```

### Layer 4 — Enrollment anchor (hard ceiling)
Regardless of all retraining, model can never move more than `MAX_TOTAL_DRIFT` from original enrollment. This is absolute and not adjustable by the continuous learning system.

---

## False Positive Taxonomy

Not all false positives are equal. Categorize them:

| Category | Signals | Pattern | Response |
|---|---|---|---|
| **Fatigue** | All timings increasing over session, rising error rate | Monotonic degradation | Widen threshold for session |
| **Different keyboard** | Consistent offset all features, rhythm preserved | Uniform location shift | Soft challenge, flag keyboard change |
| **Stress/caffeine** | Higher variance, faster tempo | Scale change | Soft challenge |
| **Physical impairment** | Specific features affected (dominant hand asymmetry) | Selective disruption | Soft challenge |
| **Genuine impostor** | Different distribution shape entirely, wrong digraph rhythms | Shape change, not just shift | Hard challenge → lockout |

**Key insight:** Categories 1-4 preserve distribution **shape** while shifting **location or scale**. Category 5 changes the shape entirely. Measure this with skewness and kurtosis of the reconstruction error distribution within a session.

---

## Fallback Layer

Never binary lock. Use graduated response:

```
Score 0.0–0.3  GREEN   Full access, log silently
Score 0.3–0.6  YELLOW  Soft challenge: re-type phrase, or captcha
                        Pass → GREEN. Fail → ORANGE
Score 0.6–0.8  ORANGE  Hard challenge: OTP, PIN, short re-enrollment
                        Pass → GREEN. Fail → RED
Score 0.8–1.0  RED     Session terminated, full re-auth required
```

The YELLOW soft challenge is the most important. It handles fatigue/keyboard-change false positives without locking out the real user. The re-typing challenge at YELLOW also gives you labeled data — you know the outcome.

### Special cases

**Cold start (first 3 sessions after enrollment):**  
Score and log but don't flag. Feed straight into retrain buffer. Model hasn't seen real usage patterns yet.

**Long absence (> 2 weeks):**  
If anomaly rate > 60% for 3 consecutive sessions, prompt 1-minute recalibration. Not full re-enrollment — just a re-anchor.

---

## Mouse Dynamics

### Per-event feature vector

```python
mouse_vector = [
    speed_px_s,          # instantaneous speed, normalized
    acceleration,        # rate of speed change
    curvature,           # path bend sharpness (key identity signal)
    angular_velocity,    # direction change rate
    action_type,         # [MM, PC, C, DD, S] one-hot
    click_dwell_ms,      # how long button held (clicks only)
    scroll_delta,        # normalized scroll amount
    time_sin,            # cyclical time encoding
    time_cos,
]
```

### Computing curvature

```python
def compute_curvature(x1, y1, x2, y2, x3, y3):
    # Three consecutive mouse positions
    # Curvature = how sharply the path bends at point 2
    v1 = np.array([x2-x1, y2-y1])
    v2 = np.array([x3-x2, y3-y2])
    cross = v1[0]*v2[1] - v1[1]*v2[0]  # 2D cross product
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6: return 0.0
    return cross / norm
```

### Minimum mouse events per window

Mouse events are sparser than keystroke events during typing. Adjust window strategy:

```python
MIN_MOUSE_EVENTS = 20  # lower threshold than keystroke minimum
# Or: use time-based windows (30s) for mouse, count-based for keyboard
# Fuse at score level — each has independent scoring
```

### Privacy note

Never store absolute x/y coordinates beyond feature extraction. After computing speed, curvature, etc., discard the raw positions. The derived features contain all identity signal with no reconstruction risk.

---

## System Workflow

### Phase 1 — Enrollment (one time, 5 minutes)

```
User opens app → Types through 3 structured segments
        ↓
Raw events: [key_code, press_ts, release_ts]
        ↓
Feature extraction per pair: [dwell, flight, digraph, key_cat, time_sin, time_cos, freq_weight]
        ↓
Train global backbone fine-tune (or train from scratch)
        ↓
Compute reconstruction error on held-out segment 3
        ↓
T_anomaly = 95th percentile of those errors
T_drift   = 3x natural mahalanobis distance variance
        ↓
Save: {model_weights, T_anomaly, T_drift, enrollment_mean, enrollment_cov,
       enrollment_latent_centroid, enrollment_latent_radius}
        ↓
App goes silent
```

### Phase 2 — Continuous evaluation (every session)

```
Background collector captures events
        ↓
Accumulate until 50 keystroke pairs (one window)
        ↓
Extract features → shape [50, 7]
        ↓
Encode → latent vector [16]
        ↓
Decode → reconstructed [50, 7]
        ↓
recon_error = MSE(input, reconstruction)
latent_dist = distance(latent, enrollment_centroid)
combined    = alpha * recon_error + (1-alpha) * latent_dist
        ↓
Compare to dynamic threshold
        ↓
Window verdict → logged to session
        ↓
Every 5 windows: compute session-level anomaly_rate
        ↓
Session ends → verdict + scores stored
              → if legitimate: add to drift buffer
```

### Phase 3 — Background retraining (nightly)

```
Enough sessions in buffer? (min 5) → NO → sleep
        ↓ YES
KS test: has drift occurred?        → NO → sleep
        ↓ YES
Classify: TEMPORARY or PERMANENT?
        ↓
TEMPORARY → widen T_anomaly ×1.5 for 24h, do not retrain
        ↓
PERMANENT → poisoning checks pass?  → NO → alert, freeze
        ↓ YES
Retrain on 30% enrollment + 70% recent
        ↓
Calibrate new threshold
        ↓
Atomic model swap
        ↓
Log retrain event
```

---

## API Layer

```
POST /enroll
  Input:  subject_id, keystroke_events[], mouse_events[]
  Output: profile_id, enrollment_quality_score, model_version

POST /score/window
  Input:  subject_id, keystroke_events[], mouse_events[]
  Output: {
    recon_error: float,
    latent_distance: float,
    combined_score: float,
    verdict: legitimate | uncertain | anomaly,
    threshold_used: float,
    threshold_type: base | session_adjusted
  }

POST /score/session
  Input:  subject_id, window_scores[]
  Output: {
    session_verdict: legitimate | uncertain | impostor,
    anomaly_rate: float,
    drift_distance: float,
    drift_type: stable | temporary | permanent,
    fp_category: fatigue | keyboard | stress | impairment | impostor | null,
    recommended_action: none | soft_challenge | hard_challenge | lockout
  }

GET /profile/{subject_id}/stats
  Output: enrollment stats, session history, drift history,
          model version, last retrain, EER estimate

GET /profile/{subject_id}/latent
  Output: enrollment latent vectors (for visualization)

POST /retrain/{subject_id}
  Input:  confirmed_legitimate_sessions[]
  Output: new_model_version, eer_delta

GET /health
  Output: uptime, inference_p99_ms, active_profiles
```

---

## Live Stats & Demo

### Real-time (updates every keystroke)
- Reconstruction error gauge — live bar vs threshold line
- Keyboard heatmap — dwell time per key (darker = longer hold)
- Last 10 digraph flight times — live bar chart
- Current WPM estimate
- Session anomaly rate

### Per-window (every 50 keystrokes)
- Reconstruction error vs personal baseline
- Latent vector dot in 2D UMAP projection (your cluster vs anonymous others)
- Window verdict chip — green / yellow / red

### Per-session
- Full session score timeline
- Anomaly windows highlighted
- Drift distance from enrollment
- False positive category if flagged

### Profile visualization
- Enrollment cluster in latent space vs other users
- Drift history: how your profile moved week-over-week
- **"Your fingerprint digraphs"** — the key pairs where you differ most from average. This is the hero visualization.
- Enrollment quality curve: EER vs keystrokes collected (live during enrollment)

### Enrollment quality display (during 5-minute test)
- Keystrokes collected so far / target
- Common digraph coverage bar
- Consistency score — are your timings stable?
- Estimated current EER
- "Keep typing to improve accuracy" progress bar

---

## Research Experiments

Minimum for a workshop paper: Experiments 1, 2, 3.  
Full journal submission: all six.

### Experiment 1 — Baseline comparison (required)
Methods: fixed threshold, one-class SVM, LSTM-AE, TCN-AE, fusion  
Metrics: EER, FAR@FRR=5%, FRR@FAR=1%  
Dataset: CMU keystroke + your collected data  
Result: TCN-AE and fusion outperform prior work

### Experiment 2 — Enrollment size curve (required)
Variable: enrollment keystrokes 200 → 5000  
Metric: EER at each size  
Result: minimum viable enrollment curve — practical result not cleanly reported in literature

### Experiment 3 — Global backbone vs independent (required)
Methods: independent TCN-AE vs global+finetune  
Especially test at low enrollment (200-500 keystrokes)  
Result: global model wins at low data — meta-learning contribution

### Experiment 4 — Temporal context
Methods: with vs without time-of-day encoding  
Metric: EER split by time bucket (morning/afternoon/night)  
Result: time encoding reduces off-hours EER

### Experiment 5 — Longitudinal drift adaptation
Collect same subjects over 4 weeks  
Methods: static model vs adaptive  
Metric: EER per week  
Result: adaptive model maintains accuracy; static degrades

### Experiment 6 — Poisoning resistance
Simulate poisoning: deliberately feed adversarial sessions into retrain buffer  
Metric: sessions until model shifts, with vs without detection layers  
Result: detection layers increase poisoning resistance by Nx

---

## Resume & SDE Framing

Frame this as a **real-time behavioral signal processing system**, not a security project.

```
BehaveGuard — Real-time Behavioral Signal Processing System

Engineered a cross-platform pipeline ingesting ~3,000 interaction
events/min, extracting 40+ temporal features via sliding-window
processing, and running ensemble ML inference (SVM + TCN-AE + VAE)
at <80ms p99 latency. Built a WebSocket-based live dashboard in React,
a Tauri cross-platform desktop agent (Win/Mac/Linux), and a Flutter
Android app. Designed REST + streaming API in FastAPI, containerized
with Docker. Global backbone meta-learning architecture achieves X%
EER with 3x less enrollment data vs independent models.
Preprint: arXiv:[id]. Dataset: huggingface.co/datasets/[id].
```

**Every sentence is an interview topic:**
- "3,000 events/min" → system design, throughput
- "40+ temporal features" → feature engineering
- "<80ms p99" → latency, performance engineering
- "WebSocket" → real-time systems
- "meta-learning" → ML depth
- "arXiv preprint" → research ability
- "published dataset" → contribution to field

**What makes interviewers stop scrolling:**
1. Live interactive demo — they can type and see their own fingerprint form
2. Every layer built — OS events → ML inference → polished UI
3. Cross-platform — web, desktop, mobile, all sharing same API
4. Numbers everywhere — EER, latency, throughput, dataset size
5. Paper submitted — shows research-grade work, not just a project

---

*Guide version 1.0 — covers full design discussion through initial planning phase*
