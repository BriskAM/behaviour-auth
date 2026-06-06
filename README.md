# BehaveGuard
> Continuous Behavioral Authentication via Keystroke Dynamics

BehaveGuard is a continuous authentication system that identifies users by **how** they type (their motor rhythm and habits) rather than what they type. It records keypress and release timing signals silently, processes them through a sliding window, and evaluates them using one-class machine learning models to detect impostors or anomalies in real time.

This repository supports two models:
1. **One-Class Support Vector Machine (SVM)**: A baseline classification model trained on aggregate statistics of sequence windows (e.g., mean, standard deviation, skewness, and kurtosis of dwell and flight times).
2. **LSTM Autoencoder (LSTM-AE)**: A deep-learning model that processes sequences of raw keystroke timing features to evaluate reconstruction error.

---

## How It Works

### Feature Engineering
For every key press and release event, BehaveGuard extracts a 7-dimensional normalized feature vector:
* **Dwell time**: Duration a key is held down.
* **Flight time**: Interval between releasing a key and pressing the next one.
* **Digraph time**: Down-to-down time between consecutive keys.
* **Key Category**: One-hot encoded category (`alphanum`, `symbol`, or `special`).
* **Digraph Frequency Weight**: Relative frequency weight based on common English digraphs.

### Model Architectures
* **SVM Baseline**: Trains on 43 aggregate window features.
* **LSTM Autoencoder**: Trains on sequence windows of shape `(50, 7)`. The model encodes the sequences into a 16-dimensional latent space (representing the user's behavioral fingerprint) and decodes them back to the original dimension. The reconstruction MSE is used as the anomaly score.

---

## Compatibility: Using Old SVM Profiles
Yes! **You can use your old trained SVM profiles without retraining.**

The loader logic inside the pipeline automatically inspects saved user profiles:
- If it detects a dictionary with `'model_type': 'lstm'`, it will instantiate the **LSTM Autoencoder** scoring flow.
- If it detects an `EnrollmentProfile` (legacy SVM profile format), it will fallback to the **One-Class SVM** scoring flow.

This allows you to test both models interchangeably.

---

## Installation

Ensure you have Python 3.12+ installed. This project uses [uv](https://github.com/astral-sh/uv) for package and environment management.

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/BriskAM/behaviour-auth.git
   cd behaviour-auth
   ```

2. Sync or install dependencies:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using standard pip
   pip install -r requirements.txt
   pip install torch
   ```

---

## Usage

BehaveGuard has a CLI interface under `main.py` to handle enrollment and live scoring.

### 1. Enrollment
Enrollment requires typing through 3 segments (Pangrams, Natural Typing, and Validation/Calibration) to record your baseline typing signature (takes about 5 minutes).

```bash
# Enroll a new subject using the LSTM Autoencoder (default)
python main.py enroll --subject alice --model lstm

# Enroll a new subject using the baseline SVM
python main.py enroll --subject alice --model svm
```

### 2. Live Scoring
Run live scoring in the background. The terminal displays a real-time status of current keystroke counts, words per minute (WPM), and anomaly verdicts for each overlapping window of 50 keystrokes.

```bash
# Run a live scoring session (auto-detects and loads model from profile)
python main.py score --subject alice --duration 300
```

---

## Testing

Verify the feature extraction, PyTorch model dimensions, wrapper fit/score, and save/load persistence:
```bash
python scratch/test_lstm.py
```
