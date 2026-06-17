# FPS Aim Performance Analyzer 🎯

A YOLO-based system for evaluating aim performance in tactical FPS games (Valorant, CS2, R6 Siege). Uses object detection to track crosshair and target positions, then computes quantitative performance metrics for players and coaches.

## Features

- **Real-Time Analysis** — Live screen capture with DXcam + YOLO detection
- **Video Analysis** — Upload gameplay recordings for frame-by-frame analysis
- **Comprehensive Metrics** — Aim Accuracy Rate, Time-to-Target, Overshoot Ratio, Crosshair Placement Consistency, Kinematics (LDLJ/SPARC), Flick vs Tracking Classification
- **Player Comparison** — Compare skill profiles across sessions or players
- **Model Training** — Built-in pipeline for training custom YOLO detection models
- **Interactive Dashboard** — Streamlit-based UI with Plotly visualizations

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Analysis Modes

- **Screen Center Mode** (no YOLO required): For Valorant, crosshair is always at screen center. Use this mode without a trained model.
- **YOLO Mode**: Train a custom model for crosshair and target detection (see Training section).

## Project Structure

```
TugasAkhirPengolahanCitra/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Python dependencies
├── config/
│   └── settings.py                 # Global configuration
├── src/
│   ├── capture/                    # Screen capture & video input
│   ├── detection/                  # YOLO object detection
│   ├── tracking/                   # Position tracking & trajectories
│   ├── metrics/                    # Aim performance metrics
│   └── visualization/              # Plotly chart builders
├── pages/                          # Streamlit dashboard pages
├── training/                       # YOLO model training pipeline
├── models/                         # Trained model weights
├── data/                           # Session data & exports
└── tests/                          # Unit tests
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Aim Accuracy Rate (AAR)** | Proportion of time crosshair is within target hitbox |
| **Time-to-Target (TTT)** | Time from target appearance to crosshair alignment |
| **Overshoot Ratio** | Frequency and magnitude of crosshair overshooting |
| **Crosshair Placement Consistency** | Variance and BCEA of crosshair positioning |
| **LDLJ** | Log Dimensionless Jerk — movement smoothness metric |
| **SPARC** | Spectral Arc Length — frequency-domain smoothness |
| **Fitts' Law Throughput** | Efficiency metric from movement time vs. difficulty |
| **Flick vs Tracking** | Classification of aiming movement patterns |

## Training Custom YOLO Model

1. **Extract Frames**: `python training/prepare_dataset.py extract gameplay.mp4 --fps 5`
2. **Annotate**: Use [Roboflow](https://roboflow.com) to label crosshairs and targets
3. **Prepare Dataset**: `python training/prepare_dataset.py split data/raw/frames`
4. **Train**: `python training/train_model.py --epochs 100 --model yolov8n.pt`

## System Requirements

- **Python**: 3.9+
- **GPU**: NVIDIA GTX 1650 or better (CUDA support)
- **Resolution**: Optimized for 1920×1080
- **OS**: Windows 10/11

## References

- Asmara et al. (2024) — YOLO-based object detection for FPS games
- XGuardian (Zhang et al., 2026) — Temporal features for aim trajectory analysis
- Fitts' Law — Movement time prediction model

## License

This project is developed for academic research purposes.
