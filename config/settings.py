"""
FPS Aim Performance Analyzer - Configuration Settings
Optimized for Valorant at 1920x1080 with NVIDIA GTX 1650
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATED_DATA_DIR = DATA_DIR / "annotated"
SESSIONS_DIR = DATA_DIR / "sessions"
EXPORTS_DIR = DATA_DIR / "exports"
TRAINING_DIR = PROJECT_ROOT / "training"

# Create directories if they don't exist
for d in [MODELS_DIR, RAW_DATA_DIR, ANNOTATED_DATA_DIR, SESSIONS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Screen Capture Settings
# ============================================================
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CAPTURE_FPS = 60  # Target capture framerate
CAPTURE_REGION = None  # None = full screen, or (left, top, right, bottom)

# ============================================================
# YOLO Detection Settings
# ============================================================
YOLO_MODEL_PATH = str(MODELS_DIR / "best.pt")  # Custom trained model
YOLO_FALLBACK_MODEL = "yolov8n.pt"  # Pretrained fallback
YOLO_CONFIDENCE = 0.5  # Minimum confidence threshold
YOLO_IOU_THRESHOLD = 0.45  # IoU threshold for NMS
YOLO_IMAGE_SIZE = 640  # Input image size for YOLO
YOLO_DEVICE = "0"  # GPU device (0 = first NVIDIA GPU, "cpu" for CPU)
YOLO_HALF_PRECISION = True  # FP16 inference for GTX 1650

# ============================================================
# Detection Classes
# ============================================================
CLASS_NAMES = {
    0: "crosshair",
    1: "enemy_head",
    2: "enemy_body",
    3: "target",
}

# ============================================================
# Crosshair Detection Settings
# ============================================================
# For Valorant, crosshair is typically at screen center
CROSSHAIR_DEFAULT_X = SCREEN_WIDTH // 2   # 960
CROSSHAIR_DEFAULT_Y = SCREEN_HEIGHT // 2  # 540
CROSSHAIR_SEARCH_RADIUS = 50  # pixels from center to search for crosshair
CROSSHAIR_USE_SCREEN_CENTER = True  # If True, assume crosshair = screen center

# ============================================================
# Aim Metrics Thresholds
# ============================================================
# Aim Accuracy
ON_TARGET_THRESHOLD = 30  # pixels - distance threshold to count as "on target"
ON_TARGET_HEAD_THRESHOLD = 15  # pixels - stricter threshold for headshots

# Time-to-Target
TTT_TARGET_APPEAR_THRESHOLD = 0.5  # confidence threshold for "target appeared"
TTT_ALIGNMENT_THRESHOLD = 25  # pixels - threshold for "crosshair aligned"

# Overshoot
OVERSHOOT_MIN_MOVEMENT = 20  # minimum pixels of movement to count as an aim event
OVERSHOOT_REVERSAL_THRESHOLD = 5  # pixels - minimum reversal to count as overshoot

# Consistency
CONSISTENCY_WINDOW_SIZE = 30  # frames - window for computing consistency

# Kinematics
KINEMATICS_SMOOTHING_WINDOW = 11  # Savitzky-Golay filter window (must be odd)
KINEMATICS_POLY_ORDER = 3  # Savitzky-Golay polynomial order

# Flick vs Tracking Classification
FLICK_MAX_DURATION_MS = 300  # Maximum duration for a flick (milliseconds)
TRACKING_MIN_DURATION_MS = 500  # Minimum duration for tracking
FLICK_VELOCITY_RATIO_THRESHOLD = 4.0  # Peak/mean velocity ratio threshold
TRACKING_CORRELATION_THRESHOLD = 0.6  # Crosshair-target velocity correlation

# ============================================================
# Aim Movement Segmentation
# ============================================================
MOVEMENT_IDLE_THRESHOLD = 3  # pixels/frame - below this = idle
MOVEMENT_MIN_DURATION_FRAMES = 3  # minimum frames to count as a movement
ENGAGEMENT_GAP_FRAMES = 15  # frames without target = engagement ends

# ============================================================
# Visualization Settings
# ============================================================
HEATMAP_RESOLUTION = (192, 108)  # Downscaled resolution for heatmaps
TRAJECTORY_LINE_WIDTH = 2
TRAJECTORY_COLORMAP = "RdYlBu_r"  # Blue(slow) -> Red(fast)

# ============================================================
# Session Settings
# ============================================================
SESSION_AUTO_SAVE = True
SESSION_FORMAT = "json"  # "json" or "csv"

# ============================================================
# Dashboard Theme
# ============================================================
DASHBOARD_THEME = {
    "primary_color": "#7C3AED",       # Purple
    "secondary_color": "#06B6D4",     # Cyan
    "accent_color": "#F59E0B",        # Amber
    "success_color": "#10B981",       # Green
    "danger_color": "#EF4444",        # Red
    "background_dark": "#0F172A",     # Slate 900
    "background_card": "#1E293B",     # Slate 800
    "text_primary": "#F8FAFC",        # Slate 50
    "text_secondary": "#94A3B8",      # Slate 400
}

# ============================================================
# Skill Level Thresholds (for classification)
# ============================================================
SKILL_LEVELS = {
    "beginner": {
        "aim_accuracy": (0, 30),        # 0-30%
        "ttt_ms": (800, float("inf")),  # >800ms
        "overshoot_ratio": (0.5, 1.0),  # 50-100%
        "consistency": (0, 0.4),        # 0-40%
    },
    "intermediate": {
        "aim_accuracy": (30, 60),
        "ttt_ms": (400, 800),
        "overshoot_ratio": (0.2, 0.5),
        "consistency": (0.4, 0.7),
    },
    "advanced": {
        "aim_accuracy": (60, 100),
        "ttt_ms": (0, 400),
        "overshoot_ratio": (0, 0.2),
        "consistency": (0.7, 1.0),
    },
}

# ============================================================
# Valorant-Specific Settings
# ============================================================
VALORANT = {
    "name": "Valorant",
    "resolution": (1920, 1080),
    "crosshair_center": True,  # Crosshair is always at screen center
    "typical_head_height_ratio": 0.35,  # Head height is ~35% from top of screen
    "ui_regions_exclude": {
        "minimap": (0, 0, 250, 250),       # Top-left minimap
        "abilities": (700, 980, 1220, 1080),  # Bottom-center abilities
        "scoreboard": (800, 0, 1120, 40),   # Top-center score
    },
}
