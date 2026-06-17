"""
Real-Time Analysis Page
========================
Live screen capture with YOLO detection and real-time metrics display.
"""

import streamlit as st
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

st.set_page_config(page_title="Analisis Real-Time - Tugas Akhir Pengolahan Citra", layout="wide")

# ============================================================
# Page Header
# ============================================================
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# Analisis Real-Time (Live Screen Capture)")
    st.markdown("Analisis gerakan aim secara langsung dengan melakukan capture layar secara real-time.")

with col_status:
    if st.session_state.get("is_capturing", False):
        st.error("PEREKAMAN LIVE AKTIF")

st.markdown("---")

# ============================================================
# Session State
# ============================================================
if "is_capturing" not in st.session_state:
    st.session_state.is_capturing = False
if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "frames_processed": 0,
        "start_time": None,
        "crosshair_positions": [],
        "target_positions": [],
        "distances": [],
        "timestamps": [],
    }
if "capture_engine" not in st.session_state:
    st.session_state.capture_engine = None

# ============================================================
# Control Panel
# ============================================================
st.markdown("### Pengaturan Capture")

col_set1, col_set2, col_set3, col_set4 = st.columns(4)

with col_set1:
    capture_fps = st.number_input("Target FPS Capture", min_value=10, max_value=120, value=60, step=10)

with col_set2:
    capture_region = st.selectbox(
        "Area Tangkapan Layar",
        ["Full Screen", "Tengah (800x600)", "Tengah (1280x720)"],
        index=0,
    )

with col_set3:
    use_yolo = st.checkbox("Gunakan YOLOv8", value=True, help="Centang untuk menggunakan model kustom best.pt")

with col_set4:
    show_preview = st.checkbox("Tampilkan Preview", value=True, help="Tampilkan visualisasi tangkapan layar")

# ============================================================
# Controls
# ============================================================
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    start_btn = st.button(
        "Mulai Capture",
        disabled=st.session_state.is_capturing,
        use_container_width=True,
        type="primary",
    )

with col_btn2:
    stop_btn = st.button(
        "Hentikan Capture",
        disabled=not st.session_state.is_capturing,
        use_container_width=True,
    )

with col_btn3:
    save_btn = st.button(
        "Simpan Hasil Sesi",
        disabled=st.session_state.session_data["frames_processed"] == 0,
        use_container_width=True,
    )

with col_btn4:
    reset_btn = st.button(
        "Reset Data",
        use_container_width=True,
    )

# ============================================================
# Handle Controls
# ============================================================
if start_btn:
    st.session_state.is_capturing = True
    st.session_state.session_data["start_time"] = time.time()
    st.session_state.session_data["frames_processed"] = 0
    st.session_state.session_data["crosshair_positions"] = []
    st.session_state.session_data["target_positions"] = []
    st.session_state.session_data["distances"] = []
    st.session_state.session_data["timestamps"] = []

    try:
        from src.capture import ScreenCapture
        region = None
        if capture_region == "Tengah (800x600)":
            left = (1920 - 800) // 2
            top = (1080 - 600) // 2
            region = (left, top, left + 800, top + 600)
        elif capture_region == "Tengah (1280x720)":
            left = (1920 - 1280) // 2
            top = (1080 - 720) // 2
            region = (left, top, left + 1280, top + 720)

        st.session_state.capture_engine = ScreenCapture(target_fps=capture_fps, region=region)
        st.session_state.capture_engine.start()
    except Exception as e:
        st.error(f"Gagal mengaktifkan capture engine: {e}")
        st.session_state.is_capturing = False

if stop_btn:
    st.session_state.is_capturing = False
    if st.session_state.capture_engine is not None:
        try:
            st.session_state.capture_engine.stop()
        except Exception:
            pass

if reset_btn:
    st.session_state.is_capturing = False
    if st.session_state.capture_engine is not None:
        try:
            st.session_state.capture_engine.stop()
        except Exception:
            pass
    st.session_state.capture_engine = None
    st.session_state.session_data = {
        "frames_processed": 0,
        "start_time": None,
        "crosshair_positions": [],
        "target_positions": [],
        "distances": [],
        "timestamps": [],
    }
    st.success("Data sesi telah di-reset.")

if save_btn:
    session = st.session_state.session_data
    if session["frames_processed"] > 0:
        save_path = settings.SESSIONS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "frames_processed": session["frames_processed"],
            "duration_seconds": time.time() - session["start_time"] if session["start_time"] else 0,
            "crosshair_positions": [pos for pos in session["crosshair_positions"]],
            "target_positions": [pos for pos in session["target_positions"]],
            "distances": [float(d) if d is not None else None for d in session["distances"]],
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        st.success(f"Sesi berhasil disimpan ke `{save_path.name}`")

st.markdown("---")

# ============================================================
# Live Dashboard
# ============================================================
if st.session_state.is_capturing:
    st.markdown("### Metrik Real-Time")

    # Metrics placeholders
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        frames_metric = st.empty()
    with m2:
        fps_metric = st.empty()
    with m3:
        accuracy_metric = st.empty()
    with m4:
        distance_metric = st.empty()
    with m5:
        elapsed_metric = st.empty()

    # Preview and chart columns
    col_preview, col_chart = st.columns([1, 1])

    with col_preview:
        preview_placeholder = st.empty()

    with col_chart:
        chart_placeholder = st.empty()

    import plotly.graph_objects as go
    import cv2

    # -------------------------------------------------------
    # REAL CAPTURE MODE
    # -------------------------------------------------------
    if st.session_state.capture_engine is not None:
        capture = st.session_state.capture_engine

        # Lazily initialise YOLO detector
        if "yolo_detector" not in st.session_state:
            st.session_state.yolo_detector = None
            if use_yolo:
                try:
                    from src.detection import YOLODetector
                    model_path = Path(settings.YOLO_MODEL_PATH)
                    if model_path.exists():
                        st.session_state.yolo_detector = YOLODetector(model_path=str(model_path))
                        st.success("Model YOLOv8 berhasil dimuat.")
                except Exception as exc:
                    st.warning(f"Gagal memuat YOLOv8: {exc}")

        detector = st.session_state.yolo_detector

        if not capture.is_capturing:
            try:
                capture.start()
            except Exception:
                pass

        st.info("Perekaman Berjalan — Silakan arahkan fokus ke game Valorant (Gunakan mode Windowed Fullscreen).")

        BATCH_SIZE = 120  # Process frames in batches then update Streamlit

        for _ in range(BATCH_SIZE):
            if not st.session_state.is_capturing:
                break

            frame, ts = capture.get_latest_frame()
            if frame is None:
                time.sleep(1 / 60)
                continue

            session = st.session_state.session_data
            session["frames_processed"] += 1
            elapsed = time.time() - session["start_time"] if session["start_time"] else 0
            t = elapsed

            h, w = frame.shape[:2]

            # Crosshair: screen centre for Valorant
            cx, cy = w / 2.0, h / 2.0
            tx, ty = None, None

            if detector is not None:
                try:
                    detections = detector.detect(frame)
                    targets = [d for d in detections if d.class_name in ["enemy_head", "enemy_body", "target", "head", "body"]]
                    if targets:
                        nearest = min(
                            targets,
                            key=lambda d: (d.center[0] - cx) ** 2 + (d.center[1] - cy) ** 2,
                        )
                        dist = np.sqrt((nearest.center[0] - cx) ** 2 + (nearest.center[1] - cy) ** 2)
                        if dist <= 350.0:  # Distance filter to prevent background clutter
                            tx, ty = nearest.center
                except Exception:
                    pass

            if tx is None:
                dist = None
            else:
                dist = float(np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))

            session["crosshair_positions"].append([cx, cy])
            session["target_positions"].append([tx, ty])
            session["distances"].append(dist)
            session["timestamps"].append(t)

            # Update metrics every 5 frames
            if session["frames_processed"] % 5 == 0:
                valid_distances = [d for d in session["distances"] if d is not None]
                on_target = sum(1 for d in valid_distances if d < settings.ON_TARGET_THRESHOLD)
                accuracy = (on_target / len(valid_distances) * 100 if valid_distances else 0)
                avg_dist = float(np.mean(valid_distances[-60:])) if valid_distances else 0

                frames_metric.metric("Frame", session["frames_processed"])
                fps_metric.metric("FPS", f"{session['frames_processed'] / max(elapsed, 0.01):.1f}")
                accuracy_metric.metric("Akurasi", f"{accuracy:.1f}%")
                distance_metric.metric("Jarak Rata-rata", f"{avg_dist:.1f} px")
                elapsed_metric.metric("Durasi Sesi", f"{elapsed:.1f} s")

            # Preview frame every 15 frames
            if show_preview and session["frames_processed"] % 15 == 0:
                try:
                    display = frame.copy()
                    cv2.drawMarker(display, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                    cv2.circle(display, (int(cx), int(cy)), 12, (0, 255, 0), 2)

                    if tx is not None and ty is not None:
                        cv2.circle(display, (int(tx), int(ty)), 16, (0, 0, 255), 2)
                        cv2.line(display, (int(cx), int(cy)), (int(tx), int(ty)), (0, 255, 255), 1)

                    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(
                        display,
                        caption=f"Frame {session['frames_processed']}",
                        use_container_width=True,
                    )
                except Exception:
                    pass

            # Chart every 30 frames
            if session["frames_processed"] % 30 == 0:
                valid_distances = [d for d in session["distances"] if d is not None]
                if len(valid_distances) > 5:
                    recent_distances = valid_distances[-200:]
                    recent_times = [session["timestamps"][i] for i, d in enumerate(session["distances"]) if d is not None][-200:]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recent_times,
                        y=recent_distances,
                        mode="lines",
                        line=dict(color="#1f77b4", width=2),
                        name="Jarak",
                    ))
                    fig.add_hline(y=settings.ON_TARGET_THRESHOLD, line_dash="dash", line_color="#2ca02c")
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        xaxis_title="Waktu (detik)",
                        yaxis_title="Jarak (pixel)",
                        title="Grafik Jarak Crosshair ke Target (Real-time)",
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(1 / capture_fps)

        if st.session_state.is_capturing:
            st.rerun()
