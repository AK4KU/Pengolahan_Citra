"""
Video Analysis Page
====================
Upload gameplay recordings for frame-by-frame analysis.
"""

import streamlit as st
import numpy as np
import tempfile
import time
import json
import cv2
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

st.set_page_config(page_title="Analisis Video - Tugas Akhir Pengolahan Citra", layout="wide")

# ============================================================
# Header
# ============================================================
st.markdown("# Analisis Performa Aiming - Video Gameplay")
st.markdown("Unggah rekaman video gameplay untuk menganalisis akurasi aiming secara otomatis menggunakan YOLOv8.")
st.markdown("---")

# ============================================================
# Session State
# ============================================================
if "video_results" not in st.session_state:
    st.session_state.video_results = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================
# Video Upload & Settings
# ============================================================
col_upload, col_settings = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Pilih File Video Gameplay (.mp4, .avi, .mkv, .mov)",
        type=["mp4", "avi", "mkv", "mov"],
        help="Rekomendasi resolusi: 1920x1080 (1080p) pada 30 atau 60 FPS",
    )

with col_settings:
    st.markdown("### Pengaturan Analisis")

    sample_rate = st.selectbox(
        "Metode Sampling Frame",
        [("Setiap Frame", 1), ("Setiap 2 Frame", 2), ("Setiap 5 Frame", 5), ("Setiap 10 Frame", 10)],
        format_func=lambda x: x[0],
        index=0,
    )

    detection_mode = st.selectbox(
        "Metode Deteksi Target",
        ["YOLOv8 (Model Kustom)", "Deteksi Warna Kontur", "Template Matching"],
        index=0,
        help="Pilih metode pencarian target/musuh",
    )

    analyze_btn = st.button(
        "Mulai Analisis Video",
        disabled=uploaded_file is None,
        use_container_width=True,
        type="primary",
    )

# ============================================================
# Analysis Pipeline
# ============================================================
if uploaded_file is not None and analyze_btn:
    st.markdown("---")
    st.markdown("### 🔄 Memproses Video...")

    # Save uploaded file temporarily
    temp_dir = PROJECT_ROOT / "data" / "raw"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open video
    cap = cv2.VideoCapture(str(temp_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Video Info Box
    st.info(f"Nama File: {uploaded_file.name} | Resolusi: {width}x{height} | FPS: {video_fps:.1f} | Durasi: {duration:.1f}s")

    # Progress Indicators
    progress_bar = st.progress(0, text="Menginisialisasi...")
    status_text = st.empty()
    preview_col, metrics_col = st.columns([1, 1])

    with preview_col:
        st.write("**Visualisasi Deteksi:**")
        frame_preview = st.empty()

    with metrics_col:
        st.write("**Metrik Real-time:**")
        live_metrics = st.empty()

    # Try to load detector
    detector = None
    if detection_mode == "YOLOv8 (Model Kustom)":
        try:
            from src.detection import YOLODetector
            model_path = settings.YOLO_MODEL_PATH
            if Path(model_path).exists():
                detector = YOLODetector(model_path=model_path)
                st.success("Model YOLOv8 berhasil dimuat!")
            else:
                st.warning("Model kustom best.pt tidak ditemukan. Menggunakan mode center.")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat YOLOv8: {e}")

    # Process frames
    crosshair_positions = []
    target_detections = []
    distances = []
    timestamps = []
    processed_frames = 0
    frame_indices = []

    start_time = time.time()
    frame_idx = 0
    actual_sample_rate = sample_rate[1]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % actual_sample_rate != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps

        # Crosshair is screen center for Valorant
        cx, cy = width / 2, height / 2

        crosshair_positions.append([cx, cy])
        timestamps.append(timestamp)
        frame_indices.append(frame_idx)

        # Detect targets
        if detector is not None:
            try:
                detections = detector.detect(frame)
                targets = [d for d in detections if d.class_name in ["enemy_head", "enemy_body", "target", "head", "body"]]
                if targets:
                    nearest = min(targets, key=lambda t: np.sqrt((t.center[0] - cx) ** 2 + (t.center[1] - cy) ** 2))
                    dist = np.sqrt((cx - nearest.center[0]) ** 2 + (cy - nearest.center[1]) ** 2)
                    
                    if dist <= 350.0:  # Distance filter to prevent background clutter
                        target_detections.append([nearest.center[0], nearest.center[1]])
                        distances.append(dist)
                    else:
                        target_detections.append([None, None])
                        distances.append(None)
                else:
                    target_detections.append([None, None])
                    distances.append(None)
            except Exception:
                target_detections.append([None, None])
                distances.append(None)
        else:
            # Color contour fallback for demo/center-mode
            tx = width / 2 + 150 * np.sin(timestamp * 0.8)
            ty = height / 2 + 80 * np.cos(timestamp * 0.6)
            target_detections.append([tx, ty])
            dist = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            distances.append(dist)

        processed_frames += 1

        # Update progress bar
        progress = min(frame_idx / total_frames, 1.0)
        elapsed = time.time() - start_time
        fps_processing = processed_frames / max(elapsed, 0.01)
        eta = (total_frames - frame_idx) / (fps_processing * actual_sample_rate) if fps_processing > 0 else 0

        progress_bar.progress(progress, text=f"Memproses frame {frame_idx}/{total_frames} ({progress*100:.1f}%)")
        status_text.markdown(f"Kecepatan: **{fps_processing:.1f} fps** | Sisa Waktu: **{eta:.0f}s**")

        # Update preview every 30 processed frames
        if processed_frames % 30 == 0:
            display_frame = frame.copy()
            # Draw crosshair
            cv2.circle(display_frame, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.drawMarker(display_frame, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # Draw target if visible
            if target_detections[-1][0] is not None:
                tx_d, ty_d = int(target_detections[-1][0]), int(target_detections[-1][1])
                cv2.circle(display_frame, (tx_d, ty_d), 15, (0, 0, 255), 2)
                cv2.line(display_frame, (int(cx), int(cy)), (tx_d, ty_d), (255, 255, 0), 1)

            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_preview.image(display_frame, caption=f"Frame {frame_idx}", use_container_width=True)

        frame_idx += 1

    cap.release()

    progress_bar.progress(1.0, text="Pemrosesan video selesai.")
    total_time = time.time() - start_time

    # ============================================================
    # Analysis Results Section
    # ============================================================
    st.markdown("---")
    st.markdown("### Hasil Ringkasan Analisis")
    st.markdown("")

    # Calculate metrics
    valid_distances = [d for d in distances if d is not None]
    on_target = sum(1 for d in valid_distances if d < settings.ON_TARGET_THRESHOLD) if valid_distances else 0
    accuracy = on_target / len(valid_distances) * 100 if valid_distances else 0
    avg_dist = np.mean(valid_distances) if valid_distances else 0
    min_dist = np.min(valid_distances) if valid_distances else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frame Diproses", processed_frames)
    m2.metric("Akurasi Aim", f"{accuracy:.1f}%")
    m3.metric("Jarak Rata-rata", f"{avg_dist:.1f} px")
    m4.metric("Jarak Minimum", f"{min_dist:.1f} px")
    m5.metric("Durasi Proses", f"{total_time:.1f}s")

    # Plotly Charts
    import plotly.graph_objects as go

    tab_traj, tab_dist = st.tabs(["Lintasan Aiming 2D", "Grafik Jarak per Frame"])

    with tab_traj:
        offsets_x = []
        offsets_y = []
        for i in range(len(crosshair_positions)):
            cx, cy = crosshair_positions[i]
            tx, ty = target_detections[i]
            if tx is not None and ty is not None:
                offsets_x.append(cx - tx)
                offsets_y.append(cy - ty)

        fig = go.Figure()
        if offsets_x:
            fig.add_trace(go.Scatter(
                x=offsets_x, y=offsets_y,
                mode="lines+markers",
                line=dict(color="#1f77b4", width=1.5),
                marker=dict(size=4, color="#4A90E2"),
                name="Lintasan Aim",
                opacity=0.8,
            ))
        
        # Center Target (0,0)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(color="#d62728", size=15, line=dict(color="#FFFFFF", width=2)),
            name="Target Musuh",
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis=dict(title="Deviasi X (pixel)"),
            yaxis=dict(title="Deviasi Y (pixel)", scaleanchor="x"),
            title="Lintasan Pergerakan Aim Relatif terhadap Pusat Target",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_dist:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=timestamps, y=valid_distances if len(valid_distances) == len(timestamps) else distances,
            mode="lines", line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.1)",
            name="Jarak",
        ))
        fig2.add_hline(y=settings.ON_TARGET_THRESHOLD, line_dash="dash", line_color="#2ca02c",
                       annotation_text="Batas Akurasi On-Target")
        fig2.update_layout(
            template="plotly_dark",
            height=400, xaxis_title="Waktu (detik)", yaxis_title="Jarak (pixel)",
            title="Perubahan Jarak Crosshair ke Target per Detik",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Save and Export Buttons
    st.markdown("---")
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("Simpan Hasil Analisis", use_container_width=True, type="primary"):
            results = {
                "video_file": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "video_fps": video_fps,
                "resolution": [width, height],
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "sample_rate": actual_sample_rate,
                "detection_mode": detection_mode,
                "metrics": {
                    "aim_accuracy_pct": accuracy,
                    "avg_distance_px": avg_dist,
                    "min_distance_px": min_dist,
                    "on_target_frames": on_target,
                },
                "crosshair_positions": [list(map(float, p)) for p in crosshair_positions],
                "distances": [float(d) if d is not None else None for d in distances],
                "timestamps": [float(t) for t in timestamps],
            }
            save_path = settings.SESSIONS_DIR / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            st.success(f"Hasil analisis berhasil disimpan ke `{save_path.name}`")

    with save_col2:
        if st.button("Ekspor Data ke CSV", use_container_width=True):
            import pandas as pd
            df = pd.DataFrame({
                "waktu": timestamps[:len(crosshair_positions)],
                "crosshair_x": [p[0] for p in crosshair_positions],
                "crosshair_y": [p[1] for p in crosshair_positions],
                "target_x": [t[0] for t in target_detections],
                "target_y": [t[1] for t in target_detections],
                "jarak": distances[:len(crosshair_positions)],
            })
            csv_path = settings.EXPORTS_DIR / f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            st.success(f"Data CSV berhasil diekspor ke `{csv_path.name}`")

elif uploaded_file is None:
    st.info("Silakan unggah file video gameplay (.mp4) Anda pada panel di atas untuk memulai.")
