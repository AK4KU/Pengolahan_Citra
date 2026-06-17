"""
Session History Page
=====================
Browse and analyze previously saved sessions.
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

st.set_page_config(page_title="Riwayat Analisis - Tugas Akhir Pengolahan Citra", layout="wide")

st.markdown("# Riwayat Sesi Analisis")
st.markdown("Telusuri dan analisis hasil sesi perekaman aim sebelumnya.")
st.markdown("---")

# ============================================================
# Load Sessions
# ============================================================
sessions_dir = settings.SESSIONS_DIR
sessions = []

if sessions_dir.exists():
    for session_file in sorted(sessions_dir.glob("*.json"), reverse=True):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            data["_filename"] = session_file.name
            data["_filepath"] = str(session_file)
            sessions.append(data)
        except Exception:
            continue

if not sessions:
    st.info("Tidak ada sesi riwayat analisis yang ditemukan. Silakan lakukan analisis video terlebih dahulu.")
else:
    st.markdown(f"### Ditemukan {len(sessions)} Sesi")

    col_list, col_detail = st.columns([1, 2])

    with col_list:
        # Build session labels for navigation
        session_options = []
        for i, s in enumerate(sessions):
            ts = s.get("timestamp", "Unknown")
            try:
                dt = datetime.fromisoformat(ts)
                display_time = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                display_time = ts

            frames = s.get("frames_processed", s.get("processed_frames", 0))
            acc = s.get("metrics", {}).get("aim_accuracy_pct", None)
            if acc is None and "distances" in s:
                dists = [d for d in s["distances"] if d is not None]
                on_target = sum(1 for d in dists if d < settings.ON_TARGET_THRESHOLD)
                acc = on_target / len(dists) * 100 if dists else 0

            label = f"Sesi {display_time} ({acc:.1f}% Akurasi)" if acc is not None else f"Sesi {display_time}"
            session_options.append(label)

        selected_idx = st.radio(
            "Pilih Sesi:",
            range(len(sessions)),
            format_func=lambda i: session_options[i],
        )

        st.markdown("---")

        # Delete button
        if st.button("Hapus Sesi Terpilih", use_container_width=True):
            filepath = Path(sessions[selected_idx]["_filepath"])
            if filepath.exists():
                filepath.unlink()
                st.success("Sesi berhasil dihapus!")
                st.rerun()

    with col_detail:
        session = sessions[selected_idx]

        # Metadata info
        ts = session.get("timestamp", "Unknown")
        try:
            dt = datetime.fromisoformat(ts)
            display_time = dt.strftime("%d %B %Y jam %H:%M:%S")
        except Exception:
            display_time = ts

        st.markdown(f"### Detail Sesi")
        st.markdown(f"**Tanggal Analisis:** {display_time}")
        st.markdown(f"**Nama File Sesi:** `{session.get('_filename', 'N/A')}`")

        if "video_file" in session:
            st.markdown(f"**Video Sumber:** `{session['video_file']}`")

        st.markdown("---")

        # Extract Metrics
        frames = session.get("frames_processed", session.get("processed_frames", 0))
        duration = session.get("duration_seconds", 0)

        distances = session.get("distances", [])
        valid_distances = [d for d in distances if d is not None]

        if valid_distances:
            on_target = sum(1 for d in valid_distances if d < settings.ON_TARGET_THRESHOLD)
            accuracy = on_target / len(valid_distances) * 100
            avg_dist = np.mean(valid_distances)
        else:
            metrics = session.get("metrics", {})
            accuracy = metrics.get("aim_accuracy_pct", 0)
            avg_dist = metrics.get("avg_distance_px", 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frame Diproses", frames)
        m2.metric("Akurasi Aim", f"{accuracy:.1f}%")
        m3.metric("Jarak Rata-rata", f"{avg_dist:.1f} px")
        m4.metric("Durasi Sesi", f"{duration:.1f} s" if duration else "N/A")

        # Charts
        if distances and len(distances) > 5:
            import plotly.graph_objects as go

            timestamps = session.get("timestamps", list(range(len(distances))))

            tab1, tab2 = st.tabs(["Grafik Jarak", "Lintasan Aiming 2D"])

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps[:len(valid_distances)],
                    y=valid_distances,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(31, 119, 180, 0.1)",
                ))
                fig.add_hline(y=settings.ON_TARGET_THRESHOLD, line_dash="dash",
                               line_color="#2ca02c", annotation_text="Batas Akurasi")
                fig.update_layout(
                    template="plotly_dark",
                    height=350, xaxis_title="Waktu (detik)", yaxis_title="Jarak (pixel)",
                    title="Jarak Crosshair ke Target per Frame",
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                positions = session.get("crosshair_positions", [])
                target_positions = session.get("target_positions", [])
                if positions and target_positions and len(positions) > 5:
                    offsets_x = []
                    offsets_y = []
                    for i in range(len(positions)):
                        cx, cy = positions[i]
                        t_pos = target_positions[i]
                        if t_pos is not None and len(t_pos) == 2 and t_pos[0] is not None:
                            tx, ty = t_pos
                            offsets_x.append(cx - tx)
                            offsets_y.append(cy - ty)
                            
                    fig2 = go.Figure()
                    if offsets_x:
                        fig2.add_trace(go.Scatter(
                            x=offsets_x, y=offsets_y,
                            mode="lines+markers",
                            line=dict(color="#1f77b4", width=1.5),
                            marker=dict(size=4, color="#4A90E2"),
                            name="Lintasan Aim",
                            opacity=0.8,
                        ))
                    
                    fig2.add_trace(go.Scatter(
                        x=[0], y=[0],
                        mode="markers",
                        marker=dict(color="#d62728", size=15, line=dict(color="#FFFFFF", width=2)),
                        name="Target Musuh",
                    ))
                    
                    fig2.update_layout(
                        template="plotly_dark",
                        height=350,
                        xaxis=dict(title="Deviasi X (pixel)"),
                        yaxis=dict(title="Deviasi Y (pixel)", scaleanchor="x"),
                        title="Lintasan Aim Relatif terhadap Target",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Data lintasan target tidak tersedia dalam sesi ini.")

        # Download Buttons
        st.markdown("---")
        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "Unduh Hasil JSON",
                data=json.dumps(session, indent=2, default=str),
                file_name=session.get("_filename", "session.json"),
                mime="application/json",
                use_container_width=True,
            )
        with exp2:
            if valid_distances:
                import pandas as pd
                df = pd.DataFrame({
                    "jarak": valid_distances,
                    "on_target": [d < settings.ON_TARGET_THRESHOLD for d in valid_distances],
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    "Unduh Hasil CSV",
                    data=csv,
                    file_name=f"session_{selected_idx}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
