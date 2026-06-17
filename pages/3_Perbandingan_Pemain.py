"""
Player Comparison Page
=======================
Compare performance profiles across multiple players or sessions.
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

st.set_page_config(page_title="Perbandingan Pemain - Tugas Akhir Pengolahan Citra", layout="wide")

st.markdown("# Perbandingan Hasil Analisis")
st.markdown("Bandingkan profil performa aim antara beberapa sesi latihan yang telah disimpan.")
st.markdown("---")

# ============================================================
# Load available sessions
# ============================================================
sessions_dir = settings.SESSIONS_DIR
all_sessions = []

if sessions_dir.exists():
    for sf in sorted(sessions_dir.glob("*.json"), reverse=True):
        try:
            with open(sf, "r") as f:
                data = json.load(f)
            data["_filename"] = sf.name
            all_sessions.append(data)
        except Exception:
            continue

if not all_sessions:
    st.info("Mode Demo: Belum ada cukup data sesi tersimpan (minimal dibutuhkan 2 sesi). Menampilkan data simulasi perbandingan pemain.")

    # Generate synthetic player data for demonstration
    np.random.seed(42)
    synthetic_players = {
        "Pemain A (Pemula)": {
            "accuracy": 22.4,
            "avg_ttt_ms": 750,
            "overshoot_ratio": 0.55,
            "consistency": 0.35,
            "level": "Pemula",
        },
        "Pemain B (Menengah)": {
            "accuracy": 48.6,
            "avg_ttt_ms": 460,
            "overshoot_ratio": 0.32,
            "consistency": 0.58,
            "level": "Menengah",
        },
        "Pemain C (Mahir)": {
            "accuracy": 82.1,
            "avg_ttt_ms": 280,
            "overshoot_ratio": 0.12,
            "consistency": 0.84,
            "level": "Mahir",
        },
    }

    import plotly.graph_objects as go
    import pandas as pd

    # Radar Chart
    st.markdown("### Grafik Radar Profil Kemampuan")

    categories = ["Akurasi", "Kecepatan Reaksi", "Presisi", "Konsistensi"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig = go.Figure()
    for i, (name, data) in enumerate(synthetic_players.items()):
        speed_score = max(0, 1 - data["avg_ttt_ms"] / 1000)
        precision_score = max(0, 1 - data["overshoot_ratio"])
        values = [
            data["accuracy"] / 100,
            speed_score,
            precision_score,
            data["consistency"],
        ]
        values.append(values[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color=colors[i], width=2),
            name=name,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        template="plotly_dark",
        height=500,
        title="Bagan Radar Kemampuan Aim",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison Table
    st.markdown("### Tabel Rincian Perbandingan")

    comparison_data = []
    for name, data in synthetic_players.items():
        comparison_data.append({
            "Nama Pemain": name,
            "Akurasi (%)": f"{data['accuracy']:.1f}%",
            "Kecepatan Reaksi (ms)": f"{data['avg_ttt_ms']} ms",
            "Rasio Overshoot": f"{data['overshoot_ratio']:.2f}",
            "Konsistensi": f"{data['consistency']:.2f}",
            "Klasifikasi Skill": data["level"],
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # Real session comparison
    st.markdown("### Pilih Sesi yang Ingin Dibandingkan")

    session_labels = []
    for s in all_sessions:
        ts = s.get("timestamp", "Unknown")
        try:
            dt = datetime.fromisoformat(ts)
            label = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            label = ts
        frames = s.get("frames_processed", s.get("processed_frames", 0))
        session_labels.append(f"Sesi {label} ({frames} frame) - {s['_filename']}")

    selected = st.multiselect(
        "Pilih minimal 2 sesi untuk dibandingkan:",
        range(len(all_sessions)),
        format_func=lambda i: session_labels[i],
        default=[0, 1] if len(all_sessions) >= 2 else [0],
    )

    if len(selected) >= 2:
        import plotly.graph_objects as go
        import pandas as pd

        colors_pool = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Compute metrics for each session
        comparison = []
        for idx in selected:
            s = all_sessions[idx]
            distances = [d for d in s.get("distances", []) if d is not None]
            if distances:
                on_target = sum(1 for d in distances if d < settings.ON_TARGET_THRESHOLD)
                accuracy = on_target / len(distances) * 100
                avg_dist = np.mean(distances)
            else:
                accuracy = s.get("metrics", {}).get("aim_accuracy_pct", 0)
                avg_dist = s.get("metrics", {}).get("avg_distance_px", 0)

            comparison.append({
                "label": session_labels[idx][:25],
                "accuracy": accuracy,
                "avg_distance": avg_dist,
                "frames": s.get("frames_processed", s.get("processed_frames", 0)),
                "distances": distances,
            })

        # Radar chart for real data comparison
        st.markdown("### Grafik Radar Perbandingan")
        categories = ["Akurasi", "Kecepatan", "Presisi"]

        fig = go.Figure()
        for i, comp in enumerate(comparison):
            speed_norm = max(0, 1 - comp["avg_distance"] / 500) if comp["avg_distance"] else 0.5
            precision = max(0, 1 - np.std(comp["distances"]) / 200) if comp["distances"] else 0.5

            values = [comp["accuracy"] / 100, speed_norm, precision]
            values.append(values[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=comp["label"],
                line=dict(color=colors_pool[i % len(colors_pool)], width=2),
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_dark",
            height=450,
            title="Profil Kemampuan per Sesi",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table Comparison
        st.markdown("### Tabel Rincian Metrik Sesi")
        table_data = []
        for comp in comparison:
            table_data.append({
                "Sesi Analisis": comp["label"],
                "Akurasi Aim (%)": f"{comp['accuracy']:.1f}%",
                "Jarak Rata-rata (pixel)": f"{comp['avg_distance']:.1f} px",
                "Total Frame": comp["frames"],
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.warning("Silakan pilih minimal 2 sesi untuk mulai menampilkan perbandingan.")
