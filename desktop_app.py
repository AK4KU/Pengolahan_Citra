"""
FPS Aim Performance Analyzer - Desktop Application
===================================================
A native CustomTkinter GUI for analyzing Valorant gameplay videos.
Optimized for presentation. Highlights YOLO detection, crosshair-target tracking,
and aim performance metrics with a premium dark-themed dashboard.
"""

import sys
import os
import time
import cv2
import threading
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GUI Imports
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Project imports
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from config import settings
from src.tracking.position_tracker import PositionTracker, TargetDetection
from src.tracking.trajectory_builder import TrajectoryBuilder
from src.metrics.session_analyzer import SessionAnalyzer, SessionReport

# Set dark theme styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AimAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("Aplikasi Analisis Performa Aiming - Desktop")
        self.geometry("1280x800")
        self.minsize(1024, 768)
        
        # Application State
        self.video_path = None
        self.detector = None
        self.is_processing = False
        self.stop_requested = False
        self.tracking_data = []
        
        # Initialize YOLO detector
        self.load_yolo_model()
        
        # Create Layout Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Build Sidebar Panel
        self.build_sidebar()
        
        # Build Main Frame with Tabs
        self.build_main_area()

    def load_yolo_model(self):
        """Pre-load the YOLO model or set up a warning state."""
        model_path = Path(settings.YOLO_MODEL_PATH)
        if model_path.exists():
            try:
                from src.detection.yolo_detector import YOLODetector
                self.detector = YOLODetector(model_path=str(model_path), confidence=settings.YOLO_CONFIDENCE)
                print("✅ YOLO Model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load YOLO detector: {e}")
        else:
            print(f"⚠️ YOLO model file not found at {model_path}. FALLBACK: Template-based tracking.")

    def build_sidebar(self):
        """Create the sidebar panel for controls & settings."""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)
        
        # App Title
        title_label = ctk.CTkLabel(
            self.sidebar, 
            text="Analisis Aiming", 
            font=ctk.CTkFont(size=22, weight="bold", family="Inter")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        subtitle_label = ctk.CTkLabel(
            self.sidebar, 
            text="Tugas Akhir Pengolahan Citra", 
            font=ctk.CTkFont(size=12, weight="normal", family="Inter")
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Divider
        divider1 = ctk.CTkFrame(self.sidebar, height=2)
        divider1.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        
        # Video Selector Button
        self.select_btn = ctk.CTkButton(
            self.sidebar, 
            text="Pilih Video Gameplay", 
            command=self.select_video,
            font=ctk.CTkFont(weight="bold", family="Inter")
        )
        self.select_btn.grid(row=3, column=0, padx=20, pady=15)
        
        self.file_label = ctk.CTkLabel(
            self.sidebar, 
            text="Tidak ada video terpilih", 
            wraplength=220,
            font=ctk.CTkFont(size=11, family="Inter")
        )
        self.file_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Configuration Settings
        settings_title = ctk.CTkLabel(
            self.sidebar, 
            text="Pengaturan Analisis", 
            font=ctk.CTkFont(size=14, weight="bold", family="Inter")
        )
        settings_title.grid(row=5, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Frame Sampling Rate
        sampling_label = ctk.CTkLabel(
            self.sidebar, 
            text="Frekuensi Sampling Frame:", 
            font=ctk.CTkFont(size=12, family="Inter")
        )
        sampling_label.grid(row=6, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.sampling_combo = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Setiap Frame", "Setiap 2 Frame", "Setiap 5 Frame"],
            font=ctk.CTkFont(family="Inter")
        )
        self.sampling_combo.grid(row=7, column=0, padx=20, pady=(0, 15))
        self.sampling_combo.set("Setiap Frame")
        
        # Action Buttons
        self.start_btn = ctk.CTkButton(
            self.sidebar, 
            text="Mulai Analisis", 
            command=self.start_analysis,
            state="disabled",
            font=ctk.CTkFont(weight="bold", family="Inter")
        )
        self.start_btn.grid(row=9, column=0, padx=20, pady=10)
        
        self.stop_btn = ctk.CTkButton(
            self.sidebar, 
            text="Reset", 
            command=self.stop_analysis,
            state="disabled",
            font=ctk.CTkFont(weight="bold", family="Inter")
        )
        self.stop_btn.grid(row=10, column=0, padx=20, pady=10)
        
        # Footer
        footer_label = ctk.CTkLabel(
            self.sidebar, 
            text="Python & YOLOv8", 
            font=ctk.CTkFont(size=10, family="Inter")
        )
        footer_label.grid(row=11, column=0, padx=20, pady=20)

    def build_main_area(self):
        """Create the tab-view frame for the main presentation content."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Tabview
        self.tabview = ctk.CTkTabview(
            self.main_frame
        )
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        self.tab_preview = self.tabview.add("Preview Video")
        self.tab_report = self.tabview.add("Laporan Analisis")
        
        # --- TAB 1: Preview Layout ---
        self.tab_preview.grid_rowconfigure(0, weight=1)
        self.tab_preview.grid_columnconfigure(0, weight=1)
        
        # Frame Container for Video
        self.video_container = ctk.CTkFrame(self.tab_preview, fg_color="#000000", corner_radius=12)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)
        
        # Video Canvas/Label
        self.video_label = ctk.CTkLabel(self.video_container, text="Panel Preview Video\n\n1. Pilih rekaman video gameplay Valorant.\n2. Klik 'Mulai Analisis' untuk mendeteksi target.", font=ctk.CTkFont(size=14, family="Inter"))
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Progress Bar and Live Stats Panel
        self.stats_panel = ctk.CTkFrame(self.tab_preview, height=70)
        self.stats_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.progress_bar = ctk.CTkProgressBar(self.stats_panel)
        self.progress_bar.pack(fill="x", padx=15, pady=(10, 5))
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(
            self.stats_panel, 
            text="Siap untuk menganalisis. Resolusi video: 1920x1080 (Valorant)", 
            font=ctk.CTkFont(size=12, family="Inter")
        )
        self.status_label.pack(side="left", padx=15, pady=(0, 5))
        
        # --- TAB 2: Metrics Report Layout ---
        self.tab_report.grid_rowconfigure(1, weight=1)
        self.tab_report.grid_columnconfigure(0, weight=1)
        
        # Cards Layout for Key Metrics
        self.metrics_grid = ctk.CTkFrame(self.tab_report, height=120, fg_color="transparent")
        self.metrics_grid.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.metrics_grid.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Create 4 Cards
        self.card_accuracy = self.create_metric_card(self.metrics_grid, 0, "AKURASI AIMING", "0.0%")
        self.card_distance = self.create_metric_card(self.metrics_grid, 1, "JARAK RATA-RATA", "0.0 px")
        self.card_overshoot = self.create_metric_card(self.metrics_grid, 2, "RASIO OVERSHOOT", "0.0%")
        self.card_skill = self.create_metric_card(self.metrics_grid, 3, "TINGKAT KEMAHIRAN", "N/A")
        
        # Matplotlib Plot Embed Frame
        self.plot_frame = ctk.CTkFrame(self.tab_report)
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        self.no_data_label = ctk.CTkLabel(
            self.plot_frame,
            text="Data analisis tidak tersedia.\nGrafik metrik akan tampil setelah pemrosesan video selesai.",
            font=ctk.CTkFont(size=14, family="Inter")
        )
        self.no_data_label.grid(row=0, column=0)
        
        # Matplotlib canvas holder
        self.canvas = None

    def create_metric_card(self, parent, col, title, initial_val):
        """Helper to create a simple dashboard card."""
        card = ctk.CTkFrame(parent, border_width=1, corner_radius=10)
        card.grid(row=0, column=col, padx=8, pady=5, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        
        title_lbl = ctk.CTkLabel(
            card, 
            text=title, 
            font=ctk.CTkFont(size=10, weight="bold", family="Inter")
        )
        title_lbl.grid(row=1, column=0, padx=10, pady=(10, 2), sticky="w")
        
        val_lbl = ctk.CTkLabel(
            card, 
            text=initial_val, 
            font=ctk.CTkFont(size=24, weight="bold", family="Inter")
        )
        val_lbl.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="w")
        
        return val_lbl

    def select_video(self):
        """Open file dialog to pick a video file."""
        file_types = [("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        selected = filedialog.askopenfilename(initialdir=str(PROJECT_ROOT / "data" / "raw"), filetypes=file_types)
        if selected:
            self.video_path = selected
            self.file_label.configure(text=Path(selected).name)
            self.start_btn.configure(state="normal")
            
            # Switch back to preview tab
            self.tabview.set("Preview Video")
            self.video_label.configure(text="Video berhasil dimuat!\n\nKlik 'Mulai Analisis' untuk mendeteksi target.")
            self.progress_bar.set(0)

    def start_analysis(self):
        """Launch the video processing thread."""
        if not self.video_path:
            return
        
        self.is_processing = True
        self.stop_requested = False
        
        # Update UI states
        self.start_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.sampling_combo.configure(state="disabled")
        self.tabview.set("Preview Video")
        
        # Start Thread
        self.thread = threading.Thread(target=self.process_video_thread, daemon=True)
        self.thread.start()

    def stop_analysis(self):
        """Request the video processing to stop and reset."""
        if self.is_processing:
            self.stop_requested = True
            self.status_label.configure(text="Menghentikan analisis...")
        else:
            # Simple reset
            self.reset_ui()

    def reset_ui(self):
        """Reset buttons and labels to starting states."""
        self.start_btn.configure(state="normal" if self.video_path else "disabled")
        self.select_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.sampling_combo.configure(state="normal")
        self.progress_bar.set(0)
        self.status_label.configure(text="Siap melakukan analisis.")
        self.video_label.configure(text="Panel Preview Video")
        self.is_processing = False

    def process_video_thread(self):
        """Main background processing thread for video frame loop."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.runOnUiThread(self.show_error, "Error opening video file.")
            self.runOnUiThread(self.reset_ui)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Verify resolution
        if width != 1920 or height != 1080:
            print(f"⚠️ Warning: Video resolution is {width}x{height}, model is trained for 1080p (1920x1080).")

        # Config frame step
        sampling_val = self.sampling_combo.get()
        step = 1
        if "2 Frame" in sampling_val:
            step = 2
        elif "5 Frame" in sampling_val:
            step = 5

        # Initialize Position Tracker
        tracker = PositionTracker()
        
        processed_frames = 0
        frame_idx = 0
        start_time = time.time()
        
        # Store raw tracking array for quick visual analytics fallback
        self.tracking_data = []

        while cap.isOpened() and not self.stop_requested:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step != 0:
                frame_idx += 1
                continue
                
            timestamp = frame_idx / video_fps if video_fps > 0 else frame_idx / 60.0
            
            # Crosshair is screen center for Valorant
            cx, cy = width / 2.0, height / 2.0
            
            # Detect target
            targets_for_tracker = []
            target_cx, target_cy = None, None
            
            if self.detector is not None:
                try:
                    detections = self.detector.detect(frame)
                    # Filter for enemy head/body
                    enemies = [d for d in detections if d.class_name in ["enemy_head", "enemy_body", "target", "head", "body"]]
                    
                    if enemies:
                        # Find nearest to crosshair
                        nearest = min(enemies, key=lambda t: np.sqrt((t.center[0] - cx)**2 + (t.center[1] - cy)**2))
                        dist = np.sqrt((nearest.center[0] - cx)**2 + (nearest.center[1] - cy)**2)
                        
                        if dist <= 350.0:  # Only track targets within 350px of crosshair to filter background clutter
                            target_cx, target_cy = nearest.center[0], nearest.center[1]
                            
                            # Populate tracker detection format
                            td = TargetDetection(
                                target_id=1,
                                x=float(target_cx),
                                y=float(target_cy),
                                width=float(nearest.width),
                                height=float(nearest.height),
                                confidence=float(nearest.confidence),
                                class_id=nearest.class_id,
                                class_name=nearest.class_name
                            )
                            targets_for_tracker.append(td)
                except Exception as e:
                    print(f"Inference error: {e}")

            # Update tracker
            tracker.update(frame_idx, timestamp, (cx, cy), targets_for_tracker)
            
            # Save raw tuple for matplotlib fallback plot
            self.tracking_data.append((timestamp, cx, cy, target_cx, target_cy))
            
            processed_frames += 1
            
            # Render video preview to UI at controlled rate (every 2 frames)
            if processed_frames % 2 == 0:
                # Draw visual elements on frame
                draw_frame = frame.copy()
                
                # Crosshair
                cv2.circle(draw_frame, (int(cx), int(cy)), 8, (0, 255, 0), 1)
                cv2.drawMarker(draw_frame, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
                
                # Detected Targets
                if target_cx is not None:
                    # Draw connection line
                    cv2.line(draw_frame, (int(cx), int(cy)), (int(target_cx), int(target_cy)), (255, 255, 0), 1)
                    # Target marker
                    cv2.circle(draw_frame, (int(target_cx), int(target_cy)), 12, (0, 0, 255), 2)
                    cv2.putText(draw_frame, "Target", (int(target_cx) + 15, int(target_cy) - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Resize and convert to PIL Image
                draw_frame = cv2.resize(draw_frame, (850, 480))
                draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(draw_frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update Video Label on main thread
                self.runOnUiThread(self.update_video_preview, img_tk)

            # Update progress bar
            progress = min(frame_idx / total_frames, 1.0)
            elapsed = time.time() - start_time
            fps_speed = processed_frames / max(elapsed, 0.01)
            
            status_text = f"Memproses: {frame_idx}/{total_frames} frame ({progress*100:.1f}%) | Kecepatan: {fps_speed:.1f} FPS"
            self.runOnUiThread(self.update_progress, progress, status_text)
            
            frame_idx += 1

        cap.release()
        
        if self.stop_requested:
            self.runOnUiThread(self.reset_ui)
            return

        # --- COMPUTE METRICS ---
        self.runOnUiThread(self.status_label.configure, text="Menganalisis data aiming...")
        
        # Extract distances
        valid_distances = []
        for row in self.tracking_data:
            tx, ty = row[3], row[4]
            if tx is not None:
                cx, cy = row[1], row[2]
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                valid_distances.append(dist)
        
        # Default fallback values
        acc_rate = 0.0
        avg_dist = 0.0
        overshoot_ratio = 0.0
        skill_level = "Pemula"
        
        if valid_distances:
            on_target = sum(1 for d in valid_distances if d < settings.ON_TARGET_THRESHOLD)
            acc_rate = (on_target / len(valid_distances)) * 100.0
            avg_dist = np.mean(valid_distances)
            
            # Simple Overshoot approximation (reversals near target)
            reversals = 0
            for i in range(2, len(valid_distances)):
                diff1 = valid_distances[i-1] - valid_distances[i-2]
                diff2 = valid_distances[i] - valid_distances[i-1]
                if diff1 * diff2 < 0 and valid_distances[i] < settings.ON_TARGET_THRESHOLD * 2:
                    reversals += 1
            overshoot_ratio = min((reversals / len(valid_distances)) * 200.0, 100.0) # scaling for display
            
            # Determine Skill Level
            if acc_rate >= 60.0 and avg_dist <= 25.0:
                skill_level = "Mahir"
            elif acc_rate >= 30.0 and avg_dist <= 50.0:
                skill_level = "Menengah"
            else:
                skill_level = "Pemula"

        # Update report cards
        self.runOnUiThread(self.card_accuracy.configure, text=f"{acc_rate:.1f}%")
        self.runOnUiThread(self.card_distance.configure, text=f"{avg_dist:.1f} px")
        self.runOnUiThread(self.card_overshoot.configure, text=f"{overshoot_ratio:.1f}%")
        self.runOnUiThread(self.card_skill.configure, text=skill_level)
        
        # Generate and Embed Plot
        self.runOnUiThread(self.generate_report_plots, valid_distances)
        
        # Finalize UI transition
        self.runOnUiThread(self.finish_analysis)

    def update_video_preview(self, img_tk):
        """Update the image label in the preview tab."""
        self.video_label.configure(image=img_tk, text="")
        self.video_label.image = img_tk  # Keep reference

    def update_progress(self, progress, status_text):
        """Update progress bar and status text safely."""
        self.progress_bar.set(progress)
        self.status_label.configure(text=status_text)

    def show_error(self, message):
        """Display error box on main thread."""
        messagebox.showerror("Error Analisis", message)

    def finish_analysis(self):
        """Handle UI changes when processing completes."""
        self.is_processing = False
        self.start_btn.configure(state="normal")
        self.select_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.sampling_combo.configure(state="normal")
        
        # Show complete message
        self.status_label.configure(text="Analisis Selesai! Laporan hasil analisis telah dibuat.")
        
        # Switch tab
        self.tabview.set("Laporan Analisis")

    def generate_report_plots(self, valid_distances):
        """Create beautiful custom dark-themed Matplotlib plots and embed them."""
        # Clear old canvas if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        if self.no_data_label.winfo_exists():
            self.no_data_label.grid_forget()

        # Style configurations
        bg_color = "#1E1E1E"  # Dark Gray
        grid_color = "#2D2D2D"  # Gray
        text_color = "#FFFFFF"  # White
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=bg_color)
        
        # --- Plot 1: 2D Aim Trajectory Path relative to Target ---
        ax1.set_facecolor(bg_color)
        ax1.spines['bottom'].set_color(grid_color)
        ax1.spines['top'].set_color(grid_color)
        ax1.spines['left'].set_color(grid_color)
        ax1.spines['right'].set_color(grid_color)
        ax1.tick_params(colors=text_color)
        ax1.grid(color=grid_color, linestyle='--', linewidth=0.5)
        
        # Compute relative offsets
        offsets_x, offsets_y = [], []
        for row in self.tracking_data:
            tx, ty = row[3], row[4]
            if tx is not None:
                cx, cy = row[1], row[2]
                offsets_x.append(cx - tx)
                offsets_y.append(cy - ty)
                
        if offsets_x:
            # Draw movement path (slow velocity = blue, fast = red-violet)
            ax1.plot(offsets_x, offsets_y, color="#1f77b4", alpha=0.7, linewidth=1.5, label="Lintasan Aim")
            # Draw points
            ax1.scatter(offsets_x, offsets_y, color="#4A90E2", s=8, alpha=0.6)
            
        # Target representation at center (0,0)
        ax1.scatter(0, 0, color="#d62728", s=150, edgecolors='#FFFFFF', linewidth=2, label="Target (Kepala)")
        ax1.axhline(0, color=grid_color, linewidth=1)
        ax1.axvline(0, color=grid_color, linewidth=1)
        
        # Labeling
        ax1.set_title("Lintasan 2D Relatif Terhadap Pusat Target", color=text_color, fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Deviasi X (px)", color=text_color)
        ax1.set_ylabel("Deviasi Y (px)", color=text_color)
        ax1.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color, loc="upper right")
        
        # --- Plot 2: Distance Timeline & Threshold ---
        ax2.set_facecolor(bg_color)
        ax2.spines['bottom'].set_color(grid_color)
        ax2.spines['top'].set_color(grid_color)
        ax2.spines['left'].set_color(grid_color)
        ax2.spines['right'].set_color(grid_color)
        ax2.tick_params(colors=text_color)
        ax2.grid(color=grid_color, linestyle='--', linewidth=0.5)
        
        if valid_distances:
            frames = range(len(valid_distances))
            ax2.plot(frames, valid_distances, color="#1f77b4", linewidth=1.5, label="Jarak Crosshair")
            
            # Draw On-Target Threshold line
            ax2.axhline(settings.ON_TARGET_THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.5, 
                        label=f"Batas Akurasi ({settings.ON_TARGET_THRESHOLD}px)")
            
            # Highlight on-target frames
            distances_np = np.array(valid_distances)
            on_target_idx = np.where(distances_np < settings.ON_TARGET_THRESHOLD)[0]
            if len(on_target_idx) > 0:
                ax2.scatter(on_target_idx, distances_np[on_target_idx], color="#2ca02c", s=12, zorder=5, label="Tepat Sasaran")
                
        # Labeling
        ax2.set_title("Grafik Jarak ke Target per Frame", color=text_color, fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Indeks Frame", color=text_color)
        ax2.set_ylabel("Jarak (pixel)", color=text_color)
        ax2.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color, loc="upper right")
        
        plt.tight_layout()
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Redraw
        plt.close(fig)

    def runOnUiThread(self, func, *args, **kwargs):
        """Safely schedule UI updates from background threads."""
        self.after(0, lambda: func(*args, **kwargs))

if __name__ == "__main__":
    app = AimAnalyzerApp()
    app.mainloop()
