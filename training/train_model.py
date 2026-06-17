"""
YOLO Model Training Script
============================
Train a custom YOLOv8 model for crosshair and target detection in FPS games.
"""

import argparse
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


def train_model(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    patience: int = 20,
    project: str = "runs/train",
    name: str = "aim_detector",
    resume: bool = False,
):
    """
    Train a YOLO model on a custom dataset.

    Args:
        data_yaml: Path to data.yaml configuration file
        base_model: Pretrained model to fine-tune from
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
        device: CUDA device (0 for first GPU, "cpu" for CPU)
        patience: Early stopping patience
        project: Project directory for saving results
        name: Run name
        resume: Whether to resume from last checkpoint
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        return

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        print(f"❌ data.yaml not found at {data_yaml}")
        print("Run prepare_dataset.py first to create the dataset structure.")
        return

    print("=" * 60)
    print("FPS Aim Performance Analyzer - YOLO Training")
    print("=" * 60)
    print(f"Base Model: {base_model}")
    print(f"Dataset:    {data_yaml}")
    print(f"Epochs:     {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Device:     {device}")
    print(f"Patience:   {patience}")
    print("=" * 60)

    # Load model
    model = YOLO(base_model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        project=project,
        name=name,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,  # Don't flip upside down (game screens don't do this)
        fliplr=0.5,  # Horizontal flip is okay
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        half=True,
        resume=resume,
        verbose=True,
        save=True,
        save_period=10,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Copy best model to models directory
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        dest = settings.MODELS_DIR / "best.pt"
        shutil.copy2(best_model_path, dest)
        print(f"✅ Best model copied to: {dest}")
    else:
        print("⚠️ Best model weights not found.")

    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for FPS aim detection")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (yolov8n/s/m/l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device (0, 1, cpu)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        patience=args.patience,
        resume=args.resume,
    )
