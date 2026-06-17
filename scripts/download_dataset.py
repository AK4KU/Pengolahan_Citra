"""
Valorant Dataset Downloader & Converter (v2)
===============================================
Downloads keremberke/valorant-object-detection via HuggingFace Hub API
and converts to YOLO format for training.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --max-images 100
    python scripts/download_dataset.py --output dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path
from io import BytesIO

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Class Mapping
# ============================================================
# HuggingFace dataset classes:
#   0: dropped spike  -> skip
#   1: enemy          -> our class 2 (enemy_body)
#   2: planted spike  -> skip
#   3: teammate       -> skip

HF_TO_YOLO = {
    0: None,  # dropped spike
    1: 2,     # enemy -> enemy_body
    2: None,  # planted spike
    3: None,  # teammate
}

OUR_CLASSES = {0: "crosshair", 1: "enemy_head", 2: "enemy_body", 3: "target"}


def download_and_convert(output_dir="dataset", max_images=None):
    """Download via HuggingFace parquet files and convert to YOLO format."""
    try:
        import pandas as pd
        from PIL import Image
        from huggingface_hub import hf_hub_download, list_repo_tree
    except ImportError:
        print("[ERROR] Required packages missing. Run:")
        print("  pip install huggingface_hub pandas pyarrow Pillow")
        sys.exit(1)

    output_path = PROJECT_ROOT / output_dir
    print("=" * 60)
    print("Valorant Dataset Downloader v2")
    print("=" * 60)
    print(f"Source:  keremberke/valorant-object-detection")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Create directory structure
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # List parquet files from the repo
    repo_id = "keremberke/valorant-object-detection"
    revision = "refs/convert/parquet"

    print("\n[INFO] Finding parquet files...")

    # Download parquet files for each split
    split_map = {
        "full/train": "train",
        "full/validation": "val",
        "full/test": "val",
    }

    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "skipped_annotations": 0,
        "per_class": {n: 0 for n in OUR_CLASSES.values()},
    }

    for hf_path_prefix, our_split in split_map.items():
        print(f"\n[DOWNLOAD] Processing {hf_path_prefix} -> {our_split}...")

        # Find parquet files for this split
        try:
            all_files = list(list_repo_tree(repo_id, path_in_repo=hf_path_prefix,
                                            revision=revision, repo_type="dataset"))
            parquet_files = [f.rfilename for f in all_files if f.rfilename.endswith(".parquet")]
        except Exception as e:
            print(f"  [WARN] Could not list files for {hf_path_prefix}: {e}")
            # Try direct known pattern
            parquet_files = [f"{hf_path_prefix}/0000.parquet"]

        if not parquet_files:
            print(f"  [WARN] No parquet files found for {hf_path_prefix}, trying default name...")
            parquet_files = [f"{hf_path_prefix}/0000.parquet"]

        for pq_file in parquet_files:
            print(f"  [DOWNLOAD] {pq_file}...")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=pq_file,
                    revision=revision,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"  [ERROR] Failed to download {pq_file}: {e}")
                continue

            # Read parquet
            df = pd.read_parquet(local_path)
            print(f"  [INFO] Loaded {len(df)} rows from {pq_file}")

            n = len(df)
            if max_images is not None:
                n = min(n, max_images)

            images_dir = output_path / "images" / our_split
            labels_dir = output_path / "labels" / our_split

            for idx in range(n):
                row = df.iloc[idx]

                # Extract image
                img_data = row.get("image", None)
                if img_data is None:
                    continue

                # Handle different image data formats
                try:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        continue

                    image = Image.open(BytesIO(img_bytes))
                except Exception:
                    continue

                img_w = row.get("width", image.width)
                img_h = row.get("height", image.height)

                # Save image
                split_prefix = hf_path_prefix.split("/")[-1]
                img_filename = f"{split_prefix}_{idx:05d}.jpg"
                img_path = images_dir / img_filename
                image.convert("RGB").save(str(img_path), "JPEG", quality=95)

                # Convert annotations
                objects = row.get("objects", {})
                if isinstance(objects, dict):
                    categories = objects.get("category", [])
                    bboxes = objects.get("bbox", [])
                elif isinstance(objects, str):
                    obj_dict = json.loads(objects)
                    categories = obj_dict.get("category", [])
                    bboxes = obj_dict.get("bbox", [])
                else:
                    categories, bboxes = [], []

                label_lines = []
                for cat_id, bbox in zip(categories, bboxes):
                    our_cls = HF_TO_YOLO.get(cat_id, None)
                    if our_cls is None:
                        stats["skipped_annotations"] += 1
                        continue

                    if len(bbox) >= 4:
                        x_min, y_min, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        x_c = max(0, min(1, (x_min + w / 2) / img_w))
                        y_c = max(0, min(1, (y_min + h / 2) / img_h))
                        w_n = max(0.001, min(1, w / img_w))
                        h_n = max(0.001, min(1, h / img_h))

                        label_lines.append(f"{our_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                        stats["total_annotations"] += 1
                        stats["per_class"][OUR_CLASSES[our_cls]] += 1

                # Save label file
                label_filename = img_filename.rsplit(".", 1)[0] + ".txt"
                with open(labels_dir / label_filename, "w") as f:
                    f.write("\n".join(label_lines))

                stats["total_images"] += 1

                if (idx + 1) % 200 == 0 or idx == n - 1:
                    print(f"    [{idx+1}/{n}] images processed...")

    # Create data.yaml
    data_yaml = f"""# Valorant Object Detection - YOLO Dataset
# Source: HuggingFace keremberke/valorant-object-detection
# Auto-generated by download_dataset.py

path: {output_path.resolve()}
train: images/train
val: images/val

names:
  0: crosshair
  1: enemy_head
  2: enemy_body
  3: target

# Note: This dataset contains 'enemy_body' (class 2) annotations.
# Crosshair uses screen center for Valorant (no detection needed).
"""
    with open(output_path / "data.yaml", "w") as f:
        f.write(data_yaml)

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Download & Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"data.yaml: {output_path / 'data.yaml'}")
    print(f"\nStats:")
    print(f"  Total images:      {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Skipped:           {stats['skipped_annotations']}")
    print(f"  Per class:")
    for name, count in stats["per_class"].items():
        print(f"    {name}: {count}")

    # Count files
    for split in ["train", "val"]:
        img_count = len(list((output_path / "images" / split).glob("*.jpg")))
        lbl_count = len(list((output_path / "labels" / split).glob("*.txt")))
        print(f"  {split}: {img_count} images, {lbl_count} labels")

    print(f"\nTo train: python training/train_model.py --data {output_path / 'data.yaml'}")

    with open(output_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Valorant dataset for YOLO training")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Max images per split")
    args = parser.parse_args()
    download_and_convert(output_dir=args.output, max_images=args.max_images)
