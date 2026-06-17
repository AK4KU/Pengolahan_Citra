"""
Roboflow Label Remapper
========================
Remaps class IDs in YOLO label files from Roboflow format to our project format.

Usage:
    1. Download dataset from Roboflow in YOLOv8 format
    2. Extract the ZIP to a folder (e.g., dataset_roboflow/)
    3. Run this script to remap and copy to the main dataset

    python scripts/remap_roboflow.py --source dataset_roboflow --dest dataset

The script will:
    - Read the data.yaml from the source to detect class mapping
    - Remap class IDs to match our project classes
    - Copy images and remapped labels to the destination dataset
"""

import argparse
import shutil
import sys
import yaml
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Our project class mapping
OUR_CLASSES = {
    "crosshair": 0,
    "enemy_head": 1,
    "enemyhead": 1,
    "enemyHead": 1,
    "enemy head": 1,
    "head": 1,
    "enemy_body": 2,
    "enemybody": 2,
    "enemyBody": 2,
    "enemy body": 2,
    "body": 2,
    "enemy": 2,
    "target": 3,
}


def detect_class_mapping(data_yaml_path):
    """Read data.yaml and build a mapping from source class ID to our class ID."""
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    source_names = config.get("names", {})
    if isinstance(source_names, list):
        source_names = {i: name for i, name in enumerate(source_names)}

    mapping = {}
    for src_id, src_name in source_names.items():
        src_id = int(src_id)
        # Try to match the source class name to our class
        name_lower = src_name.strip().lower().replace("-", "").replace("_", "")

        matched = False
        for our_name, our_id in OUR_CLASSES.items():
            if our_name.lower().replace("_", "") == name_lower:
                mapping[src_id] = our_id
                matched = True
                break

        if not matched:
            # Try partial matching
            if "head" in name_lower:
                mapping[src_id] = 1  # enemy_head
                matched = True
            elif "body" in name_lower or "enemy" in name_lower:
                mapping[src_id] = 2  # enemy_body
                matched = True
            elif "crosshair" in name_lower or "cross" in name_lower:
                mapping[src_id] = 0
                matched = True
            elif "target" in name_lower:
                mapping[src_id] = 3
                matched = True

        if not matched:
            mapping[src_id] = None  # Skip this class

    return source_names, mapping


def remap_label_file(src_path, dst_path, class_mapping):
    """Remap class IDs in a single YOLO label file."""
    lines = []
    remapped = 0
    skipped = 0

    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            src_class = int(parts[0])
            new_class = class_mapping.get(src_class, None)

            if new_class is not None:
                parts[0] = str(new_class)
                lines.append(" ".join(parts))
                remapped += 1
            else:
                skipped += 1

    with open(dst_path, "w") as f:
        f.write("\n".join(lines))

    return remapped, skipped


def remap_and_copy(source_dir, dest_dir):
    """Remap all labels from source and copy to destination dataset."""
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        print(f"[INFO] Please extract the Roboflow ZIP first.")
        sys.exit(1)

    # Find data.yaml
    data_yaml = source_dir / "data.yaml"
    if not data_yaml.exists():
        # Try README.roboflow.txt for clues, or ask user
        print(f"[ERROR] data.yaml not found in {source_dir}")
        print(f"[INFO] Make sure you extracted the Roboflow ZIP correctly.")
        sys.exit(1)

    # Detect class mapping
    source_names, class_mapping = detect_class_mapping(data_yaml)

    print("=" * 60)
    print("Roboflow Label Remapper")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_dir}")
    print(f"\nSource classes (from Roboflow data.yaml):")
    for src_id, src_name in source_names.items():
        our_id = class_mapping.get(int(src_id), None)
        our_name = {0: "crosshair", 1: "enemy_head", 2: "enemy_body", 3: "target"}.get(our_id, "SKIP")
        arrow = f"-> {our_id} ({our_name})" if our_id is not None else "-> SKIP"
        print(f"  {src_id}: {src_name} {arrow}")

    print()

    # Find source splits
    # Roboflow uses: train/, valid/, test/
    # Our project uses: images/train, images/val, labels/train, labels/val
    split_dirs = {
        "train": "train",
        "valid": "val",
        "test": "val",
    }

    total_images = 0
    total_remapped = 0
    total_skipped = 0

    for src_split, dst_split in split_dirs.items():
        # Try different directory structures
        # Structure 1: train/images/ and train/labels/
        src_images = source_dir / src_split / "images"
        src_labels = source_dir / src_split / "labels"

        if not src_images.exists():
            # Structure 2: images/train/ and labels/train/
            src_images = source_dir / "images" / src_split
            src_labels = source_dir / "labels" / src_split

        if not src_images.exists():
            print(f"[SKIP] No '{src_split}' split found")
            continue

        dst_images = dest_dir / "images" / dst_split
        dst_labels = dest_dir / "labels" / dst_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = (
            list(src_images.glob("*.jpg")) +
            list(src_images.glob("*.jpeg")) +
            list(src_images.glob("*.png"))
        )

        print(f"[PROCESS] {src_split} -> {dst_split}: {len(image_files)} images")

        for img_file in image_files:
            # Copy image (add prefix to avoid name collisions)
            dst_img_name = f"rf_{img_file.name}"
            shutil.copy2(img_file, dst_images / dst_img_name)

            # Find and remap label
            label_name = img_file.stem + ".txt"
            src_label = src_labels / label_name

            dst_label_name = f"rf_{img_file.stem}.txt"
            dst_label = dst_labels / dst_label_name

            if src_label.exists():
                remapped, skipped = remap_label_file(src_label, dst_label, class_mapping)
                total_remapped += remapped
                total_skipped += skipped
            else:
                # Create empty label file
                with open(dst_label, "w") as f:
                    f.write("")

            total_images += 1

        print(f"  Copied {len(image_files)} images + labels")

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Remap & Copy Complete!")
    print("=" * 60)
    print(f"  Images copied:      {total_images}")
    print(f"  Annotations mapped: {total_remapped}")
    print(f"  Annotations skipped:{total_skipped}")

    # Count final dataset
    for split in ["train", "val"]:
        img_count = len(list((dest_dir / "images" / split).glob("*.*")))
        lbl_count = len(list((dest_dir / "labels" / split).glob("*.txt")))
        print(f"  {split}: {img_count} images, {lbl_count} labels")

    print(f"\nDataset ready for training!")
    print(f"  python training/train_model.py --data {dest_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap Roboflow labels to project format")
    parser.add_argument("--source", default="dataset_roboflow",
                        help="Source directory (extracted Roboflow ZIP)")
    parser.add_argument("--dest", default="dataset",
                        help="Destination dataset directory")
    args = parser.parse_args()

    remap_and_copy(
        source_dir=PROJECT_ROOT / args.source,
        dest_dir=PROJECT_ROOT / args.dest,
    )
