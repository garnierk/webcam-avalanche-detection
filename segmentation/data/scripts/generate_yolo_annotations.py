#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from typing import Dict, Iterable, List, Tuple

#########
# Constants
#########

GLIDE = "glide"
LOOSE = "loose"
SLAB = "slab"
NONE = "none"

# Classes réellement détectées par YOLO
YOLO_CLASSES: List[str] = [GLIDE, LOOSE, SLAB]

# Labels de dossiers/images présents dans le dataset
ALL_IMAGE_LABELS: List[str] = [GLIDE, LOOSE, NONE, SLAB]

IDX_TO_CLASS: Dict[int, str] = {0: GLIDE, 1: LOOSE, 2: SLAB}
CLASS_TO_IDX: Dict[str, int] = {v: k for k, v in IDX_TO_CLASS.items()}

VALID_PROB = 0.1  # Proportion of training images in validation set
DEFAULT_SEEDS = [50, 100, 150]


#########
# Utils
#########

def _safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def clamp_box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Clamp une bbox YOLO normalisée pour qu'elle reste dans l'image.
    Entrée/sortie: cx, cy, w, h normalisés dans [0,1] après correction.
    """
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    x1 = clamp(x1)
    y1 = clamp(y1)
    x2 = clamp(x2)
    y2 = clamp(y2)

    new_w = x2 - x1
    new_h = y2 - y1
    new_cx = (x1 + x2) / 2.0
    new_cy = (y1 + y2) / 2.0

    return new_cx, new_cy, new_w, new_h


#########
# Conversion annotations -> YOLO
#########

def _convert_label_file(output_path: str, annotation_path: str) -> None:
    """
    Convert a VOTT annotation file into YOLO bbox format.
    Seules les classes glide/loose/slab sont autorisées ici.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann_obj: Dict = json.load(f)

    img_size = ann_obj["asset"]["size"]
    img_h = img_size["height"]
    img_w = img_size["width"]

    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Invalid image size in annotation: {annotation_path}")

    bb_lines: List[str] = []

    for region in ann_obj.get("regions", []):
        if region["type"] != "POLYGON":
            raise ValueError(f"Unsupported region type {region['type']} in {annotation_path}")

        if len(region["tags"]) > 1:
            raise ValueError(f"Bounding box has multiple labels for {annotation_path}")
        if len(region["tags"]) == 0:
            raise ValueError(f"Bounding box has no label for {annotation_path}")

        tag = region["tags"][0]
        if tag == NONE:
            # Par sécurité: NONE ne doit pas être une classe détectée.
            continue

        if tag not in CLASS_TO_IDX:
            raise ValueError(f"Unknown detection label '{tag}' in {annotation_path}")

        bbox = region["boundingBox"]
        b_h = float(bbox["height"])
        b_w = float(bbox["width"])
        b_left = float(bbox["left"])
        b_top = float(bbox["top"])

        center_x = b_left + (b_w / 2.0)
        center_y = b_top + (b_h / 2.0)

        class_idx = CLASS_TO_IDX[tag]

        norm_cx = center_x / img_w
        norm_cy = center_y / img_h
        norm_w = b_w / img_w
        norm_h = b_h / img_h

        # Corrige les bboxes qui débordent légèrement
        norm_cx, norm_cy, norm_w, norm_h = clamp_box(norm_cx, norm_cy, norm_w, norm_h)

        # Ignore les bboxes dégénérées
        if norm_w <= 0.0 or norm_h <= 0.0:
            continue

        bb_lines.append(f"{class_idx} {norm_cx:.10f} {norm_cy:.10f} {norm_w:.10f} {norm_h:.10f}")

    # On écrit toujours le fichier si on a des boxes.
    # Si aucune bbox valide, on n'écrit rien.
    if bb_lines:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for line in bb_lines:
                f.write(line + "\n")
    else:
        # Si un ancien fichier existe, on le supprime.
        if os.path.exists(output_path):
            os.remove(output_path)


def _ensure_label_tree(base_dir: str) -> None:
    """
    Create:
      base_dir/train/labels/{glide,loose,slab}
      base_dir/test/labels/{glide,loose,slab}

    Pas de labels/none : les images 'none' sont des backgrounds.
    """
    ann_train_dir = os.path.join(base_dir, "train", "labels")
    ann_test_dir = os.path.join(base_dir, "test", "labels")

    for folder in [ann_train_dir, ann_test_dir]:
        _safe_rmtree(folder)
        for label in YOLO_CLASSES:
            os.makedirs(os.path.join(folder, label), exist_ok=False)


def convert_annotation_files(data_dir: str) -> None:
    """
    Convert annotation files into the format expected by YOLO.

    - Pour glide/loose/slab: crée les labels YOLO
    - Pour none: aucune annotation YOLO écrite
    """
    data_dir = os.path.abspath(data_dir)
    _ensure_label_tree(data_dir)

    image_roots = [
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "test", "images"),
    ]

    for im_dir in image_roots:
        for root, _, filenames in os.walk(im_dir):
            for file_name in filenames:
                im_name, file_ext = os.path.splitext(file_name)

                if file_ext.lower() != ".jpg":
                    raise AssertionError(f"Bad file extension {file_name}")

                im_path = os.path.join(root, file_name)
                label = os.path.basename(os.path.dirname(im_path))

                if label not in ALL_IMAGE_LABELS:
                    raise AssertionError(f"Unknown label directory '{label}' for {im_path}")

                # NONE = background only
                if label == NONE:
                    continue

                ann_path = os.path.join(data_dir, "annotations", label, f"{im_name}.json")
                if not os.path.isfile(ann_path):
                    raise FileNotFoundError(f"Annotation not found: {ann_path}")

                output_path = os.path.join(root, f"{im_name}.txt").replace(
                    f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
                )
                _convert_label_file(output_path=output_path, annotation_path=ann_path)


#########
# Splits
#########

def create_train_valid_split(train_valid_seed: int, data_dir: str, dest_dir: str) -> None:
    """
    Create train/val/test txt files with absolute image paths.
    - test/images -> test{seed}.txt
    - train/images -> split train/val according to VALID_PROB
    """
    data_dir = os.path.abspath(data_dir)
    dest_dir = os.path.abspath(dest_dir)

    test_dir = os.path.join(data_dir, "test", "images")
    train_dir = os.path.join(data_dir, "train", "images")

    random.seed(train_valid_seed)

    for split_name in ["train", "val", "test"]:
        txt_path = os.path.join(dest_dir, f"{split_name}{train_valid_seed}.txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)

    for crawl_dir in [test_dir, train_dir]:
        for root, _, filenames in os.walk(crawl_dir):
            for file_name in filenames:
                if not file_name.lower().endswith(".jpg"):
                    continue

                if crawl_dir == test_dir:
                    split_file = "test"
                else:
                    split_file = "val" if random.random() < VALID_PROB else "train"

                txt_path = os.path.join(dest_dir, f"{split_file}{train_valid_seed}.txt")
                abs_path = os.path.abspath(os.path.join(root, file_name))

                with open(txt_path, "a", encoding="utf-8") as f_txt:
                    f_txt.write(abs_path + "\n")

    print(f"Txt files created with seed {train_valid_seed}")


#########
# YAML
#########

def write_dataset_yaml(seed: int, dest_dir: str, source_dir: str) -> str:
    """
    Write avalanchesplit{seed}.yaml into <source_dir>/segmentation/data/
    """
    dest_dir = os.path.abspath(dest_dir)
    source_dir = os.path.abspath(source_dir)

    seg_data_dir = os.path.join(source_dir, "segmentation", "data")
    os.makedirs(seg_data_dir, exist_ok=True)

    yaml_path = os.path.join(seg_data_dir, f"avalanchesplit{seed}.yaml")

    train_txt = os.path.abspath(os.path.join(dest_dir, f"train{seed}.txt"))
    val_txt = os.path.abspath(os.path.join(dest_dir, f"val{seed}.txt"))
    test_txt = os.path.abspath(os.path.join(dest_dir, f"test{seed}.txt"))

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Avalanches dataset information.\n\n")
        f.write("# Train/val/test sets as file lists of absolute image paths\n")
        f.write(f"path: {dest_dir}\n")
        f.write(f"train: {train_txt}\n")
        f.write(f"val: {val_txt}\n")
        f.write(f"test: {test_txt}\n\n")
        f.write("# Classes\n")
        f.write(f"nc: {len(YOLO_CLASSES)}\n")
        f.write("names:\n")
        for i, name in enumerate(YOLO_CLASSES):
            f.write(f"  {i}: {name}\n")

    return yaml_path


#########
# Validation / checks
#########

def check_dataset_layout(data_dir: str) -> None:
    data_dir = os.path.abspath(data_dir)
    required_dirs = [
        data_dir,
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "test", "images"),
        os.path.join(data_dir, "annotations"),
    ]
    for directory in required_dirs:
        assert os.path.isdir(directory), f"Directory not found: {directory}"


def sanity_check_generated_labels(data_dir: str) -> None:
    """
    Petit check post-génération:
    - classes uniquement 0,1,2
    - bbox dans [0,1]
    """
    allowed = {0, 1, 2}
    label_roots = [
        os.path.join(data_dir, "train", "labels"),
        os.path.join(data_dir, "test", "labels"),
    ]

    errors = []

    for label_root in label_roots:
        if not os.path.isdir(label_root):
            continue

        for root, _, filenames in os.walk(label_root):
            for file_name in filenames:
                if not file_name.endswith(".txt"):
                    continue

                txt_path = os.path.join(root, file_name)
                with open(txt_path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) != 5:
                            errors.append((txt_path, lineno, "bad_format", line))
                            continue

                        try:
                            cls = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:])
                        except Exception:
                            errors.append((txt_path, lineno, "parse_error", line))
                            continue

                        if cls not in allowed:
                            errors.append((txt_path, lineno, "bad_class", line))

                        if any(v < 0.0 or v > 1.0 for v in (x, y, w, h)):
                            errors.append((txt_path, lineno, "not_normalized", line))

                        if w <= 0.0 or h <= 0.0:
                            errors.append((txt_path, lineno, "non_positive_wh", line))

                        if x - w / 2 < 0.0 or x + w / 2 > 1.0 or y - h / 2 < 0.0 or y + h / 2 > 1.0:
                            errors.append((txt_path, lineno, "outside_image", line))

    if errors:
        print(f"[WARN] {len(errors)} label issues detected after generation:")
        for e in errors[:100]:
            print("  ", e)
    else:
        print("[OK] Generated labels passed sanity check.")


#########
# Main pipeline
#########

def prepare_dataset(data_dir: str, output_dir: str, source_dir: str, seeds: Iterable[int]) -> None:
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    check_dataset_layout(data_dir)

    # 1) write labels under train/labels and test/labels
    convert_annotation_files(data_dir=data_dir)

    # 2) optional sanity check
    sanity_check_generated_labels(data_dir=data_dir)

    # 3) recreate split dir
    _safe_rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    # 4) create split txt + yaml for each seed
    for seed in seeds:
        create_train_valid_split(seed, data_dir=data_dir, dest_dir=output_dir)
        yaml_path = write_dataset_yaml(seed, dest_dir=output_dir, source_dir=source_dir)
        print(f"YAML created: {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO labels and train/val/test split files for cluster training."
    )
    parser.add_argument(
        "--data-dir",
        default=".data",
        help="Root directory containing train/images, test/images, annotations",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where split txt and YAML files are written. Default: <data-dir>/yolo_split",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Root of the source repository where segmentation/data YAML files will be written",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds used to generate train/val splits",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(data_dir, "yolo_split")
    source_dir = os.path.abspath(args.source_dir)

    prepare_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        source_dir=source_dir,
        seeds=args.seeds,
    )
