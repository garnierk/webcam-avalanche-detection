#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from typing import Dict, Iterable

#########
# Constants
#########

GLIDE = "glide"
LOOSE = "loose"
SLAB = "slab"
NONE = "none"

IDX_TO_CLASS: Dict[int, str] = {0: GLIDE, 1: LOOSE, 2: SLAB}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
VALID_PROB = 0.1  # Proportion of training images in validation set

LABELS = ["glide", "loose", "none", "slab"]
DEFAULT_SEEDS = [50, 100, 150]


def _convert_label_file(output_path: str, annotation_path: str) -> None:
    """
    Convert a VOTT annotation file into YOLO bbox format.
    """
    with open(annotation_path, "rb") as f:
        ann_obj: Dict = json.load(f)

    img_size = ann_obj["asset"]["size"]
    _ = img_size["height"], img_size["width"]

    bb_s = []

    for region in ann_obj["regions"]:
        if region["type"] != "POLYGON":
            raise ValueError(f"Unsupported region type {region['type']}")

        assert len(region["tags"]) <= 1, f"Bounding box has multiple labels for {annotation_path}"
        assert len(region["tags"]) == 1, f"Bounding box has no label for {annotation_path}"

        b_h = region["boundingBox"]["height"]
        b_w = region["boundingBox"]["width"]
        b_left = region["boundingBox"]["left"]
        b_top = region["boundingBox"]["top"]

        center_x = b_left + (b_w / 2.0)
        center_y = b_top + (b_h / 2.0)

        class_idx = CLASS_TO_IDX[region["tags"][0]]
        norm_cen_x = center_x / img_size["width"]
        norm_cen_y = center_y / img_size["height"]
        norm_b_w = b_w / img_size["width"]
        norm_b_h = b_h / img_size["height"]

        bb_des = {
            "class": class_idx,
            "center_x": norm_cen_x,
            "center_y": norm_cen_y,
            "b_w": norm_b_w,
            "b_h": norm_b_h,
        }
        bb_s.append(bb_des)

    with open(output_path, "w") as f:
        for bb in bb_s:
            bbox_line = " ".join(
                str(v) for v in [bb["class"], bb["center_x"], bb["center_y"], bb["b_w"], bb["b_h"]]
            )
            f.write(bbox_line + "\n")


def _safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def _ensure_label_tree(base_dir: str) -> None:
    """
    Create:
      base_dir/train/labels/{glide,loose,none,slab}
      base_dir/test/labels/{glide,loose,none,slab}
    """
    ann_train_dir = os.path.join(base_dir, "train", "labels")
    ann_test_dir = os.path.join(base_dir, "test", "labels")

    for folder in [ann_train_dir, ann_test_dir]:
        _safe_rmtree(folder)
        for label in LABELS:
            os.makedirs(os.path.join(folder, label), exist_ok=False)


def convert_annotation_files(data_dir: str) -> None:
    """
    Convert annotation files into the format expected by YOLO.
    For class 'none', labels are intentionally left empty (no bbox file written),
    matching the original behavior.
    """
    data_dir = os.path.abspath(data_dir)
    _ensure_label_tree(data_dir)

    for im_dir in [os.path.join(data_dir, "train", "images"), os.path.join(data_dir, "test", "images")]:
        for root, _, filenames in os.walk(im_dir):
            for file_name in filenames:
                im_name, file_ext = os.path.splitext(file_name)
                if file_ext.lower() != ".jpg":
                    raise AssertionError(f"Bad file extension {file_name}")

                im_path = os.path.join(root, file_name)
                label = os.path.basename(os.path.dirname(im_path))

                assert label in LABELS, f"Unknown label {label}"

                if label != NONE:
                    ann_path = os.path.join(data_dir, "annotations", label, f"{im_name}.json")
                    output_path = os.path.join(root, f"{im_name}.txt").replace(
                        f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
                    )
                    _convert_label_file(output_path=output_path, annotation_path=ann_path)


def create_train_valid_split(train_valid_seed: int, data_dir: str, dest_dir: str) -> None:
    """
    Create train/val/test txt files with absolute image paths.
    """
    data_dir = os.path.abspath(data_dir)
    dest_dir = os.path.abspath(dest_dir)

    test_dir = os.path.join(data_dir, "test", "images")
    train_dir = os.path.join(data_dir, "train", "images")

    random.seed(train_valid_seed)

    # reset old files for this seed
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

                with open(txt_path, "a") as f_txt:
                    f_txt.write(abs_path + "\n")

    print(f"Txt files created with seed {train_valid_seed}")


def write_dataset_yaml(seed: int, dest_dir: str) -> str:
    """
    Write avalanchesplit{seed}.yaml next to the txt split files.
    """
    dest_dir = os.path.abspath(dest_dir)
    yaml_path = os.path.join(dest_dir, f"avalanchesplit{seed}.yaml")

    train_txt = os.path.abspath(os.path.join(dest_dir, f"train{seed}.txt"))
    val_txt = os.path.abspath(os.path.join(dest_dir, f"val{seed}.txt"))
    test_txt = os.path.abspath(os.path.join(dest_dir, f"test{seed}.txt"))

    with open(yaml_path, "w") as f:
        f.write(f"train: {train_txt}\n")
        f.write(f"val: {val_txt}\n")
        f.write(f"test: {test_txt}\n")
        f.write(f"nc: {len(LABELS)}\n")
        f.write(f"names: {LABELS}\n")

    return yaml_path


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


def prepare_dataset(data_dir: str, output_dir: str, seeds: Iterable[int]) -> None:
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    check_dataset_layout(data_dir)

    # 1) write labels under train/labels and test/labels
    convert_annotation_files(data_dir=data_dir)

    # 2) recreate split dir
    _safe_rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    # 3) create split txt + yaml for each seed
    for seed in seeds:
        create_train_valid_split(seed, data_dir=data_dir, dest_dir=output_dir)
        yaml_path = write_dataset_yaml(seed, dest_dir=output_dir)
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

    prepare_dataset(data_dir=data_dir, output_dir=output_dir, seeds=args.seeds)
