from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path

SEED = 42
VAL_RATIO = 0.10
TEST_RATIO = 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RAW_PV_DIR = Path("data/raw/plantvillage")
RECOMPOSED_DIR = Path("data/recomposed")
LEAF_SOURCE_DIR = Path("data/leaf_detector_data")
PROCESSED_ROOT = Path("data/processed")
DISEASE_ROOT = PROCESSED_ROOT / "disease_model"
LEAF_ROOT = PROCESSED_ROOT / "leaf_detector"
SUMMARY_PATH = Path("results/data_preparation_summary.json")


def clear_directory(target: Path) -> None:
    workspace = Path.cwd().resolve()
    resolved = target.resolve()
    if workspace not in resolved.parents:
        raise RuntimeError(f"Refusing to clear directory outside workspace: {resolved}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def iter_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def split_paths(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(SEED).shuffle(shuffled)
    total = len(shuffled)
    val_count = max(1, int(total * VAL_RATIO)) if total >= 10 else max(0, int(total * VAL_RATIO))
    test_count = max(1, int(total * TEST_RATIO)) if total >= 10 else max(0, int(total * TEST_RATIO))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError(f"Not enough samples to split safely: {total}")
    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]
    return train, val, test


def recomposed_source_stem(path: Path) -> str:
    match = re.match(r"^(.*)_rc_\d+$", path.stem)
    return match.group(1) if match else path.stem


def prepare_disease_model() -> dict[str, object]:
    if not RAW_PV_DIR.exists():
        raise FileNotFoundError(f"Missing PlantVillage directory: {RAW_PV_DIR}")
    if not RECOMPOSED_DIR.exists():
        raise FileNotFoundError(f"Missing recomposed directory: {RECOMPOSED_DIR}")

    originals_train_root = DISEASE_ROOT / "original" / "train"
    recomposed_train_root = DISEASE_ROOT / "recomposed" / "train"
    val_root = DISEASE_ROOT / "val"
    test_root = DISEASE_ROOT / "test"

    clear_directory(originals_train_root)
    clear_directory(recomposed_train_root)
    clear_directory(val_root)
    clear_directory(test_root)

    pv_classes = sorted(entry.name for entry in RAW_PV_DIR.iterdir() if entry.is_dir())
    rc_classes = sorted(entry.name for entry in RECOMPOSED_DIR.iterdir() if entry.is_dir())
    if pv_classes != rc_classes:
        raise ValueError("PlantVillage and recomposed class folders do not match exactly")

    link_mode = None
    summary: dict[str, object] = {"classes": {}, "total_original_train": 0, "total_val": 0, "total_test": 0, "total_recomposed_train": 0}

    for class_name in pv_classes:
        raw_images = iter_images(RAW_PV_DIR / class_name)
        if not raw_images:
            raise ValueError(f"No raw images found for class {class_name}")
        train_images, val_images, test_images = split_paths(raw_images)
        train_stems = {path.stem for path in train_images}

        recomposed_images = iter_images(RECOMPOSED_DIR / class_name)
        recomposed_train_images = [path for path in recomposed_images if recomposed_source_stem(path) in train_stems]

        for subset_root, images in (
            (originals_train_root / class_name, train_images),
            (val_root / class_name, val_images),
            (test_root / class_name, test_images),
            (recomposed_train_root / class_name, recomposed_train_images),
        ):
            subset_root.mkdir(parents=True, exist_ok=True)
            for src in images:
                method = link_or_copy(src, subset_root / src.name)
                if link_mode is None:
                    link_mode = method

        summary["classes"][class_name] = {
            "original_train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
            "recomposed_train": len(recomposed_train_images),
        }
        summary["total_original_train"] += len(train_images)
        summary["total_val"] += len(val_images)
        summary["total_test"] += len(test_images)
        summary["total_recomposed_train"] += len(recomposed_train_images)

    summary["storage_mode"] = link_mode or "unknown"
    return summary


def prepare_leaf_detector() -> dict[str, object]:
    if not LEAF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing leaf detector source directory: {LEAF_SOURCE_DIR}")

    train_root = LEAF_ROOT / "train"
    val_root = LEAF_ROOT / "val"
    test_root = LEAF_ROOT / "test"
    clear_directory(train_root)
    clear_directory(val_root)
    clear_directory(test_root)

    classes = sorted(entry.name for entry in LEAF_SOURCE_DIR.iterdir() if entry.is_dir())
    if classes != ["leaf", "non_leaf"]:
        raise ValueError(f"Expected leaf detector classes ['leaf', 'non_leaf'], found {classes}")

    link_mode = None
    summary: dict[str, object] = {"classes": {}, "total_train": 0, "total_val": 0, "total_test": 0}

    for class_name in classes:
        images = iter_images(LEAF_SOURCE_DIR / class_name)
        if not images:
            raise ValueError(f"No images found for leaf detector class {class_name}")
        train_images, val_images, test_images = split_paths(images)

        for subset_root, subset_images in (
            (train_root / class_name, train_images),
            (val_root / class_name, val_images),
            (test_root / class_name, test_images),
        ):
            subset_root.mkdir(parents=True, exist_ok=True)
            for src in subset_images:
                method = link_or_copy(src, subset_root / src.name)
                if link_mode is None:
                    link_mode = method

        summary["classes"][class_name] = {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }
        summary["total_train"] += len(train_images)
        summary["total_val"] += len(val_images)
        summary["total_test"] += len(test_images)

    summary["storage_mode"] = link_mode or "unknown"
    return summary


def main() -> None:
    results_dir = SUMMARY_PATH.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    disease_summary = prepare_disease_model()
    leaf_summary = prepare_leaf_detector()

    summary = {
        "seed": SEED,
        "disease_model": disease_summary,
        "leaf_detector": leaf_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Preparation complete")
    print(f"Summary written to: {SUMMARY_PATH}")
    print(f"Disease original train: {disease_summary['total_original_train']:,}")
    print(f"Disease val:            {disease_summary['total_val']:,}")
    print(f"Disease test:           {disease_summary['total_test']:,}")
    print(f"Disease recomposed:     {disease_summary['total_recomposed_train']:,}")
    print(f"Leaf train:             {leaf_summary['total_train']:,}")
    print(f"Leaf val:               {leaf_summary['total_val']:,}")
    print(f"Leaf test:              {leaf_summary['total_test']:,}")
    print(f"Storage mode:           {disease_summary['storage_mode']}")


if __name__ == "__main__":
    main()
