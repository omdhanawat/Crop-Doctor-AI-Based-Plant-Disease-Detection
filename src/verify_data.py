"""
Data Verification Script
=========================
Run this BEFORE training anything.
Checks every folder, every image, and prints a full health report.

WHAT IT CHECKS:
  1. All required folders exist
  2. PlantVillage — 38 classes present, image counts per class
  3. Recomposed   — 38 classes present, image counts per class
  4. Leaf detector dataset — leaf/ and non_leaf/ folders + counts
  5. Image integrity       — can every image actually be opened?
  6. Class name mismatch   — PV and recomposed folders must match exactly
  7. Imbalance warning     — flags any class with too few images
  8. Final go/no-go verdict

USAGE:
    python verify_data.py

    Optional: override default paths
    python verify_data.py --pv_dir data/raw/plantvillage
                          --rc_dir data/recomposed
                          --ld_dir data/leaf_detector
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ─────────────────────────────────────────────
# DEFAULT PATHS — edit these to match your setup
# ─────────────────────────────────────────────
DEFAULT_PV_DIR = "data/raw/plantvillage"      # original PlantVillage
DEFAULT_RC_DIR = "data/recomposed"            # background recomposed images
DEFAULT_LD_DIR = "data/leaf_detector_data"    # leaf/ and non_leaf/ folders

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

MIN_IMAGES_PER_CLASS  = 100    # warn if any class has fewer than this
MIN_LEAF_IMAGES       = 1000   # warn if leaf detector class has fewer than this
EXPECTED_PV_CLASSES   = 38


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_images(folder: Path) -> list:
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix in IMG_EXTENSIONS]


def check_image_integrity(image_paths: list, label: str) -> tuple[int, list]:
    """Try opening every image. Returns (ok_count, list_of_corrupt_paths)"""
    corrupt = []
    for p in tqdm(image_paths, desc=f"    Checking {label}", ncols=65, leave=False):
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            corrupt.append(p)
    return len(image_paths) - len(corrupt), corrupt


def print_header(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


# ─────────────────────────────────────────────
# CHECK 1: FOLDER EXISTENCE
# ─────────────────────────────────────────────
def check_folders(pv_dir, rc_dir, ld_dir) -> bool:
    print_header("1. Folder existence")
    all_ok = True

    folders = {
        "PlantVillage root":       pv_dir,
        "Recomposed root":         rc_dir,
        "Leaf detector root":      ld_dir,
        "Leaf detector — leaf/":   ld_dir / "leaf",
        "Leaf detector — non_leaf/": ld_dir / "non_leaf",
    }

    for name, path in folders.items():
        if path.exists():
            ok(f"{name}: {path}")
        else:
            fail(f"{name} NOT FOUND: {path}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────
# CHECK 2: PLANTVILLAGE CLASSES + COUNTS
# ─────────────────────────────────────────────
def check_plantvillage(pv_dir: Path) -> tuple[bool, list, dict]:
    print_header("2. PlantVillage dataset")
    all_ok = True

    class_dirs = sorted([d for d in pv_dir.iterdir() if d.is_dir()])
    pv_counts  = {}

    if len(class_dirs) == 0:
        fail("No class subdirectories found in PlantVillage directory")
        return False, [], {}

    print(f"\n  {'Class':<45} {'Images':>8}")
    print(f"  {'─'*45} {'─'*8}")

    for cls in class_dirs:
        imgs = get_images(cls)
        pv_counts[cls.name] = len(imgs)
        status = ""
        if len(imgs) < MIN_IMAGES_PER_CLASS:
            status = " <-- LOW"
            all_ok = False
        print(f"  {cls.name:<45} {len(imgs):>8}{status}")

    total = sum(pv_counts.values())
    print(f"\n  Total classes: {len(class_dirs)}")
    print(f"  Total images:  {total:,}")

    if len(class_dirs) < EXPECTED_PV_CLASSES:
        warn(f"Expected {EXPECTED_PV_CLASSES} classes, found {len(class_dirs)}")
        all_ok = False
    else:
        ok(f"All {len(class_dirs)} class folders present")

    if total < 10000:
        warn(f"Total images seems low ({total:,}). Expected ~54,000+")

    return all_ok, [d.name for d in class_dirs], pv_counts


# ─────────────────────────────────────────────
# CHECK 3: RECOMPOSED CLASSES + COUNTS
# ─────────────────────────────────────────────
def check_recomposed(rc_dir: Path, pv_class_names: list) -> tuple[bool, dict]:
    print_header("3. Recomposed dataset")
    all_ok = True

    class_dirs = sorted([d for d in rc_dir.iterdir() if d.is_dir()])
    rc_counts  = {}

    if len(class_dirs) == 0:
        fail("No class subdirectories found in recomposed directory")
        return False, {}

    print(f"\n  {'Class':<45} {'Images':>8}")
    print(f"  {'─'*45} {'─'*8}")

    for cls in class_dirs:
        imgs = get_images(cls)
        rc_counts[cls.name] = len(imgs)
        status = ""
        if len(imgs) < MIN_IMAGES_PER_CLASS:
            status = " <-- LOW"
            all_ok = False
        print(f"  {cls.name:<45} {len(imgs):>8}{status}")

    total = sum(rc_counts.values())
    print(f"\n  Total classes: {len(class_dirs)}")
    print(f"  Total images:  {total:,}")

    # Check class name match between PV and recomposed
    rc_names = {d.name for d in class_dirs}
    pv_names = set(pv_class_names)

    missing_in_rc   = pv_names - rc_names
    extra_in_rc     = rc_names - pv_names

    if missing_in_rc:
        warn(f"Classes in PlantVillage but NOT in recomposed ({len(missing_in_rc)}):")
        for c in sorted(missing_in_rc):
            print(f"       {c}")
        all_ok = False
    else:
        ok("All PlantVillage classes present in recomposed folder")

    if extra_in_rc:
        warn(f"Extra classes in recomposed not in PlantVillage ({len(extra_in_rc)}):")
        for c in sorted(extra_in_rc):
            print(f"       {c}")

    return all_ok, rc_counts


# ─────────────────────────────────────────────
# CHECK 4: LEAF DETECTOR DATASET
# ─────────────────────────────────────────────
def check_leaf_detector(ld_dir: Path) -> bool:
    print_header("4. Leaf detector dataset")
    all_ok = True

    leaf_dir     = ld_dir / "leaf"
    non_leaf_dir = ld_dir / "non_leaf"

    for name, d in [("leaf/", leaf_dir), ("non_leaf/", non_leaf_dir)]:
        if not d.exists():
            fail(f"{name} folder missing")
            all_ok = False
            continue

        imgs = get_images(d)
        count = len(imgs)

        if count < MIN_LEAF_IMAGES:
            warn(f"{name}: only {count} images (recommend >= {MIN_LEAF_IMAGES})")
            all_ok = False
        else:
            ok(f"{name}: {count:,} images")

    # Check balance between leaf and non-leaf
    if leaf_dir.exists() and non_leaf_dir.exists():
        leaf_count     = len(get_images(leaf_dir))
        non_leaf_count = len(get_images(non_leaf_dir))

        if leaf_count > 0 and non_leaf_count > 0:
            ratio = max(leaf_count, non_leaf_count) / min(leaf_count, non_leaf_count)
            if ratio > 2.0:
                warn(f"Class imbalance: leaf={leaf_count}, non_leaf={non_leaf_count} (ratio {ratio:.1f}x)")
                warn("Recommend keeping them within 2x of each other")
                all_ok = False
            else:
                ok(f"Class balance OK: leaf={leaf_count:,}, non_leaf={non_leaf_count:,}")

    return all_ok


# ─────────────────────────────────────────────
# CHECK 5: IMAGE INTEGRITY (sample check)
# ─────────────────────────────────────────────
def check_image_integrity_all(pv_dir: Path, rc_dir: Path, ld_dir: Path) -> bool:
    print_header("5. Image integrity (sample check)")
    all_ok = True

    # Sample 200 images per dataset for speed
    SAMPLE_SIZE = 200

    datasets = [
        ("PlantVillage", pv_dir),
        ("Recomposed",   rc_dir),
        ("Leaf detector", ld_dir),
    ]

    for label, root in datasets:
        all_imgs = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in IMG_EXTENSIONS:
                all_imgs.append(p)

        if len(all_imgs) == 0:
            warn(f"{label}: no images found for integrity check")
            continue

        import random
        sample = random.sample(all_imgs, min(SAMPLE_SIZE, len(all_imgs)))
        ok_count, corrupt = check_image_integrity(sample, label)

        if corrupt:
            warn(f"{label}: {len(corrupt)} corrupt images found in sample of {len(sample)}")
            for c in corrupt[:5]:
                print(f"       {c}")
            if len(corrupt) > 5:
                print(f"       ... and {len(corrupt)-5} more")
            all_ok = False
        else:
            ok(f"{label}: all {len(sample)} sampled images readable")

    return all_ok


# ─────────────────────────────────────────────
# CHECK 6: COMBINED SUMMARY + TRAINING ESTIMATE
# ─────────────────────────────────────────────
def print_training_summary(pv_counts: dict, rc_counts: dict):
    print_header("6. Training dataset summary")

    pv_total = sum(pv_counts.values())
    rc_total = sum(rc_counts.values())
    combined = pv_total + rc_total

    print(f"\n  PlantVillage originals:  {pv_total:>10,}")
    print(f"  Recomposed synthetic:    {rc_total:>10,}")
    print(f"  {'─'*35}")
    print(f"  Combined total:          {combined:>10,}")
    print()
    print(f"  Train split (80%):       {int(combined*0.8):>10,}")
    print(f"  Val   split (10%):       {int(combined*0.1):>10,}")
    print(f"  Test  split (10%):       {int(combined*0.1):>10,}")

    # Per-class stats
    all_classes = set(pv_counts) | set(rc_counts)
    per_class   = []
    for cls in all_classes:
        total = pv_counts.get(cls, 0) + rc_counts.get(cls, 0)
        per_class.append((cls, total))

    per_class.sort(key=lambda x: x[1])
    min_cls, min_count = per_class[0]
    max_cls, max_count = per_class[-1]
    avg_count = combined // max(len(all_classes), 1)

    print(f"\n  Per-class stats:")
    print(f"    Min: {min_count:,} images  ({min_cls})")
    print(f"    Max: {max_count:,} images  ({max_cls})")
    print(f"    Avg: {avg_count:,} images")

    if max_count > min_count * 3:
        warn("High class imbalance detected — consider using class_weight in training")
    else:
        ok("Class distribution looks balanced")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify all data before training")
    parser.add_argument("--pv_dir", default=DEFAULT_PV_DIR)
    parser.add_argument("--rc_dir", default=DEFAULT_RC_DIR)
    parser.add_argument("--ld_dir", default=DEFAULT_LD_DIR)
    args = parser.parse_args()

    pv_dir = Path(args.pv_dir)
    rc_dir = Path(args.rc_dir)
    ld_dir = Path(args.ld_dir)

    print(f"\n{'='*55}")
    print(f"  DATA VERIFICATION REPORT")
    print(f"{'='*55}")

    results = {}

    # Run all checks
    results["folders"]   = check_folders(pv_dir, rc_dir, ld_dir)

    if not results["folders"]:
        print(f"\n{'='*55}")
        fail("CRITICAL: Fix missing folders before continuing")
        print(f"{'='*55}\n")
        sys.exit(1)

    results["pv_ok"], pv_classes, pv_counts = check_plantvillage(pv_dir)
    results["rc_ok"], rc_counts             = check_recomposed(rc_dir, pv_classes)
    results["ld_ok"]                        = check_leaf_detector(ld_dir)
    results["integrity"]                    = check_image_integrity_all(pv_dir, rc_dir, ld_dir)

    if pv_counts and rc_counts:
        print_training_summary(pv_counts, rc_counts)

    # ── Final verdict ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL VERDICT")
    print(f"{'='*55}")

    check_names = {
        "folders":   "Folder structure",
        "pv_ok":     "PlantVillage dataset",
        "rc_ok":     "Recomposed dataset",
        "ld_ok":     "Leaf detector dataset",
        "integrity": "Image integrity",
    }

    all_passed = all(results.values())

    for key, label in check_names.items():
        status = "PASS" if results[key] else "FAIL"
        icon   = "OK  " if results[key] else "FAIL"
        print(f"  [{icon}]  {label}")

    print()
    if all_passed:
        print("  READY TO TRAIN — all checks passed")
        print("  Next step: run train_leaf_detector.py")
    else:
        print("  NOT READY — fix the FAIL items above first")
        print("  Re-run this script after fixing to confirm")

    print(f"{'='*55}\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()