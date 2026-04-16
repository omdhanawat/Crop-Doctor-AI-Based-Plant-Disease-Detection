"""
Background Recomposition Script
================================
Takes PlantVillage images (white/black background),
removes the background using rembg (U2Net AI model),
and pastes the leaf onto real outdoor photos.

INSTALL:
    pip install rembg pillow onnxruntime numpy tqdm

FOLDER STRUCTURE EXPECTED:
    data/
    ├── raw/plantvillage/
    │   ├── Tomato___Early_blight/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── ... (38 class folders)
    ├── backgrounds/
    │   ├── field1.jpg
    │   ├── grass1.jpg
    │   └── ... (200+ outdoor photos)
    └── recomposed/          <-- output goes here

USAGE:
    python recompose.py

    Optional args:
    python recompose.py --pv_dir data/raw/plantvillage
                        --bg_dir data/backgrounds
                        --out_dir data/recomposed
                        --variants 3
                        --max_per_class 500
                        --workers 4
"""

import os
import io
import random
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG DEFAULTS (override via CLI args)
# ─────────────────────────────────────────────
DEFAULT_PV_DIR        = "D:/Coding/Crop_Disease_Predection/data/raw/plantvillage dataset"
DEFAULT_BG_DIR        = "D:/Coding/Crop_Disease_Predection/data/backgrounds"
DEFAULT_OUT_DIR       = "D:/Coding/Crop_Disease_Predection/data/recomposed"
DEFAULT_VARIANTS      = 3       # synthetic images per source image
DEFAULT_MAX_PER_CLASS = 500     # cap source images per class
DEFAULT_OUTPUT_SIZE   = 224     # final image size (224x224 for MobileNetV2)
DEFAULT_WORKERS       = 2       # parallel processes (lower if RAM issues)
DEFAULT_JPEG_QUALITY  = 92      # output JPEG quality


# ─────────────────────────────────────────────
# STEP 1: REMOVE BACKGROUND
# ─────────────────────────────────────────────
def remove_background(image_path: Path, session) -> Image.Image | None:
    """
    Remove white/black background from a PlantVillage image.
    Returns RGBA PIL image with transparent background.
    Returns None if the image cannot be processed.
    """
    try:
        with open(image_path, "rb") as f:
            raw_bytes = f.read()

        # rembg removes background using U2Net model
        # output is PNG bytes with alpha channel
        output_bytes = remove(raw_bytes, session=session)
        leaf_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        return leaf_rgba

    except Exception as e:
        print(f"\n  [SKIP] {image_path.name}: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 2: LOAD AND PREPARE BACKGROUND
# ─────────────────────────────────────────────
def prepare_background(bg_path: Path, output_size: int) -> Image.Image | None:
    """
    Load a background photo and resize it to output_size x output_size.
    Applies slight random adjustments so each variant looks different.
    """
    try:
        bg = Image.open(bg_path).convert("RGBA")

        # Random crop before resize for variety
        w, h = bg.size
        crop_factor = random.uniform(0.75, 1.0)
        crop_w = int(w * crop_factor)
        crop_h = int(h * crop_factor)
        x0 = random.randint(0, w - crop_w)
        y0 = random.randint(0, h - crop_h)
        bg = bg.crop((x0, y0, x0 + crop_w, y0 + crop_h))

        # Resize to target
        bg = bg.resize((output_size, output_size), Image.LANCZOS)

        # Random brightness variation (simulates different lighting)
        brightness = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(bg.convert("RGB"))
        bg = enhancer.enhance(brightness).convert("RGBA")

        return bg

    except Exception as e:
        print(f"\n  [BG SKIP] {bg_path.name}: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 3: PASTE LEAF ONTO BACKGROUND
# ─────────────────────────────────────────────
def paste_leaf_on_background(
    leaf_rgba: Image.Image,
    bg_rgba: Image.Image,
    output_size: int
) -> Image.Image:
    """
    Scales the leaf to a random size and pastes it at a random
    position on the background. Applies edge softening to blend.
    """
    canvas_size = output_size

    # Random scale: leaf occupies 50–90% of canvas
    scale = random.uniform(0.50, 0.90)
    leaf_size = int(canvas_size * scale)
    leaf_resized = leaf_rgba.resize((leaf_size, leaf_size), Image.LANCZOS)

    # Random position: leaf fully within canvas bounds
    max_x = canvas_size - leaf_size
    max_y = canvas_size - leaf_size
    x = random.randint(0, max(0, max_x))
    y = random.randint(0, max(0, max_y))

    # Soften leaf edges: slight blur on alpha channel only
    # This makes the paste look natural instead of cut-out
    r, g, b, a = leaf_resized.split()
    a_blurred = a.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    leaf_blended = Image.merge("RGBA", (r, g, b, a_blurred))

    # Composite leaf onto background
    result = bg_rgba.copy()
    result.paste(leaf_blended, (x, y), mask=leaf_blended)

    return result.convert("RGB")


# ─────────────────────────────────────────────
# STEP 4: FINAL POST-PROCESSING
# ─────────────────────────────────────────────
def post_process(image: Image.Image) -> Image.Image:
    """
    Apply random post-processing to simulate phone camera conditions:
    - Random contrast variation
    - Random saturation shift
    - Occasional slight blur (camera shake)
    """
    # Random contrast
    contrast_factor = random.uniform(0.8, 1.2)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # Random saturation
    saturation_factor = random.uniform(0.8, 1.3)
    image = ImageEnhance.Color(image).enhance(saturation_factor)

    # Occasional blur (30% chance — simulates camera shake)
    if random.random() < 0.3:
        blur_radius = random.uniform(0.3, 0.8)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return image


# ─────────────────────────────────────────────
# CORE WORKER FUNCTION (runs per image)
# ─────────────────────────────────────────────
def process_single_image(args):
    """
    Full pipeline for one source image:
    remove bg → paste on N backgrounds → save N variants.
    Returns (success_count, skip_reason or None)
    """
    img_path, bg_paths, out_class_dir, variants, output_size, jpeg_quality = args

    try:
        # Each worker creates its own rembg session (not shareable across processes)
        session = new_session("u2net")
        leaf_rgba = remove_background(img_path, session)

        if leaf_rgba is None:
            return 0, f"bg removal failed: {img_path.name}"

        saved = 0
        for j in range(variants):
            bg_path = random.choice(bg_paths)
            bg_rgba = prepare_background(bg_path, output_size)
            if bg_rgba is None:
                continue

            result = paste_leaf_on_background(leaf_rgba, bg_rgba, output_size)
            result = post_process(result)

            out_filename = f"{img_path.stem}_rc_{j}.jpg"
            out_path = out_class_dir / out_filename
            result.save(out_path, "JPEG", quality=jpeg_quality)
            saved += 1

        return saved, None

    except Exception as e:
        return 0, str(e)


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────
def run_recomposition(
    pv_dir: str,
    bg_dir: str,
    out_dir: str,
    variants: int,
    max_per_class: int,
    output_size: int,
    workers: int,
    jpeg_quality: int,
):
    pv_path  = Path(pv_dir)
    bg_path  = Path(bg_dir)
    out_path = Path(out_dir)

    # ── Validate inputs ──────────────────────────────────────
    if not pv_path.exists():
        raise FileNotFoundError(f"PlantVillage dir not found: {pv_path}")
    if not bg_path.exists():
        raise FileNotFoundError(f"Backgrounds dir not found: {bg_path}")

    bg_files = (
        list(bg_path.glob("*.jpg")) +
        list(bg_path.glob("*.jpeg")) +
        list(bg_path.glob("*.png")) +
        list(bg_path.glob("*.JPG")) +
        list(bg_path.glob("*.JPEG"))
    )
    if len(bg_files) == 0:
        raise ValueError(f"No background images found in {bg_path}")

    print(f"\n{'='*55}")
    print(f"  Background Recomposition — PlantVillage")
    print(f"{'='*55}")
    print(f"  Source:      {pv_path}")
    print(f"  Backgrounds: {bg_path} ({len(bg_files)} images)")
    print(f"  Output:      {out_path}")
    print(f"  Variants:    {variants} per source image")
    print(f"  Max/class:   {max_per_class} source images")
    print(f"  Output size: {output_size}x{output_size}")
    print(f"  Workers:     {workers}")
    print(f"{'='*55}\n")

    # ── Discover class folders ───────────────────────────────
    class_dirs = sorted([d for d in pv_path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found in {pv_path}")

    print(f"  Found {len(class_dirs)} classes\n")

    total_saved  = 0
    total_failed = 0

    # ── Process each class ───────────────────────────────────
    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class  = out_path / class_name
        out_class.mkdir(parents=True, exist_ok=True)

        # Collect source images
        source_images = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.JPG")) +
            list(class_dir.glob("*.JPEG")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.PNG"))
        )

        # Cap source images
        if len(source_images) > max_per_class:
            source_images = random.sample(source_images, max_per_class)

        if len(source_images) == 0:
            print(f"  [SKIP] {class_name}: no images found")
            continue

        expected_output = len(source_images) * variants
        print(f"  {class_name}")
        print(f"    {len(source_images)} source → {expected_output} expected output")

        # Build args list for parallel processing
        args_list = [
            (img, bg_files, out_class, variants, output_size, jpeg_quality)
            for img in source_images
        ]

        class_saved  = 0
        class_failed = 0

        # Run with process pool for speed
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_image, a): a for a in args_list}
            with tqdm(total=len(args_list), desc=f"    Processing", leave=False, ncols=60) as pbar:
                for future in as_completed(futures):
                    saved, err = future.result()
                    class_saved  += saved
                    class_failed += (1 if err else 0)
                    pbar.update(1)

        total_saved  += class_saved
        total_failed += class_failed
        print(f"    Saved: {class_saved}  |  Skipped: {class_failed}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Total images saved:   {total_saved}")
    print(f"  Total images skipped: {total_failed}")
    print(f"  Output directory:     {out_path}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Background recomposition for PlantVillage dataset"
    )
    parser.add_argument("--pv_dir",        default=DEFAULT_PV_DIR,        help="PlantVillage root directory")
    parser.add_argument("--bg_dir",        default=DEFAULT_BG_DIR,        help="Background photos directory")
    parser.add_argument("--out_dir",       default=DEFAULT_OUT_DIR,       help="Output directory")
    parser.add_argument("--variants",      default=DEFAULT_VARIANTS,      type=int, help="Variants per image")
    parser.add_argument("--max_per_class", default=DEFAULT_MAX_PER_CLASS, type=int, help="Max source images per class")
    parser.add_argument("--output_size",   default=DEFAULT_OUTPUT_SIZE,   type=int, help="Output image size (px)")
    parser.add_argument("--workers",       default=DEFAULT_WORKERS,       type=int, help="Parallel worker processes")
    parser.add_argument("--quality",       default=DEFAULT_JPEG_QUALITY,  type=int, help="JPEG output quality")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recomposition(
        pv_dir        = args.pv_dir,
        bg_dir        = args.bg_dir,
        out_dir       = args.out_dir,
        variants      = args.variants,
        max_per_class = args.max_per_class,
        output_size   = args.output_size,
        workers       = args.workers,
        jpeg_quality  = args.quality,
    )