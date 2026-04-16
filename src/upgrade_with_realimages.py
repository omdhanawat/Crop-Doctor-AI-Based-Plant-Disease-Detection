"""
Real-World Image Integration + Model Upgrade
=============================================
Integrates your real-world labeled images into training
and runs phase 3 fine-tuning to improve real-world accuracy.

WHEN TO USE:
  You have real photos of diseased leaves organized in folders
  matching PlantVillage class names. Even 50 images per class
  will meaningfully improve real-world performance.

FOLDER STRUCTURE REQUIRED:
  your_real_images/
  ├── Tomato___Early_blight/
  │   ├── img001.jpg
  │   └── ...
  ├── Potato___Late_blight/
  │   └── ...
  └── (any PlantVillage class name)

USAGE:
  # Step 1: Verify your folders match PlantVillage names
  python src/upgrade_with_real_images.py --verify-only \
      --real_dir "path/to/your/real/images"

  # Step 2: Full upgrade (integrate + retrain)
  python src/upgrade_with_real_images.py \
      --real_dir "path/to/your/real/images"

  # Step 3: With augmentation multiplier (recommended if < 100 imgs/class)
  python src/upgrade_with_real_images.py \
      --real_dir "path/to/your/real/images" \
      --augment_factor 5
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MERGED_DIR     = Path("data/processed/disease_merged")
MODELS_DIR     = Path("models")
BACKUP_DIR     = MODELS_DIR / "backups"
LOG_DIR        = Path("logs/upgrade")

IMG_SIZE       = 224
BATCH_SIZE     = 16
EPOCHS         = 12
LR             = 3e-5          # very low LR — preserve existing knowledge
VALIDATION_SPLIT = 0.15
SEED           = 42
MIN_IMAGES     = 10            # minimum per class to include

NUM_THREADS    = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# All 38 PlantVillage class names — used for validation
PV_CLASSES = {
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
}


# ─────────────────────────────────────────────
# STEP 1: VERIFY FOLDERS
# ─────────────────────────────────────────────
def verify_real_images(real_dir: Path) -> tuple[list, list]:
    """
    Checks folder names against PV class names.
    Returns (matched_classes, unmatched_classes).
    """
    print("\n" + "="*55)
    print("  VERIFYING REAL IMAGE FOLDERS")
    print("="*55)

    if not real_dir.exists():
        print(f"\n[ERROR] Directory not found: {real_dir}")
        sys.exit(1)

    class_dirs = sorted([d for d in real_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"\n[ERROR] No subfolders found in {real_dir}")
        print("  Expected folders named like: Tomato___Early_blight/")
        sys.exit(1)

    matched   = []
    unmatched = []
    total_images = 0

    print(f"\n  {'Folder name':<55} {'Images':>7}  {'Status'}")
    print(f"  {'-'*55} {'-'*7}  {'-'*10}")

    for cls_dir in class_dirs:
        imgs  = [f for f in cls_dir.iterdir()
                 if f.is_file() and f.suffix in IMG_EXTS]
        count = len(imgs)

        if cls_dir.name in PV_CLASSES:
            status = "OK" if count >= MIN_IMAGES else f"LOW (<{MIN_IMAGES})"
            matched.append((cls_dir.name, count, cls_dir))
            total_images += count
        else:
            status = "NO MATCH"
            unmatched.append(cls_dir.name)

        flag = "[OK]" if cls_dir.name in PV_CLASSES else "[X]"
        print(f"  {flag} {cls_dir.name:<53} {count:>7}  {status}")

    print(f"\n  {'-'*55}")
    print(f"  Matched classes:   {len(matched)}")
    print(f"  Unmatched classes: {len(unmatched)}")
    print(f"  Total images:      {total_images:,}")

    if unmatched:
        print(f"\n  [WARN] These folders do not match any PlantVillage class:")
        for name in unmatched:
            print(f"         '{name}'")
        print(f"\n  They will be SKIPPED. Check spelling against PV class names.")
        print(f"  PlantVillage uses format: Plant___Disease")
        print(f"  Example: Tomato___Early_blight  (three underscores)")

    usable = [(n, c, d) for n, c, d in matched if c >= MIN_IMAGES]
    skipped_low = [(n, c, d) for n, c, d in matched if c < MIN_IMAGES]

    if skipped_low:
        print(f"\n  [WARN] These matched but have fewer than {MIN_IMAGES} images:")
        for name, count, _ in skipped_low:
            print(f"         {name}: {count} images — will be skipped")

    print(f"\n  Usable classes for training: {len(usable)}")
    print(f"  Total usable images:         {sum(c for _, c, _ in usable):,}")

    return usable, unmatched


# ─────────────────────────────────────────────
# STEP 2: COPY INTO MERGED TRAINING DIR
# ─────────────────────────────────────────────
def integrate_real_images(usable_classes: list,
                          augment_factor: int = 1) -> int:
    """
    Copies real images into the merged training directory.
    Optionally augments each image augment_factor times.
    Returns total number of images added.
    """
    print(f"\n[2/5] Integrating real images into training data...")
    print(f"      Augmentation factor: {augment_factor}x")

    total_added = 0

    for cls_name, count, src_dir in tqdm(usable_classes,
                                          desc="  Copying",
                                          ncols=65):
        dst_dir = MERGED_DIR / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in src_dir.iterdir()
                if f.is_file() and f.suffix in IMG_EXTS]

        for img_path in imgs:
            # Copy original
            dst = dst_dir / f"real_{img_path.name}"
            if not dst.exists():
                shutil.copy2(img_path, dst)
                total_added += 1

            # Generate augmented versions if requested
            if augment_factor > 1:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = image.resize((IMG_SIZE, IMG_SIZE),
                                         Image.LANCZOS)
                    for i in range(augment_factor - 1):
                        aug = _augment_pil(image, i)
                        aug_path = dst_dir / \
                            f"real_aug{i}_{img_path.stem}.jpg"
                        if not aug_path.exists():
                            aug.save(aug_path, "JPEG", quality=90)
                            total_added += 1
                except Exception as e:
                    pass  # skip corrupt images silently

    print(f"\n  Total images added to merged dir: {total_added:,}")
    return total_added


def _augment_pil(image: Image.Image, seed: int) -> Image.Image:
    """Apply a random augmentation to a PIL image."""
    import random
    random.seed(seed)

    ops = random.sample([
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        lambda img: img.rotate(random.choice([90, 180, 270])),
        lambda img: img.rotate(random.uniform(-25, 25), expand=False),
        lambda img: _adjust_brightness(img, random.uniform(0.6, 1.4)),
        lambda img: _adjust_brightness(img, random.uniform(0.7, 1.3)),
    ], k=random.randint(1, 2))

    result = image.copy()
    for op in ops:
        result = op(result)
    return result


def _adjust_brightness(img: Image.Image,
                        factor: float) -> Image.Image:
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(img).enhance(factor)


# ─────────────────────────────────────────────
# STEP 3: COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(classes_list: list) -> dict:
    counts = []
    for cls_name in classes_list:
        d = MERGED_DIR / cls_name
        if d.exists() and d.is_dir():
            c = len([f for f in d.iterdir() if f.suffix in IMG_EXTS])
        else:
            c = 0
        counts.append(c)
    
    counts = np.array(counts, dtype=np.float32)
    valid_counts = counts[counts > 0]
    
    # Avoid zero division if valid_counts is somehow empty
    if len(valid_counts) == 0:
        return {i: 1.0 for i in range(len(classes_list))}
        
    total = valid_counts.sum()
    n_cls = len(valid_counts)
    
    weights_dict = {}
    for i, c in enumerate(counts):
        if c > 0:
            weights_dict[i] = float(total / (n_cls * c))
        else:
            weights_dict[i] = 1.0  # fallback for missing classes
            
    return weights_dict


def get_data_generators():
    # Ensure strict 38-class mapping to avoid output shape mismatches
    pv_classes_list = sorted(list(PV_CLASSES))

    # Data generators — strong augmentation on real images
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale             = 1.0 / 255.0,
        validation_split    = VALIDATION_SPLIT,
        horizontal_flip     = True,
        vertical_flip       = True,
        rotation_range      = 40,
        zoom_range          = 0.3,
        brightness_range    = [0.5, 1.5],
        width_shift_range   = 0.15,
        height_shift_range  = 0.15,
        channel_shift_range = 30.0,
        shear_range         = 15,
        fill_mode           = "reflect",
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale          = 1.0 / 255.0,
        validation_split = VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        MERGED_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        classes     = pv_classes_list,
        class_mode  = "categorical",
        subset      = "training",
        seed        = SEED,
        shuffle     = True,
    )
    val_gen = val_datagen.flow_from_directory(
        MERGED_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        classes     = pv_classes_list,
        class_mode  = "categorical",
        subset      = "validation",
        seed        = SEED,
        shuffle     = False,
    )
    return train_gen, val_gen, pv_classes_list


# ─────────────────────────────────────────────
# STEP 4: PHASE 3 FINE-TUNING
# ─────────────────────────────────────────────
def run_upgrade_training() -> tuple:
    print(f"\n[3/5] Running upgrade fine-tuning...")
    print(f"      Strategy: unfreeze all layers, very low LR={LR}")
    print(f"      Epochs: {EPOCHS}")
    print(f"      Batch:  {BATCH_SIZE}")
    print(f"      Threads: {NUM_THREADS}")
    print(f"      Estimated time: 2–5 hours on CPU\n")

    # Load existing model
    keras_path  = MODELS_DIR / "disease_model.keras"
    saved_path  = MODELS_DIR / "disease_model_savedmodel"

    if keras_path.exists():
        print(f"  Loading: {keras_path}")
        model = tf.keras.models.load_model(str(keras_path))
    elif saved_path.exists():
        print(f"  Loading: {saved_path}")
        model = tf.keras.models.load_model(str(saved_path))
    else:
        print("[ERROR] No existing model found.")
        print("  Run train_disease_model.py first.")
        sys.exit(1)

    # Backup before touching it
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"disease_model_pre_upgrade_{ts}.keras"
    model.save(str(backup))
    print(f"  Backed up existing model → {backup}")

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True
    print(f"  All {len(model.layers)} layers unfrozen")

    train_gen, val_gen, pv_classes_list = get_data_generators()

    print(f"\n  Train samples: {train_gen.samples:,}")
    print(f"  Val samples:   {val_gen.samples:,}")

    # Class weights
    class_weight_dict = compute_class_weights(pv_classes_list)

    # Compile
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR),
        loss      = "categorical_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=3, name="top3_acc"),
        ],
    )

    # Callbacks
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"disease_model_upgraded_{ts}.keras"
    csv_log_path = LOG_DIR / f"upgrade_training_history_{ts}.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = str(ckpt_path),
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 4,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 2,
            min_lr   = 1e-8,
            verbose  = 1,
        )
    ]

    # Optional TensorBoard check
    try:
        import tensorboard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir        = str(LOG_DIR / f"upgrade_{ts}"),
                histogram_freq = 0,
            )
        )
        print("  TensorBoard logging enabled.")
    except (ImportError, Exception):
        print("  [WARN] TensorBoard not found or broken. Skipping TensorBoard logging...")

    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_log_path),
            separator=',',
            append=False
        )
    )

    # Train
    print(f"  Starting training at {datetime.now().strftime('%H:%M:%S')}...")
    print(f"  Note: First epoch might take a few minutes to start while preparing data.")
    
    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS,
        class_weight    = class_weight_dict,
        callbacks       = callbacks,
        verbose         = 1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n  Best val_accuracy: {best_val_acc:.4f}")

    # Save validation filenames for reproducibility
    val_filenames = val_gen.filenames
    with open(LOG_DIR / f"val_split_files_{ts}.txt", "w", encoding="utf-8") as f:
        for fname in val_filenames:
            full_path = Path(val_gen.directory) / fname
            f.write(f"{full_path}\n")

    # Load best checkpoint
    model = tf.keras.models.load_model(str(ckpt_path))
    return model, best_val_acc, ts, val_gen


# ─────────────────────────────────────────────
# STEP 4.5: RESEARCH ARTIFACTS
# ─────────────────────────────────────────────
def generate_research_artifacts(model, val_gen, ts: str):
    print(f"\n[4/5] Generating research artifacts (Confusion Matrix & Classification Report)...")
    
    # Reset validation generator
    val_gen.reset()
    
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    
    print("      Running predictions on validation set...")
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Classification Report
    report = classification_report(y_true, y_pred, labels=range(len(class_labels)), target_names=class_labels, zero_division=0)
    report_path = LOG_DIR / f"classification_report_{ts}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"      Saved Classification Report: {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(22, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Validation Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    cm_path = LOG_DIR / f"confusion_matrix_{ts}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"      Saved Confusion Matrix plot: {cm_path}")


# ─────────────────────────────────────────────
# STEP 5: EXPORT NEW TFLITE
# ─────────────────────────────────────────────
def export_upgraded_model(model, best_val_acc: float, ts: str):
    print(f"\n[5/5] Exporting upgraded model...")

    # Save new Keras model
    new_keras = MODELS_DIR / "disease_model.keras"
    model.save(str(new_keras))

    # Save SavedModel
    saved_path = MODELS_DIR / "disease_model_savedmodel"
    try:
        model.export(str(saved_path))
    except (AttributeError, Exception):
        # Fallback for older Keras versions if export() doesn't exist
        model.save(str(saved_path))

    # Export TFLite float32
    converter    = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_path))
    converter.optimizations = []
    tflite_model = converter.convert()

    # Backup old TFLite
    old_tflite = MODELS_DIR / "disease_model.tflite"
    if old_tflite.exists():
        shutil.copy2(old_tflite,
                     BACKUP_DIR / f"disease_model_pre_upgrade_{ts}.tflite")

    # Write new TFLite
    with open(old_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = old_tflite.stat().st_size / (1024 * 1024)
    print(f"  Saved: {old_tflite}")
    print(f"  Size:  {size_mb:.1f} MB")

    # Sanity check
    interp = tf.lite.Interpreter(model_path=str(old_tflite))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input:  {inp['shape']}")
    print(f"  Output: {out['shape']}")
    print(f"  Sanity check PASSED")

    # Save report
    report = {
        "upgrade_date":   datetime.now().isoformat(),
        "val_accuracy":   float(best_val_acc),
        "tflite_size_mb": round(size_mb, 2),
    }
    report_path = MODELS_DIR / f"upgrade_report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    return old_tflite


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Upgrade disease model with real-world images"
    )
    parser.add_argument(
        "--real_dir", default=r"D:\Coding\Crop_Disease_Predection\data\feedback_images",
        help="Path to your real-world images folder (default: D:\\Coding\\Crop_Disease_Predection\\data\\feedback_images)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify folders, do not train"
    )
    parser.add_argument(
        "--augment_factor", type=int, default=3,
        help="Augmentation multiplier for real images (default: 3)"
             " — use 5 if you have fewer than 100 images per class"
    )
    parser.add_argument(
        "--finish_ts", type=str, default=None,
        help="Skip training and only generate artifacts/export for this timestamp"
    )
    args = parser.parse_args()

    real_dir = Path(args.real_dir)

    print("="*55)
    print("  MODEL UPGRADE — REAL WORLD IMAGES")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Real images: {real_dir}")
    print(f"  Augment factor: {args.augment_factor}x")
    print("="*55)

    # Step 1: Verify
    usable, unmatched = verify_real_images(real_dir)

    if not usable:
        print("\n[ERROR] No usable classes found.")
        print("  Check that your folder names match PlantVillage exactly.")
        sys.exit(1)

    if args.verify_only:
        print(f"\n  Verify-only mode. No training performed.")
        print(f"  Run without --verify-only to start upgrade.")
        return

    # NEW: Skip training and finish from checkpoint if finish_ts is provided
    if args.finish_ts:
        ts = args.finish_ts
        ckpt_path = MODELS_DIR / f"disease_model_upgraded_{ts}.keras"
        if not ckpt_path.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            sys.exit(1)
            
        print(f"\n[FINISH-ONLY] Resuming from checkpoint: {ckpt_path}")
        model = tf.keras.models.load_model(str(ckpt_path))
        _, val_gen, _ = get_data_generators()
        
        # We don't have best_val_acc from history, so we'll estimate or leave as 0
        best_val_acc = 0.0 
    else:
        # Step 2: Integrate
        total_added = integrate_real_images(
            usable, augment_factor=args.augment_factor
        )

        print(f"\n  Real images added:  {total_added:,}")
        print(f"  Starting training in 5 seconds...")

        import time
        time.sleep(5)

        # Step 3: Train
        model, best_val_acc, ts, val_gen = run_upgrade_training()

    # Step 4: Research Artifacts
    generate_research_artifacts(model, val_gen, ts)

    # Step 5: Export
    tflite_path = export_upgraded_model(model, best_val_acc, ts)

    # Summary
    print("\n" + "="*55)
    print("  UPGRADE COMPLETE")
    print("="*55)
    print(f"  Best val_accuracy:  {best_val_acc:.4f}")
    print(f"  Model saved:        {tflite_path}")
    print()
    print("  NEXT STEPS:")
    print("  1. Copy new TFLite to Flutter assets:")
    print(f"     copy {tflite_path} "
          f"crop_disease_app\\assets\\models\\disease_model.tflite")
    print("  2. Rebuild APK:")
    print("     flutter build apk --release")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()