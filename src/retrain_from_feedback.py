"""
Retrain Phase 3 — From User Feedback
======================================
Downloads confirmed feedback images from Firestore + Cloudinary,
adds them to your existing training data, and runs phase 3
fine-tuning only (~2-3 hours on CPU).

WHEN TO RUN:
  Run this script when you have 500+ confirmed feedback images.
  Check count with: python src/retrain_from_feedback.py --count-only

WHAT IT DOES:
  1. Connects to Firestore and fetches all feedback documents
     where feedback = "correct" OR feedback = "wrong"
  2. Downloads each image from Cloudinary URL
  3. Places image in correct class folder using:
       - correctClass  (if feedback = "wrong")
       - predictedClass (if feedback = "correct")
  4. Runs phase 3 fine-tuning on the existing disease model
     using all data (original + recomposed + new feedback images)
  5. Exports new disease_model.tflite
  6. Saves a report of what changed

INSTALL:
  pip install firebase-admin requests tensorflow pillow tqdm

USAGE:
  # Check how many feedback images you have
  python src/retrain_from_feedback.py --count-only

  # Download images only (no training)
  python src/retrain_from_feedback.py --download-only

  # Full retrain
  python src/retrain_from_feedback.py

FIREBASE SETUP:
  1. Firebase console → Project settings → Service accounts
  2. Click "Generate new private key"
  3. Save as: firebase_credentials.json in project root
  4. NEVER commit this file to git — add to .gitignore
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageOps
import io

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FIREBASE_CREDENTIALS = Path("firebase_credentials.json")
FIRESTORE_COLLECTION = "predictions"

# Directories
FEEDBACK_DIR  = Path("data/feedback_images")   # downloaded feedback images
MERGED_DIR    = Path("data/processed/disease_merged")
MODELS_DIR    = Path("models")
LOG_DIR       = Path("logs/retrain")

# Model paths
EXISTING_MODEL = MODELS_DIR / "disease_model.keras"
BACKUP_DIR     = MODELS_DIR / "backups"

# Training config
IMG_SIZE        = 224
BATCH_SIZE      = 16
EPOCHS_PHASE3   = 10
LR_PHASE3       = 5e-5
VALIDATION_SPLIT= 0.15
SEED            = 42
MIN_IMAGES      = 100   # minimum feedback images before retraining

# CPU threading
NUM_THREADS = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ─────────────────────────────────────────────
# STEP 1: CONNECT TO FIRESTORE
# ─────────────────────────────────────────────
def init_firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not FIREBASE_CREDENTIALS.exists():
        print(f"\n[ERROR] Firebase credentials not found: {FIREBASE_CREDENTIALS}")
        print("  1. Go to Firebase console -> Project settings -> Service accounts")
        print("  2. Click 'Generate new private key'")
        print("  3. Save as firebase_credentials.json in project root")
        sys.exit(1)

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(FIREBASE_CREDENTIALS))
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("[OK] Connected to Firestore")
    return db


# ─────────────────────────────────────────────
# STEP 2: FETCH FEEDBACK DOCUMENTS
# ─────────────────────────────────────────────
def fetch_feedback_docs(db, min_confidence: float = 0.0):
    """
    Fetch all documents where feedback is 'correct' or 'wrong'.
    Skips 'skip' feedback — no confirmed label.
    """
    from firebase_admin import firestore as fs

    print("\n[1/5] Fetching feedback from Firestore...")

    collection = db.collection(FIRESTORE_COLLECTION)

    correct_docs = collection.where(
        filter=fs.FieldFilter("feedback", "==", "correct")
    ).stream()

    wrong_docs = collection.where(
        filter=fs.FieldFilter("feedback", "==", "wrong")
    ).stream()

    docs = []

    for doc in correct_docs:
        d = doc.to_dict()
        label = d.get("rawLabel") or d.get("rawPredictedClass") or d.get("predictedClass")
        if d.get("imageUrl") and label:
            docs.append({
                "id":         doc.id,
                "image_url":  d["imageUrl"],
                "label":      label,                 # confirmed correct (prefer raw label)
                "feedback":   "correct",
                "confidence": d.get("confidence", 0.0),
            })

    for doc in wrong_docs:
        d = doc.to_dict()
        label = d.get("rawLabel") or d.get("rawCorrectClass") or d.get("correctClass")
        if d.get("imageUrl") and label:
            docs.append({
                "id":         doc.id,
                "image_url":  d["imageUrl"],
                "label":      label,                 # user-corrected label (prefer raw label)
                "feedback":   "wrong",
                "confidence": d.get("confidence", 0.0),
            })

    # Filter by minimum confidence for "correct" feedback
    # Low confidence correct predictions are less reliable
    reliable = [d for d in docs
                if d["feedback"] == "wrong"
                or d["confidence"] >= min_confidence]

    print(f"  Total feedback docs:     {len(docs)}")
    print(f"    correct feedback:      {sum(1 for d in docs if d['feedback'] == 'correct')}")
    print(f"    wrong + corrected:     {sum(1 for d in docs if d['feedback'] == 'wrong')}")
    print(f"  After confidence filter: {len(reliable)}")

    return reliable


# ─────────────────────────────────────────────
# STEP 3: DOWNLOAD IMAGES
# ─────────────────────────────────────────────
def download_images(docs: list) -> dict:
    """
    Downloads images from Cloudinary URLs into:
      data/feedback_images/<class_name>/<image_id>.jpg

    Returns dict: {class_name: count}
    """
    print(f"\n[2/5] Downloading {len(docs)} images...")

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    class_counts = {}
    skipped      = 0

    for doc in tqdm(docs, desc="  Downloading", ncols=65):
        label     = doc["label"]
        image_url = doc["image_url"]
        doc_id    = doc["id"]

        # Sanitize label for folder name
        safe_label = label.replace("/", "_").replace("\\", "_").strip()
        class_dir  = FEEDBACK_DIR / safe_label
        class_dir.mkdir(parents=True, exist_ok=True)

        out_path = class_dir / f"{doc_id}.jpg"

        # Skip if already downloaded
        if out_path.exists():
            class_counts[safe_label] = class_counts.get(safe_label, 0) + 1
            continue

        try:
            response = requests.get(image_url, timeout=15)
            if response.status_code == 200:
                # Convert to JPEG and center-crop to 224x224 to maintain aspect ratio
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), method=Image.LANCZOS)
                img.save(out_path, "JPEG", quality=92)
                class_counts[safe_label] = class_counts.get(safe_label, 0) + 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"\n  [SKIP] {doc_id}: {e}")

    print(f"\n  Downloaded successfully: {sum(class_counts.values())}")
    print(f"  Skipped / failed:        {skipped}")
    print(f"\n  Per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:<50} {count:>4}")

    return class_counts


# ─────────────────────────────────────────────
# STEP 4: MERGE FEEDBACK INTO TRAINING DATA
# ─────────────────────────────────────────────
def merge_feedback_into_training():
    """
    Copies feedback images into the merged training directory
    so the phase 3 training sees them alongside existing data.
    """
    import shutil

    print("\n[3/5] Merging feedback images into training data...")

    if not FEEDBACK_DIR.exists():
        print("  No feedback images found — run --download-only first")
        return 0

    total_copied = 0

    for class_dir in FEEDBACK_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        cls_name    = class_dir.name
        out_cls_dir = MERGED_DIR / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in IMG_EXTS]

        for img in images:
            dst = out_cls_dir / f"fb_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
                total_copied += 1

    print(f"  Copied {total_copied} feedback images into merged training dir")
    return total_copied


# ─────────────────────────────────────────────
# STEP 5: PHASE 3 FINE-TUNING
# ─────────────────────────────────────────────
def run_phase3_finetuning():
    print("\n[4/5] Phase 3 fine-tuning...")

    # ── Load existing best model ───────────────────────────
    if not EXISTING_MODEL.exists():
        # Try SavedModel format
        saved_model_path = MODELS_DIR / "disease_model_savedmodel"
        if saved_model_path.exists():
            print(f"  Loading SavedModel: {saved_model_path}")
            model = tf.keras.models.load_model(str(saved_model_path))
        else:
            print(f"[ERROR] No existing model found at {EXISTING_MODEL}")
            print("  Train the base model first with train_disease_model.py")
            sys.exit(1)
    else:
        print(f"  Loading model: {EXISTING_MODEL}")
        model = tf.keras.models.load_model(str(EXISTING_MODEL))

    # ── Backup existing model ──────────────────────────────
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"disease_model_before_retrain_{ts}.keras"
    model.save(str(backup_path))
    print(f"  Backed up existing model to: {backup_path}")

    # ── Unfreeze all layers for phase 3 ───────────────────
    for layer in model.layers:
        layer.trainable = True
    print(f"  All {len(model.layers)} layers unfrozen")

    # ── Compile ────────────────────────────────────────────
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_PHASE3),
        loss      = "categorical_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    # ── Data generators ────────────────────────────────────
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale            = 1.0 / 255.0,
        validation_split   = VALIDATION_SPLIT,
        horizontal_flip    = True,
        vertical_flip      = True,
        rotation_range     = 40,
        zoom_range         = 0.3,
        brightness_range   = [0.5, 1.5],
        width_shift_range  = 0.15,
        height_shift_range = 0.15,
        channel_shift_range= 30.0,
        shear_range        = 15,
        fill_mode          = "reflect",
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale          = 1.0 / 255.0,
        validation_split = VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        MERGED_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        subset      = "training",
        seed        = SEED,
        shuffle     = True,
    )
    val_gen = val_datagen.flow_from_directory(
        MERGED_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        subset      = "validation",
        seed        = SEED,
        shuffle     = False,
    )

    print(f"\n  Train samples: {train_gen.samples:,}")
    print(f"  Val samples:   {val_gen.samples:,}")
    print(f"  Classes:       {train_gen.num_classes}")
    print(f"  Epochs:        {EPOCHS_PHASE3}")
    print(f"  LR:            {LR_PHASE3}")
    print(f"  Estimated time: 2-4 hours on CPU\n")

    # ── Compute class weights ──────────────────────────────
    class_dirs = [d for d in MERGED_DIR.iterdir() if d.is_dir()]
    counts     = np.array([
        len([f for f in d.iterdir() if f.suffix in IMG_EXTS])
        for d in class_dirs
    ], dtype=np.float32)
    total      = counts.sum()
    n_cls      = len(counts)
    weights    = total / (n_cls * counts)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

    # ── Callbacks ─────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"disease_model_retrained_{ts}.keras"

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
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir        = str(LOG_DIR / f"retrain_{ts}"),
            histogram_freq = 0,
        ),
    ]

    # ── Train ──────────────────────────────────────────────
    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS_PHASE3,
        class_weight    = class_weight_dict,
        callbacks       = callbacks,
        verbose         = 1,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n  Best val_accuracy: {best_val_acc:.4f}")
    return model, history, ckpt_path, best_val_acc


# ─────────────────────────────────────────────
# STEP 6: EXPORT NEW TFLITE
# ─────────────────────────────────────────────
def export_tflite(model, best_val_acc: float):
    print("\n[5/5] Exporting new TFLite model...")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save new Keras model as primary
    new_model_path = MODELS_DIR / "disease_model.keras"
    model.save(str(new_model_path))
    print(f"  Saved Keras model: {new_model_path}")

    # Also save as SavedModel
    saved_model_path = MODELS_DIR / "disease_model_savedmodel"
    model.save(str(saved_model_path))

    # Export TFLite float32
    converter    = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_path)
    )
    converter.optimizations = []
    tflite_model = converter.convert()

    # Backup old TFLite before overwriting
    old_tflite = MODELS_DIR / "disease_model.tflite"
    if old_tflite.exists():
        backup_tflite = BACKUP_DIR / f"disease_model_{ts}.tflite"
        import shutil
        shutil.copy2(old_tflite, backup_tflite)
        print(f"  Backed up old TFLite: {backup_tflite}")

    # Write new TFLite
    new_tflite = MODELS_DIR / "disease_model.tflite"
    with open(new_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = new_tflite.stat().st_size / (1024 * 1024)
    print(f"  New TFLite: {new_tflite}")
    print(f"  Size:       {size_mb:.1f} MB")

    # Sanity check
    interp = tf.lite.Interpreter(model_path=str(new_tflite))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input:  {inp['shape']}")
    print(f"  Output: {out['shape']}")
    print("  TFLite sanity check PASSED")

    # Save retrain report
    report = {
        "retrain_date":   datetime.now().isoformat(),
        "val_accuracy":   float(best_val_acc),
        "tflite_size_mb": round(size_mb, 2),
        "model_path":     str(new_tflite),
    }
    report_path = MODELS_DIR / f"retrain_report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    return new_tflite


# ─────────────────────────────────────────────
# PRINT FINAL SUMMARY
# ─────────────────────────────────────────────
def print_summary(num_images: int, best_val_acc: float, tflite_path: Path):
    print("\n" + "="*55)
    print("  RETRAINING COMPLETE")
    print("="*55)
    print(f"  Feedback images used:  {num_images}")
    print(f"  Best val_accuracy:     {best_val_acc:.4f}")
    print(f"  New model:             {tflite_path}")
    print()
    print("  NEXT STEP:")
    print("  Copy new TFLite to Flutter app assets:")
    print(f"  copy {tflite_path} crop_disease_app\\assets\\models\\disease_model.tflite")
    print()
    print("  Then rebuild the app:")
    print("  flutter build apk --release")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# COUNT ONLY MODE
# ─────────────────────────────────────────────
def count_feedback(db):
    from firebase_admin import firestore as fs

    collection = db.collection(FIRESTORE_COLLECTION)

    try:
        # Use aggregation queries for fast, cheap counting (1 read per 1k docs)
        correct = collection.where(filter=fs.FieldFilter("feedback", "==", "correct")).count().get()[0][0].value
        wrong = collection.where(filter=fs.FieldFilter("feedback", "==", "wrong")).count().get()[0][0].value
        skip = collection.where(filter=fs.FieldFilter("feedback", "==", "skip")).count().get()[0][0].value
    except Exception:
        # Fallback if firestore library is very old and doesn't support count()
        correct = len(list(collection.where(filter=fs.FieldFilter("feedback", "==", "correct")).stream()))
        wrong = len(list(collection.where(filter=fs.FieldFilter("feedback", "==", "wrong")).stream()))
        skip = len(list(collection.where(filter=fs.FieldFilter("feedback", "==", "skip")).stream()))

    total_usable = correct + wrong

    print("\n" + "="*55)
    print("  FEEDBACK COUNT")
    print("="*55)
    print(f"  Correct feedback:    {correct:>6}")
    print(f"  Wrong + corrected:   {wrong:>6}")
    print(f"  Skip (no label):     {skip:>6}")
    print(f"  ─────────────────────────────")
    print(f"  Total usable:        {total_usable:>6}")
    print(f"  Minimum to retrain:  {MIN_IMAGES:>6}")
    print()
    if total_usable >= MIN_IMAGES:
        print(f"  READY TO RETRAIN")
        print(f"  Run: python src/retrain_from_feedback.py")
    else:
        remaining = MIN_IMAGES - total_usable
        print(f"  NOT READY — need {remaining} more confirmed images")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Retrain disease model from user feedback"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Just show feedback counts, do not download or train",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download feedback images but do not retrain",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Min confidence for correct-feedback images (default: 0.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Retrain even if fewer than {MIN_IMAGES} images",
    )
    args = parser.parse_args()

    print("="*55)
    print("  RETRAIN FROM USER FEEDBACK")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  CPU threads: {NUM_THREADS}")
    print("="*55)

    db = init_firestore()

    # Count only mode
    if args.count_only:
        count_feedback(db)
        return

    # Fetch feedback documents
    docs = fetch_feedback_docs(db, min_confidence=args.min_confidence)

    if len(docs) == 0:
        print("\n  No usable feedback found yet.")
        print("  Collect more feedback in the app and try again.")
        return

    if len(docs) < MIN_IMAGES and not args.force:
        print(f"\n  Only {len(docs)} usable feedback images.")
        print(f"  Minimum recommended: {MIN_IMAGES}")
        print(f"  Use --force to retrain anyway (not recommended).")
        return

    # Download images
    class_counts = download_images(docs)
    total_downloaded = sum(class_counts.values())

    if args.download_only:
        print(f"\n  Download complete: {total_downloaded} images")
        print(f"  Saved to: {FEEDBACK_DIR}")
        print("  Run without --download-only to start retraining.")
        return

    # Merge into training data
    merge_feedback_into_training()

    # Run phase 3 fine-tuning
    model, history, ckpt_path, best_val_acc = run_phase3_finetuning()

    # Export new TFLite
    tflite_path = export_tflite(model, best_val_acc)

    # Summary
    print_summary(total_downloaded, best_val_acc, tflite_path)


if __name__ == "__main__":
    main()