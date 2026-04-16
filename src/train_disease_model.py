from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    import albumentations as A
except ImportError:
    A = None

MODELS_DIR = Path("models")
LOG_DIR = Path("logs/disease_model")
MERGED_DIR = Path("data/processed/disease_merged")

ORIGINALS_TRAIN_DIR = Path("data/processed/disease_model/original/train")
RECOMPOSED_TRAIN_DIR = Path("data/processed/disease_model/recomposed/train")
VAL_DIR = Path("data/processed/disease_model/val")
TEST_DIR = Path("data/processed/disease_model/test")

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 38
SEED = 42

EPOCHS_P1 = 10
EPOCHS_P2 = 20
EPOCHS_P3 = 10
LR_P1 = 1e-3
LR_P2 = 1e-4
LR_P3 = 5e-5

NUM_THREADS = os.cpu_count() or 1
try:
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
except RuntimeError:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: int
    learning_rate: float
    train_dirs: tuple[Path, ...]
    unfreeze_top_layers: int = 0
    unfreeze_all: bool = False
    use_augmentation: bool = False
    early_stopping_patience: int | None = None


def set_seed() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, ValueError):
        pass


def build_augmenter() -> A.Compose:
    if A is None:
        raise ImportError(
            "Albumentations is required for phase 3. Install `albumentations opencv-python-headless`."
        )

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, border_mode=0, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=15, val_shift_limit=12, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.25),
            A.RandomShadow(shadow_roi=(0.0, 0.35, 1.0, 1.0), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.35),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.18, alpha_coef=0.08, p=0.15),
            A.CoarseDropout(
                max_holes=6,
                max_height=int(IMG_SIZE * 0.18),
                max_width=int(IMG_SIZE * 0.18),
                min_holes=1,
                min_height=int(IMG_SIZE * 0.06),
                min_width=int(IMG_SIZE * 0.06),
                fill_value=0,
                p=0.30,
            ),
        ]
    )


def discover_class_names(reference_dir: Path) -> list[str]:
    if not reference_dir.exists():
        raise FileNotFoundError(f"Required directory not found: {reference_dir}")
    class_names = sorted(entry.name for entry in reference_dir.iterdir() if entry.is_dir())
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes in {reference_dir}, found {len(class_names)}")
    return class_names


def validate_class_structure(root_dir: Path, class_names: list[str]) -> None:
    if not root_dir.exists():
        raise FileNotFoundError(f"Required directory not found: {root_dir}")
    actual = sorted(entry.name for entry in root_dir.iterdir() if entry.is_dir())
    missing = sorted(set(class_names) - set(actual))
    extra = sorted(set(actual) - set(class_names))
    if missing or extra:
        raise ValueError(f"Class mismatch in {root_dir}. Missing: {missing or 'none'}, extra: {extra or 'none'}")


def collect_samples(roots: tuple[Path, ...], class_names: list[str]) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for label, class_name in enumerate(class_names):
        for root in roots:
            class_dir = root / class_name
            files = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                raise ValueError(f"No images found in {class_dir}")
            paths.extend(str(path) for path in files)
            labels.extend([label] * len(files))
    return paths, labels


def compute_class_weights(labels: list[int], num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("At least one class has zero training samples")
    total = counts.sum()
    weights = total / (num_classes * counts)
    return {index: float(weight) for index, weight in enumerate(weights)}


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, label


def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def apply_albumentations(image: tf.Tensor, label: tf.Tensor, augmenter: A.Compose) -> tuple[tf.Tensor, tf.Tensor]:
    def _augment(np_image: np.ndarray) -> np.ndarray:
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return augmenter(image=np_image)["image"].astype(np.float32)

    image = tf.numpy_function(_augment, [image], tf.float32)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, label


def build_dataset(paths: list[str], labels: list[int], training: bool, augmenter: A.Compose | None = None) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augmenter is not None:
        dataset = dataset.map(lambda image, label: apply_albumentations(image, label, augmenter), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def reset_merged_dir() -> None:
    resolved = MERGED_DIR.resolve()
    workspace = Path.cwd().resolve()
    if workspace not in resolved.parents:
        raise RuntimeError(f"Refusing to clear directory outside workspace: {resolved}")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)


def write_phase_manifest(phase_name: str, paths: list[str], labels: list[int], class_names: list[str]) -> None:
    phase_dir = MERGED_DIR / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples": len(paths),
        "classes": class_names,
        "class_counts": {
            class_name: int(sum(1 for label in labels if label == index))
            for index, class_name in enumerate(class_names)
        },
    }
    (phase_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="disease_classifier")
    return model, base_model


def configure_trainability(base_model: tf.keras.Model, phase: PhaseSpec) -> None:
    if phase.unfreeze_all:
        base_model.trainable = True
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        return

    if phase.unfreeze_top_layers > 0:
        base_model.trainable = True
        split_index = max(0, len(base_model.layers) - phase.unfreeze_top_layers)
        for index, layer in enumerate(base_model.layers):
            layer.trainable = index >= split_index and not isinstance(layer, tf.keras.layers.BatchNormalization)
        return

    base_model.trainable = False


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )


def get_callbacks(phase: PhaseSpec) -> tuple[list[tf.keras.callbacks.Callback], Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"disease_model_best_{phase.name}.keras"
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / f"{phase.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            histogram_freq=0,
        ),
    ]
    if phase.early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=phase.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks, ckpt_path


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        tf.saved_model.save(model, str(export_dir))


def evaluate_and_export(model: tf.keras.Model, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset, class_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    print("\n[5/6] Final evaluation")
    val_metrics = {key: float(value) for key, value in model.evaluate(val_ds, return_dict=True, verbose=1).items()}
    test_metrics = {key: float(value) for key, value in model.evaluate(test_ds, return_dict=True, verbose=1).items()}

    print("\n  Validation metrics:")
    for key, value in val_metrics.items():
        print(f"    {key}: {value:.4f}")

    print("\n  Test metrics:")
    for key, value in test_metrics.items():
        print(f"    {key}: {value:.4f}")

    print("\n[6/6] Saving and exporting")
    keras_path = MODELS_DIR / "disease_model.keras"
    saved_model_path = MODELS_DIR / "disease_model_savedmodel"
    label_map_path = MODELS_DIR / "disease_labels.json"
    summary_path = MODELS_DIR / "disease_training_summary.json"

    model.save(keras_path)
    export_saved_model(model, saved_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = []
    tflite_model = converter.convert()
    tflite_path = MODELS_DIR / "disease_model.tflite"
    tflite_path.write_bytes(tflite_model)

    label_map_path.write_text(json.dumps({str(index): name for index, name in enumerate(class_names)}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2), encoding="utf-8")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    print(f"  Keras model:   {keras_path}")
    print(f"  SavedModel:    {saved_model_path}")
    print(f"  TFLite model:  {tflite_path}")
    print(f"  Label map:     {label_map_path}")
    print(f"  Summary:       {summary_path}")
    return val_metrics, test_metrics


def print_final_summary(histories: list[tf.keras.callbacks.History], val_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE - DISEASE CLASSIFIER")
    print("=" * 55)
    for index, history in enumerate(histories, start=1):
        print(f"  Phase {index} best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Final validation accuracy:      {val_metrics['accuracy']:.4f}")
    print(f"  Final test accuracy:            {test_metrics['accuracy']:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    set_seed()
    print("=" * 55)
    print("  DISEASE CLASSIFIER TRAINING")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  CPU threads: {NUM_THREADS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 55)

    class_names = discover_class_names(VAL_DIR)
    for directory in (ORIGINALS_TRAIN_DIR, RECOMPOSED_TRAIN_DIR, VAL_DIR, TEST_DIR):
        validate_class_structure(directory, class_names)

    reset_merged_dir()
    augmenter = build_augmenter()

    val_paths, val_labels = collect_samples((VAL_DIR,), class_names)
    test_paths, test_labels = collect_samples((TEST_DIR,), class_names)
    val_ds = build_dataset(val_paths, val_labels, training=False)
    test_ds = build_dataset(test_paths, test_labels, training=False)

    phases = [
        PhaseSpec(name="phase1_originals", epochs=EPOCHS_P1, learning_rate=LR_P1, train_dirs=(ORIGINALS_TRAIN_DIR,)),
        PhaseSpec(name="phase2_originals_plus_recomposed", epochs=EPOCHS_P2, learning_rate=LR_P2, train_dirs=(ORIGINALS_TRAIN_DIR, RECOMPOSED_TRAIN_DIR), unfreeze_top_layers=50),
        PhaseSpec(name="phase3_augmented_full_finetune", epochs=EPOCHS_P3, learning_rate=LR_P3, train_dirs=(ORIGINALS_TRAIN_DIR, RECOMPOSED_TRAIN_DIR), unfreeze_all=True, use_augmentation=True, early_stopping_patience=5),
    ]

    model, base_model = build_model()
    histories: list[tf.keras.callbacks.History] = []

    for phase in phases:
        print("\n" + "=" * 55)
        print(f"  {phase.name}")
        print("=" * 55)
        train_paths, train_labels = collect_samples(phase.train_dirs, class_names)
        class_weights = compute_class_weights(train_labels, len(class_names))
        write_phase_manifest(phase.name, train_paths, train_labels, class_names)
        train_ds = build_dataset(
            train_paths,
            train_labels,
            training=True,
            augmenter=augmenter if phase.use_augmentation else None,
        )
        configure_trainability(base_model, phase)
        compile_model(model, phase.learning_rate)
        callbacks, checkpoint_path = get_callbacks(phase)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        histories.append(history)
        model = tf.keras.models.load_model(str(checkpoint_path))
        base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))

    val_metrics, test_metrics = evaluate_and_export(model, val_ds, test_ds, class_names)
    print_final_summary(histories, val_metrics, test_metrics)

