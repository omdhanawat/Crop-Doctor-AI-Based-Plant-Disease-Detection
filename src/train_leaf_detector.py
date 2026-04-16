from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf

DATA_DIR = Path("data/leaf_detector_data")
SPLIT_ROOT = Path("data/processed/leaf_detector")
MODELS_DIR = Path("models")
LOG_DIR = Path("logs/leaf_detector")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-4
VALIDATION_SPLIT = 0.15
SEED = 42

NUM_THREADS = os.cpu_count() or 1
try:
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
except RuntimeError:
    pass


def set_seed() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, ValueError):
        pass


def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(directory: Path, training: bool) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=SEED,
    )
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_split_from_single_root() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None, list[str], str]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Leaf detector dataset not found: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    class_names = list(train_ds.class_names)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, None, class_names, "validation"


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None, list[str], str]:
    train_dir = SPLIT_ROOT / "train"
    val_dir = SPLIT_ROOT / "val"
    test_dir = SPLIT_ROOT / "test"

    if train_dir.exists() and val_dir.exists():
        train_ds = make_dataset(train_dir, training=True)
        val_ds = make_dataset(val_dir, training=False)
        test_ds = make_dataset(test_dir, training=False) if test_dir.exists() else None
        class_names = list(tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
        ).class_names)
        metric_dataset_name = "test" if test_ds is not None else "validation"
        return train_ds, val_ds, test_ds, class_names, metric_dataset_name

    return make_split_from_single_root()


def verify_class_names(class_names: list[str]) -> None:
    expected = {"leaf", "non_leaf"}
    actual = set(class_names)
    if actual != expected:
        raise ValueError(f"Expected classes {sorted(expected)}, found {class_names}")


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
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)
    model = tf.keras.Model(inputs, outputs, name="leaf_detector")
    return model, base_model


def get_callbacks(model: tf.keras.Model, phase: int) -> tuple[list[tf.keras.callbacks.Callback], Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"leaf_detector_best_phase{phase}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / f"phase{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            histogram_freq=0,
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print(
                f"\n  Epoch {epoch + 1} | LR: {float(tf.keras.backend.get_value(model.optimizer.learning_rate)):.2e}"
            )
        ),
    ]
    return callbacks, checkpoint_path


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def train(model: tf.keras.Model, base_model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
    print("\n[3/5] Phase 1: training head")
    compile_model(model, LEARNING_RATE_1)
    callbacks_p1, _ = get_callbacks(model, phase=1)
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, callbacks=callbacks_p1, verbose=1)

    print("\n[4/5] Phase 2: fine-tuning top MobileNetV2 layers")
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - 30)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= freeze_until and not isinstance(layer, tf.keras.layers.BatchNormalization)

    compile_model(model, LEARNING_RATE_2)
    callbacks_p2, ckpt_p2 = get_callbacks(model, phase=2)
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, callbacks=callbacks_p2, verbose=1)
    return history1, history2, ckpt_p2


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        tf.saved_model.save(model, str(export_dir))


def export_tflite(saved_model_path: Path) -> Path:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = []
    tflite_model = converter.convert()
    tflite_path = MODELS_DIR / "leaf_detector.tflite"
    tflite_path.write_bytes(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return tflite_path


def evaluate_and_save(model: tf.keras.Model, eval_ds: tf.data.Dataset, dataset_name: str, class_names: list[str]) -> float:
    print(f"\n[5/5] Evaluating on {dataset_name} dataset...")
    results = model.evaluate(eval_ds, return_dict=True, verbose=1)
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    export_dir = MODELS_DIR / "leaf_detector_savedmodel"
    keras_path = MODELS_DIR / "leaf_detector.keras"
    model.save(keras_path)
    export_saved_model(model, export_dir)
    tflite_path = export_tflite(export_dir)

    label_map = {str(index): class_name for index, class_name in enumerate(class_names)}
    label_map_path = MODELS_DIR / "leaf_detector_labels.json"
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print(f"\n  Keras model:   {keras_path}")
    print(f"  SavedModel:    {export_dir}")
    print(f"  TFLite model:  {tflite_path}")
    print(f"  Label map:     {label_map_path}")
    return float(results["accuracy"])


def print_summary(history1, history2, metric_name: str, score: float) -> None:
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE - LEAF DETECTOR")
    print("=" * 55)
    print(f"  Phase 1 best val_accuracy: {max(history1.history['val_accuracy']):.4f}")
    print(f"  Phase 2 best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    print(f"  Final {metric_name} accuracy:    {score:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    set_seed()
    print("=" * 55)
    print("  LEAF DETECTOR TRAINING")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  CPU threads: {NUM_THREADS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 55)

    train_ds, val_ds, test_ds, class_names, metric_dataset_name = load_datasets()
    verify_class_names(class_names)
    model, base_model = build_model()
    history1, history2, best_ckpt = train(model, base_model, train_ds, val_ds)

    print(f"\n  Loading best checkpoint: {best_ckpt}")
    model = tf.keras.models.load_model(str(best_ckpt))
    eval_ds = test_ds if test_ds is not None else val_ds
    score = evaluate_and_save(model, eval_ds, metric_dataset_name, class_names)
    print_summary(history1, history2, metric_dataset_name, score)
