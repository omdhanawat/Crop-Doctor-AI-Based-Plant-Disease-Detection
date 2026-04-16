import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/disease_model.tflite")
LABELS_PATH = Path("models/disease_labels.json")
DEFAULT_TEST_IMAGES = [
    ("D:/Coding/Crop_Disease_Predection/data/raw/real_world/Apple___healthy/800113bb65efe69e.jpg", "Apple___healthy"),
    ("D:/Coding/Crop_Disease_Predection/data/raw/real_world/Apple___Apple_scab/8002cb321f8bfcdf.jpg", "Apple___Apple_scab"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TFLite disease model on one or more images")
    parser.add_argument("images", nargs="*", help="Optional image paths. If omitted, built-in sample images are used.")
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to print per image")
    parser.add_argument(
        "--expected",
        nargs="*",
        default=None,
        help="Optional expected labels in the same order as the provided images",
    )
    return parser.parse_args()


def load_labels() -> list[str]:
    with LABELS_PATH.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return [mapping[str(index)] for index in range(len(mapping))]


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def top_predictions(probabilities: np.ndarray, labels: list[str], top_k: int) -> list[tuple[str, float]]:
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [(labels[index], float(probabilities[index])) for index in indices]


def main() -> None:
    args = parse_args()
    labels = load_labels()

    if args.images:
        image_items = [(path, None) for path in args.images]
        if args.expected is not None and len(args.expected) != len(args.images):
            raise ValueError("--expected must have the same number of entries as images")
        if args.expected is not None:
            image_items = list(zip(args.images, args.expected))
    else:
        image_items = [(path, expected) for path, expected in DEFAULT_TEST_IMAGES if Path(path).exists()]
        if not image_items:
            raise FileNotFoundError("No default test images exist. Pass image paths explicitly.")

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for raw_path, expected_label in image_items:
        image_path = Path(raw_path)
        if not image_path.exists():
            print(f"SKIP {image_path}: file not found")
            continue

        batch = load_image(image_path)
        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details["index"])[0]
        predictions = top_predictions(probabilities, labels, args.top_k)
        predicted_label, predicted_prob = predictions[0]

        print(f"\nImage: {image_path}")
        if expected_label:
            status = "OK" if predicted_label == expected_label else "MISMATCH"
            print(f"Expected: {expected_label} | Predicted: {predicted_label} | Top-1: {predicted_prob:.4f} | {status}")
        else:
            print(f"Predicted: {predicted_label} | Top-1: {predicted_prob:.4f}")

        print("Top predictions:")
        for label, score in predictions:
            print(f"  {label:<45} {score:.4f}")


if __name__ == "__main__":
    main()
