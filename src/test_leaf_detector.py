import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_PATH = Path("models/leaf_detector.tflite")
LABELS_PATH = Path("models/leaf_detector_labels.json")
TEST_IMAGES = [
    ("a real leaf photo", "D:/Coding/Crop_Disease_Predection/data/raw/real_world/Apple___healthy/973896b94ba864c7.jpg"),
    ("a non-leaf object", "D:/Coding/Crop_Disease_Predection/data/Gjr5IcYXEAAl01M.jpg"),
]

with LABELS_PATH.open("r", encoding="utf-8") as handle:
    labels = json.load(handle)

leaf_label = labels.get("0", "leaf")
non_leaf_label = labels.get("1", "non_leaf")

interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    # array = tf.keras.applications.mobilenet_v2.preprocess_input(array).numpy()
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


for description, path in TEST_IMAGES:
    batch = load_image(path)
    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()
    non_leaf_prob = float(interpreter.get_tensor(output_details["index"])[0][0])
    leaf_prob = 1.0 - non_leaf_prob
    predicted_label = non_leaf_label if non_leaf_prob >= 0.5 else leaf_label

    print(
        f"{description}: {predicted_label} "
        f"(leaf_prob={leaf_prob:.4f}, non_leaf_prob={non_leaf_prob:.4f})"
    )
