import tensorflow as tf
import numpy as np

SAVED_MODEL_DIR = "models_tf/saved_model"
TFLITE_PATH = "models_tf/model_int8.tflite"


def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1,128,128,3).astype(np.float32)
        yield [data]


def convert():

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("INT8 model saved:", TFLITE_PATH)


if __name__ == "__main__":
    convert()