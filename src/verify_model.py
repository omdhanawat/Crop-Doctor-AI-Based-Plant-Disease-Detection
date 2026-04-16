import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Try to import tqdm for progress bar, fallback to simple counter
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def get_base_path():
    current = Path.cwd()
    if (current / "models").exists(): return current
    if current.name == "src" and (current.parent / "models").exists(): return current.parent
    return current

def load_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    return [labels_dict[str(i)] for i in range(len(labels_dict))]

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0 
    return np.expand_dims(img_array, axis=0)

class ModelWrapper:
    def __init__(self, model_path):
        self.is_tflite = model_path.suffix == '.tflite'
        if self.is_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(str(model_path))

    def predict(self, img_batches):
        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], img_batches)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            return self.model.predict(img_batches, verbose=0)

def verify_model(model_path_str, data_dir_str, labels_path_str, limit_per_class=None):
    base = get_base_path()
    model_path = base / model_path_str.strip()
    labels_path = base / labels_path_str.strip()
    data_path = Path(data_dir_str.strip())

    if not all([model_path.exists(), labels_path.exists(), data_path.exists()]):
        print(f"[ERROR] One or more paths are invalid.\n  Model: {model_path}\n  Labels: {labels_path}\n  Data: {data_path}")
        return

    print(f"\nLoading Model: {model_path.name}")
    mw = ModelWrapper(model_path)
    class_names = load_labels(labels_path)
    label_to_index = {name: i for i, name in enumerate(class_names)}
    
    categories = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Total categories found: {len(categories)}")
    if limit_per_class:
        print(f"Quick Check Mode: testing max {limit_per_class} images per folder.")

    correct, total = 0, 0
    results = []

    for cat_dir in categories:
        cls_name = cat_dir.name
        if cls_name not in label_to_index: continue
            
        expected_idx = label_to_index[cls_name]
        images = [f for f in cat_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if limit_per_class: images = images[:limit_per_class]
        if not images: continue
            
        cat_correct, cat_total = 0, 0
        desc = f"  Processing {cls_name[:30]}..."
        
        # Wrapped loop for progress feedback
        pbar = tqdm(images, desc=desc, leave=False) if tqdm else images
        for img_path in pbar:
            try:
                processed_img = preprocess_image(img_path)
                preds = mw.predict(processed_img)
                if np.argmax(preds[0]) == expected_idx:
                    cat_correct += 1
                cat_total += 1
            except Exception: continue

        if cat_total > 0:
            acc = (cat_correct/cat_total)*100
            print(f"  [DONE] {cls_name:<45} | Acc: {acc:>6.2f}%")
            correct += cat_correct
            total += cat_total

    if total > 0:
        print(f"\n{'='*65}\n  FINAL OVERALL ACCURACY: {(correct/total)*100:.2f}% ({correct}/{total})\n{'='*65}")
    else:
        print("\nNo images were processed. Check your paths.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/disease_model.keras")
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", default="models/disease_labels.json")
    parser.add_argument("--limit", type=int, help="Optional: limit images per class for faster check")
    args = parser.parse_args()
    verify_model(args.model, args.data, args.labels, args.limit)
