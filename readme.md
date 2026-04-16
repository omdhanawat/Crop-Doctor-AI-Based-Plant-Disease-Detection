# Crop Disease Prediction: Edge ML Pipeline

An advanced, edge-optimized Deep Learning architecture designed to classify 38+ crop diseases in real-time. The system is built around a robust, multi-phase TensorFlow training pipeline that prioritizes high accuracy on out-of-distribution (real-world) data and features an automated cloud-based continuous learning loop.

---

## 🧠 Deep Learning Architecture & Training Strategy

The core model is designed to run completely offline on mobile devices via quantization to TensorFlow Lite, requiring a model that is both lightweight and highly accurate against noisy, real-world field conditions.

### Smart Mixed Training (`smart_mixed_train.py`)
To bridge the gap between lab-controlled datasets and real-world noisy backgrounds, the model employs a sophisticated tri-dataset fusion strategy:
1. **PlantVillage Dataset:** Serves as the foundational base (38 classes).
2. **Recomposed Data:** Generates synthetic images by cutting out leaves and pasting them on complex agricultural backgrounds.
3. **Real-World Field Images:** Hard-mined, noisy datasets injected for ultimate robustness.

**Training Optimizer Strategy:**
* **Dynamic Weighting:** Real field images are weighted **3x higher** than synthetic/lab images. Underrepresented ("collapsed") classes automatically receive a **2x penalty boost**.
* **Two-Phase Fine Tuning:** The network initially trains with a frozen frozen feature-extraction backbone to preserve generalized edge detection, before completely unfreezing for low-LR (`1e-5`) aggressive fine-tuning natively preventing catastrophic forgetting.

---

## ☁️ Continuous Learning (Cloud Feedback Loop)

Model degradation is inevitable in agricultural ML due to shifting seasons, lighting conditions, and mutated disease expressions. This pipeline solves that via an active learning architectural loop (`retrain_from_feedback.py`).

**The Automated Workflow:**
1. **Edge Inference:** The Flutter app runs the `.tflite` model completely offline.
2. **Telemetry & Validation:** Users flag predictions as "Correct" or "Wrong" giving the app ground truth data. The images and metadata sync to **Firebase Firestore** and **Cloudinary**.
3. **Ingestion Script:** The retraining pipeline queries Firestore for validated edge cases and automatically downloads the images into their respective ground-truth class folders.
4. **Phase 3 Fine-Tuning:** The model undergoes a precision micro-tuning sequence integrating the exact failure permutations from the app back into its neural weights, resulting in a smarter version generated for the next app update.

---

## 🛠️ Project Execution & Scripts

The `/src` directory houses the entire ML ecosystem.

### Core Training Scripts
* `train_disease_model.py`: The foundational Phase 1 architecture build.
* `smart_mixed_train.py`: Initiates the mixed-weighting dataset routine balancing PlantVillage, Recomposed, and Real datasets.
* `retrain_from_feedback.py`: Fetches and processes edge-case JSON metadata from Firebase and executes Phase 3 fine-tuning.

### Utilities
* `convert_to_tflite.py` / `quantize_tflite.py`: Compresses and quantizes the `.keras` architecture into the `~10MB` optimal format for the Flutter engine.
* `recompose.py`: The programmatic augmentation pipeline merging leaves with synthetic real-world backgrounds.
* `dataset_split.py`: Strict mathematical split ensuring zero data leakage between train/val/test pools.

---

## ⚙️ Requirements & Execution

**Stack:** Python 3.9+, TensorFlow / Keras, OpenCV, Albumentations.

1. Init virtual environment and install ML toolkit:
   ```bash
   pip install -r requirenments.txt
   ```
2. Run automated Firebase feedback ingestion (requires `firebase_credentials.json` mapped to the workspace):
   ```bash
   # Dry-run telemetry check
   python src/retrain_from_feedback.py --count-only
   
   # Full ingest & retrain hook
   python src/retrain_from_feedback.py
   ```
3. To rebuild the foundational model from scratch:
   ```bash
   python src/smart_mixed_train.py
   ```
