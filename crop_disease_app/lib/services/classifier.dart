// lib/services/classifier.dart
//
// 2-stage cascaded inference pipeline:
//   Stage 1 → leaf_detector.tflite  (leaf vs non-leaf)
//   Stage 2 → disease_model.tflite  (38-class disease)
//
// Both models run fully offline on device.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_litert/flutter_litert.dart';

// ─────────────────────────────────────────────
// RESULT CLASSES
// ─────────────────────────────────────────────

enum DetectionStatus {
  notALeaf,       // Stage 1 rejected the image
  lowConfidence,  // Stage 2 ran but confidence below threshold
  success,        // Full pipeline succeeded
}

class ClassificationResult {
  final DetectionStatus status;
  final String? rawLabel;      // the original model label string
  final String? diseaseName;   // null if not a leaf or low confidence
  final String? plantName;     // extracted from disease label
  final double? confidence;    // 0.0 – 1.0
  final String? treatment;     // treatment text for this disease
  final bool isHealthy;        // true if disease label contains 'healthy'

  const ClassificationResult({
    required this.status,
    this.rawLabel,
    this.diseaseName,
    this.plantName,
    this.confidence,
    this.treatment,
    this.isHealthy = false,
  });

  // Convenience getters
  String get confidencePercent =>
      confidence != null ? '${(confidence! * 100).toStringAsFixed(1)}%' : '—';

  bool get succeeded => status == DetectionStatus.success;
}

// ─────────────────────────────────────────────
// MAIN CLASSIFIER CLASS
// ─────────────────────────────────────────────

class CropDiseaseClassifier {
  static const int    _imgSize             = 224;
  static const double _leafThreshold       = 0.5;   // above = non-leaf
  static const double _confidenceThreshold = 0.65;  // below = low confidence

  Interpreter? _leafInterpreter;
  Interpreter? _diseaseInterpreter;
  List<String>  _diseaseLabels = [];
  bool          _isLoaded      = false;

  // ── Load both models ───────────────────────────────────────
  Future<void> loadModels() async {
    if (_isLoaded) return;

    try {
      // Load leaf detector
      final leafOptions = InterpreterOptions()..threads = 4;
      _leafInterpreter = await Interpreter.fromAsset(
        'assets/models/leaf_detector.tflite',
        options: leafOptions,
      );

      // Load disease classifier
      final diseaseOptions = InterpreterOptions()..threads = 4;
      _diseaseInterpreter = await Interpreter.fromAsset(
        'assets/models/disease_model.tflite',
        options: diseaseOptions,
      );

      // Load disease label map
      final labelJson = await rootBundle.loadString(
        'assets/models/disease_labels.json',
      );
      final Map<String, dynamic> labelMap = json.decode(labelJson);

      // Sort by index key to ensure correct order
      final sortedKeys = labelMap.keys.toList()
        ..sort((a, b) => int.parse(a).compareTo(int.parse(b)));
      _diseaseLabels = sortedKeys.map((k) => labelMap[k] as String).toList();

      _isLoaded = true;
    } catch (e) {
      throw Exception('Failed to load models: $e');
    }
  }

  // ── Main classify method ───────────────────────────────────
  Future<ClassificationResult> classify(File imageFile) async {
    if (!_isLoaded) await loadModels();

    // Preprocess image once — reuse for both models
    final input = await _preprocessImage(imageFile);

    // ── STAGE 1: Leaf detector ─────────────────────────────
    final leafScore = _runLeafDetector(input);

    // leafScore > 0.5 means non-leaf (sigmoid output, non_leaf = index 1)
    if (leafScore > _leafThreshold) {
      return const ClassificationResult(status: DetectionStatus.notALeaf);
    }

    // ── STAGE 2: Disease classifier ────────────────────────
    final probabilities = _runDiseaseClassifier(input);
    final topIndex      = _argmax(probabilities);
    final topConfidence = probabilities[topIndex];

    if (topConfidence < _confidenceThreshold) {
      return ClassificationResult(
        status:     DetectionStatus.lowConfidence,
        confidence: topConfidence,
      );
    }

    final rawLabel  = _diseaseLabels[topIndex];
    final plant     = _extractPlantName(rawLabel);
    final disease   = _formatDiseaseName(rawLabel);
    final isHealthy = rawLabel.toLowerCase().contains('healthy');
    final treatment = _getTreatment(rawLabel);

    return ClassificationResult(
      status:      DetectionStatus.success,
      rawLabel:    rawLabel,
      diseaseName: disease,
      plantName:   plant,
      confidence:  topConfidence,
      treatment:   treatment,
      isHealthy:   isHealthy,
    );
  }

  // ── Image preprocessing ────────────────────────────────────
  Future<List<List<List<List<double>>>>> _preprocessImage(File file) async {
    final bytes    = await file.readAsBytes();
    img.Image? image = img.decodeImage(bytes);

    if (image == null) throw Exception('Could not decode image');

    // Resize to 224x224
    image = img.copyResize(image, width: _imgSize, height: _imgSize);

    // Build input tensor [1, 224, 224, 3] normalized to 0→1
    final input = List.generate(
      1,
      (_) => List.generate(
        _imgSize,
        (y) => List.generate(
          _imgSize,
          (x) {
            final pixel = image!.getPixel(x, y);
            return [
              (pixel.r / 127.5) - 1.0,
              (pixel.g / 127.5) - 1.0,
              (pixel.b / 127.5) - 1.0,
            ];
          },
        ),
      ),
    );

    return input;
  }

  // ── Stage 1: Run leaf detector ─────────────────────────────
  double _runLeafDetector(List<List<List<List<double>>>> input) {
    // Output shape: [1, 1] — sigmoid score
    final output = List.generate(1, (_) => List.filled(1, 0.0));
    _leafInterpreter!.run(input, output);
    return output[0][0];
  }

  // ── Stage 2: Run disease classifier ───────────────────────
  List<double> _runDiseaseClassifier(List<List<List<List<double>>>> input) {
    // Output shape: [1, 38] — softmax probabilities
    final output = List.generate(1, (_) => List.filled(38, 0.0));
    _diseaseInterpreter!.run(input, output);
    return output[0];
  }

  // ── Argmax ─────────────────────────────────────────────────
  int _argmax(List<double> probs) {
    int maxIdx = 0;
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  // ── Label parsing helpers ──────────────────────────────────
  String _extractPlantName(String rawLabel) {
    // PlantVillage format: "Tomato___Early_blight"
    final parts = rawLabel.split('___');
    return parts[0].replaceAll('_', ' ').replaceAll(',', '');
  }

  String _formatDiseaseName(String rawLabel) {
    final parts = rawLabel.split('___');
    if (parts.length < 2) return rawLabel.replaceAll('_', ' ');
    return parts[1]
        .replaceAll('_', ' ')
        .replaceAll('  ', ' ')
        .trim();
  }

  // ─────────────────────────────────────────────────────────
  // TREATMENT DATABASE
  // Keys match PlantVillage class names exactly.
  // ─────────────────────────────────────────────────────────
  String _getTreatment(String label) {
    const treatments = <String, String>{
      'Apple___Apple_scab':
          'Apply fungicides containing captan or myclobutanil. Remove and destroy infected leaves. Prune trees for better air circulation. Apply preventive sprays in spring.',
      'Apple___Black_rot':
          'Prune and destroy infected branches and mummified fruit. Apply copper-based fungicide. Remove cankers by cutting 15cm below infected tissue. Maintain tree vigor.',
      'Apple___Cedar_apple_rust':
          'Apply fungicides at bud break (myclobutanil or mancozeb). Remove nearby juniper or cedar trees if possible. Apply preventive sprays every 7–10 days during wet spring weather.',
      'Apple___healthy':
          'No disease detected. Continue regular monitoring, proper irrigation, and balanced fertilization. Maintain good air circulation through regular pruning.',
      'Blueberry___healthy':
          'No disease detected. Ensure soil pH 4.5–5.5. Apply mulch to retain moisture. Monitor for pests regularly.',
      'Cherry_(including_sour)___Powdery_mildew':
          'Apply sulfur-based or potassium bicarbonate fungicide. Improve air circulation by pruning. Avoid overhead irrigation. Remove and destroy infected shoots.',
      'Cherry_(including_sour)___healthy':
          'No disease detected. Continue monitoring. Prune for airflow and apply balanced fertilizer annually.',
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
          'Use resistant hybrids. Apply foliar fungicides (strobilurins or triazoles). Practice crop rotation. Improve field drainage and avoid excessive nitrogen.',
      'Corn_(maize)___Common_rust_':
          'Apply fungicides early (triazoles or strobilurins) when disease is first observed. Plant resistant varieties. Scout fields regularly during warm humid weather.',
      'Corn_(maize)___Northern_Leaf_Blight':
          'Apply fungicides at tassel emergence. Use resistant hybrids. Rotate crops — avoid continuous corn. Bury crop residue by tillage.',
      'Corn_(maize)___healthy':
          'No disease detected. Monitor regularly. Maintain proper plant density and balanced nitrogen fertilization.',
      'Grape___Black_rot':
          'Remove and destroy infected berries and mummified fruit. Apply mancozeb or myclobutanil during early season. Prune for air circulation. Apply protectant fungicides before rain.',
      'Grape___Esca_(Black_Measles)':
          'No cure currently available. Remove and destroy infected vines. Protect pruning wounds with fungicide paste. Use clean pruning tools. Avoid water stress.',
      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
          'Apply copper-based fungicides. Remove infected leaves promptly. Ensure good canopy management for air flow. Avoid wetting foliage during irrigation.',
      'Grape___healthy':
          'No disease detected. Continue canopy management, balanced fertilization, and regular scouting.',
      'Orange___Haunglongbing_(Citrus_greening)':
          'No cure — infected trees must be removed and destroyed to prevent spread. Control Asian citrus psyllid (vector) with insecticides. Use certified disease-free planting material.',
      'Peach___Bacterial_spot':
          'Apply copper-based bactericides during dormancy and early season. Use resistant varieties. Avoid overhead irrigation. Remove and destroy heavily infected shoots.',
      'Peach___healthy':
          'No disease detected. Apply preventive copper sprays during dormancy. Ensure proper thinning and balanced fertilization.',
      'Pepper,_bell___Bacterial_spot':
          'Apply copper hydroxide sprays every 5–7 days during wet weather. Use disease-free transplants. Rotate crops. Avoid working in field when plants are wet.',
      'Pepper,_bell___healthy':
          'No disease detected. Maintain proper irrigation, spacing for airflow, and balanced fertilizer application.',
      'Potato___Early_blight':
          'Apply chlorothalonil or mancozeb fungicide every 7–10 days. Remove infected foliage. Practice crop rotation. Avoid overhead irrigation and ensure adequate plant nutrition.',
      'Potato___Late_blight':
          'Apply metalaxyl or cymoxanil fungicide immediately. Remove and destroy infected plants. Avoid overhead irrigation. Use certified disease-free seed tubers next season.',
      'Potato___healthy':
          'No disease detected. Monitor regularly. Hill plants properly and maintain adequate potassium levels to reduce disease susceptibility.',
      'Raspberry___healthy':
          'No disease detected. Prune old canes after harvest. Maintain good air circulation. Monitor for aphids and spider mites.',
      'Soybean___healthy':
          'No disease detected. Scout fields regularly. Maintain proper plant population and balanced fertility.',
      'Squash___Powdery_mildew':
          'Apply potassium bicarbonate, neem oil, or sulfur fungicide at first sign. Space plants widely for airflow. Avoid excess nitrogen fertilization.',
      'Strawberry___Leaf_scorch':
          'Apply captan or thiram fungicide. Remove infected leaves. Ensure proper spacing. Avoid overhead irrigation. Use resistant varieties.',
      'Strawberry___healthy':
          'No disease detected. Renovate beds after harvest. Monitor for pests and maintain good weed control.',
      'Tomato___Bacterial_spot':
          'Apply copper bactericide every 5–7 days. Use disease-free seeds. Remove infected plant material. Avoid handling wet plants. Practice 2-year crop rotation.',
      'Tomato___Early_blight':
          'Apply chlorothalonil or mancozeb every 7–10 days. Remove lower infected leaves. Stake plants for airflow. Avoid overhead watering. Rotate crops annually.',
      'Tomato___Late_blight':
          'Apply metalaxyl or dimethomorph fungicide immediately — this disease spreads rapidly. Remove infected plants. Avoid overhead irrigation. Do not compost infected material.',
      'Tomato___Leaf_Mold':
          'Apply copper fungicide or chlorothalonil. Reduce humidity — improve greenhouse ventilation. Avoid leaf wetness. Remove infected leaves promptly.',
      'Tomato___Septoria_leaf_spot':
          'Apply mancozeb or chlorothalonil every 7–10 days. Remove and destroy infected leaves. Stake plants. Avoid wetting foliage. Practice crop rotation.',
      'Tomato___Spider_mites Two-spotted_spider_mite':
          'Apply insecticidal soap, neem oil, or miticide spray. Increase humidity — mites thrive in dry conditions. Introduce predatory mites. Remove heavily infested leaves.',
      'Tomato___Target_Spot':
          'Apply chlorothalonil or copper fungicide. Improve air circulation. Remove infected leaves. Avoid overhead irrigation. Practice crop rotation.',
      'Tomato___Tomato_mosaic_virus':
          'No cure — remove and destroy infected plants immediately. Control insect vectors. Disinfect tools with 10% bleach solution. Use virus-free seeds and resistant varieties.',
      'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
          'No cure — remove infected plants. Control whitefly vectors with insecticides or reflective mulch. Use resistant varieties. Install insect-proof nets in nurseries.',
      'Tomato___healthy':
          'No disease detected. Continue monitoring. Maintain proper staking, consistent watering, and balanced calcium nutrition to prevent BER.',
    };

    return treatments[label] ??
        'Consult a local agricultural extension officer for treatment advice specific to your region and crop variety.';
  }

  // ── Cleanup ────────────────────────────────────────────────
  void dispose() {
    _leafInterpreter?.close();
    _diseaseInterpreter?.close();
    _isLoaded = false;
  }
}

