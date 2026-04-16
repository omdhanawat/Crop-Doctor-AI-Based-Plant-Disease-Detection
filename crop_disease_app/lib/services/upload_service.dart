// lib/services/upload_service.dart

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/foundation.dart';

class UploadService {
  final String _cloudinaryUrl = 'https://api.cloudinary.com/v1_1/dplciuovv/image/upload';
  final String _uploadPreset = 'crop_disease_uploads';

  Future<String?> uploadDiagnostic({
    required File imageFile,
    required String diagnosis,
    required String confidence,
    required String rawLabel,
    required bool isHealthy,
  }) async {
    try {
      // 1. Connectivity Check (Simplified - try the request anyway if unsure)
      final List<ConnectivityResult> connectivityResult = await Connectivity().checkConnectivity();
      bool isConnected = connectivityResult.isNotEmpty && !connectivityResult.contains(ConnectivityResult.none);

      if (!isConnected) {
        debugPrint('UploadService: No internet detected via connectivity_plus.');
      }

      // 2. Upload image to Cloudinary
      var request = http.MultipartRequest('POST', Uri.parse(_cloudinaryUrl));
      request.fields['upload_preset'] = _uploadPreset;
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      debugPrint('Uploading to Cloudinary: $_cloudinaryUrl using preset: $_uploadPreset');
      var streamedResponse = await request.send().timeout(const Duration(seconds: 30));
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200 && response.statusCode != 201) {
        final error = 'Cloudinary ${response.statusCode}: ${response.body}';
        debugPrint('UPLOAD_SERVICE_ERROR: $error');
        throw Exception(error);
      }

      final responseData = json.decode(response.body);
      final imageUrl = responseData['secure_url'];

      // 3. Save to Firestore
      final docRef = await FirebaseFirestore.instance.collection('predictions').add({
        'imageUrl': imageUrl,
        'diagnosis': diagnosis,
        'confidence': confidence,
        'rawLabel': rawLabel,
        'isHealthy': isHealthy,
        'timestamp': FieldValue.serverTimestamp(),
        'status': 'verified',
      });

      return docRef.id;
    } catch (e) {
      debugPrint('Error uploading diagnostic: $e');
      if (e is SocketException) {
        debugPrint('Network error: please check internet connection.');
      }
      rethrow;
    }
  }

  static Future<String?> savePrediction({
    required File imageFile,
    required String predictedClass,
    required double confidence,
    required bool isHealthy,
  }) async {
    return await UploadService().uploadDiagnostic(
      imageFile: imageFile,
      diagnosis: predictedClass.split('___').last.replaceAll('_', ' '),
      rawLabel: predictedClass,
      confidence: '${(confidence * 100).toStringAsFixed(1)}%',
      isHealthy: isHealthy,
    );
  }

  static Future<bool> submitFeedback({
    required String docId,
    required String feedback,
    String? correctClass,
    String? rating,
  }) async {
    try {
      await FirebaseFirestore.instance.collection('predictions').doc(docId).update({
        'userFeedback': feedback,
        if (correctClass != null) 'correctClass': correctClass,
        if (rating != null) 'userRating': rating,
        'feedbackTimestamp': FieldValue.serverTimestamp(),
      });
      return true;
    } catch (e) {
      debugPrint('Error submitting feedback: $e');
      return false;
    }
  }

  static Future<void> saveFeedback({
    required String docId,
    required String feedback,
    String? correctClass,
  }) async {
    await submitFeedback(
      docId: docId,
      feedback: feedback,
      correctClass: correctClass,
    );
  }
}