// lib/screens/camera_screen.dart

import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import '../services/classifier.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with SingleTickerProviderStateMixin {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  FlashMode _flashMode = FlashMode.off;
  late AnimationController _scanController;
  late Animation<double> _scanAnimation;
  final CropDiseaseClassifier _classifier = CropDiseaseClassifier();

  @override
  void initState() {
    super.initState();
    _initCamera();
    _scanController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
    _scanAnimation = Tween<double>(begin: 0, end: 1).animate(_scanController);
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    if (_cameras != null && _cameras!.isNotEmpty) {
      _controller = CameraController(
        _cameras![0],
        ResolutionPreset.high,
        enableAudio: false,
      );

      try {
        await _controller!.initialize();
        if (mounted) {
          setState(() => _isCameraInitialized = true);
        }
      } catch (e) {
        debugPrint('Camera initialization error: $e');
      }
    }
  }

  Future<void> _takePhoto() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      final XFile photo = await _controller!.takePicture();
      final File imageFile = File(photo.path);
      
      // We can show a small "analysing" state here if we want,
      // but usually we just navigate to result screen where analysis happens
      // Actually, my Classifier takes a File and returns a result.
      // I'll run it here so I can pass both to the ResultScreen.
      
      if (!mounted) return;
      
      // Show loading overlay
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const Center(child: CircularProgressIndicator(color: Color(0xFF2DCC70))),
      );

      final result = await _classifier.classify(imageFile);
      
      if (!mounted) return;
      Navigator.pop(context); // hide loading

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(imageFile: imageFile, result: result),
        ),
      );
    } catch (e) {
      debugPrint('Error taking photo: $e');
      if (mounted) Navigator.pop(context);
    }
  }

  Future<void> _pickFromGallery() async {
    final ImagePicker picker = ImagePicker();
    try {
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);
      if (image != null && mounted) {
        final File imageFile = File(image.path);
        
        // Show loading overlay
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (context) => const Center(child: CircularProgressIndicator(color: Color(0xFF2DCC70))),
        );

        final result = await _classifier.classify(imageFile);
        
        if (!mounted) return;
        Navigator.pop(context); // hide loading

        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(imageFile: imageFile, result: result),
          ),
        );
      }
    } catch (e) {
      debugPrint('Error picking from gallery: $e');
    }
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      // Check if device has flash
      if (!(_controller!.value.description.lensDirection == CameraLensDirection.back)) {
         throw Exception("Flash only supported on back camera");
      }

      final newMode = _flashMode == FlashMode.off ? FlashMode.torch : FlashMode.off;
      await _controller!.setFlashMode(newMode);
      setState(() => _flashMode = newMode);
      
      debugPrint('Flash mode updated to: $newMode');
    } catch (e) {
      debugPrint('Error toggling flash: $e');
      if (mounted) {
        String message = "Flash not supported";
        if (e.toString().contains("torch")) message = "Torch mode not available on this device";
        if (e.toString().contains("back camera")) message = "Flash requires the back camera";
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(message),
            backgroundColor: const Color(0xFFE57373),
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _scanController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Color(0xFF2DCC70))),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          CameraPreview(_controller!),
          
          // Scanning Overlay
          _buildScanningOverlay(),
          
          // UI Elements
          SafeArea(
            child: Column(
              children: [
                _buildHeader(),
                _buildStatusRows(),
                const Spacer(),
                _buildBottomControls(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScanningOverlay() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 260,
            height: 260,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.transparent),
            ),
            child: Stack(
              children: [
                // Corners
                _buildCorner(top: 0, left: 0, rotation: 0),
                _buildCorner(top: 0, right: 0, rotation: 1),
                _buildCorner(bottom: 0, left: 0, rotation: 3),
                _buildCorner(bottom: 0, right: 0, rotation: 2),
                
                // Scanning line
                AnimatedBuilder(
                  animation: _scanAnimation,
                  builder: (context, child) {
                    return Positioned(
                      top: 260 * _scanAnimation.value,
                      left: 8,
                      right: 8,
                      child: Container(
                        height: 3,
                        decoration: BoxDecoration(
                          color: const Color(0xFF2DCC70),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF2DCC70).withValues(alpha: 0.8),
                              blurRadius: 20,
                              spreadRadius: 2,
                            ),
                          ],
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              "Place leaf inside frame",
              style: GoogleFonts.manrope(
                textStyle: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCorner({double? top, double? bottom, double? left, double? right, required int rotation}) {
    return Positioned(
      top: top, bottom: bottom, left: left, right: right,
      child: RotatedBox(
        quarterTurns: rotation,
        child: Container(
          width: 40, height: 40,
          decoration: const BoxDecoration(
            border: Border(
              top: BorderSide(color: Color(0xFF2DCC70), width: 4),
              left: BorderSide(color: Color(0xFF2DCC70), width: 4),
            ),
            borderRadius: BorderRadius.only(topLeft: Radius.circular(24)),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.arrow_back, color: Colors.white),
            style: IconButton.styleFrom(backgroundColor: Colors.black12),
          ),
          const SizedBox(width: 12),
          Text(
            "Scan Leaf",
            style: GoogleFonts.notoSerif(
              textStyle: const TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusRows() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _buildStatusTag("OPTICAL STATUS", "Active AI Syncing...", true),
          _buildStatusTag("DIAGNOSTICS", "LENS_STABILIZED_V2", false),
        ],
      ),
    );
  }

  Widget _buildStatusTag(String title, String value, bool animate) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: animate ? CrossAxisAlignment.start : CrossAxisAlignment.end,
        children: [
          Text(
            title,
            style: GoogleFonts.manrope(
              textStyle: const TextStyle(color: Color(0xFF2DCC70), fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 1),
            ),
          ),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (animate) ...[
                Container(width: 6, height: 6, decoration: const BoxDecoration(color: Color(0xFF2DCC70), shape: BoxShape.circle)),
                const SizedBox(width: 4),
              ],
              Text(
                value,
                style: GoogleFonts.manrope(textStyle: const TextStyle(color: Colors.white70, fontSize: 11)),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBottomControls() {
    return Container(
      height: 200,
      width: double.infinity,
      decoration: const BoxDecoration(
        color: Color(0xFFFDFBF7),
        borderRadius: BorderRadius.vertical(top: Radius.circular(40)),
      ),
      child: Column(
        children: [
          const SizedBox(height: 24),
          Text(
            "Hold steady for best results",
            style: GoogleFonts.manrope(
              textStyle: const TextStyle(
                color: Color(0xFF4E6E5D),
                fontSize: 13,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const Spacer(),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildSmallButton(Icons.image, "GALLERY", _pickFromGallery),
                _buildCaptureButton(),
                _buildSmallButton(_flashMode == FlashMode.off ? Icons.flash_off : Icons.flash_on, "FLASH", _toggleFlash),
              ],
            ),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildSmallButton(IconData icon, String label, VoidCallback onTap) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 48, height: 48,
            decoration: BoxDecoration(color: const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(16)),
            child: Icon(icon, color: const Color(0xFF5D4037), size: 24),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: GoogleFonts.manrope(
            textStyle: const TextStyle(color: Color(0xFF5D4037), fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 1),
          ),
        ),
      ],
    );
  }

  Widget _buildCaptureButton() {
    return GestureDetector(
      onTap: _takePhoto,
      child: Container(
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFF2DCC70), width: 4),
        ),
        child: Container(
          width: 64, height: 64,
          decoration: const BoxDecoration(color: Color(0xFFE8F5E9), shape: BoxShape.circle),
          child: const Icon(Icons.photo_camera, color: Color(0xFF2DCC70), size: 36),
        ),
      ),
    );
  }
}
