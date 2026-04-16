// lib/screens/result_screen.dart

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/classifier.dart';
import '../services/upload_service.dart';

const _allClasses = [
  'Apple___Apple_scab',
  'Apple___Black_rot',
  'Apple___Cedar_apple_rust',
  'Apple___healthy',
  'Blueberry___healthy',
  'Cherry_(including_sour)___healthy',
  'Cherry_(including_sour)___Powdery_mildew',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
  'Corn_(maize)___Common_rust_',
  'Corn_(maize)___Northern_Leaf_Blight',
  'Corn_(maize)___healthy',
  'Grape___Black_rot',
  'Grape___Esca_(Black_Measles)',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
  'Grape___healthy',
  'Orange___Haunglongbing_(Citrus_greening)',
  'Peach___Bacterial_spot',
  'Peach___healthy',
  'Pepper,_bell___Bacterial_spot',
  'Pepper,_bell___healthy',
  'Potato___Early_blight',
  'Potato___Late_blight',
  'Potato___healthy',
  'Raspberry___healthy',
  'Soybean___healthy',
  'Squash___Powdery_mildew',
  'Strawberry___Leaf_scorch',
  'Strawberry___healthy',
  'Tomato___Bacterial_spot',
  'Tomato___Early_blight',
  'Tomato___Late_blight',
  'Tomato___Leaf_Mold',
  'Tomato___Septoria_leaf_spot',
  'Tomato___Spider_mites Two-spotted_spider_mite',
  'Tomato___Target_Spot',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
  'Tomato___Tomato_mosaic_virus',
  'Tomato___healthy',
];

class ResultScreen extends StatefulWidget {
  final File imageFile;
  final ClassificationResult result;

  const ResultScreen({
    super.key,
    required this.imageFile,
    required this.result,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  String? _docId;
  bool _isUploading = false;
  bool _uploadDone = false;
  bool _uploadFailed = false;
  String? _uploadError;
  String _feedbackGiven = ''; // "correct" | "wrong" | "skip"
  bool _isSubmitting = false;

  // Daylight theme colors
  static const daylightBg = Color(0xFFF9F8F3);
  static const daylightGreenText = Color(0xFF2D4A42);
  static const daylightBrownBorder = Color(0xFFA68D7A);
  static const daylightBrownSub = Color(0xFF8D786A);
  static const warningRed = Color(0xFFDC2626);

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    if (widget.result.succeeded) {
      _autoUpload();
    }
  }

  Future<void> _autoUpload() async {
    try {
      final docId = await UploadService.savePrediction(
        imageFile: widget.imageFile,
        predictedClass: widget.result.rawLabel ?? 'unknown',
        confidence: widget.result.confidence ?? 0.0,
        isHealthy: widget.result.isHealthy,
      );
      if (mounted) {
        setState(() {
          _docId = docId;
          _isUploading = false;
          _uploadDone = true;
          _uploadFailed = false;
          _uploadError = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isUploading = false;
          _uploadFailed = true;
          _uploadError = e.toString().replaceFirst('Exception: ', '');
        });
      }
    }
  }

  Future<void> _sendFeedback(String feedback, {String? correctClass}) async {
    if (_docId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Still syncing your report to the cloud...'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    
    setState(() => _isSubmitting = true);
    
    try {
      await UploadService.saveFeedback(
        docId: _docId!,
        feedback: feedback,
        correctClass: correctClass,
      );

      if (mounted) {
        setState(() {
          _isSubmitting = false;
          _feedbackGiven = feedback;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isSubmitting = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Feedback failed: ${e.toString().replaceFirst('Exception: ', '')}'),
            backgroundColor: warningRed,
          ),
        );
      }
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: daylightBg,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildTabBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildDiagnosisTab(),
                  _buildTreatmentTab(),
                  _buildFeedbackTab(),
                ],
              ),
            ),
            _buildBottomAction(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: daylightBg.withValues(alpha: 0.9),
        border: Border(bottom: BorderSide(color: daylightBrownBorder.withValues(alpha: 0.2))),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.arrow_back, color: daylightGreenText),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Diagnostic Result",
                    style: GoogleFonts.manrope(
                      textStyle: const TextStyle(
                        color: daylightGreenText,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  Text(
                    "LABORATORY REPORT ID: ${_docId?.substring(0, 8).toUpperCase() ?? 'PENDING'}",
                    style: GoogleFonts.manrope(
                      textStyle: TextStyle(
                        color: daylightBrownSub.withValues(alpha: 0.6),
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          Stack(
            alignment: Alignment.center,
            children: [
              if (_isUploading)
                const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2, color: daylightGreenText)),
              Icon(
                _uploadDone ? Icons.cloud_done : Icons.cloud_upload,
                color: _uploadDone ? daylightGreenText : daylightBrownSub.withValues(alpha: 0.4),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTabBar() {
    return Container(
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: daylightBrownBorder.withValues(alpha: 0.1))),
      ),
      child: TabBar(
        controller: _tabController,
        labelColor: daylightGreenText,
        unselectedLabelColor: daylightBrownSub.withValues(alpha: 0.6),
        indicatorColor: daylightGreenText,
        indicatorWeight: 2,
        labelStyle: GoogleFonts.manrope(fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1),
        tabs: const [
          Tab(text: "DIAGNOSIS"),
          Tab(text: "TREATMENT"),
          Tab(text: "FEEDBACK"),
        ],
      ),
    );
  }

  Widget _buildDiagnosisTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Image Container
          Container(
            height: 280,
            width: double.infinity,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.3)),
              boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10)],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(widget.imageFile, fit: BoxFit.cover),
                  Positioned(
                    bottom: 16,
                    right: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: daylightBg.withValues(alpha: 0.8),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.2)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.eco, size: 14, color: daylightGreenText),
                          const SizedBox(width: 4),
                          Text(
                            "Natural Sample",
                            style: GoogleFonts.manrope(
                              textStyle: const TextStyle(color: daylightGreenText, fontSize: 10, fontWeight: FontWeight.bold),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          // Diagnosis Info Card
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.4)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(width: 8, height: 8, decoration: const BoxDecoration(color: warningRed, shape: BoxShape.circle)),
                    const SizedBox(width: 8),
                    Text(
                      widget.result.isHealthy ? "PLANT HEALTHY" : "DISEASE DETECTED",
                      style: GoogleFonts.manrope(
                        textStyle: const TextStyle(color: warningRed, fontSize: 10, fontWeight: FontWeight.w900, letterSpacing: 1.5),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.result.diseaseName ?? "Unknown Result",
                            style: GoogleFonts.notoSerif(
                              textStyle: const TextStyle(color: daylightGreenText, fontSize: 28, fontWeight: FontWeight.bold, height: 1.1),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            widget.result.plantName ?? "Botanical analysis",
                            style: GoogleFonts.manrope(
                              textStyle: TextStyle(color: daylightBrownSub, fontSize: 14, fontStyle: FontStyle.italic, fontWeight: FontWeight.w600),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          "CONFIDENCE",
                          style: GoogleFonts.manrope(
                            textStyle: TextStyle(color: daylightBrownSub, fontSize: 9, fontWeight: FontWeight.w900, letterSpacing: 1),
                          ),
                        ),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.baseline,
                          textBaseline: TextBaseline.alphabetic,
                          children: [
                            Text(
                              ((widget.result.confidence ?? 0.0) * 100).toStringAsFixed(1),
                              style: GoogleFonts.notoSerif(
                                textStyle: const TextStyle(color: daylightGreenText, fontSize: 24, fontWeight: FontWeight.bold),
                              ),
                            ),
                            Text(
                              "%",
                              style: GoogleFonts.notoSerif(
                                textStyle: const TextStyle(color: daylightGreenText, fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTreatmentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
           _buildSectionHeader("DETAILED TREATMENT"),
           const SizedBox(height: 16),
           Container(
             decoration: BoxDecoration(
               color: Colors.white.withValues(alpha: 0.8),
               borderRadius: BorderRadius.circular(24),
               border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.4)),
               boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10)],
             ),
             child: Column(
               children: [
                 Padding(
                   padding: const EdgeInsets.all(24),
                   child: Column(
                     crossAxisAlignment: CrossAxisAlignment.start,
                     children: [
                       Row(
                         children: [
                           const Icon(Icons.spa, color: daylightGreenText, size: 24),
                           const SizedBox(width: 8),
                           Text(
                             "Recommended Actions",
                             style: GoogleFonts.manrope(
                               textStyle: const TextStyle(color: daylightGreenText, fontSize: 18, fontWeight: FontWeight.bold),
                             ),
                           ),
                         ],
                       ),
                       const SizedBox(height: 16),
                       Text(
                         widget.result.treatment ?? "No specific treatment data found for this condition.",
                         style: GoogleFonts.manrope(
                           textStyle: TextStyle(color: daylightGreenText.withValues(alpha: 0.9), fontSize: 15, fontWeight: FontWeight.w500, height: 1.6),
                         ),
                       ),
                     ],
                   ),
                 ),
                 Container(
                   padding: const EdgeInsets.all(16),
                   decoration: BoxDecoration(
                     color: const Color(0xFFE3E0D1).withValues(alpha: 0.3),
                     borderRadius: const BorderRadius.vertical(bottom: Radius.circular(24)),
                   ),
                   child: Row(
                     crossAxisAlignment: CrossAxisAlignment.start,
                     children: [
                       const Icon(Icons.warning_amber_rounded, color: warningRed, size: 18),
                       const SizedBox(width: 8),
                       Expanded(
                         child: Text(
                           "CONSULT A LOCAL AGRONOMIST BEFORE APPLYING CHEMICAL TREATMENTS.",
                           style: GoogleFonts.manrope(
                             textStyle: TextStyle(color: daylightBrownSub, fontSize: 9, fontWeight: FontWeight.bold, fontStyle: FontStyle.italic, height: 1.4),
                           ),
                         ),
                       ),
                     ],
                   ),
                 ),
               ],
             ),
           ),
        ],
      ),
    );
  }

  Widget _buildFeedbackTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          _buildSectionHeader("COMMUNITY VALIDATION"),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.4),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.4), style: BorderStyle.solid),
            ),
            child: Column(
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.rate_review_outlined, color: daylightBrownSub, size: 24),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "VALIDATION REQUIRED",
                            style: GoogleFonts.manrope(
                              textStyle: const TextStyle(color: daylightGreenText, fontSize: 13, fontWeight: FontWeight.bold, letterSpacing: 1),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            "Was this diagnosis correct? Your input refines the analytical model.",
                            style: GoogleFonts.manrope(
                              textStyle: TextStyle(color: daylightBrownSub, fontSize: 12, fontWeight: FontWeight.w500),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 24),
                if (_uploadFailed)
                  _buildUploadFailedMessage()
                else if (_docId == null)
                  _buildSyncingMessage()
                else if (_isSubmitting)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 20),
                    child: Center(child: CircularProgressIndicator(color: daylightGreenText)),
                  )
                else if (_feedbackGiven.isEmpty)
                  Row(
                    children: [
                      _buildFeedbackButton("CORRECT", Icons.thumb_up, () => _sendFeedback('correct')),
                      const SizedBox(width: 12),
                      _buildFeedbackButton("WRONG", Icons.thumb_down, _showWrongClassPicker),
                      const SizedBox(width: 12),
                      _buildFeedbackButton("NOT SURE", Icons.help, () => _sendFeedback('skip')),
                    ],
                  )
                else
                  _buildThanksMessage(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Row(
      children: [
        const Expanded(child: Divider(color: daylightBrownBorder, thickness: 0.2)),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            title,
            style: GoogleFonts.manrope(
              textStyle: TextStyle(color: daylightBrownSub, fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 2),
            ),
          ),
        ),
        const Expanded(child: Divider(color: daylightBrownBorder, thickness: 0.2)),
      ],
    );
  }

  Widget _buildFeedbackButton(String label, IconData icon, VoidCallback onTap) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.3)),
          ),
          child: Column(
            children: [
              Icon(icon, color: daylightGreenText, size: 20),
              const SizedBox(height: 4),
              Text(
                label,
                style: GoogleFonts.manrope(
                  textStyle: const TextStyle(color: daylightGreenText, fontSize: 9, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildUploadFailedMessage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.cloud_off, color: warningRed, size: 16),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                "Report sync failed.",
                style: GoogleFonts.manrope(
                  textStyle: const TextStyle(color: warningRed, fontSize: 13, fontWeight: FontWeight.bold),
                ),
              ),
            ),
            TextButton(
              onPressed: _autoUpload,
              child: Text("RETRY", style: GoogleFonts.manrope(fontSize: 10, fontWeight: FontWeight.bold, color: daylightGreenText)),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Theme(
          data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            tilePadding: EdgeInsets.zero,
            title: Text(
              "SHOW TECHNICAL DETAILS",
              style: GoogleFonts.manrope(
                textStyle: TextStyle(color: daylightBrownSub.withValues(alpha: 0.6), fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 1),
              ),
            ),
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.1)),
                ),
                child: Text(
                  _uploadError ?? "Unknown error occurred during cloud synchronization.",
                  style: GoogleFonts.firaCode(
                    textStyle: TextStyle(color: daylightGreenText.withValues(alpha: 0.8), fontSize: 10),
                  ),
                ),
              ),
              const SizedBox(height: 12),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSyncingMessage() {
    return Row(
      children: [
        SizedBox(
          width: 16,
          height: 16,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: daylightBrownSub.withValues(alpha: 0.5),
          ),
        ),
        const SizedBox(width: 12),
        Text(
          "Syncing report to cloud...",
          style: GoogleFonts.manrope(
            textStyle: TextStyle(
              color: daylightBrownSub.withValues(alpha: 0.6),
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildThanksMessage() {
    return Row(
      children: [
        const Icon(Icons.check_circle_outline, color: daylightGreenText),
        const SizedBox(width: 12),
        Text(
          "Thank you for your feedback!",
          style: GoogleFonts.manrope(
            textStyle: const TextStyle(color: daylightGreenText, fontSize: 14, fontWeight: FontWeight.bold),
          ),
        ),
      ],
    );
  }

  void _showWrongClassPicker() {
    String? selected = _allClasses.first;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setModalState) => Padding(
          padding: EdgeInsets.only(left: 20, right: 20, top: 20, bottom: MediaQuery.of(ctx).viewInsets.bottom + 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'What is the correct disease?',
                style: GoogleFonts.manrope(textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: daylightGreenText)),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                decoration: BoxDecoration(border: Border.all(color: daylightBrownBorder.withValues(alpha: 0.3)), borderRadius: BorderRadius.circular(10)),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: selected,
                    isExpanded: true,
                    items: _allClasses.map((cls) {
                      final display = cls.replaceAll('___', ' — ').replaceAll('_', ' ');
                      return DropdownMenuItem(value: cls, child: Text(display, style: const TextStyle(fontSize: 13)));
                    }).toList(),
                    onChanged: (val) => setModalState(() => selected = val),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: daylightGreenText,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  onPressed: () {
                    Navigator.pop(ctx);
                    _sendFeedback('wrong', correctClass: selected);
                  },
                  child: Text('Submit Correction', style: GoogleFonts.manrope(textStyle: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold))),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBottomAction() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: daylightBg.withValues(alpha: 0.95),
        border: Border(top: BorderSide(color: daylightBrownBorder.withValues(alpha: 0.2))),
      ),
      child: GestureDetector(
        onTap: () => Navigator.pop(context),
        child: Container(
          height: 56,
          decoration: BoxDecoration(
            color: daylightGreenText,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [BoxShadow(color: daylightGreenText.withValues(alpha: 0.3), blurRadius: 10, offset: const Offset(0, 4))],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.photo_camera, color: Colors.white, size: 20),
              const SizedBox(width: 12),
              Text(
                "ANALYSE ANOTHER LEAF",
                style: GoogleFonts.manrope(
                  textStyle: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.5),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}