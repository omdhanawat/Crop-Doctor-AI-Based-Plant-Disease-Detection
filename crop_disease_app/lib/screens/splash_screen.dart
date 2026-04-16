import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/classifier.dart';
import 'home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;
  late Animation<double> _particleAnimation;
  final CropDiseaseClassifier _classifier = CropDiseaseClassifier();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.5, curve: Curves.easeIn),
      ),
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.8, curve: Curves.easeOutBack),
      ),
    );

    _particleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: Curves.linear,
      ),
    );

    _controller.repeat(reverse: true); // For continuous particle movement before navigation
    
    _initAndNavigate();
  }

  Future<void> _initAndNavigate() async {
    try {
      // Parallel loading and minimum branding time
      await Future.wait([
        _classifier.loadModels(),
        Future.delayed(const Duration(milliseconds: 3500)),
      ]);
      
      if (!mounted) return;
      
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => const HomeScreen(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(opacity: animation, child: child);
          },
          transitionDuration: const Duration(milliseconds: 1000),
        ),
      );
    } catch (e) {
      debugPrint('Initialization error: $e');
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF03140C),
      body: Stack(
        children: [
          // 1. Mesh Gradient Background
          Positioned.fill(
            child: Container(
              decoration: const BoxDecoration(
                gradient: RadialGradient(
                  center: Alignment(0, -0.3),
                  radius: 1.2,
                  colors: [
                    Color(0xFF084B30),
                    Color(0xFF03140C),
                  ],
                ),
              ),
            ),
          ),

          // 2. Animated Particles
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _particleAnimation,
              builder: (context, child) {
                return CustomPaint(
                  painter: ParticlePainter(progress: _controller.value),
                );
              },
            ),
          ),

          // 3. Central Brand Content
          Center(
            child: FadeTransition(
              opacity: _fadeAnimation,
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Glassmorphic Icon Wrapper
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.05),
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.white.withOpacity(0.1),
                          width: 1,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF10B981).withOpacity(0.2),
                            blurRadius: 40,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.eco_rounded,
                        size: 80,
                        color: Color(0xFF10B981),
                      ),
                    ),
                    const SizedBox(height: 48),
                    
                    // Main Title with shadow for depth
                    Text(
                      'CROP DOCTOR',
                      style: GoogleFonts.outfit(
                        fontSize: 44,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                        letterSpacing: 10,
                        shadows: [
                          Shadow(
                            color: Colors.black.withOpacity(0.3),
                            offset: const Offset(0, 4),
                            blurRadius: 10,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    
                    // Subtitle
                    Text(
                      'A I   D I A G N O S T I C S',
                      style: GoogleFonts.outfit(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                        color: const Color(0xFF10B981).withOpacity(0.8),
                        letterSpacing: 6,
                      ),
                    ),
                    
                    const SizedBox(height: 80),
                    
                    // Modern Minimal Loader
                    SizedBox(
                      width: 200,
                      child: Column(
                        children: [
                          const ClipRRect(
                            borderRadius: BorderRadius.all(Radius.circular(10)),
                            child: LinearProgressIndicator(
                              minHeight: 3,
                              backgroundColor: Color(0xFF1A1A1A),
                              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF10B981)),
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'INITIALIZING NEURAL NETWORK...',
                            style: GoogleFonts.outfit(
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              color: Colors.white.withOpacity(0.3),
                              letterSpacing: 2,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          
          // 4. Footer
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: FadeTransition(
                opacity: _fadeAnimation,
                child: Text(
                  'B E L I E V E  I N  B E T T E R  H A R V E S T',
                  style: GoogleFonts.outfit(
                    fontSize: 9,
                    fontWeight: FontWeight.w400,
                    color: Colors.white.withOpacity(0.2),
                    letterSpacing: 4,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ParticlePainter extends CustomPainter {
  final double progress;
  final List<Particle> particles;

  ParticlePainter({required this.progress})
      : particles = List.generate(25, (i) => Particle(i));

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = const Color(0xFF10B981).withOpacity(0.15);

    for (var particle in particles) {
      final double x = (particle.baseX + math.sin(progress * 2 * math.pi + particle.offset) * 20) % size.width;
      final double y = (particle.baseY - (progress * 100 + particle.speed)) % size.height;
      
      canvas.drawCircle(Offset(x, y), particle.size, paint);
      
      // Draw a subtle connecting line to another particle if close
      // (Optional: can be heavy on performance, but 25 particles is fine)
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class Particle {
  final double baseX;
  final double baseY;
  final double size;
  final double speed;
  final double offset;

  Particle(int index)
      : baseX = math.Random(index).nextDouble() * 1000,
        baseY = math.Random(index + 1).nextDouble() * 1000,
        size = math.Random(index + 2).nextDouble() * 3 + 1,
        speed = math.Random(index + 3).nextDouble() * 50 + 20,
        offset = math.Random(index + 4).nextDouble() * math.pi * 2;
}
