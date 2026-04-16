import 'package:flutter_test/flutter_test.dart';
import 'package:crop_disease_detector/main.dart';

void main() {
  testWidgets('App starts without crashing', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const CropDoctorApp());
    
    // Splash screen should be visible
    expect(find.byType(CropDoctorApp), findsOneWidget);
  });
}
