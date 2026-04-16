# android/app/proguard-rules.pro
#
# Keep TFLite classes — minifier will strip them otherwise
# causing runtime crashes on release builds

-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }

# Keep Flutter TFLite plugin
-keep class com.tfliteflutter.** { *; }

# Keep image_picker plugin
-keep class io.flutter.plugins.imagepicker.** { *; }

# Suppress warnings for missing TFLite GPU classes
# (GPU delegate is not bundled — suppress warning, not error)
-dontwarn org.tensorflow.lite.gpu.**