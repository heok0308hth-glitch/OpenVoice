# ONNX Runtime uses JNI and reflection on its Java API.
-keep class ai.onnxruntime.** { *; }
# jump3r (LAME port) wires modules by reflection-free calls, but keep it intact for LGPL relinking clarity.
-keep class de.sciss.jump3r.** { *; }
-dontwarn javax.sound.**
-keep class io.github.koreanvoice.** { *; }
