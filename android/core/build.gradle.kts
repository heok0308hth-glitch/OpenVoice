plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

val ortVersion = "1.31.0"

dependencies {
    // The Android app supplies onnxruntime-android (same `ai.onnxruntime` API); the desktop
    // artifact is only used for compiling and for the JVM tests.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    implementation("de.sciss:jump3r:1.0.5")
    // org.json is part of the Android platform; the desktop artifact is only for compiling / JVM tests
    compileOnly("org.json:json:20250107")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.json:json:20250107")
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "3g"
    testLogging { events("passed", "failed", "skipped"); showStandardStreams = true; exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
    systemProperty("modelDir", System.getProperty("modelDir") ?: "")
    systemProperty("goldenDir", System.getProperty("goldenDir") ?: "")
    systemProperty("outDir", System.getProperty("outDir") ?: "")
    for (k in listOf("cbModelDir", "cbGoldenDir", "krefDir", "study", "speakers", "tuner", "variant")) systemProperty(k, System.getProperty(k) ?: "")
}
