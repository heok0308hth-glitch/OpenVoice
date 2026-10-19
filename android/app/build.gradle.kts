plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "io.github.koreanvoice.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.koreanvoice.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 3
        versionName = "2.1.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        // Model pack location for the "lite" flavor (downloaded once, then used offline).
        buildConfigField("String", "MODEL_BASE_URLS", "\"${project.findProperty("modelBaseUrls") ?: ""}\"")
    }

    flavorDimensions += "models"
    productFlavors {
        create("full") {
            dimension = "models"
            buildConfigField("boolean", "BUNDLED_MODELS", "true")
        }
        create("lite") {
            dimension = "models"
            applicationIdSuffix = ".lite"
            versionNameSuffix = "-lite"
            buildConfigField("boolean", "BUNDLED_MODELS", "false")
        }
    }

    signingConfigs {
        create("release") {
            val ks = rootProject.file("release.keystore")
            if (ks.exists()) {
                storeFile = ks
                storePassword = System.getenv("KV_STORE_PASSWORD") ?: "koreanvoice"
                keyAlias = "koreanvoice"
                keyPassword = System.getenv("KV_KEY_PASSWORD") ?: "koreanvoice"
            }
        }
    }

    buildTypes {
        debug {
            // x86_64 only for emulator testing; release builds ship arm64-v8a.
            ndk { abiFilters += listOf("x86_64") }
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release").takeIf { it.storeFile != null } ?: signingConfigs.getByName("debug")
        }
    }

    lint {
        disable += listOf("GradleDependency", "AndroidGradlePluginVersion", "ChromeOsAbiSupport", "UseKtx", "ObsoleteSdkInt")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    androidResources {
        // ONNX files are copied out of the APK once; keeping them uncompressed makes that fast.
        noCompress += "onnx"
    }
    packaging {
        resources.excludes += listOf("META-INF/{AL2.0,LGPL2.1}", "META-INF/DEPENDENCIES")
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    val composeBom = platform("androidx.compose:compose-bom:2025.05.01")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0")
    implementation("androidx.lifecycle:lifecycle-service:2.9.0")
    implementation("androidx.core:core-ktx:1.16.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
}

// "full" flavor: join the model pack parts (../modelpack) into APK assets at build time.
val modelAssetsDir = layout.buildDirectory.dir("generated/modelAssets")
val joinModelParts by tasks.registering {
    val packDir = rootProject.file("modelpack")
    inputs.dir(packDir)
    outputs.dir(modelAssetsDir)
    doLast {
        val manifest = groovy.json.JsonSlurper().parse(File(packDir, "manifest.json")) as Map<*, *>
        val out = modelAssetsDir.get().asFile.resolve("models").apply { mkdirs() }
        @Suppress("UNCHECKED_CAST")
        // the optional high-quality decoder is never bundled (downloaded on demand from the app)
        (manifest["files"] as Map<String, Map<String, Any>>).filterKeys { it != "s3gen_hq.onnx" }.forEach { (name, info) ->
            File(out, name).outputStream().use { o ->
                (info["parts"] as List<String>).forEach { p -> File(packDir, p).inputStream().use { it.copyTo(o) } }
            }
            check(File(out, name).length() == (info["size"] as Number).toLong()) { "size mismatch: $name" }
        }
    }
}
android.sourceSets.getByName("full").assets.srcDir(modelAssetsDir)
tasks.configureEach {
    // every task of the "full" variants that may read the generated assets
    if (name.contains("Full")) dependsOn(joinModelParts)
}
