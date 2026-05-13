plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.ocrtoanki.ocr_to_anki"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.ocrtoanki.ocr_to_anki"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    signingConfigs {
        maybeCreate("release").apply {
            val keystorePath = System.getenv("KEYSTORE_PATH")
            if (!keystorePath.isNullOrEmpty()) {
                storeFile = file(keystorePath)
                storePassword = System.getenv("KEYSTORE_PASSWORD") ?: ""
                keyAlias = System.getenv("KEY_ALIAS") ?: "release"
                keyPassword = System.getenv("KEY_PASSWORD") ?: ""
            }
        }
    }

    buildTypes {
        release {
            val useReleaseSigning = !System.getenv("KEYSTORE_PATH").isNullOrEmpty()
            signingConfig = if (useReleaseSigning) {
                signingConfigs.getByName("release")
            } else {
                // Fallback to debug signing when no keystore is configured.
                // This means local dev builds and CI builds without secrets
                // will use different keys — in-app updates will fail with
                // "package conflicts". Uninstall and reinstall in that case.
                signingConfigs.getByName("debug")
            }
        }
    }

    // Native binaries (llama-server, llama-mtmd-cli) ship as lib*.so under
    // jniLibs/arm64-v8a. Modern Android (API 23+) keeps native libs inside
    // the APK and serves them via mmap for dlopen, but execve needs a real
    // file path. useLegacyPackaging=true forces extraction to nativeLibraryDir
    // (/data/app/.../lib/arm64/) at install time so we can spawn them.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

flutter {
    source = "../.."
}
