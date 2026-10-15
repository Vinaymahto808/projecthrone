plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.faceattendance"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.faceattendance"
        minSdk = 24
        targetSdk = 34
        versionCode = 2
        versionName = "2.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // CameraX: live preview, frame analysis, still capture
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ML Kit on-device face detection (gate for capture; recognition is server-side)
    implementation("com.google.mlkit:face-detection:16.1.7")

    // HTTP client for the Face Attendance API
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
