package com.example.faceattendance

import android.os.Handler
import android.os.Looper
import android.util.Size
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Shared camera pipeline used by both screens:
 *  - Preview into a PreviewView
 *  - ImageAnalysis + ML Kit, reporting face count per frame via [onFaces]
 *  - ImageCapture for still JPEGs sent to the server via [capture]
 */
class FaceCamera(
    private val activity: AppCompatActivity,
    private val previewView: PreviewView,
    private val onFaces: (count: Int) -> Unit,
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private var imageCapture: ImageCapture? = null

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.25f)
            .build()
    )

    fun start() {
        val future = ProcessCameraProvider.getInstance(activity)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(executor, ::analyze) }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setJpegQuality(88)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(Size(960, 1280), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                        ).build()
                )
                .build()
            imageCapture = capture
            provider.unbindAll()
            provider.bindToLifecycle(activity, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis, capture)
        }, ContextCompat.getMainExecutor(activity))
    }

    /** Take a still JPEG. Callback runs on the main thread. */
    fun capture(onResult: (Result<ByteArray>) -> Unit) {
        val capture = imageCapture ?: run {
            onResult(Result.failure(IllegalStateException("Camera not ready")))
            return
        }
        capture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }
                image.close()
                activity.runOnUiThread { onResult(Result.success(bytes)) }
            }

            override fun onError(exception: ImageCaptureException) {
                activity.runOnUiThread { onResult(Result.failure(exception)) }
            }
        })
    }

    /** Capture [count] stills in quick succession for the server's liveness check. */
    fun captureBurst(count: Int, intervalMs: Long, onResult: (Result<List<ByteArray>>) -> Unit) {
        val frames = ArrayList<ByteArray>(count)
        fun next() {
            capture { res ->
                res.onSuccess { bytes ->
                    frames.add(bytes)
                    if (frames.size >= count) onResult(Result.success(frames))
                    else ui.postDelayed({ next() }, intervalMs)
                }
                res.onFailure { onResult(Result.failure(it)) }
            }
        }
        next()
    }

    fun shutdown() {
        executor.shutdown()
        detector.close()
    }

    @OptIn(ExperimentalGetImage::class)
    private fun analyze(proxy: ImageProxy) {
        val mediaImage = proxy.image ?: run { proxy.close(); return }
        val input = InputImage.fromMediaImage(mediaImage, proxy.imageInfo.rotationDegrees)
        detector.process(input)
            .addOnSuccessListener { faces -> activity.runOnUiThread { onFaces(faces.size) } }
            .addOnFailureListener { activity.runOnUiThread { onFaces(0) } }
            .addOnCompleteListener { proxy.close() }
    }
}
