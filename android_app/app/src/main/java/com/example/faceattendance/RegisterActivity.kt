package com.example.faceattendance

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/**
 * Employee enrolment: details form + capture of REQUIRED face photos, then a
 * single multipart upload to /register. The server extracts embeddings with
 * the recognition model and stores them; this screen never stores faces.
 */
class RegisterActivity : AppCompatActivity() {

    private lateinit var nameLayout: TextInputLayout
    private lateinit var codeLayout: TextInputLayout
    private lateinit var nameInput: TextInputEditText
    private lateinit var codeInput: TextInputEditText
    private lateinit var deptInput: TextInputEditText
    private lateinit var passwordLayout: TextInputLayout
    private lateinit var passwordInput: TextInputEditText
    private lateinit var previewView: PreviewView
    private lateinit var ring: FaceRingView
    private lateinit var captureStatus: TextView
    private lateinit var captureBtn: MaterialButton
    private lateinit var submitBtn: MaterialButton
    private lateinit var progress: ProgressBar

    private lateinit var camera: FaceCamera
    private val photos = mutableListOf<ByteArray>()
    private var faceDetected = false
    private var busy = false

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) camera.start() else captureStatus.text = getString(R.string.camera_denied)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_register)

        nameLayout = findViewById(R.id.nameLayout)
        codeLayout = findViewById(R.id.codeLayout)
        nameInput = findViewById(R.id.nameInput)
        codeInput = findViewById(R.id.codeInput)
        deptInput = findViewById(R.id.deptInput)
        passwordLayout = findViewById(R.id.passwordLayout)
        passwordInput = findViewById(R.id.passwordInput)
        previewView = findViewById(R.id.previewView)
        ring = findViewById(R.id.ring)
        captureStatus = findViewById(R.id.captureStatus)
        captureBtn = findViewById(R.id.captureButton)
        submitBtn = findViewById(R.id.submitButton)
        progress = findViewById(R.id.progress)

        findViewById<View>(R.id.backBtn).setOnClickListener { finish() }
        captureBtn.setOnClickListener { capture() }
        submitBtn.setOnClickListener { submit() }
        findViewById<View>(R.id.retakeBtn).setOnClickListener {
            photos.clear(); updateCaptureUi()
        }

        camera = FaceCamera(this, previewView) { count ->
            faceDetected = count == 1
            ring.setDetected(faceDetected)
            if (!busy) updateCaptureUi()
        }
        updateCaptureUi()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            camera.start() else cameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        super.onDestroy()
        camera.shutdown()
    }

    private fun updateCaptureUi() {
        val done = photos.size >= REQUIRED
        captureStatus.text = when {
            done -> getString(R.string.capture_done, photos.size)
            faceDetected -> getString(R.string.capture_ready, photos.size, REQUIRED)
            else -> getString(R.string.capture_no_face, photos.size, REQUIRED)
        }
        captureStatus.setTextColor(
            ContextCompat.getColor(this, if (done || faceDetected) R.color.success else R.color.danger)
        )
        captureBtn.isEnabled = faceDetected && !done && !busy
        captureBtn.text = getString(R.string.capture_n, photos.size + 1, REQUIRED)
        submitBtn.isEnabled = done && !busy
    }

    private fun capture() {
        if (busy) return
        busy = true
        captureBtn.isEnabled = false
        camera.capture { res ->
            busy = false
            res.onSuccess { photos.add(it) }
            res.onFailure { Snackbar.make(findViewById(R.id.root), it.message ?: "Capture failed", Snackbar.LENGTH_SHORT).show() }
            updateCaptureUi()
        }
    }

    private fun submit() {
        val name = nameInput.text?.toString()?.trim().orEmpty()
        val code = codeInput.text?.toString()?.trim().orEmpty()
        val dept = deptInput.text?.toString()?.trim().orEmpty()
        val password = passwordInput.text?.toString().orEmpty()
        nameLayout.error = if (name.isEmpty()) getString(R.string.required) else null
        codeLayout.error = if (code.isEmpty()) getString(R.string.required) else null
        passwordLayout.error =
            if (password.isNotEmpty() && password.length < 4) getString(R.string.password_short) else null
        if (name.isEmpty() || code.isEmpty() || passwordLayout.error != null) return

        busy = true
        progress.visibility = View.VISIBLE
        submitBtn.isEnabled = false
        captureBtn.isEnabled = false
        ApiClient.register(this, name, code, dept, password, photos) { res ->
            busy = false
            progress.visibility = View.GONE
            res.onFailure {
                Snackbar.make(findViewById(R.id.root), it.message ?: "Registration failed", Snackbar.LENGTH_LONG).show()
                updateCaptureUi()
            }
            res.onSuccess { json ->
                val saved = json.optInt("images_saved")
                Snackbar.make(findViewById(R.id.root), getString(R.string.registered_ok, name, saved), Snackbar.LENGTH_LONG).show()
                findViewById<View>(R.id.root).postDelayed({ finish() }, 1800)
            }
        }
    }

    companion object {
        const val REQUIRED = 5
    }
}
