package com.example.faceattendance

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.snackbar.Snackbar
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Kiosk screen.
 *
 * SCANNING  : ring red/green from on-device detection. When a face has been
 *             steady for ~1.2 s the frame is captured and sent to /verify.
 * VERIFYING : waiting for the server (model inference).
 * RESULT    : employee card filled from the server; Check In / Check Out call
 *             /attendance/mark. Returns to SCANNING after a short hold.
 *
 * Mismatch rule: if a face keeps being seen but is not recognised for
 * MISMATCH_MS (40 s), a "Face mismatch" alert is shown with Register /
 * Employee login options. The timer resets when the person leaves the frame.
 */
class MainActivity : AppCompatActivity() {

    private enum class State { SCANNING, VERIFYING, RESULT }

    private lateinit var previewView: PreviewView
    private lateinit var ring: FaceRingView
    private lateinit var statusCard: MaterialCardView
    private lateinit var status: TextView
    private lateinit var clockTime: TextView
    private lateinit var clockDate: TextView
    private lateinit var avatar: TextView
    private lateinit var empName: TextView
    private lateinit var empId: TextView
    private lateinit var empDept: TextView
    private lateinit var shiftStatus: TextView
    private lateinit var checkIn: MaterialButton
    private lateinit var checkOut: MaterialButton
    private lateinit var rescan: MaterialButton
    private lateinit var historyBtn: MaterialButton
    private lateinit var loginBtn: MaterialButton
    private lateinit var registerBtn: MaterialButton
    private lateinit var emptyLog: TextView
    private lateinit var logList: RecyclerView
    private lateinit var connection: TextView

    private lateinit var camera: FaceCamera
    private val ui = Handler(Looper.getMainLooper())

    private var state = State.SCANNING
    private var faceDetected = false
    private var faceSince = 0L
    private var lastFaceAt = 0L
    private var lastResultAt = 0L
    private var mismatchSince = 0L
    private var dialogOpen = false
    private var employee: JSONObject? = null
    private var checkedIn = false
    private var score = 0.0

    private val entries = mutableListOf<LogEntry>()
    private val adapter = LogAdapter(entries)

    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("EEEE, dd MMM yyyy", Locale.getDefault())

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) camera.start() else showStatus(false, getString(R.string.camera_denied))
        }

    private val clockTick = object : Runnable {
        override fun run() {
            val now = Date()
            clockTime.text = timeFormat.format(now)
            clockDate.text = dateFormat.format(now)
            ui.postDelayed(this, 1000)
        }
    }

    private val backToScan = Runnable { resetToScanning() }

    // ------------------------------------------------------------ lifecycle
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()

        logList.layoutManager = LinearLayoutManager(this)
        logList.adapter = adapter

        checkIn.setOnClickListener { mark("IN") }
        checkOut.setOnClickListener { mark("OUT") }
        rescan.setOnClickListener { resetToScanning() }
        historyBtn.setOnClickListener { openHistory() }
        loginBtn.setOnClickListener { startActivity(Intent(this, LoginActivity::class.java)) }
        registerBtn.setOnClickListener { askAdmin { startActivity(Intent(this, RegisterActivity::class.java)) } }
        findViewById<TextView>(R.id.settingsBtn).setOnClickListener { askPin { showSettings() } }

        camera = FaceCamera(this, previewView, ::onFaces)
        resetToScanning()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            camera.start() else cameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onResume() {
        super.onResume()
        ui.post(clockTick)
        checkServer()
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(clockTick)
        ui.removeCallbacks(backToScan)
    }

    override fun onDestroy() {
        super.onDestroy()
        camera.shutdown()
    }

    private fun bindViews() {
        previewView = findViewById(R.id.previewView)
        ring = findViewById(R.id.ring)
        statusCard = findViewById(R.id.statusCard)
        status = findViewById(R.id.status)
        clockTime = findViewById(R.id.clockTime)
        clockDate = findViewById(R.id.clockDate)
        avatar = findViewById(R.id.avatar)
        empName = findViewById(R.id.empName)
        empId = findViewById(R.id.empId)
        empDept = findViewById(R.id.empDept)
        shiftStatus = findViewById(R.id.shiftStatus)
        checkIn = findViewById(R.id.checkInButton)
        checkOut = findViewById(R.id.checkOutButton)
        rescan = findViewById(R.id.rescanButton)
        historyBtn = findViewById(R.id.historyButton)
        loginBtn = findViewById(R.id.loginButton)
        registerBtn = findViewById(R.id.registerButton)
        emptyLog = findViewById(R.id.emptyLog)
        logList = findViewById(R.id.logList)
        connection = findViewById(R.id.connection)
    }

    // ------------------------------------------------------------- scanning
    private fun onFaces(count: Int) {
        val detected = count == 1
        faceDetected = detected
        ring.setDetected(detected)
        val now = System.currentTimeMillis()

        if (detected) {
            lastFaceAt = now
        } else if (mismatchSince != 0L && now - lastFaceAt > PERSON_GONE_MS) {
            mismatchSince = 0L   // person walked away: forget the mismatch streak
        }
        if (state != State.SCANNING || dialogOpen) return

        if (!detected) {
            faceSince = 0L
            showStatus(false, getString(if (count > 1) R.string.face_many else R.string.face_none))
            return
        }
        if (faceSince == 0L) faceSince = now
        val held = now - faceSince
        if (held < HOLD_MS) {
            if (mismatchSince == 0L) showStatus(true, getString(R.string.face_hold))
        } else if (now - lastResultAt > COOLDOWN_MS) {
            captureAndVerify()
        }
    }

    private fun captureAndVerify() {
        state = State.VERIFYING
        showStatus(true, getString(R.string.verifying))
        // A short burst gives the server frames to run its liveness check.
        camera.captureBurst(BURST_FRAMES, BURST_GAP_MS) { captured ->
            captured.onFailure { fail(it.message ?: "Capture failed") }
            captured.onSuccess { frames ->
                ApiClient.verify(this, frames) { res -> onVerifyResult(res) }
            }
        }
    }

    private fun onVerifyResult(res: Result<JSONObject>) {
        val now = System.currentTimeMillis()
        lastResultAt = now
        res.onFailure { fail(it.message ?: "Verification failed") }
        res.onSuccess { json ->
            val best = json.optDouble("score", 0.0)
            if (!json.optBoolean("matched")) {
                onMismatch(best, now)
                return@onSuccess
            }
            val emp = json.optJSONObject("employee")
            if (emp == null || emp.isNull("id")) {
                showStatus(false, getString(R.string.face_no_record, emp?.optString("name") ?: "?"))
                state = State.SCANNING
                return@onSuccess
            }
            mismatchSince = 0L
            score = best
            state = State.RESULT
            applyEmployee(emp, json.optJSONObject("today"))
            showStatus(true, getString(R.string.verified_as, emp.optString("name"), score))
            ui.removeCallbacks(backToScan)
            ui.postDelayed(backToScan, RESULT_HOLD_MS)
        }
    }

    /** Face seen but not recognised: count the streak, alert after MISMATCH_MS. */
    private fun onMismatch(best: Double, now: Long) {
        state = State.SCANNING
        faceSince = 0L
        if (mismatchSince == 0L) mismatchSince = now
        val elapsed = now - mismatchSince
        if (elapsed >= MISMATCH_MS) {
            mismatchSince = 0L
            showStatus(false, getString(R.string.mismatch_title))
            showMismatchDialog(best, (elapsed / 1000).toInt())
        } else {
            val left = ((MISMATCH_MS - elapsed) / 1000).toInt()
            showStatus(false, getString(R.string.face_unknown_retry, best, left))
        }
    }

    private fun showMismatchDialog(best: Double, seconds: Int) {
        dialogOpen = true
        AlertDialog.Builder(this)
            .setTitle(R.string.mismatch_title)
            .setMessage(getString(R.string.mismatch_message, seconds, best))
            .setPositiveButton(R.string.register) { _, _ ->
                askAdmin { startActivity(Intent(this, RegisterActivity::class.java)) }
            }
            .setNeutralButton(R.string.employee_login) { _, _ ->
                startActivity(Intent(this, LoginActivity::class.java))
            }
            .setNegativeButton(R.string.dismiss, null)
            .setOnDismissListener {
                dialogOpen = false
                faceSince = 0L
                lastResultAt = System.currentTimeMillis()
                showStatus(false, getString(R.string.face_none))
            }
            .show()
    }

    private fun fail(message: String) {
        Snackbar.make(findViewById(R.id.root), message, Snackbar.LENGTH_LONG).show()
        state = State.SCANNING
        faceSince = 0L
        showStatus(false, getString(R.string.face_none))
    }

    // --------------------------------------------------------------- result
    private fun applyEmployee(emp: JSONObject, today: JSONObject?) {
        employee = emp
        val name = emp.optString("name", "—")
        empName.text = name
        empId.text = emp.optString("emp_code", "—")
        empDept.text = emp.optString("department").ifBlank { getString(R.string.dept_unknown) }
        avatar.text = initials(name)

        entries.clear()
        today?.optJSONArray("events")?.let { arr ->
            for (i in 0 until arr.length()) {
                val e = arr.getJSONObject(i)
                entries.add(LogEntry(e.optString("event_type"), e.optString("ts").substringAfter(' ')))
            }
        }
        checkedIn = today?.optBoolean("checked_in") ?: false
        adapter.notifyDataSetChanged()
        refreshLogUi()
    }

    private fun mark(type: String) {
        val emp = employee ?: return
        ui.removeCallbacks(backToScan)
        checkIn.isEnabled = false
        checkOut.isEnabled = false
        ApiClient.mark(this, emp.getInt("id"), type, score) { res ->
            res.onFailure {
                Snackbar.make(findViewById(R.id.root), it.message ?: "Failed", Snackbar.LENGTH_LONG).show()
                updateButtons()
                ui.postDelayed(backToScan, RESULT_HOLD_MS)
            }
            res.onSuccess { json ->
                applyEmployee(json.getJSONObject("employee"), json.optJSONObject("today"))
                val time = json.optJSONObject("event")?.optString("ts")?.substringAfter(' ') ?: ""
                val late = json.optJSONObject("grading")?.optInt("late_min", 0) ?: 0
                val msg = if (type == "IN" && late > 0)
                    getString(R.string.checked_in_late_msg, time, late)
                else getString(if (type == "IN") R.string.checked_in_msg else R.string.checked_out_msg, time)
                Snackbar.make(findViewById(R.id.root), msg, Snackbar.LENGTH_LONG).show()
                ui.postDelayed(backToScan, 3000)
            }
        }
    }

    private fun openHistory() {
        val emp = employee ?: return
        ui.removeCallbacks(backToScan)
        startActivity(
            Intent(this, HistoryActivity::class.java)
                .putExtra(HistoryActivity.EXTRA_EMPLOYEE, emp.toString())
        )
    }

    private fun resetToScanning() {
        ui.removeCallbacks(backToScan)
        state = State.SCANNING
        employee = null
        faceSince = 0L
        entries.clear()
        adapter.notifyDataSetChanged()
        avatar.text = "?"
        empName.text = getString(R.string.scan_prompt)
        empId.text = "—"
        empDept.text = getString(R.string.scan_hint)
        checkedIn = false
        refreshLogUi()
        showStatus(false, getString(R.string.face_none))
    }

    private fun updateButtons() {
        val ready = state == State.RESULT && employee != null
        checkIn.isEnabled = ready && !checkedIn
        checkOut.isEnabled = ready && checkedIn
        historyBtn.isEnabled = ready
        rescan.isEnabled = state != State.VERIFYING
    }

    private fun refreshLogUi() {
        emptyLog.visibility = if (entries.isEmpty()) TextView.VISIBLE else TextView.GONE
        val first = entries.firstOrNull()
        shiftStatus.text = when {
            employee == null -> getString(R.string.status_idle)
            first == null -> getString(R.string.status_not_in)
            first.type == "IN" -> getString(R.string.status_in_since, first.time)
            else -> getString(R.string.status_done, first.time)
        }
        updateButtons()
    }

    private fun showStatus(ok: Boolean, text: String) {
        status.text = text
        status.setTextColor(ContextCompat.getColor(this, if (ok) R.color.success else R.color.danger))
        statusCard.setCardBackgroundColor(ContextCompat.getColor(this, if (ok) R.color.success_soft else R.color.danger_soft))
    }

    private fun initials(name: String): String =
        name.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.take(2)
            .joinToString("") { it.first().uppercaseChar().toString() }.ifEmpty { "?" }

    // ----------------------------------------------------------- admin bits
    private fun checkServer() {
        connection.text = getString(R.string.server_checking, ApiClient.baseUrl(this))
        ApiClient.health(this) { res ->
            res.onSuccess {
                connection.text = getString(R.string.server_ok, it.optInt("employees"), it.optString("model").substringBefore(" "))
                connection.setTextColor(ContextCompat.getColor(this, R.color.success))
            }
            res.onFailure {
                connection.text = getString(R.string.server_down, ApiClient.baseUrl(this))
                connection.setTextColor(ContextCompat.getColor(this, R.color.danger))
            }
        }
    }

    private fun askPin(onOk: () -> Unit) {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            hint = getString(R.string.admin_pin)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.admin_required)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                if (input.text.toString() == ApiClient.adminPin(this)) onOk()
                else Snackbar.make(findViewById(R.id.root), R.string.wrong_pin, Snackbar.LENGTH_SHORT).show()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun showSettings() {
        val url = EditText(this).apply { setText(ApiClient.baseUrl(this@MainActivity)); hint = "http://host:8000" }
        val device = EditText(this).apply { setText(ApiClient.deviceName(this@MainActivity)); hint = getString(R.string.device_name) }
        val key = EditText(this).apply { setText(ApiClient.deviceKey(this@MainActivity)); hint = getString(R.string.device_key) }
        val pin = EditText(this).apply {
            setText(ApiClient.adminPin(this@MainActivity)); hint = getString(R.string.admin_pin)
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 16, 48, 0)
            addView(url); addView(device); addView(key); addView(pin)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.settings)
            .setView(box)
            .setPositiveButton(R.string.save) { _, _ ->
                ApiClient.setBaseUrl(this, url.text.toString())
                ApiClient.setDeviceName(this, device.text.toString())
                ApiClient.setDeviceKey(this, key.text.toString())
                if (pin.text.length >= 4) ApiClient.setAdminPin(this, pin.text.toString())
                ApiClient.clearAdmin(this)   // URL/key may have changed: force a fresh device + admin sign-in
                checkServer()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    /** Server admin sign-in, required before registering staff. */
    private fun askAdmin(onOk: () -> Unit) {
        val user = EditText(this).apply {
            setText(ApiClient.adminUser(this@MainActivity)); hint = getString(R.string.admin_user)
        }
        val pass = EditText(this).apply {
            hint = getString(R.string.admin_password)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 16, 48, 0)
            addView(user); addView(pass)
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.admin_sign_in)
            .setView(box)
            .setPositiveButton(R.string.sign_in, null)
            .setNegativeButton(android.R.string.cancel, null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                ApiClient.adminLogin(this, user.text.toString().trim(), pass.text.toString()) { res ->
                    res.onSuccess { dialog.dismiss(); onOk() }
                    res.onFailure {
                        Snackbar.make(findViewById(R.id.root), it.message ?: getString(R.string.login_failed), Snackbar.LENGTH_LONG).show()
                    }
                }
            }
        }
        dialog.show()
    }

    companion object {
        private const val HOLD_MS = 1200L          // face must be steady before auto-verify
        private const val COOLDOWN_MS = 4000L      // min gap between verifications
        private const val RESULT_HOLD_MS = 12000L  // how long the result stays on screen
        private const val MISMATCH_MS = 40_000L    // unrecognised streak before the alert
        private const val PERSON_GONE_MS = 6000L   // no face this long resets the streak
        private const val BURST_FRAMES = 3         // frames per verify (server liveness check)
        private const val BURST_GAP_MS = 140L      // gap between burst frames
    }
}
