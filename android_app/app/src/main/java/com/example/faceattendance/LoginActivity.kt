package com.example.faceattendance

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/** Employee self-service login (employee code + password) -> HistoryActivity. */
class LoginActivity : AppCompatActivity() {

    private lateinit var codeLayout: TextInputLayout
    private lateinit var passwordLayout: TextInputLayout
    private lateinit var codeInput: TextInputEditText
    private lateinit var passwordInput: TextInputEditText
    private lateinit var loginBtn: MaterialButton
    private lateinit var progress: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        codeLayout = findViewById(R.id.codeLayout)
        passwordLayout = findViewById(R.id.passwordLayout)
        codeInput = findViewById(R.id.codeInput)
        passwordInput = findViewById(R.id.passwordInput)
        loginBtn = findViewById(R.id.loginButton)
        progress = findViewById(R.id.progress)

        findViewById<View>(R.id.backBtn).setOnClickListener { finish() }
        loginBtn.setOnClickListener { login() }
    }

    private fun login() {
        val code = codeInput.text?.toString()?.trim().orEmpty()
        val password = passwordInput.text?.toString().orEmpty()
        codeLayout.error = if (code.isEmpty()) getString(R.string.required) else null
        passwordLayout.error = if (password.isEmpty()) getString(R.string.required) else null
        if (code.isEmpty() || password.isEmpty()) return

        loginBtn.isEnabled = false
        progress.visibility = View.VISIBLE
        ApiClient.login(this, code, password, HistoryActivity.DEFAULT_DAYS) { res ->
            progress.visibility = View.GONE
            loginBtn.isEnabled = true
            res.onFailure {
                Snackbar.make(findViewById(R.id.root), it.message ?: getString(R.string.login_failed), Snackbar.LENGTH_LONG).show()
            }
            res.onSuccess { json ->
                startActivity(
                    Intent(this, HistoryActivity::class.java)
                        .putExtra(HistoryActivity.EXTRA_EMPLOYEE, json.getJSONObject("employee").toString())
                        .putExtra(HistoryActivity.EXTRA_HISTORY, json.optJSONObject("history")?.toString())
                )
                finish()
            }
        }
    }
}
