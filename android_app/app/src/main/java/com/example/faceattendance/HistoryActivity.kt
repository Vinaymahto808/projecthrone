package com.example.faceattendance

import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.snackbar.Snackbar
import org.json.JSONObject

/**
 * Per-day attendance history for one employee.
 * Opened from the kiosk after a face match (fetches /employees/{id}/history)
 * or from LoginActivity (history already included in the login response).
 */
class HistoryActivity : AppCompatActivity() {

    private lateinit var avatar: TextView
    private lateinit var name: TextView
    private lateinit var code: TextView
    private lateinit var dept: TextView
    private lateinit var daysPresent: TextView
    private lateinit var totalHours: TextView
    private lateinit var period: TextView
    private lateinit var empty: TextView
    private lateinit var list: RecyclerView
    private lateinit var progress: ProgressBar

    private val days = mutableListOf<HistoryDay>()
    private val adapter = HistoryAdapter(days)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        avatar = findViewById(R.id.avatar)
        name = findViewById(R.id.empName)
        code = findViewById(R.id.empId)
        dept = findViewById(R.id.empDept)
        daysPresent = findViewById(R.id.daysPresent)
        totalHours = findViewById(R.id.totalHours)
        period = findViewById(R.id.period)
        empty = findViewById(R.id.emptyHistory)
        list = findViewById(R.id.historyList)
        progress = findViewById(R.id.progress)
        findViewById<View>(R.id.backBtn).setOnClickListener { finish() }

        list.layoutManager = LinearLayoutManager(this)
        list.adapter = adapter

        val employee = JSONObject(intent.getStringExtra(EXTRA_EMPLOYEE) ?: "{}")
        bindEmployee(employee)

        val preloaded = intent.getStringExtra(EXTRA_HISTORY)
        if (preloaded != null) {
            bindHistory(JSONObject(preloaded))
        } else {
            progress.visibility = View.VISIBLE
            ApiClient.history(this, employee.optInt("id"), DEFAULT_DAYS) { res ->
                progress.visibility = View.GONE
                res.onFailure { Snackbar.make(findViewById(R.id.root), it.message ?: "Failed", Snackbar.LENGTH_LONG).show() }
                res.onSuccess { bindHistory(it.optJSONObject("history") ?: JSONObject()) }
            }
        }
    }

    private fun bindEmployee(emp: JSONObject) {
        val n = emp.optString("name", "—")
        name.text = n
        code.text = emp.optString("emp_code", "—")
        dept.text = emp.optString("department").ifBlank { getString(R.string.dept_unknown) }
        avatar.text = n.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.take(2)
            .joinToString("") { it.first().uppercaseChar().toString() }.ifEmpty { "?" }
    }

    private fun bindHistory(h: JSONObject) {
        daysPresent.text = h.optInt("days_present").toString()
        totalHours.text = getString(R.string.hours_fmt, h.optDouble("total_hours", 0.0))
        period.text = getString(R.string.period, h.optInt("days", DEFAULT_DAYS))

        days.clear()
        val rows = h.optJSONArray("rows")
        if (rows != null) {
            for (i in 0 until rows.length()) {
                val r = rows.getJSONObject(i)
                days.add(
                    HistoryDay(
                        date = r.optString("date"),
                        firstIn = r.optString("first_in", "").takeIf { !r.isNull("first_in") },
                        lastOut = r.optString("last_out", "").takeIf { !r.isNull("last_out") },
                        hours = r.optDouble("hours", 0.0),
                        complete = r.optBoolean("complete"),
                    )
                )
            }
        }
        adapter.notifyDataSetChanged()
        empty.visibility = if (days.isEmpty()) View.VISIBLE else View.GONE
    }

    companion object {
        const val EXTRA_EMPLOYEE = "employee"
        const val EXTRA_HISTORY = "history"
        const val DEFAULT_DAYS = 30
    }
}
