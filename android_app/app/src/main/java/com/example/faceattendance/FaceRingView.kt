package com.example.faceattendance

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.min

/**
 * Overlay placed on top of the square camera preview. It paints the page
 * background over the corners, punches a circular hole so the preview shows
 * through, and draws a coloured ring: GREEN = face detected, RED = no face.
 */
class FaceRingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detected = false

    private val maskPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.bg_page)
    }
    private val holePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
    }
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
    }

    private val green = ContextCompat.getColor(context, R.color.success)
    private val red = ContextCompat.getColor(context, R.color.danger)

    init {
        // CLEAR xfermode needs an off-screen layer to punch a transparent hole.
        setLayerType(LAYER_TYPE_HARDWARE, null)
    }

    fun setDetected(value: Boolean) {
        if (detected != value) {
            detected = value
            invalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val size = min(width, height).toFloat()
        val cx = width / 2f
        val cy = height / 2f
        val stroke = size * 0.035f
        val radius = size / 2f - stroke

        val color = if (detected) green else red
        ringPaint.strokeWidth = stroke
        ringPaint.color = color
        glowPaint.strokeWidth = stroke * 2.2f
        glowPaint.color = (color and 0x00FFFFFF) or 0x33000000

        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), maskPaint)
        canvas.drawCircle(cx, cy, radius, holePaint)
        canvas.drawCircle(cx, cy, radius, glowPaint)
        canvas.drawCircle(cx, cy, radius, ringPaint)
    }
}
