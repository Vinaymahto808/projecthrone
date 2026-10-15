package com.example.faceattendance

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import java.text.SimpleDateFormat
import java.util.Locale

/** One day of attendance as returned by /employees/{id}/history. */
data class HistoryDay(
    val date: String,
    val firstIn: String?,
    val lastOut: String?,
    val hours: Double,
    val complete: Boolean,
)

class HistoryAdapter(private val items: List<HistoryDay>) :
    RecyclerView.Adapter<HistoryAdapter.Holder>() {

    class Holder(view: View) : RecyclerView.ViewHolder(view) {
        val date: TextView = view.findViewById(R.id.date)
        val inOut: TextView = view.findViewById(R.id.inOut)
        val hours: TextView = view.findViewById(R.id.hours)
        val pill: MaterialCardView = view.findViewById(R.id.pill)
        val pillText: TextView = view.findViewById(R.id.pillText)
    }

    private val parse = SimpleDateFormat("yyyy-MM-dd", Locale.US)
    private val pretty = SimpleDateFormat("EEE, dd MMM", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_history, parent, false))

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val d = items[position]
        val ctx = holder.itemView.context
        holder.date.text = try { pretty.format(parse.parse(d.date)!!) } catch (e: Exception) { d.date }
        holder.inOut.text = ctx.getString(
            R.string.day_in_out,
            d.firstIn?.take(5) ?: "—",
            d.lastOut?.take(5) ?: "—",
        )
        holder.hours.text = ctx.getString(R.string.hours_fmt, d.hours)
        holder.pillText.text = ctx.getString(if (d.complete) R.string.present else R.string.incomplete)
        holder.pillText.setTextColor(ContextCompat.getColor(ctx, if (d.complete) R.color.success else R.color.brand_primary))
        holder.pill.setCardBackgroundColor(
            ContextCompat.getColor(ctx, if (d.complete) R.color.success_soft else R.color.warning_soft)
        )
    }

    override fun getItemCount(): Int = items.size
}
