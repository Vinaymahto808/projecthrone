package com.example.faceattendance

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView

/** One attendance event: type is "IN" or "OUT", time is HH:mm:ss. */
data class LogEntry(val type: String, val time: String)

class LogAdapter(private val items: List<LogEntry>) :
    RecyclerView.Adapter<LogAdapter.Holder>() {

    class Holder(view: View) : RecyclerView.ViewHolder(view) {
        val dot: View = view.findViewById(R.id.dot)
        val type: TextView = view.findViewById(R.id.type)
        val time: TextView = view.findViewById(R.id.time)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_log, parent, false)
        return Holder(view)
    }

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val entry = items[position]
        val ctx = holder.itemView.context
        val isIn = entry.type == "IN"
        holder.type.text = ctx.getString(if (isIn) R.string.check_in else R.string.check_out)
        holder.time.text = entry.time
        holder.dot.backgroundTintList = ContextCompat.getColorStateList(
            ctx, if (isIn) R.color.success else R.color.brand_primary
        )
    }

    override fun getItemCount(): Int = items.size
}
