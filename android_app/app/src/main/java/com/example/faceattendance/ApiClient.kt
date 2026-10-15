package com.example.faceattendance

import android.content.Context
import android.os.Handler
import android.os.Looper
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Thin client for server.py. All callbacks run on the main thread.
 *
 * Two kinds of auth (see server.py / security.py):
 *  - the kiosk itself signs in with its device name + key (/auth/device-token)
 *    and caches the short-lived Bearer token; this covers /verify,
 *    /attendance/mark and history.
 *  - privileged actions (registering staff) require a server admin login
 *    (/auth/login). The token is cached until it expires.
 *
 * Device + admin settings live in SharedPreferences ("settings"). A local PIN
 * only guards the on-device settings screen so it keeps working offline.
 */
object ApiClient {

    const val DEFAULT_URL = "http://10.49.227.221:8000"
    const val DEFAULT_DEVICE = "kiosk-1"
    const val DEFAULT_DEVICE_KEY = "kiosk-key-change-me"
    private const val PREFS = "settings"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // first request may load the model
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    private val main = Handler(Looper.getMainLooper())
    private val jpeg = "image/jpeg".toMediaType()
    private val json = "application/json".toMediaType()

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    // ------------------------------------------------------------- settings
    fun baseUrl(ctx: Context): String =
        prefs(ctx).getString("server_url", DEFAULT_URL)!!.trim().trimEnd('/')

    fun setBaseUrl(ctx: Context, url: String) =
        prefs(ctx).edit().putString("server_url", url.trim()).apply()

    fun deviceName(ctx: Context): String =
        prefs(ctx).getString("device_name", DEFAULT_DEVICE)!!.trim()

    fun setDeviceName(ctx: Context, name: String) =
        prefs(ctx).edit().putString("device_name", name.trim()).apply()

    fun deviceKey(ctx: Context): String =
        prefs(ctx).getString("device_key", DEFAULT_DEVICE_KEY)!!.trim()

    fun setDeviceKey(ctx: Context, key: String) =
        prefs(ctx).edit().putString("device_key", key.trim()).apply()

    /** Local PIN that only guards the on-device settings screen (works offline). */
    fun adminPin(ctx: Context): String =
        prefs(ctx).getString("admin_pin", "1234")!!

    fun setAdminPin(ctx: Context, pin: String) =
        prefs(ctx).edit().putString("admin_pin", pin).apply()

    fun adminUser(ctx: Context): String =
        prefs(ctx).getString("admin_user", "admin")!!

    fun setAdminUser(ctx: Context, user: String) =
        prefs(ctx).edit().putString("admin_user", user.trim()).apply()

    fun hasAdmin(ctx: Context): Boolean = validToken(ctx, "admin_token", "admin_token_exp") != null

    fun clearAdmin(ctx: Context) =
        prefs(ctx).edit().remove("admin_token").remove("admin_token_exp").apply()

    // --------------------------------------------------------------- tokens
    private fun validToken(ctx: Context, tokenKey: String, expKey: String): String? {
        val tok = prefs(ctx).getString(tokenKey, null) ?: return null
        val exp = prefs(ctx).getLong(expKey, 0L)
        val now = System.currentTimeMillis() / 1000
        return if (exp - now > 60) tok else null
    }

    private fun storeToken(ctx: Context, obj: JSONObject, tokenKey: String, expKey: String) {
        prefs(ctx).edit()
            .putString(tokenKey, obj.getString("token"))
            .putLong(expKey, obj.optLong("expires", System.currentTimeMillis() / 1000 + 3600))
            .apply()
    }

    /** Exchange the configured device key for a kiosk token. */
    fun deviceLogin(ctx: Context, cb: (Result<Unit>) -> Unit) {
        val payload = JSONObject().put("device", deviceName(ctx)).put("key", deviceKey(ctx))
        run(
            Request.Builder().url("${baseUrl(ctx)}/auth/device-token")
                .post(payload.toString().toRequestBody(json)).build()
        ) { res ->
            res.onSuccess { storeToken(ctx, it, "device_token", "device_token_exp") }
            cb(res.map { })
        }
    }

    /** Server admin sign-in; stores the admin token on success. */
    fun adminLogin(ctx: Context, username: String, password: String, cb: (Result<JSONObject>) -> Unit) {
        val payload = JSONObject().put("username", username).put("password", password)
        run(
            Request.Builder().url("${baseUrl(ctx)}/auth/login")
                .post(payload.toString().toRequestBody(json)).build()
        ) { res ->
            res.onSuccess {
                storeToken(ctx, it, "admin_token", "admin_token_exp")
                setAdminUser(ctx, username)
            }
            cb(res)
        }
    }

    // -------------------------------------------------------------- requests
    fun health(ctx: Context, cb: (Result<JSONObject>) -> Unit) =
        run(Request.Builder().url("${baseUrl(ctx)}/health").get().build(), cb)

    /**
     * Identify a face from a short burst of frames. The server uses the extra
     * frames for its liveness / anti-spoof check, so send [frames] images.
     */
    fun verify(ctx: Context, frames: List<ByteArray>, cb: (Result<JSONObject>) -> Unit) =
        withDevice(ctx, cb) { token ->
            val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            frames.forEachIndexed { i, bytes ->
                body.addFormDataPart("images", "frame_$i.jpg", bytes.toRequestBody(jpeg))
            }
            Request.Builder().url("${baseUrl(ctx)}/verify")
                .header("Authorization", "Bearer $token")
                .post(body.build()).build()
        }

    fun register(
        ctx: Context,
        name: String,
        empCode: String,
        department: String,
        password: String,
        images: List<ByteArray>,
        cb: (Result<JSONObject>) -> Unit,
    ) = withAdmin(ctx, cb) { token ->
        val builder = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("name", name)
            .addFormDataPart("emp_code", empCode)
            .addFormDataPart("department", department)
            .addFormDataPart("password", password)
        images.forEachIndexed { i, bytes ->
            builder.addFormDataPart("images", "face_$i.jpg", bytes.toRequestBody(jpeg))
        }
        Request.Builder().url("${baseUrl(ctx)}/register")
            .header("Authorization", "Bearer $token")
            .post(builder.build()).build()
    }

    fun mark(ctx: Context, employeeId: Int, type: String, score: Double, cb: (Result<JSONObject>) -> Unit) =
        withDevice(ctx, cb) { token ->
            val payload = JSONObject()
                .put("employee_id", employeeId)
                .put("event_type", type)
                .put("score", score)
                .put("source", "android-kiosk")
            Request.Builder().url("${baseUrl(ctx)}/attendance/mark")
                .header("Authorization", "Bearer $token")
                .post(payload.toString().toRequestBody(json)).build()
        }

    /** Employee self-service login (public): returns employee, today and history. */
    fun login(ctx: Context, empCode: String, password: String, days: Int, cb: (Result<JSONObject>) -> Unit) {
        val payload = JSONObject().put("emp_code", empCode).put("password", password).put("days", days)
        run(
            Request.Builder().url("${baseUrl(ctx)}/employee/login")
                .post(payload.toString().toRequestBody(json)).build(), cb
        )
    }

    /** History for an employee already identified by face on the kiosk. */
    fun history(ctx: Context, employeeId: Int, days: Int, cb: (Result<JSONObject>) -> Unit) =
        withDevice(ctx, cb) { token ->
            Request.Builder().url("${baseUrl(ctx)}/employees/$employeeId/history?days=$days")
                .header("Authorization", "Bearer $token").get().build()
        }

    fun setPassword(ctx: Context, employeeId: Int, password: String, cb: (Result<JSONObject>) -> Unit) =
        withDevice(ctx, cb) { token ->
            val payload = JSONObject().put("password", password)
            Request.Builder().url("${baseUrl(ctx)}/employees/$employeeId/password")
                .header("Authorization", "Bearer $token")
                .post(payload.toString().toRequestBody(json)).build()
        }

    // --------------------------------------------------------------- plumbing
    private fun withDevice(ctx: Context, cb: (Result<JSONObject>) -> Unit, build: (String) -> Request) {
        validToken(ctx, "device_token", "device_token_exp")?.let { run(build(it), cb); return }
        deviceLogin(ctx) { res ->
            res.onFailure { cb(Result.failure(it)) }
            res.onSuccess {
                val tok = validToken(ctx, "device_token", "device_token_exp")
                if (tok == null) cb(Result.failure(IOException("Device sign-in failed")))
                else run(build(tok), cb)
            }
        }
    }

    private fun withAdmin(ctx: Context, cb: (Result<JSONObject>) -> Unit, build: (String) -> Request) {
        val tok = validToken(ctx, "admin_token", "admin_token_exp")
        if (tok == null) cb(Result.failure(IOException("Admin sign-in required")))
        else run(build(tok), cb)
    }

    private fun run(request: Request, cb: (Result<JSONObject>) -> Unit) {
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                main.post { cb(Result.failure(IOException("Cannot reach server: ${e.message}"))) }
            }

            override fun onResponse(call: Call, response: Response) {
                val text = response.body?.string().orEmpty()
                val result: Result<JSONObject> = try {
                    val obj = if (text.isBlank()) JSONObject() else JSONObject(text)
                    if (response.isSuccessful) Result.success(obj)
                    else Result.failure(IOException(obj.optString("detail", "HTTP ${response.code}")))
                } catch (e: Exception) {
                    Result.failure(IOException("Bad response (${response.code}): ${text.take(200)}"))
                }
                main.post { cb(result) }
            }
        })
    }
}
