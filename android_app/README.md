# Face Attendance (Android)

Kiosk + registration client for the Face Attendance API (`server.py` in the parent folder). Face *detection* runs on the phone with ML Kit to drive the green/red ring and gate captures; face *recognition* runs on the server with the DeepFace-based model, so the phone never stores face data.

## Screens

**Kiosk (main screen)**
1. Header with company name, live clock, server connection status, and an employee card that fills in after a match.
2. Circular front-camera preview. Ring green = one face detected, red = none (or more than one). After the face is steady for about a second, a short **burst of 3 frames** is captured and sent to `/verify`, which runs recognition plus an anti-spoof **liveness** check (motion, same face across frames, image quality). Printed photos and frozen screens are rejected.
3. On a match the card shows name, employee code, department, today's state, and today's events. **Check In** / **Check Out** call `/attendance/mark`; the server enforces IN before OUT. The screen returns to scanning after 12 s or after an action.
4. Unknown face: red pill "Not recognised · keep looking at the camera · N s" and scanning continues. If the same unrecognised face keeps being seen for **40 seconds**, a **Face mismatch** alert appears with Register / Employee login / Dismiss. The timer resets when the person leaves the frame for 6 s or when a match succeeds.
5. **My history** (enabled after a match) opens the employee's per-day attendance history. **Employee login** opens the same history after entering employee code + password, for staff who were not recognised or want to check from elsewhere.
6. ⚙ (Kiosk settings) is protected by a local admin PIN (default `1234`) and holds the server URL, the **device name + device key** (the kiosk signs in to the server with these), and the PIN. **Register employee** requires a **server admin sign-in** (username + password, bootstrap `admin` / `admin123`) because enrolment is a privileged server action.

**Employee login / Attendance history**
- Login: employee code + the password given at registration (admins can set one later via the API).
- History: days present and total hours for the last 30 days, then one row per day with first In, last Out, hours and a Present/Open pill.

**Register employee**
1. Full name, employee code, department, optional password (min 4 characters) for the history login.
2. Capture 5 face photos; the button is enabled only while exactly one face is in the circle. Retake clears them.
3. Submit uploads everything to `/register`. The server rejects photos without a single face and needs at least 3 good ones.

## Build / run

Toolchain lives in `%LOCALAPPDATA%\Android` (portable JDK 17 + SDK); no Android Studio required.

```powershell
cd C:\Users\Galaxy\Desktop\carrier\android_app
.\build_apk.ps1            # APK -> app\build\outputs\apk\debug\app-debug.apk
.\build_apk.ps1 run        # build + install + launch on a USB phone
```

Xiaomi/Redmi phones block USB installs until **Developer options → Install via USB** is enabled (Mi account sign-in needed). Alternative: copy the APK to the phone and tap it in the Files app.

## First-time setup on the phone

1. Start the server on the PC: `.\run_server.ps1` (parent folder). Note the `http://<pc-ip>:8000` it prints, and the bootstrap kiosk key it logs (`kiosk-1` / `kiosk-key-change-me` by default).
2. In the app, tap ⚙, enter PIN `1234`, set the server URL and the **device name + device key**, Save. The connection line in the header turns green when the server answers.
3. Tap **Register employee**; when asked, sign in with a server admin account (bootstrap `admin` / `admin123`), fill the form, capture 5 photos, Submit.
4. Stand in front of the kiosk: the face is verified and Check In / Check Out become active.

Phone and PC must be on the same Wi-Fi, and Windows Firewall must allow TCP 8000 (see parent README).

## Customise

| What | Where |
| --- | --- |
| Company name, texts | `res/values/strings.xml` |
| Colours | `res/values/colors.xml` |
| Phone vs tablet sizes | `res/values/dimens.xml`, `res/values-sw600dp/dimens.xml` |
| Default server URL, device name/key, PIN storage | `ApiClient.kt` |
| Auto-verify timing (hold, cooldown, result hold) | `MainActivity.kt` companion object |
| Number of registration photos | `RegisterActivity.REQUIRED` |

## Files

| File | Purpose |
| --- | --- |
| `MainActivity.kt` | Kiosk state machine: scanning → verifying → result, Check In/Out, admin dialogs |
| `RegisterActivity.kt` | Enrolment form + 5-photo capture + upload |
| `FaceCamera.kt` | CameraX preview, ML Kit face count, JPEG still + burst capture |
| `ApiClient.kt` | OkHttp calls to `/auth/*`, `/health`, `/verify`, `/register`, `/attendance/mark` with token auth |
| `FaceRingView.kt` | Mask + circular hole + green/red ring |
| `LogAdapter.kt`, `item_log.xml` | Today's events list |
| `activity_main.xml`, `activity_register.xml` | Responsive layouts |
