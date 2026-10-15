# build_apk.ps1 - Build (and optionally install) the Face Attendance APK.
#
#   .\build_apk.ps1                 -> builds app\build\outputs\apk\debug\app-debug.apk
#   .\build_apk.ps1 installDebug    -> builds and installs on the connected phone
#   .\build_apk.ps1 run             -> builds, installs, and launches the app
#
# Uses the portable JDK 17 and Android SDK under %LOCALAPPDATA%\Android,
# so no Android Studio or system-wide JAVA_HOME is required.

param([string]$Task = "assembleDebug")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$androidHome = Join-Path $env:LOCALAPPDATA "Android"
$sdk = Join-Path $androidHome "Sdk"
$jdk = Get-ChildItem (Join-Path $androidHome "jdk") -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "bin\java.exe") } |
    Select-Object -First 1

if (-not $jdk) {
    Write-Host "JDK not found under $androidHome\jdk. Run the setup again or install Android Studio." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $sdk "platforms"))) {
    Write-Host "Android SDK platforms not found under $sdk. SDK setup has not finished." -ForegroundColor Red
    exit 1
}

$env:JAVA_HOME = $jdk.FullName
$env:ANDROID_HOME = $sdk
$env:ANDROID_SDK_ROOT = $sdk
$env:Path = "$($jdk.FullName)\bin;$sdk\platform-tools;$env:Path"

Write-Host "JAVA_HOME    = $env:JAVA_HOME"
Write-Host "ANDROID_HOME = $env:ANDROID_HOME"

$launch = $false
if ($Task -eq "run") { $Task = "installDebug"; $launch = $true }

& .\gradlew.bat $Task --console=plain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Task -eq "assembleDebug") {
    $apk = Join-Path $root "app\build\outputs\apk\debug\app-debug.apk"
    Write-Host ""
    Write-Host "APK ready: $apk" -ForegroundColor Green
    Write-Host "Install on a connected phone with:  adb install -r `"$apk`""
}

if ($launch) {
    & adb shell am start -n com.example.faceattendance/.MainActivity
}
