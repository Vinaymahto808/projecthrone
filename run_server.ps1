# run_server.ps1 - Start the Face Attendance API for the Android kiosk app.
#
#   .\run_server.ps1            -> http://0.0.0.0:8000  (LAN access for phones)
#   .\run_server.ps1 -Port 9000
#
# Dashboard: http://<this-pc-ip>:8000/      Health: http://<this-pc-ip>:8000/health
# The Android app's server URL must point at this PC's LAN IP (shown below).

param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Virtual environment not found at .venv. See README.md (Offline face attendance)." -ForegroundColor Red
    exit 1
}

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress
Write-Host "Face Attendance API starting on port $Port" -ForegroundColor Green
foreach ($ip in $ips) { Write-Host "  Phone/app URL : http://$ip`:$Port" }
Write-Host "  Dashboard     : http://localhost:$Port/   (sign in with an admin account)"
Write-Host "  First run bootstraps admin 'admin' / 'admin123' and kiosk key 'kiosk-key-change-me'."
Write-Host "If the phone cannot connect, allow the port once (admin PowerShell):"
Write-Host "  netsh advfirewall firewall add rule name=`"Face Attendance API`" dir=in action=allow protocol=TCP localport=$Port"
Write-Host ""

& $python -m uvicorn server:app --host 0.0.0.0 --port $Port
