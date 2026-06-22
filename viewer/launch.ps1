param(
  [string]$Package = "cv_world_arm",
  [string]$Mode = "robot",
  [int]$Port = 9190
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$alive = $false
try {
  $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2
  $alive = $response.StatusCode -eq 200
} catch {
  $alive = $false
}

if (-not $alive) {
  Start-Process -WindowStyle Minimized python -ArgumentList "viewer\server.py", "--host", "127.0.0.1", "--port", "$Port"
  Start-Sleep -Seconds 2
}

$url = "http://127.0.0.1:$Port/?package=$([uri]::EscapeDataString($Package))&mode=$Mode"
Write-Host "Opening $url"
Start-Process $url
