$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $scriptDir "logs"
$stdoutLog = Join-Path $logDir "webapp.out.log"
$stderrLog = Join-Path $logDir "webapp.err.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Process `
  -FilePath python `
  -ArgumentList (Join-Path $scriptDir "run_webapp.py") `
  -WorkingDirectory $scriptDir `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog

Write-Output "PySERA web app started."
Write-Output "URL: http://127.0.0.1:8050"
Write-Output "Logs:"
Write-Output "  $stdoutLog"
Write-Output "  $stderrLog"
