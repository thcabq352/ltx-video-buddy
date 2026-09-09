# VIDEO BUDDY — Windows installer
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) {
  Write-Host "FAIL  Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick Add Python to PATH."
  exit 1
}
& $py.Source install.py @args
exit $LASTEXITCODE
