param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    "GEMINI_API_KEY=your_key_here" | Out-File -FilePath ".env" -Encoding ascii
}

if (-not (Test-Path "output\fasta")) {
    New-Item -ItemType Directory -Path "output\fasta" -Force | Out-Null
}

Write-Host "Setup complete."
Write-Host "Run server: .\.venv\Scripts\python.exe server.py"
