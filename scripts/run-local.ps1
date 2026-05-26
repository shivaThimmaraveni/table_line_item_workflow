param(
  [string]$HostName = "0.0.0.0",
  [int]$Port = 8080
)

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"
python -m pip install -U pip
pip install -r requirements.txt

Write-Host "Starting service at http://$HostName`:$Port/docs"
uvicorn app.main:app --host $HostName --port $Port --reload
