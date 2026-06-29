param(
    [string]$symbol = "BTCUSDT",
    [string]$interval = "5m",
    [int]$limit = 1000
)

$env:PYTHONPATH = "D:\CWT prediction"
& "C:\Users\ag065\AppData\Local\Programs\Python\Python311\python.exe" "D:\CWT prediction\run_prediction.py" --symbol $symbol --interval $interval --limit $limit
if ($LASTEXITCODE -eq 124) { Write-Error "Timeout (60s) - prediction may have partially completed" }
