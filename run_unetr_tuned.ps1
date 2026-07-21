# PowerShell script to run the tuned UNETR sweep robustly with thermal cooldown restarts
$ErrorActionPreference = "Stop"

do {
    Write-Host "Starting tuned UNETR sweep..."
    .venv\Scripts\python.exe run_phase3_robust.py --config configs/sweep_unetr_tuned.yaml --models unetr --out outputs/phase3_unetr_tuned --suffix _tuned
    $lastExitCode = $LASTEXITCODE
    if ($lastExitCode -eq 42) {
        Write-Host "Thermal cooldown triggered. Sleeping for 120 seconds to let the GPU cool down..."
        Start-Sleep -Seconds 120
    }
} while ($lastExitCode -eq 42)

Write-Host "Tuned UNETR sweep complete. Running final analysis..."
.venv\Scripts\python.exe scripts/analyze_probe1.py --summary outputs/phase3_unetr_tuned/phase3_summary.jsonl
