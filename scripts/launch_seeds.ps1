<#
.SYNOPSIS
    Launch a training script with several seeds in parallel, in the background.

.DESCRIPTION
    Seed 0 is saved as test <Test>, seed S > 0 as test <Test><S> (e.g. 43, 431, 432, 433, 434),
    the same convention as tests 311-314, 351-354, 401-404, 411-414 and 421-424.
    Each run writes logs\test<N>_train.out / .err (tqdm progress goes to .err). A run whose
    runs\Test\<N> or checkpoints\Test\<N> folder already exists is skipped, so nothing is overwritten.
    The PIDs are printed and saved to logs\launch_<Test>.txt.

.EXAMPLE
    powershell -File scripts\launch_seeds.ps1 -Script train\grid_training_43.py -Test 43
    powershell -File scripts\launch_seeds.ps1 -Script train\grid_training_44.py -Test 44 -Seeds 0,1
#>
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [Parameter(Mandatory = $true)][int]$Test,
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [string[]]$ExtraArgs = @()
)

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python not found: $python" }
# With stdout redirected to a file, Python on Windows writes cp1252 and crashes on emoji in prints.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path $Script)) { throw "Script not found: $Script" }
New-Item -ItemType Directory -Force logs | Out-Null

$launched = @()
foreach ($seed in $Seeds) {
    $n = if ($seed -eq 0) { "$Test" } else { "$Test$seed" }
    if ((Test-Path "runs\Test\$n") -or (Test-Path "checkpoints\Test\$n")) {
        Write-Host "test $n (seed $seed): runs\Test\$n or checkpoints\Test\$n exists, skipped"
        continue
    }
    $arguments = @($Script, "--seed", "$seed", "--test_number", $n) + $ExtraArgs
    $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
        -RedirectStandardOutput "logs\test${n}_train.out" -RedirectStandardError "logs\test${n}_train.err" `
        -WindowStyle Hidden -PassThru
    $line = "test $n (seed $seed): PID $($p.Id) started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host $line
    $launched += $line
}
if ($launched) { $launched | Out-File -FilePath "logs\launch_$Test.txt" -Encoding utf8 -Append }
Write-Host ""
Write-Host "Progress: Get-Content logs\test<N>_train.err -Tail 1   (or: tensorboard --logdir runs\Test\$Test)"
Write-Host "When all have finished: powershell -File scripts\evaluate_seeds.ps1 -Test $Test -Vs 42,35,40"
