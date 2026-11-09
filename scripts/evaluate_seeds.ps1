<#
.SYNOPSIS
    Evaluate the seeds of a test and compare them with other configurations.

.DESCRIPTION
    Seeds follow the launch_seeds.ps1 convention (seed 0 = test <Test>, seed S = test <Test><S>).
    1. Evaluates every seed that has a checkpoint but no plots\evals\eval_<N>_e1000_s0.json
       (scripts/eval_policy_modes.py, 1000 episodes greedy + sampled), in parallel.
    2. Seed plots and training curves: scripts/analyze_tests.py -> plots\seeds_<Test>\
    3. Mean ± std over seeds and Welch t-tests against each configuration in -Vs (each given by
       the test number of its seed 0; its seeds are found the same way):
       scripts/seed_stats.py -> plots\seeds_<Test>\seeds_stats_vs_<V>.md
    4. Group plot of <Test> and the -Vs configurations: scripts/plot_seed_groups.py ->
       plots\seeds_<Test>_groups\

.EXAMPLE
    powershell -File scripts\evaluate_seeds.ps1 -Test 43 -Vs 42,35,40
#>
param(
    [Parameter(Mandatory = $true)][int]$Test,
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [int[]]$Vs = @(42)
)

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"

function Get-SeedTests([int]$t, [int[]]$seeds) {
    $seeds | ForEach-Object { if ($_ -eq 0) { "$t" } else { "$t$_" } }
}
function Get-Checkpoint([string]$n) {
    Get-ChildItem "checkpoints\Test\$n\*\final.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
}

$tests = @(Get-SeedTests $Test $Seeds | Where-Object { Get-Checkpoint $_ })
$missing = @(Get-SeedTests $Test $Seeds | Where-Object { -not (Get-Checkpoint $_) })
if ($missing) { Write-Host "Not finished (no final.pth), left out: $($missing -join ', ')" }
if ($tests.Count -lt 2) { throw "Need at least 2 finished seeds of test $Test" }

# 1. Evaluations, in parallel
$procs = @()
foreach ($n in $tests) {
    if (-not (Test-Path "plots\evals\eval_${n}_e1000_s0.json")) {
        Write-Host "evaluating test $n ..."
        $procs += Start-Process -FilePath $python -ArgumentList @("scripts\eval_policy_modes.py", $n, "--episodes", "1000", "--seed", "0") `
            -WorkingDirectory $root -RedirectStandardOutput "logs\eval_$n.log" -RedirectStandardError "logs\eval_${n}.err" `
            -WindowStyle Hidden -PassThru
    }
}
if ($procs) { $procs | Wait-Process; foreach ($n in $tests) { Get-Content "logs\eval_$n.log" -ErrorAction SilentlyContinue | Select-String "test $n" } }

# 2. Seed plots
$labels = $tests | ForEach-Object { "$_" }
& $python scripts\analyze_tests.py @tests --labels @labels --out "plots\seeds_$Test" | Select-String "Written"

# 3. Statistics against each configuration in -Vs
$groupArgs = @("--group", "Test $Test") + $tests
foreach ($v in $Vs) {
    $other = @(Get-SeedTests $v $Seeds | Where-Object { Test-Path "plots\evals\eval_${_}_e1000_s0.json" })
    if ($other.Count -lt 2) { Write-Host "test ${v}: fewer than 2 evaluated seeds, skipped"; continue }
    & $python scripts\seed_stats.py @tests --name "Test $Test" --vs @other --vs-name "Test $v" --no-eval `
        --out "plots\seeds_$Test\seeds_stats_vs_$v.md" | Out-Null
    Write-Host "==== test $Test vs test $v"
    Get-Content "plots\seeds_$Test\seeds_stats_vs_$v.md" -Encoding UTF8 | Select-String "^## |^\| (Success \(total\)|simple_map|Timeouts)"
    $groupArgs += @("--group", "Test $v") + $other
}

# 4. Group plot
& $python scripts\plot_seed_groups.py @groupArgs --out "plots\seeds_${Test}_groups"
