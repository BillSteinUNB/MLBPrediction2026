$ErrorActionPreference = "Stop"
$env:MLB_OPTUNA_N_JOBS = "2"

$trainingData = "data\training\ParquetDefault_2018_no2020.parquet"
$oddsDb = "OddsScraper\data\mlb_odds.db;data\mlb_odds_oddsportal.db"
$failures = @()

& ".\.venv\Scripts\python.exe" `
    "scripts\filter_training_data_exclude_seasons.py" `
    --input "data\training\ParquetDefault.parquet" `
    --output $trainingData `
    --exclude-season "2020"

function Run-Workflow {
    param(
        [string]$Label,
        [string]$Hypothesis,
        [string]$FeatureSelectionMode = "flat",
        [int]$ForcedDeltaCount = 0,
        [string]$MuDeltaMode = "off",
        [int]$Folds = 3,
        [int]$Simulations = 1000,
        [int]$StarterInnings = 5,
        [bool]$EnableMarketPriors = $true
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running $Label"
    Write-Host "============================================================"

    $cmd = @(
        ".\.venv\Scripts\python.exe"
        "scripts\run_run_count_research_workflow.py"
        "--training-data"; $trainingData
        "--start"; "2018"
        "--end"; "2025"
        "--holdout"; "2025"
        "--Folds"; "$Folds"
        "--feature-selection-mode"; $FeatureSelectionMode
        "--forced-delta-count"; "$ForcedDeltaCount"
        "--XGBWork"; "4"
        "--simulations"; "$Simulations"
        "--starter-innings"; "$StarterInnings"
        "--tracker-run-label"; $Label
        "--tracker-hypothesis"; $Hypothesis
    )

    if ($EnableMarketPriors) {
        $cmd += "--enable-market-priors"
        $cmd += "--historical-odds-db"
        $cmd += $oddsDb
    }

    if ($MuDeltaMode -ne "off") {
        $cmd += "--mu-delta-mode"
        $cmd += $MuDeltaMode
    }

    & $cmd[0] $cmd[1..($cmd.Length - 1)]

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label"
        $script:failures += $Label
    }
}

Run-Workflow `
    -Label "window2018_test_01_flat_baseline" `
    -Hypothesis "2018 start, no 2020, flat selection, priors on, baseline control."

Run-Workflow `
    -Label "window2018_test_02_no_market_priors" `
    -Hypothesis "2018 start, no 2020, test whether market priors hurt broader-window learning." `
    -EnableMarketPriors $false

Run-Workflow `
    -Label "window2018_test_03_mu_gap_only" `
    -Hypothesis "2018 start, no 2020, mu_delta gap_only on flat baseline." `
    -MuDeltaMode "gap_only"

Run-Workflow `
    -Label "window2018_test_04_mu_gap_linear" `
    -Hypothesis "2018 start, no 2020, mu_delta gap_linear on flat baseline." `
    -MuDeltaMode "gap_linear"

Run-Workflow `
    -Label "window2018_test_05_mu_anchor_bundle" `
    -Hypothesis "2018 start, no 2020, mu_delta anchor_bundle on flat baseline." `
    -MuDeltaMode "anchor_bundle"

Run-Workflow `
    -Label "window2018_test_06_forced_delta_4" `
    -Hypothesis "2018 start, no 2020, flat selection with forced_delta_count 4." `
    -ForcedDeltaCount 4

Run-Workflow `
    -Label "window2018_test_07_forced_delta_8" `
    -Hypothesis "2018 start, no 2020, flat selection with forced_delta_count 8." `
    -ForcedDeltaCount 8

Run-Workflow `
    -Label "window2018_test_08_forced_delta_12" `
    -Hypothesis "2018 start, no 2020, flat selection with forced_delta_count 12." `
    -ForcedDeltaCount 12

Run-Workflow `
    -Label "window2018_test_09_bucketed" `
    -Hypothesis "2018 start, no 2020, bucketed selection with priors on." `
    -FeatureSelectionMode "bucketed"

Run-Workflow `
    -Label "window2018_test_10_grouped" `
    -Hypothesis "2018 start, no 2020, grouped selection with priors on." `
    -FeatureSelectionMode "grouped"

Run-Workflow `
    -Label "window2018_test_11_bucketed_delta_4" `
    -Hypothesis "2018 start, no 2020, bucketed selection plus forced_delta_count 4." `
    -FeatureSelectionMode "bucketed" `
    -ForcedDeltaCount 4

Run-Workflow `
    -Label "window2018_test_12_bucketed_delta_8" `
    -Hypothesis "2018 start, no 2020, bucketed selection plus forced_delta_count 8." `
    -FeatureSelectionMode "bucketed" `
    -ForcedDeltaCount 8

Run-Workflow `
    -Label "window2018_test_13_flat_folds_4" `
    -Hypothesis "2018 start, no 2020, flat selection with 4 CV folds." `
    -Folds 4

Run-Workflow `
    -Label "window2018_test_14_flat_folds_5" `
    -Hypothesis "2018 start, no 2020, flat selection with 5 CV folds." `
    -Folds 5

Run-Workflow `
    -Label "window2018_test_15_bucketed_folds_4" `
    -Hypothesis "2018 start, no 2020, bucketed selection with 4 CV folds." `
    -FeatureSelectionMode "bucketed" `
    -Folds 4

Run-Workflow `
    -Label "window2018_test_16_flat_folds_4_delta_4" `
    -Hypothesis "2018 start, no 2020, flat selection with 4 folds and forced_delta_count 4." `
    -Folds 4 `
    -ForcedDeltaCount 4

Run-Workflow `
    -Label "window2018_test_17_flat_folds_4_delta_8" `
    -Hypothesis "2018 start, no 2020, flat selection with 4 folds and forced_delta_count 8." `
    -Folds 4 `
    -ForcedDeltaCount 8

Run-Workflow `
    -Label "window2018_test_18_stage4_starter_4" `
    -Hypothesis "2018 start, no 2020, flat baseline with Stage 4 starter innings set to 4." `
    -StarterInnings 4

Run-Workflow `
    -Label "window2018_test_19_stage4_starter_6" `
    -Hypothesis "2018 start, no 2020, flat baseline with Stage 4 starter innings set to 6." `
    -StarterInnings 6

Run-Workflow `
    -Label "window2018_test_20_stage4_sim_2000" `
    -Hypothesis "2018 start, no 2020, flat baseline with 2000 Stage 4 simulations." `
    -Simulations 2000

Write-Host ""
Write-Host "============================================================"
Write-Host "Batch Complete"
Write-Host "============================================================"

if ($failures.Count -gt 0) {
    Write-Host "Failures:"
    $failures | ForEach-Object { Write-Host "  $_" }
}
else {
    Write-Host "All 20 runs completed."
}
