param(
    [string]$MacUser = "bill",
    [string]$MacHost,
    [string]$RemoteRepoPath = "/Users/bill/Code/MLBTracker",
    [string]$LocalRepoPath = "C:\Users\bills\Documents\Personal Code\MLBPrediction2026",
    [string]$PipelineDate = "today"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($MacHost)) {
    throw "MacHost is required. Example: -MacHost 100.93.108.96"
}

$remoteOddsDb = "$MacUser@${MacHost}:$RemoteRepoPath/OddsScraper/data/mlb_odds.db"
$remoteMarketState = "$MacUser@${MacHost}:$RemoteRepoPath/OddsScraper/data/live_market_state.json"
$remoteGameState = "$MacUser@${MacHost}:$RemoteRepoPath/OddsScraper/data/live_game_state.json"
$localOddsDb = Join-Path $LocalRepoPath "OddsScraper\data\mlb_odds.db"
$localMarketState = Join-Path $LocalRepoPath "OddsScraper\data\live_market_state.json"
$localGameState = Join-Path $LocalRepoPath "OddsScraper\data\live_game_state.json"

Write-Host "Pulling latest Mac odds database..."
scp $remoteOddsDb $localOddsDb

Write-Host "Pulling latest Mac market-state export..."
scp $remoteMarketState $localMarketState

Write-Host "Pulling latest Mac game-state export..."
scp $remoteGameState $localGameState

Write-Host "Refreshing cached slate from local data..."
Push-Location $LocalRepoPath
try {
    & ".\.venv\Scripts\python.exe" -m src.pipeline.daily --date $PipelineDate --mode prod --dry-run --db-path data\mlb.db
    & ".\.venv\Scripts\python.exe" -m src.ops.live_season_tracker sync-game-state --input-path OddsScraper\data\live_game_state.json --db-path data\mlb.db
}
finally {
    Pop-Location
}

Write-Host "Sync and refresh complete."
