param(
  [int]$EpochsWDR = 50,
  [int]$EpochsCoD = 50,
  [double]$Beta = 0.45,
  [double]$Alpha = 0.01
)

$ErrorActionPreference = "Stop"

Write-Host "=== 0) Env check & setup ==="
# ensure Python can import project modules
$env:PYTHONPATH = (Get-Location).Path

# 1) start OSRM
Write-Host "=== 1) Start OSRM ==="
powershell -ExecutionPolicy Bypass -File .\scripts\osrm_up.ps1

# 2) batch map-matching (full sample)
Write-Host "=== 2) Batch map-matching (full) ==="
python -m data_proc.01b_match_tdrive_batch --outshard datasets\matched_parts --chunk 120 --flush-every 50

# 3) concat matched shards
Write-Host "=== 3) Concat matched shards ==="
python -m data_proc.01c_concat_matched_parts --indir datasets\matched_parts --out datasets\matched_legs.parquet

# 4) preprocessing pipeline
Write-Host "=== 4) Run 02/03/04 preprocessing ==="
python -m data_proc.02_build_links
python -m data_proc.03_agg_speed_10min
python -m data_proc.04_split_sets

# 5) train WDR
Write-Host "=== 5) Train WDR ==="
python -m trainers.train_wdr --epochs $EpochsWDR --ckpt wdr_full.pt

# 6) train CoDriver (triplet multi-task)
Write-Host ("=== 6) Train CoDriver (beta={0}, alpha={1}) ===" -f $Beta, $Alpha)
$ck = ("codriver_beta{0}.pt" -f ($Beta.ToString().Replace('.','')))
python -m trainers.train_codriver --beta $Beta --alpha $Alpha --epochs $EpochsCoD --ckpt $ck

# 7) evaluation
Write-Host "=== 7) Evaluate overall / sparse-dense ==="
python -m evals.eval_overall
python -m evals.eval_sparse_dense --nsparse 50 --ndense 500

# 8) t-SNE visualization
Write-Host "=== 8) t-SNE visualization ==="
$fig = ("outputs\figs\tsne_beta{0}.png" -f ($Beta.ToString().Replace('.','')))
python -m viz.tsne_driver_emb --ckpt ("outputs\" + $ck) --out $fig

Write-Host "=== DONE ==="
