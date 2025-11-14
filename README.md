# CoDriver-ETA (Beijing / OSRM / WDR mini) — Reproduction Notes

This repository documents an end-to-end pipeline to reproduce a simplified version of **CoDriver ETA** on the **Beijing T-Drive dataset**, using **OSRM** for map-matching and a lightweight **Wide & Deep (WDR)** model for link-level speed prediction and downstream ETA estimation.

The implementation is designed and tested on **Windows + PowerShell**, with **Docker** for OSRM and **Python** for data processing and modelling.

---

## 0) Environment & Dependencies

**OS / Shell**

- Windows 10 / 11  
- PowerShell

**Core dependencies**

- Docker Desktop (for `osrm/osrm-backend`)
- Python 3.13 (in a virtual environment, e.g. `.venv`)

**Optional GPU stack**

- PyTorch 2.6.0 + CUDA 12.4  
- Verified with `torch.cuda.is_available() == True` on an RTX 4070

---

## Project Structure

The main project layout is as follows (only key paths listed):

```text
codriver_eta/
  configs/
    pickup.yaml
    trip.yaml

  data_raw/
    tdrive/
      taxi_log_2008_by_id/        # T-Drive raw txt files (not in repo)

  roadnet/
    beijing.osm.pbf               # OSM Beijing road network

  data_proc/
    01_match_osrm.py              # OSRM /match
    02_build_links.py             # link & trip-link construction
    03_agg_speed_10min.py         # 10-min link speed aggregation
    04_split_sets.py              # time-based splits

  datasets/                        # intermediate products (not in repo)

  models/
    wdr.py
    codriver.py

  trainers/
    train_wdr.py
    train_codriver.py

  evals/
    eval_overall.py
    eval_sparse_dense.py
    eval_dense2sparse.py
    eval_eta_simple.py

  viz/
    plot_hparams.py
    tsne_driver_emb.py

  scripts/
    osrm_docker_cmds.md
    run_all_pickup.sh
    run_all_trip.sh

  outputs/
    tables/
    figs/
Note: Large data and processed artifacts under roadnet/, datasets/, and data_raw/ are not stored in this GitHub repo due to size limits. See About large data and OSRM files.

##1) Start OSRM (routing & map-matching)
Assuming beijing.osm.pbf has already been preprocessed into beijing.osrm (see Appendix if preprocessing is needed), OSRM is launched via Docker as:

powershell
docker run -d --name osrm-beijing -p 5000:5000 `
  -v "C:\Users\zyzmc\PycharmProjects\codriver_eta\roadnet:/data" osrm/osrm-backend `
  osrm-routed --algorithm mld --threads 4 --max-matching-size 5000 /data/beijing.osrm
Health check:

powershell
curl "http://localhost:5000/nearest/v1/driving/116.397,39.908?number=1"
If port 5000 is already in use, the old container can be removed and restarted:

powershell
docker rm -f osrm-beijing
# then rerun the docker run command above
2) Map-match T-Drive (100 drivers)
Raw T-Drive trajectories are map-matched to the OSRM road network. The script uses moderate chunk sizes and radius to avoid "Too many coordinates" errors and timeouts:

powershell
python .\data_proc\01_match_osrm.py `
  --in_dir ".\data_raw\tdrive" `
  --pattern "taxi_log_2008_by_id\*.txt" `
  --fmt tdrive `
  --annotations true `
  --radius_m 150 `
  --probe_min_ratio 0.02 `
  --max_recursion_depth 3 `
  --early_stop_empty 6 `
  --chunk_points 50 `
  --overlap 2 `
  --downsample_every 1 `
  --min_move_m 1.0 `
  --workers 1 `
  --max_points_per_file 5000 `
  --max_files 100
Main outputs

datasets/matched_legs.csv

datasets/matched_legs.parquet

Practical notes

If "TooBig" / "Too many trace coordinates" appears, consider reducing --chunk_points or increasing OSRM’s --max-matching-size.

If "Read timed out" occurs:

Keep --workers 1

Possibly increase --radius_m

Ensure Docker has sufficient CPU/RAM.

3) Build link tables
Unique links are compacted and trip-level link sequences are built:

powershell
python .\data_proc\02_build_links.py `
  --in_csv .\datasets\matched_legs.csv `
  --out_links .\datasets\links.parquet `
  --out_trip_links .\datasets\trip_links.parquet
Outputs

datasets/links.parquet (and optional .csv export)

datasets/trip_links.parquet (and optional .csv export)

4) Aggregate 10-minute link speeds
Link-level speeds are aggregated over fixed 10-minute windows:

powershell
python .\data_proc\03_agg_speed_10min.py `
  --in_trip_links .\datasets\trip_links.parquet `
  --out_speed .\datasets\speed_10min.parquet `
  --window_min 10
Output

datasets/link_speed_10min.parquet (and optional .csv)

5) Time-based splits
The T-Drive subset used here covers 2008-02-02 ~ 2008-02-08. Time-based Train/Val/Test splits are created as follows:

powershell
python .\data_proc\04_split_sets.py `
  --links .\datasets\links.parquet `
  --trip_links .\datasets\trip_links.parquet `
  --speed_10min .\datasets\link_speed_10min.parquet `
  --out_splits .\datasets\splits_trip.pkl `
  --train_until "2008-02-06 00:00:00" `
  --val_until   "2008-02-07 00:00:00" `
  --test_until  "2008-02-09 00:00:00"
Outputs

datasets/splits_trip.pkl

datasets/link_speed_train.parquet

datasets/link_speed_val.parquet

datasets/link_speed_test.parquet

(Exact filenames may vary depending on the implementation in 04_split_sets.py.)

6) Train a simple Wide & Deep model (WDR)
A compact Wide & Deep model is trained on (link_id, time_bucket) to predict avg_speed_mps. Training is launched as a module from the project root so that package imports (e.g. models/) resolve correctly:

powershell
python -m trainers.train_wdr `
  --links .\datasets\links.parquet `
  --speed_10min .\datasets\link_speed_10min.parquet `
  --splits .\datasets\splits_trip.pkl `
  --out_dir .\outputs\wdr `
  --batch_size 131072 `
  --num_workers 8 `
  --epochs 5 `
  --wide_cross true
Output

outputs/wdr_simple.pt (model checkpoint; actual filename may depend on script arguments)

GPU usage tips

Increase --batch_size to better utilize VRAM.

Set --num_workers > 0 to remove input pipeline bottlenecks.

Ensure a CUDA-enabled PyTorch build (torch.version.cuda, torch.cuda.is_available()).

7) Predict link speeds & evaluate ETA
Predicted link speeds are generated for all relevant (link_id, time_bucket) pairs and then accumulated along trip_links to compute ETAs. Missing pairs are “self-healed” via fallback statistics, and overall MAE/MAPE metrics are reported:

powershell
python -m evals.eval_eta_simple `
  --ckpt .\outputs\wdr_simple.pt `
  --links .\datasets\links.parquet `
  --trip_links .\datasets\trip_links.parquet `
  --speed_10min .\datasets\link_speed_10min.parquet `
  --out_pred_speed .\outputs\pred_speed_10min.parquet `
  --out_eta_table .\outputs\tables\eta_overall.csv
Outputs

outputs/pred_speed_10min.parquet

outputs/tables/eta_overall.csv (aggregate ETA metrics)

What “good” performance means in this setup
Because this reproduction focuses on a simplified pipeline and a subset of drivers/time, the goal is stable, reasonable performance rather than exactly matching the original paper’s metrics.

Empirically:

Training MAE on link speed decreases steadily.

Validation MAE typically converges in the range ≈ 0.45–0.60 m/s under the 10-minute aggregation setup.

ETA evaluation yields consistent MAE (in seconds) across splits, with variation depending on:

coverage of links and time windows,

chunking configuration in map-matching,

subset of drivers and periods included.

The focus is on robust, interpretable trends (e.g. improvements over baselines and consistent behaviour across splits), rather than overfitting to a particular numeric target.

Troubleshooting Notes
Some issues observed during the reproduction and their resolutions:

PowerShell line continuation issues
All multi-line commands use the PowerShell backtick ` at the end of each continued line.
If copy/paste breaks, commands can be run as a single long line instead.

OSRM timeouts / “Too many coordinates”

Use --workers 1 in 01_match_osrm.py for stability.

Keep --chunk_points moderate (e.g. 50).

Optionally increase --radius_m.

Ensure Docker has sufficient CPU/RAM allocated.

ModuleNotFoundError: models

Ensure commands like trainers.train_wdr and evals.eval_eta_simple are invoked from the project root using python -m ....

Low GPU utilization

Increase --batch_size.

Increase --num_workers.

Allow for a warm-up phase.

Avoid splitting data into overly small shards.

Appendix — Building beijing.osrm from beijing.osm.pbf
If the OSRM graph files have not yet been prepared, they can be generated from beijing.osm.pbf in roadnet/ via:

powershell
# 1) Extract
docker run --rm -v "C:\Users\zyzmc\PycharmProjects\codriver_eta\roadnet:/data" osrm/osrm-backend `
  osrm-extract -p /opt/car.lua /data/beijing.osm.pbf

# 2) Partition
docker run --rm -v "C:\Users\zyzmc\PycharmProjects\codriver_eta\roadnet:/data" osrm/osrm-backend `
  osrm-partition /data/beijing.osrm

# 3) Customize
docker run --rm -v "C:\Users\zyzmc\PycharmProjects\codriver_eta\roadnet:/data" osrm/osrm-backend `
  osrm-customize /data/beijing.osrm
After these steps, return to Start OSRM (routing & map-matching) and launch osrm-routed.

About Large data and OSRM files (not included in this repo)
Due to GitHub's file size limits, several large data/OSRM files are not tracked by git and are therefore not included in this repository.
To fully reproduce the experiments, these files need to be obtained manually.

OSRM graph data
Place the following files under roadnet/:

roadnet/beijing.osrm

roadnet/beijing.osrm.cell_metrics

Processed datasets
Place the following files under datasets/:

datasets/matched_legs.csv

datasets/trip_links.csv

Raw / intermediate T-Drive data
Place the following under data_raw/tdrive/:

data_raw/tdrive/tdrive_oneweek_extracted.csv

data_raw/tdrive/tdrive_points.parquet

(and other raw log files under data_raw/tdrive/taxi_log_2008_by_id/)

How to download these files
Download the prepared data archive:

codriver_eta_data from: https://pan.quark.cn/s/45263683a59c

Extract the archive into the root folder of this project so that the following paths exist:

roadnet/beijing.osrm

roadnet/beijing.osrm.cell_metrics

datasets/matched_legs.csv

datasets/trip_links.csv

data_raw/tdrive/tdrive_oneweek_extracted.csv

data_raw/tdrive/tdrive_points.parquet

Verify that the folder structure matches the paths above. After that, all preprocessing and training scripts described in this README can be executed.

Future Extensions toward the Original CoDriver Paper
This reproduction currently focuses on:

OSRM-based map-matching on a subset of T-Drive;

a simplified Wide & Deep model for link-level speed prediction; and

a basic ETA evaluation pipeline.

To move closer to the original CoDriver ETA paper, the following extensions are planned:

Incorporation of driver embeddings and auxiliary driving-style tasks (see train_codriver.py);

Systematic evaluation of sparse vs. dense drivers with proper ablations;

More comprehensive ETA benchmarking with richer feature sets and additional baselines.

Credits
OSRM backend

T-Drive: public Beijing taxi GPS dataset

Model and pipeline scripts under data_proc/, trainers/, and evals/ in this repository.
