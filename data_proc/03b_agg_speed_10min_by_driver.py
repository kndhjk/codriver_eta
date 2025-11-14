# data_proc/03b_agg_speed_10min_by_driver.py
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SRC = DATASETS / "trip_links.parquet"
OUT = DATASETS / "link_speed_driver_10min.parquet"
WINDOW_S = 600

def slice_into_windows(t0, t1, window_s=WINDOW_S):
    if t1 <= t0: return []
    i0 = int(np.floor(t0 / window_s))
    out = []
    cur = t0
    idx = i0
    while cur < t1:
        win_start = idx * window_s
        win_end   = win_start + window_s
        seg_end   = min(t1, win_end)
        overlap   = max(0.0, seg_end - cur)
        if overlap > 0:
            out.append((idx, overlap))
        cur = seg_end
        idx += 1
    return out

def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    df = pd.read_parquet(SRC)

    # 兜底 duration
    if "duration_s" not in df.columns and {"distance_m","speed_mps"} <= set(df.columns):
        df["duration_s"] = df["distance_m"] / df["speed_mps"].replace(0, np.nan)
    for c in ["trip_id","driver_id","link_id","seq","duration_s"]:
        if c not in df.columns:
            raise ValueError(f"trip_links 缺列: {c}")

    rows = []
    for trip_id, g in df.groupby("trip_id", sort=False):
        g = g.sort_values("seq").reset_index(drop=True)
        dur = g["duration_s"].astype(float).fillna(0.0).clip(lower=0.0)
        rel_t1 = dur.cumsum().values
        rel_t0 = rel_t1 - dur.values
        for i, r in g.iterrows():
            t0, t1 = float(rel_t0[i]), float(rel_t1[i])
            if t1 <= t0: continue
            d = float(r.get("distance_m", np.nan))
            if not np.isfinite(d) and np.isfinite(r.get("speed_mps", np.nan)):
                d = float(r["speed_mps"]) * float(r["duration_s"])
            parts = slice_into_windows(t0, t1)
            for win_idx, overlap in parts:
                frac = overlap / float(r["duration_s"]) if r["duration_s"] and r["duration_s"] > 0 else 0.0
                rows.append({
                    "trip_id":   r["trip_id"],
                    "driver_id": r["driver_id"],
                    "link_id":   r["link_id"],
                    "rel_window_idx": int(win_idx),
                    "part_duration_s": overlap,
                    "part_distance_m": d * frac if np.isfinite(d) else np.nan,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[info] empty result.")
        return
    agg = out.groupby(["link_id","rel_window_idx","driver_id"], as_index=False).agg(
        total_distance_m=("part_distance_m","sum"),
        total_duration_s=("part_duration_s","sum"),
        count_segments=("part_duration_s","count"),
    )
    agg["avg_speed_mps"] = agg["total_distance_m"] / agg["total_duration_s"].replace(0, np.nan)
    agg = agg.sort_values(["driver_id","rel_window_idx","link_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    agg.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8")
    print(f"[ok] {len(agg)} rows -> {OUT}")
    print(agg.head())

if __name__ == "__main__":
    main()
