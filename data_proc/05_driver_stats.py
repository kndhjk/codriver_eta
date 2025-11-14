# data_proc/05_driver_stats.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets"
INP = DS / "link_speed_driver_10min.parquet"
OUT = DS / "drivers.parquet"
OUT_CSV = DS / "drivers.csv"

def canon_str(s):
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    except Exception:
        return s.astype(str)

def main():
    df = pd.read_parquet(INP)
    # 兼容不同列名
    if "driver_id" not in df.columns:
        raise SystemExit("link_speed_driver_10min.parquet 缺少 driver_id 列，请确认 03 脚本已按包含 driver_id 的版本运行。")

    df["driver_id_str"] = canon_str(df["driver_id"])
    df["avg_speed_mps"] = pd.to_numeric(df["avg_speed_mps"], errors="coerce").astype(float)

    # 每个司机的平均速度（基于所有10min窗口）
    drv_stat = (
        df.groupby("driver_id_str")["avg_speed_mps"]
          .mean()
          .reset_index(name="avg_speed_mps_driver")
          .dropna()
    )
    if len(drv_stat) < 3:
        # 数据很少时，仍然给出中间分桶
        drv_stat["speed_bin"] = 1
    else:
        # 用三分位把司机分成：0=慢，1=中，2=快
        q1 = drv_stat["avg_speed_mps_driver"].quantile(1/3)
        q2 = drv_stat["avg_speed_mps_driver"].quantile(2/3)
        def to_bin(v):
            if v <= q1: return 0
            if v >= q2: return 2
            return 1
        drv_stat["speed_bin"] = drv_stat["avg_speed_mps_driver"].apply(to_bin)

    # 一些可用的统计信息（可扩展）
    drv_stat["n_windows"] = df.groupby("driver_id_str")["avg_speed_mps"].size().reindex(drv_stat["driver_id_str"]).values

    drv_stat.to_parquet(OUT, index=False)
    drv_stat.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[ok] drivers: {len(drv_stat)} rows -> {OUT} / {OUT_CSV}")
    print(drv_stat.head())

if __name__ == "__main__":
    main()
