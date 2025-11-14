# -*- coding: utf-8 -*-
# 从 tdrive_points.parquet 读取标准列，过滤北京范围，可选只取一周，导出 CSV: lon,lat,t(ISO)

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "datasets/tdrive/tdrive_points.parquet"
OUT = ROOT / "data_raw/tdrive/tdrive_oneweek_extracted.csv"

# 北京大致经纬范围（稍放宽）
LON_MIN, LON_MAX = 114.0, 118.0
LAT_MIN, LAT_MAX =  38.0,  42.0

ONLY_ONE_WEEK = True  # 如需全量导出，设为 False

def main():
    if not SRC.exists():
        print(f"[err] not found: {SRC.resolve()}")
        return

    # 读取 parquet
    df = pd.read_parquet(SRC, columns=["taxi_id","timestamp","lon","lat"])
    # 基本清洗
    df = df.dropna(subset=["timestamp","lon","lat"])
    # 过滤北京范围
    mask = (
        df["lon"].between(LON_MIN, LON_MAX) &
        df["lat"].between(LAT_MIN, LAT_MAX)
    )
    df = df[mask].copy()

    # 只取最早起始日的一周（可选）
    if ONLY_ONE_WEEK:
        t0 = df["timestamp"].min().normalize()  # 当天 00:00
        t1 = t0 + pd.Timedelta(days=7)
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]

    # 导出为 lon,lat,t（ISO 字符串）
    out_df = pd.DataFrame({
        "lon": df["lon"].astype(np.float64),
        "lat": df["lat"].astype(np.float64),
        "t":   df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"[ok] saved -> {OUT.resolve()}  rows={len(out_df)}")

if __name__ == "__main__":
    main()
