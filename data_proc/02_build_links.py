# data_proc/02_build_links.py
# 从 datasets/matched_legs.parquet 构建：
# 1) 唯一路段表 (links.parquet)
# 2) 行程-路段序列表 (trip_links.parquet)

import os
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUT_LINKS = DATASETS_DIR / "links.parquet"
OUT_TRIP_LINKS = DATASETS_DIR / "trip_links.parquet"
SRC_MATCHED = DATASETS_DIR / "matched_legs.parquet"

def stable_uint64_from_pair(u: int, v: int) -> np.uint64:
    """用 MD5 的低 8 字节生成稳定 uint64 link_id"""
    s = f"{u}->{v}".encode("utf-8")
    h = hashlib.md5(s).digest()  # 16 bytes
    low8 = h[-8:]                 # 8 bytes
    return np.frombuffer(low8, dtype=">u8")[0]  # big-endian 转 uint64

def build_links(df: pd.DataFrame) -> pd.DataFrame:
    # 基本清洗
    need_cols = ["u_node","v_node","distance_m","duration_s","speed_mps"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 生成 link_id 与 key
    df["link_key"] = df["u_node"].astype(str) + "->" + df["v_node"].astype(str)
    df["link_id"]  = df.apply(lambda r: stable_uint64_from_pair(int(r["u_node"]), int(r["v_node"])), axis=1)

    # 统计（对异常值更稳：长度用中位数）
    agg = df.groupby(["link_id","link_key","u_node","v_node"], as_index=False).agg(
        count=("link_id","size"),
        length_m_median=("distance_m","median"),
        length_m_mean=("distance_m","mean"),
        duration_s_mean=("duration_s","mean"),
        speed_mps_mean=("speed_mps","mean"),
    )

    # 规范列顺序/类型
    cols = [
        "link_id","link_key","u_node","v_node",
        "count","length_m_median","length_m_mean",
        "duration_s_mean","speed_mps_mean"
    ]
    agg = agg[cols]
    # 单位提示：速度 m/s；需要 km/h 时 = *3.6
    return agg

def build_trip_links(df: pd.DataFrame, drop_consecutive_duplicates: bool = True) -> pd.DataFrame:
    # 生成 link_id 与 key
    df = df.copy()
    df["link_key"] = df["u_node"].astype(str) + "->" + df["v_node"].astype(str)
    df["link_id"]  = df.apply(lambda r: stable_uint64_from_pair(int(r["u_node"]), int(r["v_node"])), axis=1)

    # 确保有排序锚点
    if "row_idx" not in df.columns:
        df["row_idx"] = np.arange(len(df))

    sort_cols = [c for c in ["trip_id","matching_idx","leg_idx","seg_idx","row_idx"] if c in df.columns]
    df = df.sort_values(sort_cols)

    # 可选：去掉同一 trip 内的“连续重复 link”
    if drop_consecutive_duplicates:
        def _drop_run(g: pd.DataFrame) -> pd.DataFrame:
            keep = [True]
            for i in range(1, len(g)):
                keep.append(g["link_id"].iloc[i] != g["link_id"].iloc[i-1])
            return g.loc[keep]
        df = df.groupby("trip_id", group_keys=False).apply(_drop_run)

    # 为每个 trip 编序号
    df["seq"] = df.groupby("trip_id").cumcount()

    # 选择输出列
    out_cols = [
        "trip_id","driver_id","seq",
        "link_id","link_key","u_node","v_node",
        "distance_m","duration_s","speed_mps"
    ]
    # 缺失列填补
    for c in out_cols:
        if c not in df.columns:
            df[c] = None
    df_out = df[out_cols].reset_index(drop=True)
    return df_out

def main():
    if not SRC_MATCHED.exists():
        raise FileNotFoundError(f"Not found: {SRC_MATCHED}")

    df = pd.read_parquet(SRC_MATCHED)
    # 基本校验
    must = ["trip_id","driver_id","u_node","v_node"]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in matched_legs: {missing}")

    # 1) links 表
    links = build_links(df)
    links.to_parquet(OUT_LINKS, index=False)
    # 同时给个 csv 方便查看
    links_csv = OUT_LINKS.with_suffix(".csv")
    links.to_csv(links_csv, index=False, encoding="utf-8")
    print(f"[ok] links: {len(links)} rows -> {OUT_LINKS} / {links_csv}")

    # 2) trip_links 表
    trip_links = build_trip_links(df, drop_consecutive_duplicates=True)
    trip_links.to_parquet(OUT_TRIP_LINKS, index=False)
    trip_links_csv = OUT_TRIP_LINKS.with_suffix(".csv")
    trip_links.to_csv(trip_links_csv, index=False, encoding="utf-8")
    print(f"[ok] trip_links: {len(trip_links)} rows -> {OUT_TRIP_LINKS} / {trip_links_csv}")

    # 打印几个样例
    print("\n[links head]")
    print(links.head())
    print("\n[trip_links head]")
    print(trip_links.head())

if __name__ == "__main__":
    main()
