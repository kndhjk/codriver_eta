# evals/eval_eta_simple.py
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from models.wdr import SimpleWDR  # 用到模型只为加载 -> 预测 link×time 的速度表

WINDOW_SEC = 600         # 10min
WEEK_SLOTS = 7*24*6      # 1008

def _hash_start_idx(trip_id: str) -> int:
    h = hashlib.md5(trip_id.encode("utf-8")).digest()
    return int.from_bytes(h[:2], "big") % WEEK_SLOTS  # 0..1007

def load_ckpt(ckpt_path, n_links, n_times, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SimpleWDR(n_links=n_links, n_times=n_times)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--links", type=str, required=True)
    ap.add_argument("--trip_links", type=str, required=True)
    ap.add_argument("--speed_10min", type=str, required=True)
    ap.add_argument("--out_pred_speed", type=str, required=True)
    ap.add_argument("--out_eta_table", type=str, required=True)
    args = ap.parse_args()

    links = pd.read_parquet(args.links)
    tl = pd.read_parquet(args.trip_links)
    speed_grid = pd.read_parquet(args.speed_10min)

    # --- 0) 规范列名 & 基础兜底 ---
    if "duration_s" not in tl.columns:
        # 极端兜底：如果没有 duration_s，就用 links 的均速和长度推回（不推荐，但可跑通）
        tl = tl.merge(links[["link_id","length_m_mean","speed_mps_mean"]],
                      on="link_id", how="left")
        tl["duration_s"] = tl["duration_s"].fillna(tl["length_m_mean"]/tl["speed_mps_mean"].replace(0, np.nan))
        tl["duration_s"] = tl["duration_s"].fillna(tl["duration_s"].median())
    tl["duration_s"] = tl["duration_s"].fillna(0).astype(float)
    tl["seq"] = tl["seq"].astype(int)

    # --- 1) 用累计时长推进时间窗（关键改动）---
    tl = tl.sort_values(["trip_id","seq"]).copy()
    # 每个 trip 的起始窗用 trip_id 的 hash 来打散
    start_idx = tl[["trip_id"]].drop_duplicates().copy()
    start_idx["start_bin"] = start_idx["trip_id"].apply(_hash_start_idx)
    tl = tl.merge(start_idx, on="trip_id", how="left")

    tl["cum_s"] = tl.groupby("trip_id")["duration_s"].cumsum().fillna(0)
    tl["rel_window_idx"] = (tl["start_bin"] + (tl["cum_s"] // WINDOW_SEC).astype(int)) % WEEK_SLOTS
    tl["rel_window_idx"] = tl["rel_window_idx"].astype(int)

    # --- 2) 预测速度表（或者直接用 speed_grid）---
    # 这里直接用已训练好的 speed_grid（你之前已保存过）
    grid = speed_grid.rename(columns={"avg_speed_mps":"pred_speed_mps"}).copy()
    grid["rel_window_idx"] = grid["rel_window_idx"].astype(int)

    # --- 3) 合并预测速度 ---
    tl = tl.merge(grid[["link_id","rel_window_idx","pred_speed_mps"]],
                  on=["link_id","rel_window_idx"], how="left")

    # --- 4) 回填策略：先用 per-link 均速，再用全局中位数 ---
    per_link_speed = links[["link_id","speed_mps_mean"]].copy()
    tl = tl.merge(per_link_speed, on="link_id", how="left")
    n_total = len(tl)
    n_nan_before = tl["pred_speed_mps"].isna().sum()

    tl["pred_speed_mps"] = tl["pred_speed_mps"].fillna(tl["speed_mps_mean"])
    global_med = tl["pred_speed_mps"].median(skipna=True)
    tl["pred_speed_mps"] = tl["pred_speed_mps"].fillna(global_med)

    n_nan_after = tl["pred_speed_mps"].isna().sum()
    print(f"[merge] pred_speed NaN before: {n_nan_before}/{n_total}  after: {n_nan_after}/{n_total}")

    # --- 5) 导出 per-(link, timebin) 的预测（可用于可视化/调参）---
    Path(args.out_pred_speed).parent.mkdir(parents=True, exist_ok=True)
    out_grid = tl.groupby(["link_id","rel_window_idx"], as_index=False)["pred_speed_mps"].median()
    out_grid.to_parquet(args.out_pred_speed, index=False)
    print(f"[ok] saved pred speeds -> {args.out_pred_speed}")

    # --- 6) 计算 ETA：sum(length / speed) ---
    # 用 links 的 length；若缺，回填 median
    tl = tl.merge(links[["link_id","length_m_mean"]], on="link_id", how="left")
    length_med = tl["length_m_mean"].median(skipna=True)
    tl["length_m_mean"] = tl["length_m_mean"].fillna(length_med)

    tl["eta_s"] = tl["length_m_mean"] / tl["pred_speed_mps"].replace(0, np.nan)
    tl["eta_s"] = tl["eta_s"].fillna(tl["duration_s"])  # 兜底再兜底

    # 对每个 trip 求和得到整段 ETA 预测
    # 同时需要实际 “真值 ETA”：这里用 sum(duration_s) 当近似（没有真实任务标注时）
    eta = tl.groupby("trip_id").agg(
        pred_eta_s=("eta_s","sum"),
        gt_eta_s=("duration_s","sum")
    ).reset_index()

    eta["ae"] = (eta["pred_eta_s"] - eta["gt_eta_s"]).abs()
    # 避免除 0：若 gt 很小则忽略或加一个小 epsilon
    eps = 1.0
    eta["ape"] = eta["ae"] / eta["gt_eta_s"].clip(lower=eps)

    mae = eta["ae"].mean()
    mape = (eta["ape"].mean() * 100.0)
    print(f"[ETA overall] N = {len(eta)} MAE(s) = {mae:.2f} MAPE(%) = {mape:.2f}")

    Path(args.out_eta_table).parent.mkdir(parents=True, exist_ok=True)
    eta.to_csv(args.out_eta_table, index=False, encoding="utf-8")
    print(f"[ok] saved ETA table -> {args.out_eta_table}")

if __name__ == "__main__":
    main()
