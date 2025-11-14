# data_proc/04_split_sets.py
# 将 datasets/link_speed_10min.parquet 切成 train/val/test
# 优先按 rel_window_idx 做时间切分；若窗口种类不足则退化为随机切分。

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
SRC = DATASETS_DIR / "link_speed_10min.parquet"

OUT_TRAIN = DATASETS_DIR / "link_speed_train.parquet"
OUT_VAL   = DATASETS_DIR / "link_speed_val.parquet"
OUT_TEST  = DATASETS_DIR / "link_speed_test.parquet"

RATIOS: Tuple[float,float,float] = (0.8, 0.1, 0.1)  # train/val/test
SEED = 42

def time_split(df: pd.DataFrame, ratios=RATIOS):
    # 按 rel_window_idx（相对窗口）做“时间”切分
    uniq = np.sort(df["rel_window_idx"].unique())
    if len(uniq) < 3:
        return None  # 窗口值太少，不适合时间切分

    n = len(uniq)
    n_train = max(1, int(n * ratios[0]))
    n_val   = max(1, int(n * ratios[1]))
    # 保证三者覆盖全部窗口
    n_test  = max(1, n - n_train - n_val)

    edges_train = uniq[:n_train]
    edges_val   = uniq[n_train:n_train+n_val]
    edges_test  = uniq[n_train+n_val:]

    df_train = df[df["rel_window_idx"].isin(edges_train)]
    df_val   = df[df["rel_window_idx"].isin(edges_val)]
    df_test  = df[df["rel_window_idx"].isin(edges_test)]
    return df_train, df_val, df_test

def random_split(df: pd.DataFrame, ratios=RATIOS, seed=SEED):
    # 退化方案：随机切分（可按 link_id 分组随机）
    rng = np.random.default_rng(seed)
    # 以 (link_id, rel_window_idx) 为组，防止同一窗内的同一路段被拆到不同集合
    grp_keys = df[["link_id","rel_window_idx"]].drop_duplicates().reset_index(drop=True)
    n = len(grp_keys)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    train_keys = grp_keys.iloc[idx[:n_train]]
    val_keys   = grp_keys.iloc[idx[n_train:n_train+n_val]]
    test_keys  = grp_keys.iloc[idx[n_train+n_val:]]

    def _sel(keys):
        m = df.merge(keys, on=["link_id","rel_window_idx"], how="inner")
        return m

    return _sel(train_keys), _sel(val_keys), _sel(test_keys)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Not found: {SRC}")
    df = pd.read_parquet(SRC)

    # 基本检查
    need = ["link_id","rel_window_idx","total_distance_m","total_duration_s","avg_speed_mps"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns in {SRC.name}: {miss}")

    # 优先时间切分
    split = time_split(df)
    if split is None:
        print("[warn] too few distinct rel_window_idx; fallback to random split.")
        split = random_split(df)

    df_train, df_val, df_test = split

    for out_path, part in [(OUT_TRAIN, df_train),(OUT_VAL, df_val),(OUT_TEST, df_test)]:
        part = part.reset_index(drop=True)
        part.to_parquet(out_path, index=False)
        # 方便查看
        part.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8")
        print(f"[ok] {len(part):5d} rows -> {out_path}")

    print("\n[head train]")
    print(df_train.head())

if __name__ == "__main__":
    main()
