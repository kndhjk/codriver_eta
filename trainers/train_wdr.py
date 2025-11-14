# trainers/train_wdr.py  (fixed: use string keys for link_id to avoid float precision loss)
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.wdr import SimpleWDR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def canon_link_id_series(s: pd.Series) -> pd.Series:
    """
    Canonicalize link_id column to string without precision loss.
    Works whether source is UInt64, Int64, object, or float64 (with potential NaN).
    """
    # 优先尝试无符号整型
    try:
        # to_numeric 再转成 pandas 的可空 UInt64，防止精度丢失
        s2 = pd.to_numeric(s, errors="coerce").astype("UInt64")
        return s2.astype(str)
    except Exception:
        # 兜底：直接转字符串
        return s.astype(str)

def canon_time_series(s: pd.Series) -> pd.Series:
    # 确保窗口索引是 int64；NaN -> -1（后面会被编码器覆盖）
    x = pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int64)
    return x

class LinkWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, link2idx: dict, time2idx: dict, target_col="avg_speed_mps"):
        self.df = df.reset_index(drop=True)
        self.link2idx = link2idx
        self.time2idx = time2idx
        self.target_col = target_col

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        link_idx = self.link2idx[r["link_id_str"]]
        time_idx = self.time2idx[int(r["rel_window_idx"])]
        y = float(r[self.target_col])
        return np.int64(link_idx), np.int64(time_idx), np.float32(y)

def build_indexers(df_train, df_val, df_test):
    # link_id 映射用字符串，避免 64 位整数在 float64 中精度丢失
    link_ids = pd.concat([
        df_train["link_id_str"], df_val["link_id_str"], df_test["link_id_str"]
    ]).unique()
    link2idx = {str(x): i for i, x in enumerate(sorted(link_ids))}

    times = pd.concat([
        df_train["rel_window_idx"], df_val["rel_window_idx"], df_test["rel_window_idx"]
    ]).unique()
    time2idx = {int(x): i for i, x in enumerate(sorted(times))}
    return link2idx, time2idx

def mape(pred, target, eps=1e-6):
    return torch.mean(torch.abs((pred - target) / torch.clamp(target.abs(), min=eps)))

def train_one_epoch(model, loader, optim):
    model.train()
    tot_mae = tot_mape = n = 0
    for link_idx, time_idx, y in loader:
        link_idx = torch.from_numpy(np.asarray(link_idx)).to(DEVICE)
        time_idx = torch.from_numpy(np.asarray(time_idx)).to(DEVICE)
        y = torch.from_numpy(np.asarray(y)).to(DEVICE)

        optim.zero_grad()
        yhat = model(link_idx, time_idx)
        loss = torch.nn.functional.l1_loss(yhat, y)  # MAE
        loss.backward()
        optim.step()

        with torch.no_grad():
            tot_mae  += torch.nn.functional.l1_loss(yhat, y, reduction="sum").item()
            tot_mape += mape(yhat, y).item() * len(y)
            n += len(y)
    return tot_mae / n, tot_mape / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_mae = tot_mape = n = 0
    for link_idx, time_idx, y in loader:
        link_idx = torch.from_numpy(np.asarray(link_idx)).to(DEVICE)
        time_idx = torch.from_numpy(np.asarray(time_idx)).to(DEVICE)
        y = torch.from_numpy(np.asarray(y)).to(DEVICE)
        yhat = model(link_idx, time_idx)
        tot_mae  += torch.nn.functional.l1_loss(yhat, y, reduction="sum").item()
        tot_mape += mape(yhat, y).item() * len(y)
        n += len(y)
    return tot_mae / n, tot_mape / n

def main():
    # 1) 读数据（用 parquet，避免 CSV 的类型波动）
    train_path = DATASETS_DIR / "link_speed_train.parquet"
    val_path   = DATASETS_DIR / "link_speed_val.parquet"
    test_path  = DATASETS_DIR / "link_speed_test.parquet"
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_test  = pd.read_parquet(test_path)

    # 2) 规范列类型
    for df in (df_train, df_val, df_test):
        df["link_id_str"]    = canon_link_id_series(df["link_id"])
        df["rel_window_idx"] = canon_time_series(df["rel_window_idx"])
        # 目标列也确保 float
        df["avg_speed_mps"]  = pd.to_numeric(df["avg_speed_mps"], errors="coerce").astype(float)

    # 3) 编码器
    link2idx, time2idx = build_indexers(df_train, df_val, df_test)
    n_links, n_times = len(link2idx), len(time2idx)

    # 4) Dataset/DataLoader
    ds_train = LinkWindowDataset(df_train, link2idx, time2idx)
    ds_val   = LinkWindowDataset(df_val,   link2idx, time2idx)
    ds_test  = LinkWindowDataset(df_test,  link2idx, time2idx)

    dl_train = DataLoader(ds_train, batch_size=64,  shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=256, shuffle=False, num_workers=0)
    dl_test  = DataLoader(ds_test,  batch_size=256, shuffle=False, num_workers=0)

    # 5) 模型
    from models.wdr import SimpleWDR
    model = SimpleWDR(n_links=n_links, n_times=n_times,
                      emb_dim_link=32, emb_dim_time=8,
                      mlp_hidden=(128,64), wide_cross=True).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)

    best_val = float("inf")
    ckpt_path = OUT_DIR / "wdr_simple.pt"

    # 6) 训练
    for epoch in range(1, 51):
        tr_mae, tr_mape = train_one_epoch(model, dl_train, optim)
        va_mae, va_mape = evaluate(model, dl_val)
        print(f"[epoch {epoch:02d}] train MAE={tr_mae:.4f} MAPE={tr_mape:.4f} | val MAE={va_mae:.4f} MAPE={va_mape:.4f}")
        if va_mae < best_val:
            best_val = va_mae
            torch.save({
                "model": model.state_dict(),
                "link2idx": link2idx,
                "time2idx": time2idx,
            }, ckpt_path)
            print(f"[ckpt] saved -> {ckpt_path}")

    # 7) 测试
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        te_mae, te_mape = evaluate(model, dl_test)
        print(f"[test] MAE={te_mae:.4f}  MAPE={te_mape:.4f}")

if __name__ == "__main__":
    main()
