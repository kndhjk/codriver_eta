# trainers/train_wdr.py
import os, argparse, pickle, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.wdr import SimpleWDR

# ========= 新增：GPU 与性能开关 =========
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # Ampere/Lovelace 对 TF32 友好

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ========= 示例数据集（按你的数据结构改）=========
class SpeedDataset(Dataset):
    """
    传入已经编码好的 link_idx, time_idx, y (avg_speed_mps)
    """
    def __init__(self, link_idx, time_idx, y):
        self.link_idx = torch.as_tensor(link_idx, dtype=torch.long)
        self.time_idx = torch.as_tensor(time_idx, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.link_idx[i], self.time_idx[i], self.y[i]

def load_train_val_test(links_fp, speed10_fp, splits_fp):
    """
    这里仅演示流程：假设你已经根据 03/04 脚本生成了
    - links.parquet （含所有 link_id 的全集）
    - link_speed_10min.parquet （link_id, rel_window_idx, avg_speed_mps）
    - splits_trip.pkl （里面有 train/val/test 的 rel_window_idx 范围或掩码）
    你自己的实际编码方式照旧——只要最后能得到 (link_idx, time_idx, y) 即可。
    """
    links = pd.read_parquet(links_fp)
    speed = pd.read_parquet(speed10_fp)

    # 构建 link_id -> 索引
    link_ids = links["link_id"].unique()
    link_id2idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_id2idx)

    # 假设 rel_window_idx 已经是「离散时间桶」，直接压成 0..T-1
    t_ids = np.sort(speed["rel_window_idx"].unique())
    t_map = {t:i for i,t in enumerate(t_ids)}
    n_times = len(t_map)

    # 加载切分
    with open(splits_fp, "rb") as f:
        splits = pickle.load(f)  # 你自己的字典结构：包含 train/val/test 的选择条件
    # 简单示例：按 rel_window_idx 切（你也可以用 mask 或别的键）
    def _to_xy(df):
        lk = df["link_id"].map(link_id2idx).values
        tm = df["rel_window_idx"].map(t_map).values
        y  = df["avg_speed_mps"].values
        # 去掉 NaN/无效
        ok = np.isfinite(lk) & np.isfinite(tm) & np.isfinite(y)
        return lk[ok].astype(np.int64), tm[ok].astype(np.int64), y[ok].astype(np.float32)

    train_mask = speed["rel_window_idx"] < splits["val_start_idx"]  # 示例用（按 04_split_sets.py 的逻辑改）
    val_mask   = (speed["rel_window_idx"] >= splits["val_start_idx"]) & (speed["rel_window_idx"] < splits["test_start_idx"])
    test_mask  = speed["rel_window_idx"] >= splits["test_start_idx"]

    link_tr, time_tr, y_tr = _to_xy(speed.loc[train_mask])
    link_va, time_va, y_va = _to_xy(speed.loc[val_mask])
    link_te, time_te, y_te = _to_xy(speed.loc[test_mask])

    return (n_links, n_times, (link_tr,time_tr,y_tr), (link_va,time_va,y_va), (link_te,time_te,y_te))

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    cnt = 0
    for link_idx, time_idx, y in loader:
        link_idx = link_idx.to(device, non_blocking=True)
        time_idx = time_idx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ========= 关键：AMP 混合精度 =========
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=torch.float16):
            pred = model(link_idx, time_idx)
            loss = mse(pred, y)

        scaler.scale(loss).zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item() * y.size(0)
        cnt += y.size(0)
    return total_loss / max(1, cnt)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total = 0.0
    cnt = 0
    for link_idx, time_idx, y in loader:
        link_idx = link_idx.to(device, non_blocking=True)
        time_idx = time_idx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=torch.float16):
            pred = model(link_idx, time_idx)
            total += mse(pred, y).item()
            cnt += y.size(0)
    rmse = math.sqrt(total / max(1, cnt))
    return rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True)
    ap.add_argument("--speed_10min", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=131072)  # 4070 很能装，按显存再调
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--emb_dim_link", type=int, default=32)
    ap.add_argument("--emb_dim_time", type=int, default=8)
    ap.add_argument("--wide_cross", type=lambda s: s.lower()!="false", default=True)
    args = ap.parse_args()

    device = get_device()
    print("Using device:", device)

    n_links, n_times, tr, va, te = load_train_val_test(args.links, args.speed_10min, args.splits)
    print(f"n_links={n_links}, n_times={n_times}")

    # ⚠️ 注意：如果 n_links*n_times 很大，wide_cross 会占显存，必要时关掉
    model = SimpleWDR(
        n_links=n_links,
        n_times=n_times,
        emb_dim_link=args.emb_dim_link,
        emb_dim_time=args.emb_dim_time,
        wide_cross=args.wide_cross
    ).to(device)

    # DataLoader：pin_memory + 非阻塞搬运，加速主机->GPU
    train_ds = SpeedDataset(*tr)
    val_ds   = SpeedDataset(*va)
    test_ds  = SpeedDataset(*te)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device)
        val_rmse = evaluate(model, val_loader, device)
        print(f"[epoch {ep}] train_loss={tr_loss:.5f}  val_RMSE={val_rmse:.5f}")

        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(),
                        "n_links": n_links, "n_times": n_times,
                        "emb_dim_link": args.emb_dim_link,
                        "emb_dim_time": args.emb_dim_time,
                        "wide_cross": args.wide_cross},
                       os.path.join(args.out_dir, "best.pt"))

    test_rmse = evaluate(model, test_loader, device)
    print(f"[test] RMSE={test_rmse:.5f}")

if __name__ == "__main__":
    main()
