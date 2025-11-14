# trainers/train_codriver.py  (覆盖原文件，支持 --beta/--alpha/--epochs)
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.codriver import CoDriverSimple

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def canon_str(s):
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    except Exception:
        return s.astype(str)
def canon_int(s):
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int64)

class LWDataset(Dataset):
    def __init__(self, df, link2idx, time2idx, drv2idx, target="avg_speed_mps"):
        self.df=df.reset_index(drop=True)
        self.link2idx=link2idx; self.time2idx=time2idx; self.drv2idx=drv2idx
        self.target=target
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        return (
            np.int64(self.link2idx[r["link_id_str"]]),
            np.int64(self.time2idx[int(r["rel_window_idx"])]),
            np.int64(self.drv2idx[r["driver_id_str"]]),
            np.float32(r[self.target]),
        )

def build_indexers(df):
    link2idx = {s:i for i,s in enumerate(sorted(df["link_id_str"].unique()))}
    time2idx = {int(t):i for i,t in enumerate(sorted(df["rel_window_idx"].unique()))}
    drv2idx  = {s:i for i,s in enumerate(sorted(df["driver_id_str"].unique()))}
    return link2idx,time2idx,drv2idx

def split_random(df, ratios=(0.8,0.1,0.1), seed=42):
    rng = np.random.default_rng(seed)
    keys = df[["link_id_str","rel_window_idx","driver_id_str"]].drop_duplicates().reset_index(drop=True)
    idx = np.arange(len(keys)); rng.shuffle(idx)
    n=len(idx); ntr=int(n*ratios[0]); nval=int(n*ratios[1])
    ktr=keys.iloc[idx[:ntr]]; kval=keys.iloc[idx[ntr:ntr+nval]]; kte=keys.iloc[idx[ntr+nval:]]
    def sel(k): return df.merge(k, on=["link_id_str","rel_window_idx","driver_id_str"], how="inner")
    return sel(ktr), sel(kval), sel(kte)

def triplet_loss(emb, anchor_idx, pos_idx, neg_idx, margin=0.01, l2norm=True):
    a = emb[anchor_idx]; p = emb[pos_idx]; n = emb[neg_idx]
    if l2norm:
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        p = torch.nn.functional.normalize(p, p=2, dim=-1)
        n = torch.nn.functional.normalize(n, p=2, dim=-1)
    d_ap = torch.sum((a-p)**2, dim=-1)
    d_an = torch.sum((a-n)**2, dim=-1)
    return torch.relu(d_ap - d_an + margin).mean()

def sample_triplets(drv_bin, batch_driver_idx, drv2bin_t, rng, n_triplets=128):
    bins = {b: torch.where(drv2bin_t == b)[0].cpu().numpy() for b in [0,1,2]}
    anchors, positives, negatives = [], [], []
    batch_drivers = batch_driver_idx.cpu().numpy()
    for _ in range(n_triplets):
        if len(batch_drivers)==0: break
        a = rng.choice(batch_drivers)
        b = int(drv_bin[a])
        pool_p = bins.get(b, np.array([], dtype=np.int64))
        if len(pool_p)==0: continue
        p = int(rng.choice(pool_p))
        b_neg = 2 if b==0 else (0 if b==2 else rng.choice([0,2]))
        pool_n = bins.get(b_neg, np.array([], dtype=np.int64))
        if len(pool_n)==0:
            other = [bb for bb in [0,1,2] if bb!=b and len(bins[bb])>0]
            if not other: continue
            b_neg = int(rng.choice(other)); pool_n = bins[b_neg]
        n = int(rng.choice(pool_n))
        anchors.append(a); positives.append(p); negatives.append(n)
    if not anchors: return None
    device = drv2bin_t.device
    return (torch.tensor(anchors, dtype=torch.long, device=device),
            torch.tensor(positives, dtype=torch.long, device=device),
            torch.tensor(negatives, dtype=torch.long, device=device))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01, help="triplet margin")
    ap.add_argument("--beta",  type=float, default=0.45, help="aux weight")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ckpt", type=str, default="codriver_simple.pt")
    args = ap.parse_args()

    lw_path = DATASETS / "link_speed_driver_10min.parquet"
    drv_path = DATASETS / "drivers.parquet"
    df = pd.read_parquet(lw_path)
    drivers = pd.read_parquet(drv_path)

    df["link_id_str"] = canon_str(df["link_id"])
    df["driver_id_str"] = canon_str(df["driver_id"])
    df["rel_window_idx"] = canon_int(df["rel_window_idx"])
    df["avg_speed_mps"] = pd.to_numeric(df["avg_speed_mps"], errors="coerce").astype(float)

    if "driver_id_str" not in drivers.columns:
        drivers["driver_id_str"] = canon_str(drivers["driver_id"] if "driver_id" in drivers.columns else drivers.index)
    if "speed_bin" not in drivers.columns:
        raise ValueError("drivers.parquet 缺少 speed_bin，请先运行 05_driver_stats.py")

    df_tr, df_va, df_te = split_random(df)

    link2idx, time2idx, drv2idx = build_indexers(pd.concat([df_tr, df_va, df_te], ignore_index=True))
    n_links, n_times, n_drivers = len(link2idx), len(time2idx), len(drv2idx)

    ds_tr = LWDataset(df_tr, link2idx, time2idx, drv2idx)
    ds_va = LWDataset(df_va, link2idx, time2idx, drv2idx)
    ds_te = LWDataset(df_te, link2idx, time2idx, drv2idx)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False)

    # driver bins aligned to drv2idx
    drv_bin = np.zeros(n_drivers, dtype=np.int64)
    for d_str, idx in drv2idx.items():
        row = drivers[drivers["driver_id_str"]==d_str]
        drv_bin[idx] = int(row["speed_bin"].iloc[0]) if len(row) else 1
    drv_bin_t = torch.tensor(drv_bin, dtype=torch.long, device=DEVICE)

    model = CoDriverSimple(n_links, n_times, n_drivers,
                           emb_dim_link=32, emb_dim_time=8, emb_dim_driver=16,
                           mlp_hidden=(128,64), wide_cross=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
    rng = np.random.default_rng(123)

    def run_epoch(loader, train=True):
        if train: model.train()
        else:     model.eval()
        tot_mae=tot_mape=n=0
        for link_idx, time_idx, drv_idx, y in loader:
            link_idx=torch.as_tensor(link_idx,device=DEVICE)
            time_idx=torch.as_tensor(time_idx,device=DEVICE)
            drv_idx =torch.as_tensor(drv_idx, device=DEVICE)
            y=torch.as_tensor(y,device=DEVICE)
            yhat=model(link_idx,time_idx,drv_idx)
            main_loss = torch.nn.functional.l1_loss(yhat,y)
            aux = torch.tensor(0.0, device=DEVICE)
            if train and args.beta>0:
                trip = sample_triplets(drv_bin, drv_idx, drv_bin_t, rng, n_triplets=min(128,len(drv_idx)))
                if trip is not None:
                    a,p,n_idx = trip
                    aux = triplet_loss(model.driver_emb.weight, a,p,n_idx, margin=args.alpha, l2norm=True)
            loss = (1-args.beta)*main_loss + args.beta*aux
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                tot_mae  += torch.nn.functional.l1_loss(yhat,y,reduction="sum").item()
                tot_mape += (torch.abs((yhat - y) / torch.clamp(y.abs(), min=1e-6))).sum().item()
                n += len(y)
        return tot_mae/n, tot_mape/n

    best=1e18; ckpt=OUT_DIR / args.ckpt
    for ep in range(1, args.epochs+1):
        tr_mae,tr_mape = run_epoch(dl_tr, train=True)
        va_mae,va_mape = run_epoch(dl_va, train=False)
        print(f"[epoch {ep:02d}] train MAE={tr_mae:.4f} MAPE={tr_mape:.4f} | val MAE={va_mae:.4f} MAPE={va_mape:.4f}")
        if va_mae<best:
            best=va_mae
            torch.save({"model":model.state_dict(),"link2idx":link2idx,"time2idx":time2idx,"drv2idx":drv2idx}, ckpt)
            print(f"[ckpt] saved -> {ckpt}")

    # test
    state=torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"])
    te_mae, te_mape = run_epoch(dl_te, train=False)
    print(f"[test] MAE={te_mae:.4f} MAPE={te_mape:.4f}")

if __name__ == "__main__":
    main()
