# evals/eval_sparse_dense.py  (覆盖原文件)
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.codriver import CoDriverSimple

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
CKPT = ROOT / "outputs" / "codriver_simple.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def canon_str(s):
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    except Exception:
        return s.astype(str)
def canon_int(s):
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int64)

class LWDataset(Dataset):
    def __init__(self, df, link2idx, time2idx, drv2idx):
        self.df=df.reset_index(drop=True)
        self.link2idx=link2idx; self.time2idx=time2idx; self.drv2idx=drv2idx
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        x=(np.int64(self.link2idx[r["link_id_str"]]),
           np.int64(self.time2idx[int(r["rel_window_idx"])]),
           np.int64(self.drv2idx[r["driver_id_str"]]))
        y=np.float32(r["avg_speed_mps"])
        return x,y

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_mae=tot_mape=n=0
    for (link_idx,time_idx,drv_idx), y in loader:
        link_idx=torch.as_tensor(link_idx,device=DEVICE)
        time_idx=torch.as_tensor(time_idx,device=DEVICE)
        drv_idx=torch.as_tensor(drv_idx,device=DEVICE)
        y=torch.as_tensor(y,device=DEVICE)
        yhat=model(link_idx,time_idx,drv_idx)
        tot_mae  += torch.nn.functional.l1_loss(yhat,y,reduction="sum").item()
        tot_mape += (torch.abs((yhat-y)/torch.clamp(y.abs(),min=1e-6))).sum().item()
        n+=len(y)
    return tot_mae/n, tot_mape/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsparse", type=int, default=5, help="订单数 ≤nsparse 记为 sparse")
    ap.add_argument("--ndense",  type=int, default=10, help="订单数 ≥ndense 记为 dense")
    ap.add_argument("--suffix",  type=str, default="", help="结果文件名后缀，如 _beta045")
    args = ap.parse_args()

    lw = pd.read_parquet(DATASETS / "link_speed_driver_10min.parquet")
    tl = pd.read_parquet(DATASETS / "trip_links.parquet")

    lw["link_id_str"]=canon_str(lw["link_id"])
    lw["driver_id_str"]=canon_str(lw["driver_id"])
    lw["rel_window_idx"]=canon_int(lw["rel_window_idx"])
    lw["avg_speed_mps"]=pd.to_numeric(lw["avg_speed_mps"],errors="coerce").astype(float)

    order_cnt = tl.groupby("driver_id")["trip_id"].nunique().reset_index(name="n_orders")
    order_cnt["driver_id_str"] = canon_str(order_cnt["driver_id"])
    lw = lw.merge(order_cnt[["driver_id_str","n_orders"]], on="driver_id_str", how="left").fillna({"n_orders":0})

    keys = lw[["link_id_str","rel_window_idx","driver_id_str"]].drop_duplicates().sample(frac=1.0, random_state=42)
    n=len(keys); ntr=int(n*0.8); nva=int(n*0.1)
    te_keys = keys.iloc[ntr+nva:]
    df_te = lw.merge(te_keys, on=["link_id_str","rel_window_idx","driver_id_str"], how="inner")

    state = torch.load(CKPT, map_location=DEVICE)
    link2idx=state["link2idx"]; time2idx=state["time2idx"]; drv2idx=state["drv2idx"]
    model = CoDriverSimple(n_links=len(link2idx), n_times=len(time2idx), n_drivers=len(drv2idx)).to(DEVICE)
    model.load_state_dict(state["model"])

    N_SPARSE = args.nsparse
    N_DENSE  = args.ndense

    buckets = {
        f"sparse(≤{N_SPARSE})": df_te[df_te["n_orders"]<=N_SPARSE],
        f"dense(≥{N_DENSE})":   df_te[df_te["n_orders"]>=N_DENSE],
        "all": df_te
    }

    rows=[]
    for name, sub in buckets.items():
        if len(sub)==0:
            rows.append({"bucket":name, "N":0, "MAE":np.nan, "MAPE":np.nan})
            print(name, 0, "nan", "nan")
            continue
        dl = DataLoader(LWDataset(sub, link2idx, time2idx, drv2idx), batch_size=1024, shuffle=False)
        mae,mape = evaluate(model, dl)
        rows.append({"bucket":name, "N":len(sub), "MAE":mae, "MAPE":mape})
        print(name, len(sub), mae, mape)

    out = pd.DataFrame(rows).sort_values("bucket")
    out_path = ROOT / "outputs" / "tables" / f"sparse_dense{args.suffix}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[ok] -> {out_path}")

if __name__ == "__main__":
    main()
