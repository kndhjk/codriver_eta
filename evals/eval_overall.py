# evals/eval_overall.py  (可传参后缀)
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.codriver import CoDriverSimple

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)
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
        self.df = df.reset_index(drop=True)
        self.link2idx=link2idx; self.time2idx=time2idx; self.drv2idx=drv2idx
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r = self.df.iloc[i]
        x = (np.int64(self.link2idx[r["link_id_str"]]),
             np.int64(self.time2idx[int(r["rel_window_idx"])]),
             np.int64(self.drv2idx[r["driver_id_str"]]))
        y = np.float32(r["avg_speed_mps"])
        return x, y

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_mae = tot_mape = n = 0
    for (link_idx, time_idx, drv_idx), y in loader:
        link_idx = torch.as_tensor(link_idx, device=DEVICE)
        time_idx = torch.as_tensor(time_idx, device=DEVICE)
        drv_idx  = torch.as_tensor(drv_idx,  device=DEVICE)
        y        = torch.as_tensor(y,        device=DEVICE)
        yhat = model(link_idx, time_idx, drv_idx)
        tot_mae  += torch.nn.functional.l1_loss(yhat, y, reduction="sum").item()
        tot_mape += (torch.abs((yhat - y) / torch.clamp(y.abs(), min=1e-6))).sum().item()
        n += len(y)
    return tot_mae/n, tot_mape/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", type=str, default="", help="文件名后缀，如 _beta045")
    args = ap.parse_args()

    lw_path = DATASETS / "link_speed_driver_10min.parquet"
    df = pd.read_parquet(lw_path)
    df["link_id_str"] = canon_str(df["link_id"])
    df["driver_id_str"] = canon_str(df["driver_id"])
    df["rel_window_idx"] = canon_int(df["rel_window_idx"])
    df["avg_speed_mps"] = pd.to_numeric(df["avg_speed_mps"], errors="coerce").astype(float)

    keys = df[["link_id_str","rel_window_idx","driver_id_str"]].drop_duplicates().sample(frac=1.0, random_state=42)
    n=len(keys); ntr=int(n*0.8); nva=int(n*0.1)
    te_keys = keys.iloc[ntr+nva:]
    df_te = df.merge(te_keys, on=["link_id_str","rel_window_idx","driver_id_str"], how="inner")

    state = torch.load(CKPT, map_location=DEVICE)
    link2idx = state["link2idx"]; time2idx = state["time2idx"]; drv2idx = state["drv2idx"]
    model = CoDriverSimple(n_links=len(link2idx), n_times=len(time2idx), n_drivers=len(drv2idx)).to(DEVICE)
    model.load_state_dict(state["model"])

    dl_te = DataLoader(LWDataset(df_te, link2idx, time2idx, drv2idx), batch_size=1024, shuffle=False)
    mae, mape = evaluate(model, dl_te)
    out = pd.DataFrame([{"model":"codriver_simple", "MAE":mae, "MAPE":mape, "N":len(df_te)}])
    out.to_csv(OUT_TAB / f"overall{args.suffix}.csv", index=False, encoding="utf-8")
    print(out)

if __name__ == "__main__":
    main()
