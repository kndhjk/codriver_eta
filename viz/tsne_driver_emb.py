# viz/tsne_driver_emb.py
# 可视化 CoDriver 司机嵌入的 t-SNE，并按 speed_bin(慢/中/快) 上色
# 用法（项目根目录执行）：
#   python -m viz.tsne_driver_emb --ckpt outputs/codriver_beta045.pt --out outputs/figs/tsne_beta045.png
#   # 也可一次画多张：
#   python -m viz.tsne_driver_emb --ckpt outputs/codriver_beta045.pt outputs/codriver_beta000.pt

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch

def load_ckpt(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu")
    # 1) 找到 driver embedding
    weight = None
    if isinstance(state, dict):
        # 常见：state["model"]["driver_emb.weight"]
        for k in ["driver_emb.weight", "module.driver_emb.weight"]:
            if isinstance(state.get("model", {}), dict) and k in state["model"]:
                weight = state["model"][k]
                break
        # 退而求其次：在顶层keys里搜
        if weight is None:
            for k, v in state.items():
                if isinstance(k, str) and k.endswith("driver_emb.weight"):
                    weight = v
                    break
    if weight is None:
        raise RuntimeError(f"无法在 {ckpt_path} 中找到 driver_emb.weight")

    W = weight.detach().cpu().numpy()
    # 2) 找到 drv2idx 映射
    drv2idx = None
    for key in ["drv2idx", ("meta","drv2idx")]:
        if isinstance(key, tuple):
            if key[0] in state and isinstance(state[key[0]], dict) and key[1] in state[key[0]]:
                drv2idx = state[key[0]][key[1]]
                break
        else:
            if key in state:
                drv2idx = state[key]
                break
    if drv2idx is None:
        # 有些保存的是 idx2drv
        if "idx2drv" in state:
            idx2drv = state["idx2drv"]
            drv2idx = {d:i for i,d in enumerate(idx2drv)}
        else:
            raise RuntimeError(f"无法在 {ckpt_path} 中找到 drv2idx / idx2drv")
    # 转成 str key，保证与 drivers.parquet 对齐
    drv2idx = {str(k): int(v) for k,v in drv2idx.items()}
    return W, drv2idx

def ensure_drivers_parquet(datasets_dir: Path) -> pd.DataFrame:
    drivers_pq = datasets_dir / "drivers.parquet"
    if drivers_pq.exists():
        df = pd.read_parquet(drivers_pq)
        # 规范列名
        if "driver_id_str" not in df.columns:
            # 尝试从其他列派生
            if "driver_id" in df.columns:
                df["driver_id_str"] = df["driver_id"].astype(str)
            else:
                raise RuntimeError("drivers.parquet 缺少 driver_id_str / driver_id")
        if "speed_bin" not in df.columns:
            # 没有 speed_bin 就用三分位重算
            if "avg_speed_mps_driver" not in df.columns:
                raise RuntimeError("drivers.parquet 缺少 speed_bin 和 avg_speed_mps_driver，无法上色")
            df = _add_speed_bin(df)
        return df[["driver_id_str","avg_speed_mps_driver","speed_bin"]]
    else:
        # 兜底：从 link_speed_driver_10min.parquet 计算
        lsd = datasets_dir / "link_speed_driver_10min.parquet"
        if not lsd.exists():
            raise RuntimeError("缺少 drivers.parquet 且缺少 link_speed_driver_10min.parquet，无法构建司机统计")
        d = pd.read_parquet(lsd)
        if "driver_id" not in d.columns or "avg_speed_mps" not in d.columns:
            raise RuntimeError("link_speed_driver_10min.parquet 需包含 driver_id, avg_speed_mps")
        tmp = (
            d.assign(driver_id_str=d["driver_id"].astype(str))
             .groupby("driver_id_str")["avg_speed_mps"].mean().reset_index(name="avg_speed_mps_driver")
        )
        tmp = _add_speed_bin(tmp)
        return tmp

def _add_speed_bin(drv_df: pd.DataFrame) -> pd.DataFrame:
    x = drv_df["avg_speed_mps_driver"].astype(float)
    if len(x) < 3:
        drv_df["speed_bin"] = 1
        return drv_df
    q1, q2 = x.quantile(1/3), x.quantile(2/3)
    def to_bin(v):
        if v <= q1: return 0
        if v >= q2: return 2
        return 1
    drv_df["speed_bin"] = x.apply(to_bin)
    return drv_df

def run_tsne(emb_subset: np.ndarray, seed: int = 42):
    # 首选 TSNE，不可用则退化到 PCA
    try:
        from sklearn.manifold import TSNE
        n = len(emb_subset)
        if n < 2:
            raise RuntimeError("参与可视化的司机不足 2 个。")
        # 合理设置 perplexity
        perf = max(5, min(30, n//3))
        X = TSNE(n_components=2, init="pca", learning_rate="auto",
                 perplexity=perf, random_state=seed).fit_transform(emb_subset)
        return X
    except Exception as e:
        from sklearn.decomposition import PCA
        X = PCA(n_components=2, random_state=seed).fit_transform(emb_subset)
        return X

def plot_scatter(X, color_bin, title: str, out_png: Path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap = mpl.cm.get_cmap("tab10", 3)
    labels = {0:"slow", 1:"medium", 2:"fast"}
    plt.figure(figsize=(6.4, 4.8), dpi=140)
    for b in [0,1,2]:
        idx = (color_bin == b)
        if idx.sum()==0: continue
        plt.scatter(X[idx,0], X[idx,1], s=16, alpha=0.85, label=labels[b], c=[cmap(b)])
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=1.2, frameon=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[ok] saved -> {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True, help="一个或多个模型 ckpt 路径（.pt）")
    parser.add_argument("--out", default=None, help="输出文件名（仅在单个 ckpt 时生效）")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DS = ROOT / "datasets"

    drivers = ensure_drivers_parquet(DS)  # driver_id_str, avg_speed_mps_driver, speed_bin

    for i, ck in enumerate(args.ckpt):
        ckpt_path = Path(ck)
        if not ckpt_path.exists():
            # 允许传相对 outputs/ 路径的文件名
            maybe = ROOT / ck
            if maybe.exists():
                ckpt_path = maybe
            else:
                raise SystemExit(f"ckpt 不存在: {ck}")
        print(f"[load] {ckpt_path}")
        W, drv2idx = load_ckpt(ckpt_path)

        # 和 embedding 对齐
        df = drivers.copy()
        df = df[df["driver_id_str"].isin(drv2idx.keys())].copy()
        if df.empty:
            raise RuntimeError("没有与模型匹配的 driver_id。检查保存的 drv2idx 是否与 drivers.parquet 一致。")
        df["emb_idx"] = df["driver_id_str"].map(drv2idx).astype(int)
        df = df.sort_values("emb_idx")
        # 边界保护
        keep = (df["emb_idx"] >= 0) & (df["emb_idx"] < W.shape[0])
        df = df[keep]
        emb = W[df["emb_idx"].to_numpy()]
        if len(df) < 2:
            raise RuntimeError("用于可视化的司机数量 < 2。")

        X = run_tsne(emb, seed=42)

        # 输出名
        if args.out and len(args.ckpt)==1:
            out_png = Path(args.out)
        else:
            out_name = f"tsne_{ckpt_path.stem}.png"
            out_png = ROOT / "outputs" / "figs" / out_name
        title = f"Driver embedding t-SNE ({ckpt_path.stem})"
        plot_scatter(X, df["speed_bin"].to_numpy(), title, out_png)

if __name__ == "__main__":
    main()
