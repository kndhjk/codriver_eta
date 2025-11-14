# data_raw/combine_data.py
import argparse, os, glob, sys
import pandas as pd
from pathlib import Path

def read_one(path):
    df = pd.read_csv(
        path,
        header=None,
        names=["taxi_id", "timestamp", "lon", "lat"],
        usecols=[0, 1, 2, 3],
        dtype={"taxi_id": "int32", "lon": "float32", "lat": "float32"},
        skipinitialspace=True
    )

    df = df.drop_duplicates()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp","lon","lat"])
    # 粗过滤北京范围，去掉明显脏点（可注释）
    df = df[(df["lon"].between(115.5,117.5)) & (df["lat"].between(39.4,40.6))]
    return df

def main(src_dir, out_path, recursive=True):
    # 允许 .txt / .TXT；支持递归
    patterns = ["*.txt", "*.TXT"] if not recursive else ["**/*.txt", "**/*.TXT"]
    files = []
    for pat in patterns:
        files.extend(Path(src_dir).glob(pat))
    files = sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    if not files:
        print(f"No .txt found under {src_dir}")
        # 打印一下附近的目录帮助定位
        base = Path(src_dir)
        if base.exists():
            print("Children of src_dir:")
            for p in base.iterdir():
                print("  -", p)
        else:
            print("src_dir does NOT exist:", base)
        sys.exit(1)

    chunks = (read_one(str(p)) for p in files)
    df = pd.concat(chunks, ignore_index=True).sort_values(["taxi_id","timestamp"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path}  rows={len(df)}  taxis={df['taxi_id'].nunique()}")

if __name__ == "__main__":
    # 默认路径基于脚本文件位置（而不是当前工作目录）
    here = Path(__file__).resolve()
    default_src = here.parent / "tdrive" / "taxi_log_2008_by_id"
    default_out = here.parents[1] / "datasets" / "tdrive" / "tdrive_points.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(default_src))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--no-recursive", action="store_true", help="禁用递归搜索")
    args = parser.parse_args()

    main(args.src, args.out, recursive=not args.no_recursive)
