# data_proc/01b_match_tdrive_batch.py
from pathlib import Path
import argparse, time, json
import pandas as pd
import numpy as np
import requests

BASE = "http://localhost:5000"

def osrm_match(points):
    coord_str = ";".join([f"{lon},{lat}" for lon,lat in points])
    params = {"geometries":"geojson", "annotations":"true", "overview":"false"}
    r = requests.get(f"{BASE}/match/v1/driving/{coord_str}", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        raise RuntimeError(js)
    return js

def parse_match(js, trip_id, driver_id):
    rows = []
    for mi, m in enumerate(js.get("matchings", [])):
        for li, leg in enumerate(m.get("legs", [])):
            ann = leg.get("annotation", {}) or {}
            nodes = ann.get("nodes", []) or []
            dist = ann.get("distance", []) or []
            dur  = ann.get("duration", []) or []
            speed= ann.get("speed", []) or []
            for k in range(len(nodes)-1):
                rows.append({
                    "trip_id": trip_id,
                    "driver_id": driver_id,
                    "matching_idx": mi,
                    "leg_idx": li,
                    "seg_idx": k,
                    "u_node": nodes[k],
                    "v_node": nodes[k+1],
                    "link_id": f"{nodes[k]}->{nodes[k+1]}",
                    "distance_m": float(dist[k]) if k < len(dist) else None,
                    "duration_s": float(dur[k])  if k < len(dur)  else None,
                    "speed_mps":  float(speed[k])if k < len(speed)else None,
                })
    return rows

def read_points_generic(f: Path, max_points=5000):
    # 尽量兼容 CSV/PLT 两列经纬度（经度在前、纬度在后）
    if f.suffix.lower()==".csv":
        df = pd.read_csv(f)
    else:
        df = pd.read_csv(f, header=None)
    cols = [str(c).lower() for c in df.columns]
    if "longitude" in cols and "latitude" in cols:
        lon = df.iloc[:, cols.index("longitude")]
        lat = df.iloc[:, cols.index("latitude")]
    else:
        lon = df.iloc[:,0]; lat = df.iloc[:,1]
    pts = list(zip(lon.astype(float).tolist(), lat.astype(float).tolist()))
    if len(pts) > max_points:
        step = max(1, len(pts)//max_points)
        pts = pts[::step]
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdir", default="data_raw/tdrive")
    ap.add_argument("--outshard", default="datasets/matched_parts")
    ap.add_argument("--chunk", type=int, default=120, help="每个 /match 请求的点数")
    ap.add_argument("--flush-every", type=int, default=50, help="累积多少片写一个分片文件")
    args = ap.parse_args()

    rawdir = Path(args.rawdir); outdir = Path(args.outshard)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in rawdir.rglob("*") if p.suffix.lower() in [".csv",".plt"]])
    if not files:
        print(f"[err] 未发现轨迹文件于 {rawdir}")
        return

    shard_rows = []
    shard_id = 0
    written = 0

    for fi, f in enumerate(files):
        driver = f.stem
        try:
            pts = read_points_generic(f)
        except Exception as e:
            print(f"[skip] 读入失败: {f} -> {e}")
            continue
        if len(pts) < 3:
            print(f"[skip] 点太少: {f}")
            continue

        CHUNK = max(20, args.chunk)
        part = 0
        for ci in range(0, len(pts), CHUNK):
            chunk = pts[ci:ci+CHUNK]
            if len(chunk) < 3: break
            trip_id = f"{driver}#part{part:04d}"
            part += 1
            # 已存在的分片检查（可选：根据 trip_id 去重；这里简单交给后续 groupby 去重）
            try:
                js = osrm_match(chunk)
                rows = parse_match(js, trip_id=trip_id, driver_id=driver)
                shard_rows.extend(rows)
                written += 1
                print(f"[ok] {f.name} {trip_id}: {len(rows)} segs (batch={written})")
                time.sleep(0.15)
            except Exception as e:
                print(f"[warn] match 失败: {f.name} {trip_id}: {e}")

            # 定期落地一个分片
            if written % args.flush_every == 0 and shard_rows:
                df = pd.DataFrame(shard_rows)
                shard_path = outdir / f"matched_part_{shard_id:05d}.parquet"
                df.to_parquet(shard_path, index=False)
                print(f"[save] {shard_path} ({len(df)} rows)")
                shard_rows.clear()
                shard_id += 1

    # 收尾
    if shard_rows:
        df = pd.DataFrame(shard_rows)
        shard_path = outdir / f"matched_part_{shard_id:05d}.parquet"
        df.to_parquet(shard_path, index=False)
        print(f"[save] {shard_path} ({len(df)} rows)")

    print("[done] 批量匹配完成，已生成分片。请运行 01c_concat_matched_parts 合并。")

if __name__ == "__main__":
    main()
