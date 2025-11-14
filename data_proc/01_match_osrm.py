# -*- coding: utf-8 -*-
import os, sys, time, argparse, re, math, random
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import requests
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- 全局配置 ----------------
DEFAULT_BASE = os.environ.get("OSRM_BASE", "http://127.0.0.1:5000")  # 用 127.0.0.1 比 localhost 更稳
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# 北京经纬度大致范围（用于识别/清洗）
LON_MIN, LON_MAX = 114.0, 118.0
LAT_MIN, LAT_MAX =  38.0,  42.0

def in_beijing(lon: float, lat: float) -> bool:
    return (LON_MIN <= lon <= LON_MAX) and (LAT_MIN <= lat <= LAT_MAX)

# ---------------- 小工具 ----------------
def haversine_m(lon1, lat1, lon2, lat2):
    if any(map(lambda x: x is None or not np.isfinite(x), [lon1, lat1, lon2, lat2])):
        return np.nan
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def dedupe_and_smooth(points: List[Tuple[float,float]],
                      min_move_m=3.0, dedupe_seconds=2, downsample_every=1):
    """无时间戳时：按空间阈值去抖 + 简单去重 + 下采样"""
    if not points: return points
    cleaned = []
    last_keep = None
    same_count = 0
    for p in points:
        if last_keep is None:
            cleaned.append(p); last_keep = p; same_count = 1
            continue
        d = haversine_m(last_keep[0], last_keep[1], p[0], p[1])
        if not np.isfinite(d):
            continue
        # 完全重复点
        if d < 1e-6:
            same_count += 1
            if same_count <= max(1, dedupe_seconds):  # 保留少量重复
                cleaned.append(p)
            continue
        # 小于移动阈值：丢弃（抖动）
        if d < min_move_m:
            continue
        cleaned.append(p)
        last_keep = p
        same_count = 1
    # 下采样
    if downsample_every > 1:
        cleaned = cleaned[::downsample_every]
    return cleaned

# ---------------- 会话&重试&节流 ----------------
def make_session(max_retries: int, backoff_factor: float, pool_maxsize: int = 128) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,           # 指数退避，0.8~1.2 比较合适
        status_forcelist=[429, 502, 503, 504],   # 典型可重试状态
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

class Throttle:
    """简单漏桶：保证两次请求之间至少间隔 min_interval 秒"""
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._last_ts = 0.0

    def wait(self, extra_sleep: float = 0.0, jitter: float = 0.0):
        if self.min_interval > 0:
            now = time.time()
            delta = now - self._last_ts
            need = self.min_interval - delta
            if need > 0:
                time.sleep(need)
        if extra_sleep > 0:
            # 叠加用户指定 sleep + 抖动
            jitter_s = random.uniform(0, max(0.0, jitter))
            time.sleep(extra_sleep + jitter_s)
        self._last_ts = time.time()

# ---------------- OSRM 基础 ----------------
def osrm_match(points: List[Tuple[float, float]],
               annotations: bool,
               radius_m: Optional[int],
               base_url: str,
               session: requests.Session,
               timeout: int,
               throttle: Throttle,
               sleep_sec: float,
               sleep_jitter: float) -> Dict:
    if len(points) < 2:
        raise ValueError("At least 2 points are required for /match")

    throttle.wait(extra_sleep=sleep_sec, jitter=sleep_jitter)

    coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in points])
    params = {"geometries": "geojson", "overview": "false"}
    if annotations:
        params["annotations"] = "true"
    if radius_m and radius_m > 0:
        params["radiuses"] = ";".join([str(radius_m)] * len(points))

    r = session.get(f"{base_url}/match/v1/driving/{coord_str}", params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        raise RuntimeError(f"OSRM match error: {js}")
    return js

def osrm_route(points: List[Tuple[float, float]],
               base_url: str,
               session: requests.Session,
               timeout: int,
               throttle: Throttle,
               sleep_sec: float,
               sleep_jitter: float) -> Dict:
    if len(points) < 2:
        raise ValueError("At least 2 points are required for /route")

    throttle.wait(extra_sleep=sleep_sec, jitter=sleep_jitter)

    # /route 仅用少量途径点，避免 400
    if len(points) > 3:
        pts = [points[0], points[len(points)//2], points[-1]]
    else:
        pts = points
    coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in pts])
    params = {"geometries": "geojson", "overview": "false"}
    r = session.get(f"{base_url}/route/v1/driving/{coord_str}", params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        raise RuntimeError(f"OSRM route error: {js}")
    return js

def parse_match_to_rows(js: Dict, trip_id: str, driver_id: str) -> List[Dict]:
    rows = []
    for mi, m in enumerate(js.get("matchings", []) or []):
        for li, leg in enumerate(m.get("legs", []) or []):
            ann = leg.get("annotation", {}) or {}
            nodes = ann.get("nodes", []) or []
            dist  = ann.get("distance", []) or []
            dur   = ann.get("duration", []) or []
            speed = ann.get("speed", []) or []
            for k in range(max(0, len(nodes) - 1)):
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

def off_network_ratio(js: Dict, n_points: int) -> float:
    """根据 tracepoints 计算贴路比例（非 None 的占比）"""
    tps = js.get("tracepoints", None)
    if not isinstance(tps, list) or n_points <= 0:
        return 0.0
    ok = sum(1 for t in tps if t is not None)
    return ok / float(n_points)

# ---------------- 分段匹配（带递归上限/提前放弃/兜底） ----------------
def chunk_with_overlap(seq: List[Tuple[float, float]], chunk_size: int = 100, overlap: int = 1):
    n = len(seq)
    if n <= chunk_size:
        yield seq
        return
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        yield seq[i:j]
        if j == n:
            break
        i = j - overlap

def _match_segment(
    seg: List[Tuple[float,float]],
    trip_id: str, driver_id: str, part: int,
    *,
    annotations: bool, radius_m: Optional[int],
    probe_min_ratio: float,
    max_recursion_depth: int,
    early_stop_empty: int,
    route_fallback: bool,
    base_url: str,
    request_timeout: int,
    max_retries: int,
    backoff_factor: float,
    min_request_interval: float,
    sleep_sec: float,
    sleep_jitter: float,
) -> List[Dict]:
    """对单个 seg 做匹配，内含递归切分与提前放弃逻辑 + 节流"""
    results: List[Dict] = []
    empty_streak = 0
    throttle = Throttle(min_request_interval)
    sess = make_session(max_retries=max_retries, backoff_factor=backoff_factor, pool_maxsize=128)

    def recurse(s: List[Tuple[float,float]], depth: int, tag: str):
        nonlocal results, empty_streak
        if len(s) < 2:
            return
        if depth > max_recursion_depth:
            print(f"[warn] {trip_id}#part{part}: reach max depth, give up (len={len(s)})")
            return
        try:
            js = osrm_match(
                s, annotations=annotations, radius_m=radius_m,
                base_url=base_url, session=sess, timeout=request_timeout,
                throttle=throttle, sleep_sec=sleep_sec, sleep_jitter=sleep_jitter
            )
            ratio = off_network_ratio(js, len(s))
            if ratio < probe_min_ratio:
                print(f"[warn] {trip_id}#part{part}: off-network probe<{probe_min_ratio:.2f}, skip (len={len(s)})")
                empty_streak += 1
                if empty_streak >= early_stop_empty:
                    print(f"[warn] {trip_id}#part{part}: empty after recursive split (early stop)")
                    return
                # 再往下切半尝试一次
                mid = max(2, len(s)//2)
                recurse(s[:mid], depth+1, tag+"L")
                recurse(s[mid-1:], depth+1, tag+"R")
                return
            rows = parse_match_to_rows(js, trip_id=f"{trip_id}#part{part}{tag}", driver_id=driver_id)
            if rows:
                results.extend(rows)
                empty_streak = 0
            else:
                # 没有 annotation 也可能为 0；尝试再切
                mid = max(2, len(s)//2)
                if mid >= len(s):
                    # 兜底：route
                    if route_fallback:
                        try:
                            _ = osrm_route(
                                s, base_url=base_url, session=sess, timeout=request_timeout,
                                throttle=throttle, sleep_sec=sleep_sec, sleep_jitter=sleep_jitter
                            )
                        except Exception:
                            pass
                    print(f"[warn] {trip_id}#part{part}: empty after recursive split")
                    return
                recurse(s[:mid], depth+1, tag+"L")
                recurse(s[mid-1:], depth+1, tag+"R")
        except requests.HTTPError as e:
            print(f"[warn] hard fail (len={len(s)}): {e}")
            if len(s) <= 3:
                if route_fallback:
                    try:
                        _ = osrm_route(
                            s, base_url=base_url, session=sess, timeout=request_timeout,
                            throttle=throttle, sleep_sec=sleep_sec, sleep_jitter=sleep_jitter
                        )
                    except Exception:
                        pass
                print(f"[warn] {trip_id}#part{part}: empty after recursive split")
                return
            mid = max(2, len(s)//2)
            recurse(s[:mid], depth+1, tag+"L")
            recurse(s[mid-1:], depth+1, tag+"R")
        except requests.ReadTimeout:
            print(f"[warn] {trip_id}#part{part}: Read timeout (len={len(s)}), split and retry")
            if len(s) <= 3:
                print(f"[warn] {trip_id}#part{part}: empty after recursive split")
                return
            mid = max(2, len(s)//2)
            recurse(s[:mid], depth+1, tag+"L")
            recurse(s[mid-1:], depth+1, tag+"R")
        except Exception as e:
            print(f"[warn] match failed for {trip_id} part {part}: {e}")
            if len(s) <= 3:
                print(f"[warn] {trip_id}#part{part}: empty after recursive split")
                return
            mid = max(2, len(s)//2)
            recurse(s[:mid], depth+1, tag+"L")
            recurse(s[mid-1:], depth+1, tag+"R")

    recurse(seg, 0, "")
    try:
        sess.close()
    except Exception:
        pass
    return results

def batch_match(points: List[Tuple[float,float]], trip_id: str, driver_id: str,
                chunk_size: int, overlap: int,
                annotations: bool = False, radius_m: Optional[int] = None,
                probe_min_ratio: float = 0.18,
                max_recursion_depth: int = 2,
                early_stop_empty: int = 3,
                route_fallback: bool = True,
                base_url: str = DEFAULT_BASE,
                request_timeout: int = 120,
                max_retries: int = 3,
                backoff_factor: float = 0.8,
                min_request_interval: float = 0.02,
                sleep_sec: float = 0.03,
                sleep_jitter: float = 0.01,
                **kwargs) -> pd.DataFrame:   # ←← 仍接收多余参数，避免 CLI 传参出错

    all_rows: List[Dict] = []
    if len(points) < 2:
        return pd.DataFrame(all_rows)

    # 预过滤：去掉非法值
    cleaned = [(lo, la) for (lo, la) in points if np.isfinite(lo) and np.isfinite(la) and abs(lo) <= 180 and abs(la) <= 90]
    if len(cleaned) < 2:
        return pd.DataFrame(all_rows)

    # 抽样检查（北京范围）
    sample = cleaned[::max(1, len(cleaned)//50)]
    frac_bj = np.mean([in_beijing(lo, la) for lo, la in sample]) if sample else 0.0
    if frac_bj < 0.2:
        print(f"[warn] trip {trip_id}: too few points in Beijing bbox (sample_in_bbox={frac_bj:.2f}). Check lon/lat parsing.")
    else:
        print(f"[info] trip {trip_id}: sample_in_bbox={frac_bj:.2f}, first_point={cleaned[0]}")

    part = 0
    for seg in chunk_with_overlap(cleaned, chunk_size, overlap):
        rows = _match_segment(
            seg, trip_id, driver_id, part,
            annotations=annotations,
            radius_m=radius_m,
            probe_min_ratio=probe_min_ratio,
            max_recursion_depth=max_recursion_depth,
            early_stop_empty=early_stop_empty,
            route_fallback=route_fallback,
            base_url=base_url,
            request_timeout=request_timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            min_request_interval=min_request_interval,
            sleep_sec=sleep_sec,
            sleep_jitter=sleep_jitter,
        )
        if rows:
            all_rows.extend(rows)
        part += 1

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["trip_id","driver_id","matching_idx","leg_idx","seg_idx","u_node","v_node"])
    return df

# ---------------- 数据加载（鲁棒识别 lon/lat） ----------------
def _score_lon_lat_pair(df_num: pd.DataFrame, c1, c2) -> Tuple[float, bool]:
    lon1, lat1 = pd.to_numeric(df_num[c1], errors="coerce"), pd.to_numeric(df_num[c2], errors="coerce")
    lat2, lon2 = lon1, lat1
    s1 = np.mean((lon1 >= LON_MIN) & (lon1 <= LON_MAX) & (lat1 >= LAT_MIN) & (lat1 <= LAT_MAX))
    s2 = np.mean((lon2 >= LON_MIN) & (lon2 <= LON_MAX) & (lat2 >= LAT_MIN) & (lat2 <= LAT_MAX))
    if s2 > s1:
        return s2, True
    return s1, False

def _choose_lon_lat_columns(df: pd.DataFrame) -> Tuple[str, str]:
    name_map = {c: str(c).strip().lower() for c in df.columns}
    lon_name = next((c for c,n in name_map.items() if n in ("lon","lng","longitude")), None)
    lat_name = next((c for c,n in name_map.items() if n in ("lat","latitude")), None)
    if lon_name and lat_name:
        return lon_name, lat_name
    bad_keys = {"id","taxi_id","car_id","vehicle","veh","vid","time","timestamp","date","datetime"}
    candidate_cols = [c for c in df.columns if str(c).strip().lower() not in bad_keys]
    dfn = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in dfn.columns if pd.api.types.is_numeric_dtype(dfn[c])]
    best = None
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            s, swap = _score_lon_lat_pair(dfn, c1, c2)
            if best is None or s > best[0]:
                best = (s, c1, c2, swap)
    if best and best[0] > 0:
        _, c1, c2, swap = best
        return (c2, c1) if swap else (c1, c2)
    if len(num_cols) >= 2:
        a, b = num_cols[0], num_cols[1]
        med_a = dfn[a].abs().median()
        med_b = dfn[b].abs().median()
        return (a,b) if med_a >= med_b else (b,a)
    raise ValueError("Cannot detect lon/lat columns.")

def _parse_tdrive_txt(path: Path) -> List[Tuple[float,float]]:
    pts: List[Tuple[float,float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 4: continue
            try:
                lon = float(parts[2]); lat = float(parts[3])
                if np.isfinite(lon) and np.isfinite(lat):
                    pts.append((lon, lat))
                continue
            except:
                pass
            nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", ln)]
            for i in range(len(nums)-1):
                a, b = nums[i], nums[i+1]
                if in_beijing(a,b):
                    pts.append((a,b))
                    break
    return pts

def load_points_from_file(path: Path, fmt: str = "auto",
                          max_points: Optional[int] = None,
                          min_move_m: float = 3.0,
                          dedupe_seconds: int = 2,
                          downsample_every: int = 1) -> List[Tuple[float,float]]:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.suffix.lower() == ".plt":
            fmt = "plt"
        elif path.suffix.lower() == ".txt":
            fmt = "tdrive"
        else:
            fmt = "csv"

    points: List[Tuple[float,float]] = []
    if fmt == "plt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        data = lines[6:] if len(lines) > 6 else lines
        for line in data:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0]); lon = float(parts[1])
                    points.append((lon, lat))
                except:
                    continue
    elif fmt == "tdrive":
        points = _parse_tdrive_txt(path)
    else:
        try:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(path, sep=r"[,\s;]+", engine="python", header=None, on_bad_lines="skip")
        cols_lower = {c: str(c).lower() for c in df.columns}
        if set(["lon","lat"]).issubset(set(cols_lower.values())):
            inv = {v:k for k,v in cols_lower.items()}
            lo, la = inv["lon"], inv["lat"]
            dfn = df[[lo,la]].apply(pd.to_numeric, errors="coerce")
            for lon, lat in zip(dfn[lo], dfn[la]):
                if pd.notna(lon) and pd.notna(lat):
                    points.append((float(lon), float(lat)))
        else:
            lo, la = _choose_lon_lat_columns(df)
            dfn = df[[lo,la]].apply(pd.to_numeric, errors="coerce")
            for lon, lat in zip(dfn[lo], dfn[la]):
                if pd.notna(lon) and pd.notna(lat):
                    points.append((float(lon), float(lat)))

    # 清洗：去除非法值；整体经纬对调判断
    cleaned = []
    for lon, lat in points:
        if not (np.isfinite(lon) and np.isfinite(lat)): continue
        if abs(lon) > 180 or abs(lat) > 90: continue
        cleaned.append((lon, lat))
    if not cleaned:
        return []

    sample = cleaned[::max(1, len(cleaned)//200)]
    frac_ok = np.mean([in_beijing(lo, la) for lo, la in sample]) if sample else 0.0
    frac_sw = np.mean([in_beijing(la, lo) for lo, la in sample]) if sample else 0.0
    if frac_sw > frac_ok:
        cleaned = [(la, lo) for (lo, la) in cleaned]

    # 去重/去抖/下采样
    cleaned = dedupe_and_smooth(cleaned, min_move_m=min_move_m,
                                dedupe_seconds=dedupe_seconds,
                                downsample_every=downsample_every)

    if max_points and len(cleaned) > max_points:
        cleaned = cleaned[:max_points]
    return cleaned

def iter_trips(in_dir: Path, pattern: str) -> Iterable[Tuple[str,str,Path]]:
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    if not patterns:
        patterns = ["*.*"]
    seen = set()
    for pat in patterns:
        for p in sorted(in_dir.rglob(pat)):
            if not p.is_file(): continue
            if p in seen: continue
            seen.add(p)
            trip_id = p.stem
            m = re.match(r"(\d+)", trip_id)
            driver_id = m.group(1) if m else trip_id
            yield trip_id, driver_id, p

# ---------------- 单文件处理（供并行调度） ----------------
def process_one_file(fp: Path, args) -> Optional[pd.DataFrame]:
    try:
        pts = load_points_from_file(
            fp, fmt=args.fmt,
            max_points=(args.max_points_per_file or None),
            min_move_m=args.min_move_m,
            dedupe_seconds=args.dedupe_seconds,
            downsample_every=args.downsample_every
        )
        if len(pts) < 2:
            print(f"[skip] too few points: {fp}")
            return None
        df = batch_match(
            pts, trip_id=fp.stem, driver_id=re.match(r'(\d+)', fp.stem).group(1) if re.match(r'(\d+)', fp.stem) else fp.stem,
            chunk_size=args.chunk_points, overlap=args.overlap,
            annotations=args.annotations,
            radius_m=args.radius_m,
            probe_min_ratio=args.probe_min_ratio,
            max_recursion_depth=args.max_recursion_depth,
            early_stop_empty=args.early_stop_empty,
            route_fallback=args.route_fallback,
            base_url=args.base,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
            min_request_interval=args.min_request_interval,
            sleep_sec=args.sleep_sec,
            sleep_jitter=args.sleep_jitter,
        )
        if df is None or df.empty:
            print(f"[warn] {fp.name}: empty match result")
            return None
        print(f"[ok] {fp.name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[warn] failed on {fp}: {e}")
        return None

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE, help="OSRM base URL, e.g., http://127.0.0.1:5000")
    ap.add_argument("--in_dir", type=str, default=str(ROOT / "data_raw" / "tdrive"))
    ap.add_argument("--pattern", type=str, default="*.csv,*.txt", help="多个用逗号分隔，如 *.csv,*.txt 或 *.plt")
    ap.add_argument("--fmt", type=str, default="auto", choices=["auto","csv","plt","tdrive"])
    # 更稳的默认值
    ap.add_argument("--chunk_points", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=2)
    # 节流/限速/重试/超时
    ap.add_argument("--sleep_sec", type=float, default=0.03, help="每次请求额外 sleep 秒数")
    ap.add_argument("--sleep_jitter", type=float, default=0.01, help="每次请求附加随机抖动上限秒数")
    ap.add_argument("--min_request_interval", type=float, default=0.02, help="两次请求间的最小间隔（漏桶）")
    ap.add_argument("--request_timeout", type=int, default=120, help="单次请求超时秒数（读超时）")
    ap.add_argument("--max_retries", type=int, default=3, help="requests 层重试次数")
    ap.add_argument("--backoff_factor", type=float, default=0.8, help="指数退避基数（越大退避越久）")
    # 其他
    ap.add_argument("--max_files", type=int, default=0, help="0 表示不限制")
    ap.add_argument("--max_points_per_file", type=int, default=5000, help="0 表示不限制")
    ap.add_argument("--out", type=str, default=str(DATASETS_DIR / "matched_legs.csv"))
    ap.add_argument("--out_parquet", type=str, default="")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（按文件粒度）。建议先用 1，稳了再调大。")
    ap.add_argument("--annotations", type=lambda s: s.lower()!="false", default=False, help="是否返回 annotation（默认关闭以提速）")
    ap.add_argument("--radius_m", type=int, default=80, help="match radiuses 容忍半径（米），0 表示不用")
    ap.add_argument("--probe_min_ratio", type=float, default=0.12, help="tracepoints 贴路最小比例，低于则判为 off-network")
    ap.add_argument("--max_recursion_depth", type=int, default=2, help="单段匹配递归切分的最大层数")
    ap.add_argument("--early_stop_empty", type=int, default=4, help="连续空结果计数达到该值则提前放弃该段")
    ap.add_argument("--route_fallback", type=lambda s: s.lower()!="false", default=True, help="match 多次失败是否退化到 /route 兜底")
    ap.add_argument("--min_move_m", type=float, default=2.0, help="小于该位移视为抖动丢弃（略微放宽）")
    ap.add_argument("--dedupe_seconds", type=int, default=2, help="允许保留的连续重复点数量（无时间戳场景近似）")
    ap.add_argument("--downsample_every", type=int, default=1, help="下采样步长")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    # 探活 OSRM
    try:
        sess = make_session(max_retries=args.max_retries, backoff_factor=args.backoff_factor)
        r = sess.get(f"{args.base}/nearest/v1/driving/116.397,39.908?number=1", timeout=min(5, args.request_timeout))
        r.raise_for_status()
    except Exception as e:
        print(f"[fatal] OSRM not reachable at {args.base}: {e}")
        sys.exit(2)
    finally:
        try:
            sess.close()
        except Exception:
            pass

    # 遍历文件
    files = []
    n = 0
    for trip_id, driver_id, fp in iter_trips(in_dir, args.pattern):
        if args.max_files and n >= args.max_files:
            break
        files.append(fp)
        n += 1
    if not files:
        print("[info] no input files found.")
        return

    rows_all = []

    if args.workers and args.workers > 1:
        # 注意：多进程会加大 OSRM 压力，先确认单进程稳再调大
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one_file, fp, args): fp for fp in files}
            for fut in as_completed(futs):
                df = fut.result()
                if df is not None and not df.empty:
                    rows_all.append(df)
    else:
        for fp in files:
            df = process_one_file(fp, args)
            if df is not None and not df.empty:
                rows_all.append(df)

    if not rows_all:
        print("[info] No rows matched. Check your input format/pattern and OSRM.")
        return

    out_csv = Path(args.out)
    out_parquet = Path(args.out_parquet) if args.out_parquet else out_csv.with_suffix(".parquet")

    df_all = pd.concat(rows_all, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False, encoding="utf-8")
    try:
        df_all.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"[warn] failed to write parquet: {e}")
    print(f"[ok] Saved {len(df_all)} rows -> {out_parquet} / {out_csv}")

if __name__ == "__main__":
    main()
