#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动完成：
  1) 下载 T-Drive 一周样本到 data_raw/tdrive/
     - 优先从微软官方页面抓取 OneDrive 直链
     - 若检测到 Kaggle API 凭据（KAGGLE_USERNAME/KAGGLE_KEY），可走 Kaggle 备用
  2) 下载北京 OSM PBF 到 roadnet/beijing.osm.pbf
     - 优先 beijing 分区，失败则回退 hebei 分区（含北京与天津）
"""
import os
import re
import sys
import json
import time
import shutil
import zipfile
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# ----------------------------
# 路径与常量
# ----------------------------
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "setup_tdrive_osm.py") else Path.cwd()
TD_DIR = ROOT / "data_raw" / "tdrive"
OSM_DIR = ROOT / "roadnet"
OSM_FILE = OSM_DIR / "beijing.osm.pbf"

MS_TDRIVE_PAGE = "https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/"  # 官方页
KAGGLE_DATASET = "arashnic/tdriver"  # 备用
GEOFABRIK_BJ = "https://download.geofabrik.de/asia/china/beijing.html"
GEOFABRIK_HEBEI = "https://download.geofabrik.de/asia/china/hebei.html"

HDRS = {"User-Agent": "Mozilla/5.0 (compatible; TDriveFetcher/1.0)"}


# ----------------------------
# 小工具
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def stream_download(url: str, dst: Path, headers=None, desc=None):
    with requests.get(url, headers=headers or HDRS, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=desc or dst.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if total > 0:
                        bar.update(len(chunk))
        tmp.rename(dst)
    return dst


def try_extract(archive: Path, outdir: Path):
    """尽力解压 zip / tar.gz；其他后缀保留原文件并提示。"""
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(outdir)
            return True
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(outdir)
            return True
    except Exception as e:
        print(f"[warn] 解压失败（{archive.name}）：{e}")
    return False


# ----------------------------
# T-Drive 下载
# ----------------------------
def fetch_tdrive_from_ms(outdir: Path):
    print("[T-Drive] 从微软官方页面抓取 OneDrive 下载链接…")
    html = requests.get(MS_TDRIVE_PAGE, headers=HDRS, timeout=60).text
    # 页面里有 “Download the Trajectory Data” 链接 -> 1drv.ms
    m = re.search(r'href="(https://1drv\.ms/[^"]+)"', html)
    if not m:
        raise RuntimeError("未找到 OneDrive 下载链接（页面结构可能更新）。")
    onedrive_url = m.group(1)
    print(f"[T-Drive] OneDrive 链接：{onedrive_url}")

    # 直接跟随重定向获取实际文件（OneDrive 通常会 302 到下载）
    with requests.get(onedrive_url, headers=HDRS, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        final_url = r.url
    # 猜测文件名
    filename = re.findall(r'/([^/?#]+)$', final_url)
    fname = filename[0] if filename else f"tdrive_oneweek.{int(time.time())}"
    dst = outdir / fname
    print(f"[T-Drive] 开始下载：{final_url}")
    stream_download(final_url, dst, desc="T-Drive")

    # 若是压缩包，尝试解压
    extracted = try_extract(dst, outdir)
    if extracted:
        print(f"[T-Drive] 已解压到：{outdir}")
    else:
        print(f"[T-Drive] 已保存：{dst}（如为 .rar/.7z，请手动解压或安装相应库解压）")


def fetch_tdrive_from_kaggle(outdir: Path):
    """需要你在系统环境变量配置 Kaggle API：KAGGLE_USERNAME 和 KAGGLE_KEY"""
    try:
        import subprocess
        print("[T-Drive] 检测到 Kaggle 凭据，尝试通过 Kaggle 下载…")
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(outdir), "-o"]
        print(f"[T-Drive] 执行：{' '.join(cmd)}")
        subprocess.check_call(cmd)
        # Kaggle 会下载一个 zip，尝试解压
        for z in outdir.glob("*.zip"):
            try_extract(z, outdir)
        print(f"[T-Drive] Kaggle 下载完成：{outdir}")
        return True
    except Exception as e:
        print(f"[warn] Kaggle 下载失败：{e}")
        return False


def download_tdrive():
    ensure_dir(TD_DIR)
    # 若目录下已有文件，认为已完成
    if any(TD_DIR.iterdir()):
        print(f"[T-Drive] 目录非空，跳过下载：{TD_DIR}")
        return

    # 优先微软官方；失败则尝试 Kaggle（前提：配置了 Kaggle API）
    try:
        fetch_tdrive_from_ms(TD_DIR)  # 官方样本页（含 1周一万车） :contentReference[oaicite:4]{index=4}
    except Exception as e:
        print(f"[warn] 微软源失败：{e}")
        if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            ok = fetch_tdrive_from_kaggle(TD_DIR)  # Kaggle 备用镜像 :contentReference[oaicite:5]{index=5}
            if not ok:
                raise
        else:
            raise


# ----------------------------
# OSM 北京 PBF 下载
# ----------------------------
def find_osm_pbf(url: str) -> str:
    """在 Geofabrik 页面中找到 .osm.pbf 直链"""
    html = requests.get(url, headers=HDRS, timeout=60).text
    m = re.search(r'href="([^"]+?\.osm\.pbf)"', html)
    if not m:
        raise RuntimeError("未在页面中找到 .osm.pbf 链接")
    href = m.group(1)
    if href.startswith("/"):
        # 相对路径
        base = re.match(r'^(https?://[^/]+)', url).group(1)
        return base + href
    elif href.startswith("http"):
        return href
    else:
        return url.rstrip("/") + "/" + href

def download_osm_beijing():
    ensure_dir(OSM_DIR)
    if OSM_FILE.exists():
        print(f"[OSM] 已存在：{OSM_FILE}，跳过下载。")
        return
    try:
        print("[OSM] 尝试从 Geofabrik 北京分区抓取 beijing.osm.pbf…")
        bj_url = find_osm_pbf(GEOFABRIK_BJ)  # 北京分区页 :contentReference[oaicite:6]{index=6}
        print(f"[OSM] 下载链接：{bj_url}")
        stream_download(bj_url, OSM_FILE, desc="Beijing PBF")
    except Exception as e:
        print(f"[warn] 北京分区失败，尝试回退到河北(含京津)分区：{e}")
        hebei_url = find_osm_pbf(GEOFABRIK_HEBEI)  # 河北(含京津)页 :contentReference[oaicite:7]{index=7}
        print(f"[OSM] 下载链接（回退）：{hebei_url}")
        stream_download(hebei_url, OSM_FILE, desc="Beijing PBF (fallback)")


# ----------------------------
# 主流程
# ----------------------------
def main():
    print(f"[root] 项目根目录：{ROOT}")
    print(f"[step] 准备目录…")
    ensure_dir(TD_DIR)
    ensure_dir(OSM_DIR)

    print(f"[step] 下载 T-Drive…")
    download_tdrive()

    print(f"[step] 下载北京 OSM PBF…")
    download_osm_beijing()

    print("\n✅ 完成：")
    print(f" - T-Drive 目录：{TD_DIR}  （请查看并解压产生的文件/子目录）")
    print(f" - OSM PBF：{OSM_FILE}")
    print("\n后续步骤提示：")
    print("  1) 启动 OSRM 并进行 /match 地图匹配（参考 OSRM API Match 文档）。")
    print("  2) 运行你的 01_match_osrm.py 等脚本做路段序列化与 10 分钟窗口均速聚合。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] 用户中断。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] 失败：{e}")
        sys.exit(2)
