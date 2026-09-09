#!/usr/bin/env python3
"""Download AlfWorld data via GitHub proxy when github.com is unreachable.

Usage (DAVIS env):
  python scripts/alfworld_download_mirror.py
  python scripts/alfworld_download_mirror.py --proxy https://ghproxy.net/
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from os.path import join as pjoin

import requests
from tqdm import tqdm

try:
    from alfworld.info import ALFWORLD_DATA, ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    from alfworld.utils import mkdirs
except ImportError as exc:
    raise SystemExit("Activate DAVIS env and install alfworld first.") from exc

FILES = [
    ("https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip", "."),
    ("https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip", "."),
    ("https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip", "."),
    # MaskRCNN only needed for AlfredThorEnv / vision; skip with --text-only
    ("https://github.com/alfworld/alfworld/releases/download/0.2.2/mrcnn_alfred_objects_sep13_004.pth", "detectors"),
]


def proxied(url: str, proxy_prefix: str) -> str:
    proxy_prefix = proxy_prefix.rstrip("/") + "/"
    if url.startswith(proxy_prefix):
        return url
    return proxy_prefix + url


def download(url: str, dst_dir: str, force: bool = False) -> str:
    mkdirs(dst_dir)
    filename = url.rstrip("/").split("/")[-1]
    path = pjoin(dst_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0 and not force:
        print(f"[skip] {path}")
        return path

    temp_dir = mkdirs(pjoin(tempfile.gettempdir(), "alfworld"))
    temp_path = pjoin(temp_dir, filename)
    # drop corrupt 0-byte leftovers
    if os.path.isfile(temp_path) and os.path.getsize(temp_path) == 0:
        os.remove(temp_path)

    headers = {}
    mode = "ab"
    resume = 0
    if os.path.isfile(temp_path):
        resume = os.path.getsize(temp_path)
        if resume:
            headers["Range"] = f"bytes={resume}-"

    print(f"[get] {url}")
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = resume + int(r.headers.get("Content-Length") or 0)
        pbar = tqdm(
            unit="B",
            initial=resume,
            unit_scale=True,
            total=total or None,
            desc=filename,
        )
        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()

    shutil.move(temp_path, path)
    print(f"[ok] {path} ({os.path.getsize(path)} bytes)")
    return path


def unzip(filename: str, dst: str, force: bool = False) -> None:
    with zipfile.ZipFile(filename) as zf:
        names = zf.namelist()
        skipped = 0
        for name in tqdm(names, desc=f"Extract {os.path.basename(filename)}"):
            out = pjoin(dst, name)
            if os.path.isfile(out) and not force:
                skipped += 1
                continue
            zf.extract(name, dst)
        if skipped:
            print(f"{skipped} files skipped (use -f to overwrite).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=ALFWORLD_DATA)
    ap.add_argument(
        "--proxy",
        default=os.environ.get("ALFWORLD_GH_PROXY", "https://ghproxy.net/"),
        help="GitHub proxy prefix",
    )
    ap.add_argument("--text-only", action="store_true", default=True,
                    help="Skip MaskRCNN weights (default for DAVIS text AlfWorld)")
    ap.add_argument("--with-mrcnn", action="store_true",
                    help="Also download MaskRCNN detector weights")
    ap.add_argument("-f", "--force", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    mkdirs(data_dir)
    print(f"ALFWORLD_DATA = {data_dir}")
    print(f"proxy         = {args.proxy}")

    want_mrcnn = args.with_mrcnn
    for url, sub in FILES:
        if url.endswith(".pth") and not want_mrcnn:
            print("[skip] mrcnn (text-only)")
            continue
        dst = data_dir if sub == "." else pjoin(data_dir, sub)
        mirrored = proxied(url, args.proxy)
        path = download(mirrored, dst_dir=dst if not url.endswith(".zip") else data_dir,
                        force=args.force_download)
        # zip targets always land in data_dir; non-zip already in dst
        if path.endswith(".zip"):
            unzip(path, dst=data_dir, force=args.force)
            os.remove(path)

    logic_dir = mkdirs(pjoin(data_dir, "logic"))
    for src, name in ((ALFRED_PDDL_PATH, "alfred.pddl"), (ALFRED_TWL2_PATH, "alfred.twl2")):
        dst = pjoin(logic_dir, name)
        if not os.path.isfile(dst) or args.force:
            shutil.copy(src, dst)
            print(f"[copy] {dst}")
        else:
            print(f"[skip] {dst}")

    # sanity
    tw = pjoin(data_dir, "json_2.1.1")
    print(f"\nDone. Check: ls {tw}")
    if os.path.isdir(tw):
        for split in ("train", "valid_seen", "valid_unseen"):
            p = pjoin(tw, split)
            print(f"  {split}: {'OK' if os.path.isdir(p) else 'MISSING'} -> {p}")


if __name__ == "__main__":
    main()
