#!/usr/bin/env python3
"""Download PlotQA from GitHub and convert to load_from_disk format."""

from __future__ import annotations

import json, os, sys, zipfile, io, tarfile
from pathlib import Path
from typing import Dict, List
import urllib.request

import PIL.Image
from datasets import Dataset, DatasetDict, ClassLabel

PLOTQA_REPO = "https://github.com/Nitesh060/PlotQA/archive/refs/heads/master.zip"
PLOTQA_TAR = "https://github.com/Nitesh060/PlotQA/archive/refs/heads/master.tar.gz"

FIGURQA_URL = "https://github.com/nitesh060/FigureQA/archive/refs/heads/master.zip"
FIGURQA_TAR = "https://github.com/nitesh060/FigureQA/archive/refs/heads/master.tar.gz"

OUTPUT_ROOT = Path("/root/autodl-tmp/data/datasets")


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  saved {dest.stat().st_size / 1_000_000:.1f} MB")


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        root_name = names[0].split("/")[0]
        zf.extractall(extract_to)
    return extract_to / root_name


def extract_tar(tar_path: Path, extract_to: Path) -> Path:
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        root_name = tf.getnames()[0].split("/")[0]
        tf.extractall(extract_to)
    return extract_to / root_name


def build_plotqa_dataset(repo_root: Path) -> DatasetDict:
    data_dir = repo_root / "plotqa"
    splits: Dict[str, List[Dict]] = {}

    for split_name in ["train", "val", "test"]:
        json_path = data_dir / f"qa_{split_name}.json"
        if not json_path.exists():
            # Try alternate naming
            candidates = list(data_dir.glob(f"*{split_name}*.json"))
            if not candidates:
                print(f"  WARNING: no JSON found for split {split_name}")
                continue
            json_path = candidates[0]

        print(f"  Loading {json_path} ...")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        images = []

        for i, item in enumerate(data):
            question = item.get("question", item.get("query", ""))
            gold = item.get("answer", item.get("label", ""))
            if not isinstance(gold, list):
                gold = [str(gold)]

            image = None
            if "image" in item:
                image_data = item["image"]
                if isinstance(image_data, dict) and "bytes" in image_data:
                    image = PIL.Image.open(io.BytesIO(image_data["bytes"]))
                elif isinstance(image_data, bytes):
                    image = PIL.Image.open(io.BytesIO(image_data))
            elif "image_index" in item:
                img_idx = int(item["image_index"])
                img_path = find_image(repo_root, img_idx, split_name)
                if img_path and img_path.exists():
                    image = PIL.Image.open(img_path)

            images.append({
                "image": image,
                "query": question,
                "label": gold,
            })

        if images:
            splits[split_name] = images
            print(f"  {split_name}: {len(images)} samples")

    if not splits:
        raise RuntimeError("No splits found in PlotQA repo")

    datasets = {}
    for split_name, records in splits.items():
        datasets[split_name] = Dataset.from_list(records)

    return DatasetDict(datasets)


def find_image(repo_root: Path, img_index: int, split_name: str) -> Path | None:
    candidates = list(repo_root.rglob(f"**/{img_index}.png")) + \
                 list(repo_root.rglob(f"**/{img_index:06d}.png")) + \
                 list(repo_root.rglob(f"**/*_{img_index}.png"))
    if candidates:
        return candidates[0]
    return None


def download_plotqa():
    output = OUTPUT_ROOT / "plotqa"
    if (output / "dataset_dict.json").exists():
        print(f"PlotQA already exists at {output}")
        return

    cache_dir = OUTPUT_ROOT / "plotqa_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "plotqa.zip"
    if not zip_path.exists():
        download_file(PLOTQA_REPO, zip_path)

    if zip_path.suffix == ".gz":
        repo_root = extract_tar(zip_path, cache_dir)
    else:
        repo_root = extract_zip(zip_path, cache_dir)

    print(f"Repo root: {repo_root}")
    print(f"Contents: {sorted([p.name for p in repo_root.iterdir()])[:30]}")

    ds = build_plotqa_dataset(repo_root)
    ds.save_to_disk(str(output))
    print(f"Saved PlotQA to {output}")
    for s in ds:
        print(f"  {s}: {len(ds[s])}")


def download_figureqa():
    output = OUTPUT_ROOT / "figureqa"
    if (output / "dataset_dict.json").exists():
        print(f"FigureQA already exists at {output}")
        return

    cache_dir = OUTPUT_ROOT / "figureqa_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "figureqa.zip"
    if not zip_path.exists():
        download_file(FIGURQA_URL, zip_path)

    if zip_path.suffix == ".gz":
        repo_root = extract_tar(zip_path, cache_dir)
    else:
        repo_root = extract_zip(zip_path, cache_dir)

    print(f"Repo root: {repo_root}")
    print(f"Contents: {sorted([p.name for p in repo_root.iterdir()])[:30]}")

    ds = build_plotqa_dataset(repo_root)
    ds.save_to_disk(str(output))
    print(f"Saved FigureQA to {output}")
    for s in ds:
        print(f"  {s}: {len(ds[s])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["plotqa","figureqa","all"], default="plotqa")
    args = parser.parse_args()

    if args.dataset in ("plotqa", "all"):
        download_plotqa()
    if args.dataset in ("figureqa", "all"):
        download_figureqa()
