"""
Dataset Preparation & Exploratory Data Analysis (EDA)
======================================================
Run this BEFORE training to:
  1. Validate folder/file structure
  2. Generate EDA visualizations
  3. Print dataset statistics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

VIZ_DIR = Path("Visualization-Image")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate_structure(root: Path) -> dict:
    """
    Expected structure:
        root/
          <ClassName>/
              img001.jpg
              img002.jpg
              ...
    Returns dict: class_name -> list[Path]
    """
    print("\n🔍  Validating dataset structure …")
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if not class_dirs:
        print(f"  ✗  No sub-directories found in {root}")
        sys.exit(1)

    data = {}
    issues = []
    for cls_dir in class_dirs:
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        if len(imgs) == 0:
            issues.append(f"  ⚠  {cls_dir.name}: NO images found")
        data[cls_dir.name] = sorted(imgs)

    if issues:
        print("\n".join(issues))
    else:
        print(f"  ✓  {len(class_dirs)} class folders found, all contain images.")
    return data


# ─────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────
def compute_stats(data: dict) -> dict:
    stats = {}
    total = 0
    for cls, imgs in data.items():
        n = len(imgs)
        total += n
        stats[cls] = n
    stats["__total__"] = total
    return stats


def sample_image_sizes(data: dict, n_sample: int = 20) -> list:
    sizes = []
    all_imgs = [p for imgs in data.values() for p in imgs]
    rng = np.random.default_rng(42)
    sample = rng.choice(all_imgs, min(n_sample * len(data), len(all_imgs)), replace=False)
    for p in tqdm(sample, desc="  Sampling image sizes", ncols=80):
        try:
            w, h = Image.open(p).size
            sizes.append((w, h))
        except Exception:
            pass
    return sizes


# ─────────────────────────────────────────────
#  EDA PLOTS
# ─────────────────────────────────────────────
def _save(fig, name: str):
    path = VIZ_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Saved: {path}")


def plot_images_per_class(stats: dict):
    classes = [k for k in stats if k != "__total__"]
    counts  = [stats[k] for k in classes]
    n = len(classes)

    fig, ax = plt.subplots(figsize=(14, max(8, n // 2.5)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n))
    bars = ax.barh(range(n), counts, color=colors, edgecolor="#222233", alpha=0.9)

    mean_count = np.mean(counts)
    ax.axvline(mean_count, color="#FFD54F", lw=1.5, ls="--", label=f"Mean: {mean_count:.1f}")

    ax.set_yticks(range(n))
    ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=8, color="#CCCCCC")
    ax.set_title("Images per Class", color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Images", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA", axis="x")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(facecolor="#1A1D27", labelcolor="white", edgecolor="#333344")
    ax.grid(axis="x", alpha=0.12, color="white")

    for bar, count in zip(bars, counts):
        ax.text(count + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", color="white", fontsize=8)

    _save(fig, "eda_images_per_class.png")


def plot_split_pie(stats: dict, val_ratio: float, test_ratio: float):
    total = stats["__total__"]
    n_test  = int(total * test_ratio)
    n_val   = int(total * val_ratio)
    n_train = total - n_test - n_val

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    sizes  = [n_train, n_val, n_test]
    labels = [f"Train\n{n_train} ({n_train/total*100:.1f}%)",
              f"Val\n{n_val} ({n_val/total*100:.1f}%)",
              f"Test\n{n_test} ({n_test/total*100:.1f}%)"]
    colors = ["#4FC3F7", "#FF7043", "#66BB6A"]
    explode = [0.04, 0.04, 0.04]

    wedges, texts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                           startangle=90, textprops={"color": "white", "fontsize": 12},
                           wedgeprops={"edgecolor": "#0F1117", "linewidth": 2})
    ax.set_title("Dataset Split Distribution", color="white", fontsize=14, fontweight="bold", pad=18)

    _save(fig, "eda_split_distribution.png")


def plot_image_size_distribution(sizes: list):
    if not sizes:
        return
    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    for ax, vals, title, color in zip(
        axes,
        [widths, heights],
        ["Image Width Distribution", "Image Height Distribution"],
        ["#4FC3F7", "#FF7043"]
    ):
        ax.set_facecolor("#1A1D27")
        ax.hist(vals, bins=30, color=color, edgecolor="#222233", alpha=0.85)
        ax.axvline(np.mean(vals), color="#FFD54F", lw=2, ls="--",
                   label=f"Mean: {np.mean(vals):.0f}px")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Pixels", color="#AAAAAA")
        ax.set_ylabel("Count", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.legend(facecolor="#1A1D27", labelcolor="white", edgecolor="#333344")
        ax.grid(alpha=0.12, color="white")

    fig.suptitle("Image Size Distribution (sampled)", color="white",
                 fontsize=14, fontweight="bold")
    _save(fig, "eda_image_sizes.png")


def plot_class_count_histogram(stats: dict):
    counts = [v for k, v in stats.items() if k != "__total__"]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")
    ax.hist(counts, bins=15, color="#7E57C2", edgecolor="#222233", alpha=0.85)
    ax.set_title("Distribution of Images-per-Class Counts", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Images per Class", color="#AAAAAA")
    ax.set_ylabel("Number of Classes", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(alpha=0.12, color="white")
    _save(fig, "eda_class_count_histogram.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main(args):
    root = Path(args.data_root)
    if not root.exists():
        print(f"✗  Data root not found: {root}")
        sys.exit(1)

    data  = validate_structure(root)
    stats = compute_stats(data)
    total = stats["__total__"]
    n_classes = len(data)

    print("\n📊  Dataset Statistics")
    print("─" * 40)
    print(f"  Classes       : {n_classes}")
    print(f"  Total images  : {total}")
    print(f"  Avg per class : {total / n_classes:.1f}")
    print(f"  Min per class : {min(v for k,v in stats.items() if k != '__total__')}")
    print(f"  Max per class : {max(v for k,v in stats.items() if k != '__total__')}")

    n_test  = int(total * args.test_ratio)
    n_val   = int(total * args.val_ratio)
    n_train = total - n_test - n_val
    print(f"\n  Train split   : {n_train} ({n_train/total*100:.1f}%)")
    print(f"  Val split     : {n_val}   ({n_val/total*100:.1f}%)")
    print(f"  Test split    : {n_test}  ({n_test/total*100:.1f}%)")

    print("\n🎨  Generating EDA visualizations …")
    plot_images_per_class(stats)
    plot_split_pie(stats, args.val_ratio, args.test_ratio)
    plot_class_count_histogram(stats)

    if args.sample_sizes:
        sizes = sample_image_sizes(data)
        plot_image_size_distribution(sizes)

    # Save class list
    class_list = [k for k in stats if k != "__total__"]
    out = {"classes": class_list, "stats": stats}
    with open(VIZ_DIR.parent / "dataset_info.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓  dataset_info.json written.")
    print(f"\n  ✓  All EDA plots saved to: {VIZ_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="./data")
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--test_ratio",  type=float, default=0.10)
    p.add_argument("--sample_sizes", action="store_true",
                   help="Sample and plot image resolution distribution (slower)")
    main(p.parse_args())