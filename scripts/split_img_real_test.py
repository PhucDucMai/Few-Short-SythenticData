#!/usr/bin/env python3
"""
Split Img-Real-Test into two directories, each with 37 class folders.
Each class folder in the targets will contain between 20 and 30 images when possible.
"""
import random
import shutil
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "Img-Real-Test"
DEST_PARENT = SRC.parent
DEST_A = DEST_PARENT / "Img-Real-Test-1"
DEST_B = DEST_PARENT / "Img-Real-Test-2"


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_list(src_files, dest_dir):
    ensure_dir(dest_dir)
    for f in src_files:
        shutil.copy2(f, dest_dir / f.name)


def run():
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return

    ensure_dir(DEST_A)
    ensure_dir(DEST_B)

    classes = sorted([p for p in SRC.iterdir() if p.is_dir()])
    print(f"Found {len(classes)} class folders in {SRC}")
    if len(classes) != 37:
        print("Warning: expected 37 class folders. Script will still proceed.")

    summary = []
    for cls in classes:
        imgs = [p for p in cls.iterdir() if p.is_file() and is_image(p)]
        total = len(imgs)
        if total == 0:
            print(f"Skipping empty class {cls.name}")
            continue

        random.shuffle(imgs)

        # Decide sizes for each target
        min_n = 20
        max_n = 30

        if total >= 40:
            # try disjoint partition with random sizes
            n1 = random.randint(min_n, max_n)
            n2 = random.randint(min_n, max_n)
            if n1 + n2 > total:
                n2 = max(min_n, total - n1)
            if n2 < min_n:
                # fallback: evenly split to reach at least min_n if possible
                n1 = min(max_n, total // 2)
                n2 = min(max_n, total - n1)
        elif total >= min_n:
            # sample for A, sample for B allowing overlap
            n1 = random.randint(min_n, min(max_n, total))
            n2 = random.randint(min_n, min(max_n, total))
        else:
            # too few images: copy all to both
            n1 = total
            n2 = total

        # Select indices
        if total >= 40 and n1 + n2 <= total:
            sel1 = imgs[:n1]
            sel2 = imgs[n1:n1 + n2]
        else:
            sel1 = random.sample(imgs, min(n1, total))
            sel2 = random.sample(imgs, min(n2, total))

        copy_list(sel1, DEST_A / cls.name)
        copy_list(sel2, DEST_B / cls.name)

        summary.append((cls.name, total, len(sel1), len(sel2)))

    print("\nSummary per class: (class, total, copied_to_A, copied_to_B)")
    for s in summary:
        print(s)


if __name__ == "__main__":
    run()
