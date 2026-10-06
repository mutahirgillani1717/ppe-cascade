"""Convert the Kaggle Hard Hat Detection dataset (Pascal VOC XML) to YOLO format.

Makes a fixed 80/10/10 train/val/test split (seed 42) and writes data.yaml.
Usage: python scripts/prepare_data.py --src <downloaded_folder> --out data/hardhat_yolo
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

CLASSES = ["helmet", "head", "person"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def to_yolo(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, xmin), min(float(w), xmax)
    ymin, ymax = max(0.0, ymin), min(float(h), ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    return (
        (xmin + xmax) / 2 / w,
        (ymin + ymax) / 2 / h,
        (xmax - xmin) / w,
        (ymax - ymin) / h,
    )


def parse_labels(xml_path, w, h):
    root = ET.parse(xml_path).getroot()
    lines, skipped = [], Counter()
    for obj in root.iter("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASSES:
            skipped[name] += 1
            continue
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        yolo = to_yolo(box, w, h)
        if yolo is None:
            skipped["bad_box"] += 1
            continue
        lines.append(f"{CLASSES.index(name)} " + " ".join(f"{v:.6f}" for v in yolo))
    return lines, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with the images and xml files")
    ap.add_argument("--out", default="data/hardhat_yolo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out).resolve()
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    xmls = {p.stem: p for p in src.rglob("*.xml")}
    stems = sorted(set(images) & set(xmls))
    print(f"images: {len(images)} | xml files: {len(xmls)} | matched pairs: {len(stems)}")
    if not stems:
        raise SystemExit("No matching image/xml pairs found. Check --src.")

    random.Random(args.seed).shuffle(stems)
    n = len(stems)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    skipped_total = Counter()
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        class_counts = Counter()
        for stem in items:
            img = images[stem]
            with Image.open(img) as im:
                w, h = im.size
            lines, skipped = parse_labels(xmls[stem], w, h)
            skipped_total.update(skipped)
            for line in lines:
                class_counts[CLASSES[int(line.split()[0])]] += 1
            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        print(f"{split:5s}: {len(items):5d} images | boxes: {dict(class_counts)}")

    if skipped_total:
        print("skipped objects:", dict(skipped_total))

    yaml = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (out / "data.yaml").write_text(yaml)
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()