"""Sanity checks for the YOLO-format dataset. Usage: python scripts/verify_data.py data/hardhat_yolo"""
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
splits = ["train", "val", "test"]
seen = {}
problems = 0

for s in splits:
    imgs = {p.stem for p in (root / "images" / s).iterdir()}
    lbls = {p.stem for p in (root / "labels" / s).glob("*.txt")}
    seen[s] = imgs
    counts, empty, bad = Counter(), 0, 0
    for f in (root / "labels" / s).glob("*.txt"):
        text = f.read_text().strip()
        if not text:
            empty += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            ok = len(parts) == 5 and parts[0] in {"0", "1", "2"} and all(0 <= float(v) <= 1 for v in parts[1:])
            if ok:
                counts[parts[0]] += 1
            else:
                bad += 1
    print(f"{s:5s} images={len(imgs)} labels={len(lbls)} missing={len(imgs ^ lbls)} "
          f"empty={empty} bad_lines={bad} boxes(helmet,head,person)=({counts['0']},{counts['1']},{counts['2']})")
    problems += len(imgs ^ lbls) + bad

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = len(seen[a] & seen[b])
    print(f"overlap {a}/{b}: {overlap}")
    problems += overlap

print("PASS" if problems == 0 else f"CHECK: {problems} problems")