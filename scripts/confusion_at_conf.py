"""Confusion matrix at a fixed confidence threshold with IoU matching.

Rows = predicted class, columns = true class, last row/column = background.
A ground-truth box with no match is a miss (background row).
A prediction with no match is a false alarm (background column).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def read_gt(label_path, w, h):
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            x, y, bw, bh = map(float, p[1:])
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            classes.append(c)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), classes


def save_plot(mat, labels, title, path, fmt):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title(title)
    cutoff = mat.max() / 2 if mat.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    color="white" if mat[i, j] > cutoff else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="data/hardhat_yolo")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to match a prediction to a ground-truth box")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    nc = len(names)
    root = Path(args.data_root)
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    def predict_in_chunks(paths):
        for i in range(0, len(paths), args.chunk):
            yield from model.predict(
                source=[str(f) for f in paths[i:i + args.chunk]],
                conf=args.conf, iou=0.6, imgsz=args.imgsz,
                batch=args.chunk, verbose=False,
            )

    mat = np.zeros((nc + 1, nc + 1))
    for r in predict_in_chunks(files):
        h, w = r.orig_shape
        gt_boxes, gt_cls = read_gt(lbl_dir / (Path(r.path).stem + ".txt"), w, h)
        pred_boxes = r.boxes.xyxy.cpu()
        pred_cls = r.boxes.cls.int().tolist()
        used_gt, used_pred = set(), set()
        if len(gt_cls) and len(pred_cls):
            iou = box_iou(gt_boxes, pred_boxes)
            pairs = (iou >= args.iou).nonzero()
            order = iou[pairs[:, 0], pairs[:, 1]].argsort(descending=True)
            for k in order.tolist():
                g, p = pairs[k].tolist()
                if g in used_gt or p in used_pred:
                    continue
                used_gt.add(g)
                used_pred.add(p)
                mat[pred_cls[p], gt_cls[g]] += 1
        for g in range(len(gt_cls)):
            if g not in used_gt:
                mat[nc, gt_cls[g]] += 1
        for p in range(len(pred_cls)):
            if p not in used_pred:
                mat[pred_cls[p], nc] += 1

    labels = names + ["background"]
    col = mat.sum(axis=0, keepdims=True)
    norm = np.divide(mat, col, out=np.zeros_like(mat), where=col > 0)

    tag = f"{args.split}_conf{args.conf}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_plot(mat, labels, f"Confusion ({args.split}, conf {args.conf}, counts)", out / f"cm_{tag}_counts.png", ".0f")
    save_plot(norm, labels, f"Confusion ({args.split}, conf {args.conf}, normalised by true class)", out / f"cm_{tag}_norm.png", ".2f")
    (out / f"cm_{tag}.json").write_text(json.dumps({"labels": labels, "matrix_pred_by_true": mat.tolist()}, indent=2))

    idx = {n: i for i, n in enumerate(names)}
    helmet, head = idx["helmet"], idx["head"]
    print(f"\nImages: {len(files)} | conf {args.conf} | match IoU {args.iou}")
    print("True head predicted as helmet (hidden violation):", int(mat[helmet, head]))
    print("True helmet predicted as head (false alarm)     :", int(mat[head, helmet]))
    print("True head missed entirely                       :", int(mat[nc, head]))
    print("True helmet missed entirely                     :", int(mat[nc, helmet]))
    print("Helmet predicted where nothing exists           :", int(mat[helmet, nc]))
    print("Head predicted where nothing exists             :", int(mat[head, nc]))
    print("\nPer class at this threshold:")
    for n in names:
        i = idx[n]
        tp = mat[i, i]
        rec = tp / mat[:, i].sum() if mat[:, i].sum() else 0
        prec = tp / mat[i, :].sum() if mat[i, :].sum() else 0
        print(f"  {n:7s} precision {prec:.3f} | recall {rec:.3f} | true boxes {int(mat[:, i].sum())}")


if __name__ == "__main__":
    main()