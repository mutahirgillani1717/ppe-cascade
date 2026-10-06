"""Evaluate a YOLO model on a split and save metrics to JSON."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/baseline_test.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project="results",
        name=f"baseline_{args.split}_plots",
        exist_ok=True,
    )
    box = m.box
    per_class = {
        m.names[int(c)]: {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        }
        for i, c in enumerate(box.ap_class_index)
    }
    result = {
        "weights": args.weights,
        "split": args.split,
        "imgsz": args.imgsz,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()