"""Train the YOLO baseline. Run on Colab with a GPU."""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True, help="output folder (a Google Drive path on Colab)")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,
        seed=0,
        workers=2,
        project=args.project,
        name="yolo11s_baseline",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()