# ppe-cascade

Confidence-routed helmet and PPE detection. A fast YOLO detector handles most frames,
uncertain detections are verified by a small vision-language model, and anything still
uncertain is flagged for a human instead of guessed.

## Status
- [x] Data pipeline and fixed train/val/test split
- [x] YOLO11s baseline
- [ ] Tracking (ByteTrack)
- [ ] VLM verifier and confidence router
- [ ] Benchmark: YOLO only vs VLM only vs cascade
- [ ] Docker, API, demo video

## Baseline results (test set)
| Model | mAP50 | mAP50-95 | helmet AP50 | head AP50 |
|---|---|---|---|---|
| YOLO11s | TBD | TBD | TBD | TBD |