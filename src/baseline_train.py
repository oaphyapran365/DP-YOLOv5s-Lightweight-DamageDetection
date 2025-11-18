"""
Post-Disaster Damage Detection (YOLOv5s)

Copyright (C) 2025  Honghui Xu, Md Abdullahil Oaphy,
Kennesaw State University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------
baseline_train.py
Train a baseline YOLOv5s model on the building-damage dataset (no DP).

This script is a thin wrapper around Ultralytics YOLOv5's built-in train.py,
so users can reproduce the baseline checkpoint we compare against in the paper.

Usage:
    python baseline_train.py \
        --data data/dataset.yaml \
        --weights yolov5s.pt \
        --epochs 50 \
        --batch 16 \
        --imgsz 640 \
        --run_name baseline_yolov5s

Notes:
- We disable Weights & Biases (wandb) by default to avoid external logging.
- This expects that the YOLOv5 repo is available locally and that its
  train.py module can be imported.

-------------------------------------------------------------------------------
"""

import os, sys, argparse

# Don't log to wandb by default
os.environ["WANDB_DISABLED"] = "true"

# Add YOLOv5 repo to path.
# In your GitHub layout, you'll either:
#  (a) vendor YOLOv5 into a subfolder like third_party/yolov5
#  (b) or tell users to `git clone https://github.com/ultralytics/yolov5` and set PYTHONPATH.
#
# We'll assume YOLOv5/ is next to src/ in the repo root.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolov5_dir = f"{repo_root}/YOLOv5"
sys.path.insert(0, yolov5_dir)

import train as yolo_train  # YOLOv5's train.py

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dataset yaml (same format as YOLOv5 expects)")
    ap.add_argument("--weights", type=str, default="yolov5s.pt", help="initial weights")
    ap.add_argument("--cfg", type=str, default="models/yolov5s.yaml", help="model config within YOLOv5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--run_name", type=str, default="baseline_yolov5s", help="experiment name under runs/train/")
    return ap

def main():
    args = build_argparser().parse_args()

    # We translate our args to YOLOv5's train.run() signature.
    # YOLOv5 train.run(...) takes many kwargs. We'll send the important ones.
    yolo_train.run(
        imgsz=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        data=args.data,
        cfg=args.cfg,
        weights=args.weights,
        project="runs/train",
        name=args.run_name
    )

if __name__ == "__main__":
    main()
