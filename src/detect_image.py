
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
detect_image.py
Run building-damage detection on images using a (potentially DP-trained) YOLOv5 model.

Key features:
- Automatically unwraps an Opacus-wrapped detection head (GradSampleModule),
  so you can run inference without needing Opacus at runtime.
- Saves a clean checkpoint (so downstream responders don't need to know about DP internals).
- Filters out any detection below a confidence threshold (default 0.50).
- Hides confidence score on the visualization overlay. Boxes are labeled ONLY with
  damage class strings like "light damage", "severe damage", etc.
- Uses YOLOv5's built-in detect.py under the hood for the actual inference and plotting.

Typical usage:
    python detect_image.py \
        --yolov5_dir YOLOv5 \
        --weights runs/dp_yolov5/your_dp_run/dp_finetune_yolov5fmt.pt \
        --source assets/test_images \
        --out_dir runs/inference \
        --run_name dp_detect_clean \
        --conf_thres 0.50 \
        --imgsz 640

After running:
- Annotated images will be written to <out_dir>/<run_name>/
- Bounding boxes will show only class labels — NO confidence %
-------------------------------------------------------------------------------
"""

import os
import sys
import copy
import torch
from pathlib import Path
from contextlib import contextmanager
import argparse

# ------------------------------------------------------------------
# Small helper: load checkpoint and unwrap DP head if present
# ------------------------------------------------------------------
def load_and_clean_checkpoint(weights_path: str):
    """
    Loads a YOLOv5-format checkpoint that may contain an Opacus GradSampleModule
    in the final Detect head, unwraps it if needed, and returns:
        model_clean (nn.Module),
        epoch (int or -1)
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)  # local/trusted
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        epoch = ckpt.get("epoch", -1)
    else:
        # in case someone passes raw .pt with just the model
        model = ckpt
        epoch = -1

    # If the Detect head was still wrapped by Opacus (GradSampleModule),
    # unwrap it so we can run plain inference without Opacus.
    try:
        from opacus.grad_sample import GradSampleModule
        if isinstance(model.model[-1], GradSampleModule):
            model.model[-1] = copy.deepcopy(model.model[-1]._module)
            print("[detect_image] ✔ Unwrapped GradSampleModule on Detect head.")
    except Exception as e:
        # It's fine if Opacus isn't available or head is not wrapped
        print("[detect_image] GradSampleModule unwrap note:", e)

    # Move everything to CPU for saving a generic checkpoint
    model = model.cpu()

    return model, epoch


def save_clean_checkpoint(model, epoch: int, clean_out_path: str):
    """
    Save a new checkpoint that:
      - has plain model (no DP wrapper),
      - no optimizer state,
      - safe to ship for inference.
    """
    out_path = Path(clean_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model,
            "ema": None,
            "updates": 0,
            "optimizer": None,
            "epoch": epoch,
        },
        out_path,
    )
    print(f"[detect_image] ✅ Saved clean weights: {out_path}")
    return str(out_path)


# ------------------------------------------------------------------
# CLI + runner
# ------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Run YOLOv5 building-damage detection on images with confidence filtering and label-only overlays."
    )
    ap.add_argument(
        "--yolov5_dir",
        type=str,
        required=True,
        help="Path to local YOLOv5 repo root (must contain detect.py, models/, utils/ ...)",
    )
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained .pt (possibly DP-wrapped) checkpoint.",
    )
    ap.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file, directory of images, or glob (same semantics as YOLOv5 detect.py).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="runs/inference",
        help="Parent output dir for annotated results.",
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default="dp_detect_clean",
        help="Name of this inference run (subfolder in out_dir).",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size (square).",
    )
    ap.add_argument(
        "--conf_thres",
        type=float,
        default=0.50,
        help="Confidence threshold: only keep detections ≥ this.",
    )
    ap.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    ap.add_argument(
        "--max_det",
        type=int,
        default=1000,
        help="Maximum detections per image.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', '0', '1', ...",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    # ------------------------------------------------------------------
    # 1. Add YOLOv5 repo to path and import detect.run
    # ------------------------------------------------------------------
    yolov5_dir = Path(args.yolov5_dir).resolve()
    assert yolov5_dir.exists(), f"YOLOv5 dir not found: {yolov5_dir}"
    sys.path.insert(0, str(yolov5_dir))
    os.chdir(yolov5_dir)

    try:
        # Some older Opacus-pickled heads reference forbid_accumulation_hook.
        # We'll create a dummy no-op if it's missing.
        import opacus.privacy_engine as pe  # noqa
        if not hasattr(pe, "forbid_accumulation_hook"):
            from contextlib import contextmanager

            @contextmanager
            def forbid_accumulation_hook(*_args, **_kwargs):
                yield

            pe.forbid_accumulation_hook = forbid_accumulation_hook
    except Exception as e:
        # If Opacus isn't installed at all, that's okay for inference.
        print("[detect_image] Opacus patch note:", e)

    # now import detect.run dynamically from YOLOv5
    from detect import run as yolo_detect

    # ------------------------------------------------------------------
    # 2. Load the (possibly DP-wrapped) checkpoint and produce a clean version
    # ------------------------------------------------------------------
    model_obj, epoch = load_and_clean_checkpoint(args.weights)

    clean_weights_path = Path(args.weights).with_name(
        Path(args.weights).stem + "_CLEAN_FOR_INFERENCE.pt"
    )
    clean_weights_path = save_clean_checkpoint(model_obj, epoch, str(clean_weights_path))

    # ------------------------------------------------------------------
    # 3. Determine device to run inference on
    # ------------------------------------------------------------------
    if args.device == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device

    half_precision = dev != "cpu"  # we only run half() on GPU for speed

    # ------------------------------------------------------------------
    # 4. Run YOLOv5's detect() with our inference-time policies:
    #    - conf_thres = args.conf_thres
    #    - hide_conf = True (so only class labels, not percentages)
    # ------------------------------------------------------------------
    print("[detect_image] Starting inference...")
    yolo_detect(
        weights=str(clean_weights_path),
        source=args.source,
        imgsz=(args.imgsz, args.imgsz),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=dev,
        max_det=args.max_det,
        project=args.out_dir,
        name=args.run_name,
        exist_ok=True,
        half=half_precision,
        hide_conf=True,   # <-- IMPORTANT: label only (no "xx%")
        # leave save_conf=False (default) so label txt files won't include conf
    )

    print()
    print("============================================================")
    print("[detect_image] Inference complete.")
    print(f"[detect_image] Results (annotated .jpg/.png) in: {Path(args.out_dir)/args.run_name}")
    print("  • Each bounding box = one building / damage assessment.")
    print("  • Low-confidence (< conf_thres) detections are dropped.")
    print("  • Overlays show damage class ONLY (no confidence %).")
    print("============================================================")


if __name__ == "__main__":
    main()
