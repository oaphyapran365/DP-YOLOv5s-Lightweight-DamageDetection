
"""
detect_video.py
Run building-damage detection on a video (UAV / aerial footage) using a YOLOv5 model.

Features:
- Loads a standard YOLOv5 .pt model (including your DP-finetuned weights after cleaning).
- Performs per-frame inference with letterbox preprocessing, then rescales boxes back.
- Drops any detection below --conf_thres.
- Draws bounding boxes with ONLY the damage class label (no confidence %).
- Uses a fixed color scheme aligned with triage semantics:
    "no damage"       -> sky color
    "light damage"    -> blue
    "moderate damage" -> orange
    "severe damage"   -> red
- Writes an annotated .mp4 and (optionally) periodic debug frames for QA.
- Supports --vid_stride to only write every Nth frame to the output video, which helps on CPU.

Example:
    python detect_video.py \
        --yolov5_dir YOLOv5 \
        --weights runs/dp_yolov5/clean/dp_finetune_clean_yolov5fmt.pt \
        --video_source assets/demo_video.mp4 \
        --out_mp4 runs/inference/demo_out.mp4 \
        --imgsz 640 \
        --conf_thres 0.35 \
        --vid_stride 2
"""

import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------------
# Fixed color mapping (OpenCV BGR)
# You asked for:
#   "no damage": sky color
#   "light damage": blue
#   "moderate damage": orange
#   "severe damage": red
#
# NOTE: OpenCV uses BGR, not RGB.
# We'll approximate "sky color" with a light sky-ish blue.
# --------------------------------------------------------------------------------
FIXED_COLORS = {
    "no damage":       (255, 230, 150),  # light sky-ish tone (B,G,R)
    "light damage":    (255,   0,   0),  # blue (in BGR, blue is (255,0,0))
    "moderate damage": (  0, 165, 255),  # orange
    "severe damage":   (  0,   0, 255),  # red
}


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Run YOLOv5 building-damage detection on video with fixed colors and label-only boxes."
    )
    ap.add_argument(
        "--yolov5_dir",
        type=str,
        required=True,
        help="Path to local YOLOv5 repo (must contain models/, utils/, etc.).",
    )
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to inference-ready YOLOv5 .pt weights (clean DP model is fine).",
    )
    ap.add_argument(
        "--video_source",
        type=str,
        required=True,
        help="Path to input video file. Must be readable by OpenCV.",
    )
    ap.add_argument(
        "--out_mp4",
        type=str,
        required=True,
        help="Path where the annotated .mp4 will be written.",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square). Larger can help detect small buildings but is slower.",
    )
    ap.add_argument(
        "--conf_thres",
        type=float,
        default=0.35,
        help="Confidence threshold. Detections below this are discarded.",
    )
    ap.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression.",
    )
    ap.add_argument(
        "--vid_stride",
        type=int,
        default=2,
        help="Write 1 out of every N frames to output video (to keep output filesize + CPU cost reasonable). "
             "Use 1 to write every frame.",
    )
    ap.add_argument(
        "--line_w",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    ap.add_argument(
        "--save_dbg",
        action="store_true",
        help="If set, dump periodic debug frames with overlays to --dbg_dir.",
    )
    ap.add_argument(
        "--dbg_dir",
        type=str,
        default="runs/inference/video_debug_frames",
        help="Where to dump debug frames if --save_dbg is set.",
    )
    ap.add_argument(
        "--dbg_every",
        type=int,
        default=150,
        help="Save one debug frame every N frames if --save_dbg is set.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or CUDA device like '0'.",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    # -------------------------------------------------------------------------
    # 1. Prep YOLOv5 environment and import runtime pieces
    # -------------------------------------------------------------------------
    yolov5_dir = Path(args.yolov5_dir).resolve()
    assert yolov5_dir.exists(), f"[detect_video] YOLOv5 dir not found: {yolov5_dir}"
    sys.path.insert(0, str(yolov5_dir))
    os.chdir(yolov5_dir)

    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, check_img_size, scale_boxes
    from utils.augmentations import letterbox as yv5_letterbox
    from utils.torch_utils import select_device
    from utils.plots import Annotator, colors  # YOLOv5's default palette (fallback)

    # -------------------------------------------------------------------------
    # 2. Device / model setup
    # -------------------------------------------------------------------------
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
        device = select_device("0" if use_cuda else "cpu")
    else:
        device = select_device(args.device)
        use_cuda = (str(device) != "cpu")

    model = DetectMultiBackend(args.weights, device=device)
    # model.names may be list or dict depending on export
    if isinstance(model.names, dict):
        names = [model.names[i] for i in sorted(model.names.keys())]
    else:
        names = list(model.names)
    stride = int(model.stride)
    imgsz = check_img_size(args.imgsz, s=stride)

    # half precision only on GPU
    half = bool(use_cuda)
    if half:
        model.model.half()

    print(f"[detect_video] Model device: {device}")
    print(f"[detect_video] Classes: {names}")
    print(f"[detect_video] Using imgsz={imgsz}, conf_thres={args.conf_thres}, iou_thres={args.iou_thres}")

    # optional warmup for TensorRT-ish speed on GPU
    if use_cuda:
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

    # -------------------------------------------------------------------------
    # 3. Open the input video with OpenCV
    # -------------------------------------------------------------------------
    video_source_path = Path(args.video_source).resolve()
    assert video_source_path.exists(), f"[detect_video] Cannot find video: {video_source_path}"

    cap = cv2.VideoCapture(str(video_source_path))
    ok, first = cap.read()
    assert ok and first is not None, f"[detect_video] Cannot read frames from: {video_source_path}"

    # Prepare output writer
    out_path = Path(args.out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first.shape[1])
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or first.shape[0])

    # We only WRITE every Nth frame, so output fps ~ fps_in / N.
    write_fps = max(fps_in / max(1, args.vid_stride), 1.0)
    writer = cv2.VideoWriter(str(out_path), fourcc, write_fps, (W, H))

    # Debug frame dump directory
    if args.save_dbg:
        dbg_dir = Path(args.dbg_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)
    else:
        dbg_dir = None

    # -------------------------------------------------------------------------
    # Helper: preprocess with YOLOv5's letterbox, return tensor + letterbox_hw
    # -------------------------------------------------------------------------
    def preprocess_letterbox(im_bgr):
        """
        Returns:
            im (torch.Tensor): model input BCHW, normalized [0,1], half/float based on device
            lb_shape (tuple): (H, W) of the letterboxed image (for later scale_boxes)
        """
        lb_img = yv5_letterbox(im_bgr, new_shape=imgsz, stride=stride, auto=True)[0]
        # BGR -> RGB, HWC -> CHW
        im_chw = lb_img[:, :, ::-1].transpose(2, 0, 1).copy()
        im_t = torch.from_numpy(im_chw).to(device)
        im_t = im_t.half() if half else im_t.float()
        im_t /= 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)
        return im_t, lb_img.shape[:2]  # (H, W)

    # -------------------------------------------------------------------------
    # 4. Main loop over frames
    # -------------------------------------------------------------------------
    frame_idx = 0
    total_kept = 0
    t0 = time.time()

    print("[detect_video] ▶️  Starting video inference ...")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Preprocess
        im_in, letterbox_hw = preprocess_letterbox(frame_bgr)

        # Inference + NMS
        with torch.no_grad():
            pred = model(im_in)
            pred = non_max_suppression(
                pred,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=1000
            )[0]

        # We'll annotate on a copy of the original frame
        out_frame = frame_bgr.copy()
        annotator = Annotator(out_frame, line_width=args.line_w)
        kept_this_frame = 0

        if pred is not None and len(pred):
            # map boxes from letterboxed tensor space -> original frame space
            pred[:, :4] = scale_boxes(
                (letterbox_hw[0], letterbox_hw[1]),  # letterbox height,width
                pred[:, :4],
                frame_bgr.shape                      # original frame shape
            ).round()

            for *xyxy, conf, cls in pred:
                if float(conf) < float(args.conf_thres):
                    continue

                x1, y1, x2, y2 = [
                    int(v.item()) if hasattr(v, "item") else int(v)
                    for v in xyxy
                ]
                # discard degenerate boxes (can happen on edge cases)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_idx = int(cls)
                label_text = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)

                # choose color: prefer our fixed scheme, fall back to YOLOv5 palette
                custom_color = FIXED_COLORS.get(label_text.lower(), None)
                box_color = custom_color if custom_color is not None else colors(cls_idx, True)

                # draw box + label (label only, NO confidence %)
                annotator.box_label((x1, y1, x2, y2), label_text, color=box_color)
                kept_this_frame += 1

        total_kept += kept_this_frame

        # light logging for situational awareness
        if frame_idx % 30 == 0:
            print(f"[detect_video] Frame {frame_idx:05d}: kept {kept_this_frame} boxes")

        # periodic debug frame dump for QA / after-action review
        if args.save_dbg and (frame_idx % args.dbg_every == 0):
            dbg_path = Path(dbg_dir) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(dbg_path), annotator.result())

        # write e.g. every 2nd frame to reduce output size / speed up CPU
        if args.vid_stride <= 1 or (frame_idx % args.vid_stride == 0):
            writer.write(annotator.result())

    # clean up
    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"[detect_video] ✅ Saved annotated video: {out_path}")
    print(f"[detect_video] Frames processed: {frame_idx}")
    print(f"[detect_video] Total kept boxes: {total_kept}")
    print(f"[detect_video] Elapsed: {elapsed:.1f}s")
    print(
        "[detect_video] Settings → "
        f"device={device}, imgsz={imgsz}, conf_thres={args.conf_thres}, "
        f"iou_thres={args.iou_thres}, vid_stride={args.vid_stride}"
    )
    if args.save_dbg:
        print(f"[detect_video] 🔎 Debug frames were saved to: {dbg_dir}")


if __name__ == "__main__":
    main()


