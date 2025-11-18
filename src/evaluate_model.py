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
evaluate_model.py
Lightweight deployment-readiness evaluation for the Building Damage Assessment model.

This script reports three things for a given trained model checkpoint:

1. Checkpoint size:
   - Useful for understanding storage / transmission cost for field deployment.

2. Layer sparsity profile:
   - Iterates through Conv2d layers and reports how many weights are zero.
   - Produces an overall sparsity percentage and a per-layer breakdown.
   - Higher sparsity (more zeros) generally means lower memory bandwidth needs.

3. Inference latency (synthetic benchmark):
   - Runs N forward passes on random input at the target resolution
     and reports mean per-frame runtime (ms).
   - This is not full end-to-end FPS, but it's a good relative speed check
     between different checkpoints on the same hardware.

Typical usage
-------------
python evaluate_model.py \
    --yolov5_dir YOLOv5 \
    --weights runs/dp_yolov5/pruned/model_slimmed.pt \
    --imgsz 640 \
    --iters 50
-------------------------------------------------------------------------------
"""

import os
import sys
import time
import torch
import argparse
import pandas as pd
from pathlib import Path
import torch.nn as nn


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Evaluate model footprint, sparsity, and latency for deployment readiness."
    )
    ap.add_argument(
        "--yolov5_dir",
        type=str,
        required=True,
        help="Path to your local YOLOv5 repo (must contain models/, utils/, etc.).",
    )
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Path to the model checkpoint to evaluate. "
            "Must be a YOLOv5-format checkpoint like {'model': <DetectionModel>, ...}."
        ),
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size to benchmark (input will be [1,3,imgsz,imgsz]).",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of forward passes to average for latency.",
    )
    return ap


def pretty_mb(p: Path) -> str:
    """Return file size in MB, or 'missing' if not found."""
    if not p.exists():
        return "missing"
    return f"{p.stat().st_size / 1e6:.2f} MB"


def load_checkpoint_for_analysis(weights_path: Path):
    """
    Loads the checkpoint in Python and returns (model_module, epoch_hint).

    The checkpoint should look like:
        {'model': <nn.Module>, 'epoch': X, ...}
    The returned model is not moved to CUDA here; we just inspect it.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        epoch_hint = ckpt.get("epoch", None)
    else:
        # Fall back if user gave us a raw module
        model = ckpt
        epoch_hint = None

    if not hasattr(model, "state_dict"):
        raise RuntimeError(
            "This checkpoint does not contain a full model module. "
            "Pass the YOLOv5-format checkpoint (not just a bare state_dict)."
        )

    # put in eval mode for consistency
    model.eval()
    return model, epoch_hint


def measure_sparsity(model: torch.nn.Module):
    """
    Walk through Conv2d modules and compute:
        - number of parameters
        - number of non-zero parameters
        - per-layer sparsity %
    Also return global sparsity across all Conv2d layers.
    """
    rows = []
    total_params = 0
    total_nonzeros = 0

    for layer_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            numel = w.numel()
            nnz = (w != 0).sum().item()

            rows.append(
                {
                    "layer": layer_name,
                    "type": "Conv2d",
                    "params": numel,
                    "nonzeros": nnz,
                    "sparsity_%": 100.0 * (1.0 - (nnz / max(1, numel))),
                }
            )

            total_params += numel
            total_nonzeros += nnz

    overall_sparsity = 100.0 * (1.0 - (total_nonzeros / max(1, total_params)))

    df = pd.DataFrame(rows).sort_values("sparsity_%", ascending=False)
    return overall_sparsity, total_nonzeros, total_params, df


def benchmark_latency(yolov5_dir: Path, weights_path: Path, imgsz: int, iters: int):
    """
    Runs a synthetic throughput test:
    - Loads model into DetectMultiBackend (same loader used at inference time).
    - Warms up.
    - Measures avg forward time over N iters at resolution imgsz x imgsz.

    Returns avg_ms_per_infer.
    """
    # Make sure YOLOv5 internals are importable
    sys.path.insert(0, str(yolov5_dir))
    os.chdir(yolov5_dir)

    from models.common import DetectMultiBackend
    from utils.general import check_img_size
    from utils.torch_utils import select_device

    device = select_device("0" if torch.cuda.is_available() else "cpu")
    dm = DetectMultiBackend(str(weights_path), device=device)
    stride = int(dm.stride)
    imgsz = check_img_size(imgsz, s=stride)

    # Create dummy input
    x = torch.randn(1, 3, imgsz, imgsz).to(device)

    # use half precision if on CUDA for realistic throughput
    if device.type != "cpu":
        dm.model.half()
        x = x.half()

    # optional warmup
    if device.type != "cpu":
        dm.warmup(imgsz=(1, 3, imgsz, imgsz))

    # one dry run
    _ = dm(x)
    if device.type != "cpu":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = dm(x)
    if device.type != "cpu":
        torch.cuda.synchronize()
    dt = time.time() - t0

    avg_ms = (dt / max(1, iters)) * 1000.0
    return avg_ms, device.type, imgsz


def main():
    args = build_argparser().parse_args()

    weights_path = Path(args.weights).resolve()
    yolov5_dir = Path(args.yolov5_dir).resolve()

    # --- 1) File size report
    print("== Checkpoint Size ==")
    print(f"{weights_path.name}: {pretty_mb(weights_path)}")

    # --- 2) Load model into CPU and inspect sparsity
    print("\n== Sparsity Profile ==")
    model, epoch_hint = load_checkpoint_for_analysis(weights_path)
    overall_sparsity, total_nonzeros, total_params, df_layers = measure_sparsity(model)

    print(
        f"Overall Conv2d sparsity: {overall_sparsity:.2f}%  "
        f"(nonzeros {total_nonzeros:,} / {total_params:,})"
    )

    # Show the 10 most-sparse conv layers for a quick glance
    topk = min(10, len(df_layers))
    if topk > 0:
        print("\nTop sparse Conv2d layers:")
        print(df_layers.head(topk).to_string(index=False))

    # --- 3) Latency benchmark
    print("\n== Latency Benchmark ==")
    avg_ms, dev_type, checked_size = benchmark_latency(
        yolov5_dir=yolov5_dir,
        weights_path=weights_path,
        imgsz=args.imgsz,
        iters=args.iters,
    )
    print(
        f"Avg inference: {avg_ms:.2f} ms @ {checked_size} "
        f"(device={dev_type}, iters={args.iters})"
    )

    # --- 4) Metadata / summary
    print("\n== Summary ==")
    print(f"Epoch hint: {epoch_hint if epoch_hint is not None else 'n/a'}")
    print(f"Total Conv2d params inspected: {total_params:,}")
    print(f"Total Conv2d nonzeros:         {total_nonzeros:,}")
    print(f"Reported sparsity:             {overall_sparsity:.2f}%")
    print("This checkpoint is ready for downstream deployment tests (UAV feed, video, etc.).")


if __name__ == "__main__":
    main()

