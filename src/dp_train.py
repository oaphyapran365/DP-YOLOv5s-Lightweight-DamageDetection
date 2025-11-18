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
dp_train.py
Differentially Private fine-tuning of a YOLOv5-style building-damage detector.

This script:
1. Loads a baseline YOLOv5s checkpoint (trained normally).
2. Freezes almost the whole network except:
   - the final detection head
   - optionally the last C3 block in the neck
3. Wraps ONLY the detection head in Opacus' PrivacyEngine to apply DP-SGD.
   => This gives per-sample gradient clipping + Gaussian noise at the head,
      which bounds what the model can "memorize" about any one training image.
4. Optimizes those unfrozen layers for EPOCHS steps on your disaster dataset.
5. Tracks privacy budget (epsilon) using the accountant in Opacus.
6. Saves 2 artifacts:
   - dp_finetune_yolov5fmt.pt  (full model object for inference)
   - dp_finetune_last_state_dict.pt (state_dict + metadata)

Usage (example):
    python dp_train.py \
        --data data/dataset.yaml \
        --cfg  YOLOv5/models/yolov5s.yaml \
        --baseline_ckpt runs/train/best.pt \
        --project runs/dp_yolov5 \
        --run_name yolov5s_dp_headonly_sigma0.50 \
        --epochs 50 \
        --batch 32 \
        --imgsz 640 \
        --lr 0.002 \
        --weight_decay 5e-4 \
        --sigma 0.50 \
        --clip 1.5 \
        --delta 1e-5

Important assumptions:
- You have a local YOLOv5 checkout at repo_root/YOLOv5
- You have installed:
    pip install opacus
    pip install -r YOLOv5/requirements.txt
- Your dataset YAML matches YOLOv5 format
  (contains: train, val, nc, names)
-------------------------------------------------------------------------------
"""

import os
import sys
import time
import re
import yaml
import copy
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Helper: add YOLOv5 repo to path based on this file's location
# We assume repo layout:
#   repo_root/
#       YOLOv5/
#       src/dp_train.py  <-- this file
# ------------------------------------------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_THIS_DIR)
_YOLOV5_DIR = os.path.join(_REPO_ROOT, "YOLOv5")
sys.path.insert(0, _YOLOV5_DIR)

from utils.general import (
    check_dataset,
    check_img_size,
    colorstr,
    intersect_dicts,
    increment_path,
)
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss
from utils.autoanchor import check_anchors
from models.yolo import Model
from val import run as val_run
from opacus.grad_sample import GradSampleModule
from opacus import PrivacyEngine


# ------------------------------------------------------------------
# small util: BN freeze
# ------------------------------------------------------------------
def _set_bn_eval(m: nn.Module):
    """Put all BatchNorm layers in eval mode (stops them from updating stats)."""
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
        m.eval()


def _find_last_c3(model_module: nn.Module):
    """Return the last C3 block in the YOLOv5 neck/head graph."""
    last = None
    # model.model is a nn.ModuleList([...])
    for m in list(model_module.model)[::-1]:
        if m.__class__.__name__ == "C3":
            last = m
            break
    return last


def _strip_gsm_hooks(m: nn.Module):
    """
    Clean up stale autograd_grad_sample_hooks if the model was wrapped/unwrapped
    before. This prevents double-registration in Opacus.
    """
    if hasattr(m, "autograd_grad_sample_hooks"):
        for h in list(m.autograd_grad_sample_hooks):
            try:
                h.remove()
            except Exception:
                pass
        m.autograd_grad_sample_hooks = []
    for c in m.children():
        _strip_gsm_hooks(c)


def _parse_results_txt(results_file: Path):
    """
    YOLOv5 val() writes results.txt with lines like:
    'all ... P ... R ... mAP@0.5 ... mAP@0.5:0.95 ...'
    We'll grab the last 'all' line.
    """
    if not results_file.exists():
        return None
    txt = results_file.read_text()
    for line in txt.splitlines()[::-1]:
        if line.strip().startswith("all"):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            vals = [float(x) for x in nums]
            # heuristic: last 4 floats are P, R, mAP50, mAP50-95
            if len(vals) >= 4:
                return vals[-4], vals[-3], vals[-2], vals[-1]
    return None


# ------------------------------------------------------------------
# arg parser
# ------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Head-only DP fine-tuning for YOLOv5s damage model")
    ap.add_argument("--data", type=str, required=True,
                    help="dataset yaml with train/val paths, nc, names")
    ap.add_argument("--cfg", type=str, required=True,
                    help="YOLOv5 model cfg (e.g. YOLOv5/models/yolov5s.yaml)")
    ap.add_argument("--baseline_ckpt", type=str, required=True,
                    help="baseline non-DP checkpoint to start from (e.g. runs/train/best.pt)")
    ap.add_argument("--project", type=str, default="runs/dp_yolov5",
                    help="output parent dir for this DP run")
    ap.add_argument("--run_name", type=str, default="dp_run",
                    help="subdir name inside --project")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3,
                    help="learning rate for fine-tune")
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--sigma", type=float, default=0.50,
                    help="noise multiplier for DP-SGD")
    ap.add_argument("--clip", type=float, default=1.5,
                    help="per-sample grad norm clip")
    ap.add_argument("--delta", type=float, default=1e-5,
                    help="privacy delta for epsilon accounting")
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto', '0', 'cpu', etc.")
    return ap


# ------------------------------------------------------------------
# main training logic
# ------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()

    # Make output dir
    os.makedirs(args.project, exist_ok=True)
    save_dir = increment_path(Path(args.project) / args.run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dp_train] Saving run artifacts to: {save_dir}")

    # For reproducibility / cleanliness
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # --------------------------
    # Device select
    # --------------------------
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_str = "0"
            print(f"[dp_train] Using GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            device_str = "cpu"
            print("[dp_train] CUDA not available; using CPU.")
    elif args.device == "cpu":
        device = torch.device("cpu")
        device_str = "cpu"
    else:
        # user passed "0", "1", etc.
        device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")
        device_str = args.device
    torch.backends.cudnn.benchmark = True

    # --------------------------
    # Load dataset config
    # --------------------------
    data_cfg = check_dataset(args.data)
    nc = int(data_cfg["nc"])
    names = data_cfg["names"]

    # --------------------------
    # Build YOLOv5 model skeleton
    # --------------------------
    model = Model(args.cfg, ch=3, nc=nc)

    # --- SAFE load of baseline checkpoint weights ---
    ckpt_path = Path(args.baseline_ckpt)
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 0, \
        f"Checkpoint missing or empty: {ckpt_path}"

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e_safe:
        print(f"[dp_train] weights_only=True failed: {e_safe}")
        print("[dp_train] Retrying with weights_only=False (assumes trusted checkpoint).")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw = (ckpt.get("ema") or ckpt.get("model") or ckpt) if isinstance(ckpt, dict) else ckpt
    src_sd = raw.float().state_dict() if hasattr(raw, "state_dict") else raw

    # Only load matching keys (ignore anchors mismatch)
    src_sd = intersect_dicts(src_sd, model.state_dict(), exclude=["anchors"])
    model.load_state_dict(src_sd, strict=False)
    model.names = names

    # --------------------------
    # Freeze everything first
    # --------------------------
    for p in model.parameters():
        p.requires_grad = False

    # Keep batchnorms frozen (important for stability in DP fine-tune)
    model.apply(_set_bn_eval)

    # Move model to device and init detection head biases if available
    model = model.to(device).float()
    if hasattr(model, "_initialize_biases"):
        model._initialize_biases()

    # --------------------------
    # Pick layers we DO want to train under DP:
    #   - detection head (model.model[-1])
    #   - also optionally last C3 in the neck to give it some flex
    # --------------------------
    detect_head = model.model[-1]  # Detect()
    last_c3 = _find_last_c3(model)

    for p in detect_head.parameters():
        p.requires_grad = True
    if last_c3 is not None:
        for p in last_c3.parameters():
            p.requires_grad = True

    # --------------------------
    # Hyperparameters / augment
    # Reuse YOLOv5 hyp.scratch-low but soften the heavy augmentations
    # --------------------------
    hyp_path = Path(_YOLOV5_DIR) / "data" / "hyps" / "hyp.scratch-low.yaml"
    hyp = yaml.safe_load(open(hyp_path))

    hyp_updates = {
        "lr0": args.lr,
        "lrf": 0.05,
        "momentum": 0.937,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 2.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # keep geometric augment mild; don't explode privacy noise with crazy distorts
        "hsv_h": hyp.get("hsv_h", 0.015),
        "hsv_s": hyp.get("hsv_s", 0.7),
        "hsv_v": hyp.get("hsv_v", 0.4),
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.1,
        "copy_paste": 0.0,
    }
    hyp.update(hyp_updates)
    model.hyp = hyp

    # --------------------------
    # Data loaders (YOLOv5 builtin)
    # --------------------------
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.imgsz, s=gs)

    train_loader = create_dataloader(
        path=data_cfg["train"],
        imgsz=imgsz,
        batch_size=args.batch,
        stride=gs,
        hyp=hyp,
        augment=True,
        cache=None,
        rect=False,
        rank=-1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
    )[0]

    # sanity: anchors
    try:
        check_anchors(train_loader.dataset, model=model, thr=4.0, imgsz=imgsz)
    except Exception as e:
        print("[dp_train] anchor check warning:", e)

    val_src = data_cfg.get("val") or data_cfg.get("valid") or data_cfg.get("test")
    val_loader = create_dataloader(
        path=val_src,
        imgsz=imgsz,
        batch_size=max(1, args.batch // 4),
        stride=gs,
        hyp=hyp,
        augment=False,
        cache=None,
        rect=True,
        rank=-1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("val: "),
        shuffle=False,
    )[0]

    # --------------------------
    # Clean any stale grad_sample hooks on the head
    # --------------------------
    if isinstance(detect_head, GradSampleModule):
        detect_head = detect_head._module
    _strip_gsm_hooks(detect_head)
    detect_head = copy.deepcopy(detect_head).to(device)
    model.model[-1] = detect_head  # re-attach clean copy

    # --------------------------
    # Build optimizers
    # We'll keep 2 param groups:
    #   ndp_params = last_c3 (non-DP SGD)
    #   dp_params  = detect head (DP-SGD via Opacus)
    # --------------------------
    dp_params = list(model.model[-1].parameters())
    dp_ids = {id(p) for p in dp_params}
    ndp_params = [
        p for p in model.parameters()
        if p.requires_grad and (id(p) not in dp_ids)
    ]

    optim_ndp = (
        torch.optim.SGD(
            ndp_params,
            lr=args.lr,
            momentum=0.937,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
        if ndp_params
        else None
    )

    optim_dp = torch.optim.SGD(
        dp_params,
        lr=args.lr,
        momentum=0.937,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    steps_per_epoch = max(1, len(train_loader))
    sched_ndp = (
        torch.optim.lr_scheduler.OneCycleLR(
            optim_ndp,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        if optim_ndp
        else None
    )
    sched_dp = torch.optim.lr_scheduler.OneCycleLR(
        optim_dp,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # --------------------------
    # Attach Opacus PrivacyEngine ONLY to detection head optimizer
    # --------------------------
    privacy_engine = PrivacyEngine()
    dp_head, optim_dp, train_loader = privacy_engine.make_private(
        module=model.model[-1],
        optimizer=optim_dp,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.clip,
    )

    # After wrapping, restore YOLOv5 Detect() routing attrs like .f (feature indices)
    # so forward() still works downstream.
    orig = detect_head
    for attr in ("f", "i", "type", "np"):
        if hasattr(orig, attr):
            setattr(dp_head, attr, getattr(orig, attr))

    model.model[-1] = dp_head

    # ComputeLoss needs to see an *unwrapped* head for targets assignment
    unwrapped = dp_head._module if isinstance(dp_head, GradSampleModule) else dp_head
    model.model[-1] = unwrapped
    compute_loss = ComputeLoss(model)
    model.model[-1] = dp_head

    model.train()

    # --------------------------
    # TRAIN LOOP
    # --------------------------
    start = time.time()
    eps = float("nan")

    for epoch in range(args.epochs):
        model.train()
        running_sum = 0.0
        nb = len(train_loader)

        for _, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device, non_blocking=True)

            if optim_ndp:
                optim_ndp.zero_grad(set_to_none=True)
            optim_dp.zero_grad(set_to_none=True)

            preds = model(imgs)
            loss, loss_items = compute_loss(preds, targets)
            loss.backward()

            if optim_ndp:
                optim_ndp.step()
                if sched_ndp:
                    sched_ndp.step()

            optim_dp.step()
            sched_dp.step()

            running_sum += loss_items.sum().item()

        # track privacy after each epoch
        try:
            eps = privacy_engine.get_epsilon(delta=args.delta)
        except Exception:
            eps = float("nan")

        print(
            f"[dp_train] Epoch {epoch+1:>3}/{args.epochs} "
            f"| loss {(running_sum/nb):.4f} | ε≈{eps:.2f}"
        )

    # --------------------------
    # VALIDATION / METRICS
    # --------------------------
    model.eval()
    final_val_dir = save_dir / "val_final"
    final_val_dir.mkdir(parents=True, exist_ok=True)

    P = R = mAP50 = mAP5095 = float("nan")

    try:
        ret = val_run(
            data=data_cfg,
            model=model,
            dataloader=val_loader,
            imgsz=imgsz,
            batch_size=max(1, args.batch // 4),
            device=device_str,
            half=False,
            task="val",
            verbose=False,
            save_dir=final_val_dir,
            plots=False,
        )

        # YOLOv5 val.run() usually returns (P, R, mAP50, mAP5095, ...)
        if isinstance(ret, (list, tuple)) and len(ret) > 0:
            first = ret[0] if isinstance(ret[0], (list, tuple)) else ret
            try:
                P, R, mAP50, mAP5095 = [float(x) for x in first[:4]]
            except Exception:
                pass

    except Exception as e:
        print("[dp_train] val_run warning:", e)

    # fallback: parse results.txt if metrics didn't parse
    if any(x != x for x in [P, R, mAP50, mAP5095]):  # NaN check
        parsed = _parse_results_txt(final_val_dir / "results.txt")
        if parsed:
            P, R, mAP50, mAP5095 = parsed

    print(
        f"[dp_train] Final metrics:"
        f" P {P:.3f} | R {R:.3f} | mAP@0.5 {mAP50:.3f} | mAP@0.5:.95 {mAP5095:.3f}"
    )

    # --------------------------
    # SAVE CHECKPOINTS
    # --------------------------
    ckpt_sd = save_dir / "dp_finetune_last_state_dict.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epsilon": eps,
            "names": names,
            "nc": nc,
        },
        ckpt_sd,
    )

    ckpt_full = save_dir / "dp_finetune_yolov5fmt.pt"
    torch.save(
        {
            "model": model,
            "ema": None,
            "updates": 0,
            "optimizer": None,
            "epoch": args.epochs - 1,
        },
        ckpt_full,
    )

    dur_min = (time.time() - start) / 60.0
    print(
        f"[dp_train] ✅ Done in {dur_min:.1f} min | "
        f"Saved:\n  • {ckpt_sd}\n  • {ckpt_full}\n"
        f"Final ε≈{eps:.2f}"
    )


if __name__ == "__main__":
    main()
