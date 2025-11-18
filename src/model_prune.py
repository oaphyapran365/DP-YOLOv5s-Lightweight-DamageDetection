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
model_prune.py
Offline model slimming / sparsification for the trained YOLOv5 damage assessment model.

What this script does
---------------------
1. Loads a trained YOLOv5 checkpoint (including differentially private fine-tuned models).
   - The loader is robust to Opacus-wrapped detection heads used during DP training.
2. Reconstructs a fresh YOLOv5 model with the correct number of classes and class names.
3. Applies magnitude-based sparsification to convolutional layers in the backbone/neck
   while leaving the final detection head intact.
   - Lower-magnitude weights in those conv layers are set to zero.
   - The goal is to reduce model density and memory footprint for downstream deployment.
4. Saves two artifacts:
   (a) A ready-to-run YOLOv5-format checkpoint.
   (b) A light state_dict-only checkpoint for custom loaders.

Usage (example)
---------------
python model_prune.py \
    --yolov5_dir YOLOv5 \
    --data_yaml dataset-OC/data.yaml \
    --cfg_yaml models/yolov5s.yaml \
    --weights_in runs/dp_yolov5/final/dp_finetune_clean_yolov5fmt.pt \
    --weights_out runs/dp_yolov5/pruned/model_slimmed.pt \
    --amount 0.30
-------------------------------------------------------------------------------
"""

import os
import sys
import copy
import torch
import argparse
from pathlib import Path
from contextlib import contextmanager


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Post-training sparsification / slimming for YOLOv5 building-damage model."
    )
    ap.add_argument(
        "--yolov5_dir",
        type=str,
        required=True,
        help="Path to local YOLOv5 repo (must contain models/, utils/, etc.).",
    )
    ap.add_argument(
        "--data_yaml",
        type=str,
        required=True,
        help="Dataset .yaml (used to recover class names and nc).",
    )
    ap.add_argument(
        "--cfg_yaml",
        type=str,
        default="models/yolov5s.yaml",
        help="YOLOv5 model config .yaml used to rebuild the architecture.",
    )
    ap.add_argument(
        "--weights_in",
        type=str,
        required=True,
        help=(
            "Trained checkpoint to slim. This can be your DP fine-tuned model. "
            "Should be in 'yolov5fmt.pt' style, i.e. {'model': <DetectionModel>, ...}."
        ),
    )
    ap.add_argument(
        "--weights_out",
        type=str,
        required=True,
        help="Output path for the slimmed YOLOv5-format checkpoint.",
    )
    ap.add_argument(
        "--amount",
        type=float,
        default=0.30,
        help=(
            "Fraction of the smallest-magnitude Conv2d weights to zero out in each prunable layer. "
            "Example: 0.30 means ~30% of weights in each eligible conv layer get zeroed."
        ),
    )
    return ap


def patch_opacus_pickle_guard():
    """
    During DP training we used Opacus. Some checkpoints refer to
    opacus.privacy_engine.forbid_accumulation_hook, which may not exist
    in other environments. We inject a no-op stub so torch.load() works.
    """
    try:
        import opacus.privacy_engine as pe
        if not hasattr(pe, "forbid_accumulation_hook"):
            @contextmanager
            def forbid_accumulation_hook(*args, **kwargs):
                # no-op shim for legacy pickles
                yield
            pe.forbid_accumulation_hook = forbid_accumulation_hook

            # Also expose at builtins level so pickle can resolve it by qualname
            import builtins
            builtins.forbid_accumulation_hook = forbid_accumulation_hook
    except Exception as e:
        print("[model_prune] Opacus patch note:", e)


def load_trained_checkpoint(weights_in: Path):
    """
    Safely load a model checkpoint that may include DP wrappers.
    Returns (raw_model_like, epoch).
    """
    ckpt = torch.load(weights_in, map_location="cpu", weights_only=False)
    epoch = None
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", None)
        raw = ckpt.get("model", ckpt)
    else:
        raw = ckpt

    # Force float32 for safety
    if hasattr(raw, "float"):
        raw = raw.float()

    # If the Detect head was still wrapped by Opacus' GradSampleModule,
    # unwrap it so it's a standard nn.Module.
    try:
        from opacus.grad_sample import GradSampleModule
        if hasattr(raw, "model") and isinstance(raw.model[-1], GradSampleModule):
            raw.model[-1] = copy.deepcopy(raw.model[-1]._module)
            print("✔ Unwrapped GradSampleModule on Detect head.")
    except Exception as e:
        print("[model_prune] GradSampleModule unwrap note:", e)

    # Get a plain state_dict (covers both nn.Module and dict-of-tensors cases)
    state_dict = raw.state_dict() if hasattr(raw, "state_dict") else raw
    return state_dict, epoch


def rebuild_yolov5_model(yolov5_dir: Path, cfg_yaml: str, data_yaml: str, state_dict: dict):
    """
    Rebuild a fresh YOLOv5 model with correct nc and class names,
    then load the provided state_dict.
    """
    # Add YOLOv5 repo to path and import
    sys.path.insert(0, str(yolov5_dir))
    os.chdir(yolov5_dir)

    from models.yolo import Model
    from utils.general import check_dataset, intersect_dicts

    data_info = check_dataset(data_yaml)
    nc = int(data_info["nc"])
    names = data_info["names"]

    # Create new model with correct number of classes
    model = Model(cfg_yaml, ch=3, nc=nc)

    # Some checkpoints won't have identical keys (anchors etc). Safely intersect.
    ref_sd = model.state_dict()
    load_sd = intersect_dicts(state_dict, ref_sd, exclude=["anchors"])
    missing, unexpected = model.load_state_dict(load_sd, strict=False)
    if missing or unexpected:
        print("[model_prune] load_state_dict info",
              "missing:", missing, "unexpected:", unexpected)

    model.names = names
    model.eval()
    return model, names, nc


def apply_magnitude_sparsity(model, amount: float):
    """
    Apply magnitude-based sparsification layer-by-layer to Conv2d modules
    in the backbone/neck, while leaving the final Detect head intact.

    We:
    - Identify YOLOv5's Detect head (model.model[-1]) and collect its params.
    - For each Conv2d NOT belonging to that head, set the lowest-magnitude
      weights to zero (fraction = amount).
    - We then 'remove' the pruning reparam so the zeros are baked into weight.data.
    """
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    detect_head = model.model[-1]
    protected_param_ids = {id(p) for p in detect_head.parameters()}

    layers_modified = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # skip convs that are part of detection head
            if any(id(p) in protected_param_ids for p in layer.parameters()):
                continue

            # magnitude-based mask, applied per-layer
            prune.l1_unstructured(layer, name="weight", amount=amount)

            # make it permanent (remove pruning wrapper, keep zeros in .weight)
            prune.remove(layer, "weight")
            layers_modified += 1

    print(f"[model_prune] Applied magnitude sparsification to {layers_modified} Conv2d layers "
          f"(amount={amount*100:.0f}%).")
    return model


def save_outputs(model, names, nc, weights_out: Path, epoch_hint):
    """
    Save:
    (1) a YOLOv5-format checkpoint that DetectMultiBackend can load directly.
    (2) a light state_dict-only file for custom loading or analysis.
    """
    weights_out = Path(weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Full-style checkpoint
    ckpt_full = {
        "model": model,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "epoch": epoch_hint if epoch_hint is not None else -1,
    }
    torch.save(ckpt_full, weights_out)
    print(f"[model_prune] 💾 Saved slimmed model: {weights_out}")

    # 2. State dict only (more portable)
    state_only_path = weights_out.with_name(weights_out.stem + "_state_dict.pt")
    ckpt_state = {
        "model": model.state_dict(),
        "names": names,
        "nc": nc,
    }
    torch.save(ckpt_state, state_only_path)
    print(f"[model_prune] 💾 Saved state_dict-only: {state_only_path}")


def main():
    args = build_argparser().parse_args()

    yolov5_dir  = Path(args.yolov5_dir).resolve()
    weights_in  = Path(args.weights_in).resolve()
    weights_out = Path(args.weights_out).resolve()
    data_yaml   = str(Path(args.data_yaml).resolve())
    cfg_yaml    = str(Path(args.cfg_yaml).resolve())

    # 0. Safety checks
    assert yolov5_dir.exists(), f"[model_prune] YOLOv5 dir not found: {yolov5_dir}"
    assert weights_in.exists(), f"[model_prune] weights_in not found: {weights_in}"
    assert Path(data_yaml).exists(), f"[model_prune] data_yaml not found: {data_yaml}"
    assert Path(cfg_yaml).exists(), f"[model_prune] cfg_yaml not found: {cfg_yaml}"

    # 1. Patch Opacus pickle symbols if needed
    patch_opacus_pickle_guard()

    # 2. Load training checkpoint and unwrap DP head if present
    state_dict, epoch_hint = load_trained_checkpoint(weights_in)

    # 3. Rebuild YOLOv5 with correct nc/names and load weights
    model, names, nc = rebuild_yolov5_model(
        yolov5_dir=yolov5_dir,
        cfg_yaml=cfg_yaml,
        data_yaml=data_yaml,
        state_dict=state_dict,
    )

    # 4. Apply magnitude-based sparsification to Conv2d layers (except Detect head)
    model = apply_magnitude_sparsity(model, amount=args.amount)

    # 5. Save final artifacts
    save_outputs(model, names, nc, weights_out, epoch_hint)

    print("[model_prune] ✅ Done. Model has been slimmed and exported.")


if __name__ == "__main__":
    main()
