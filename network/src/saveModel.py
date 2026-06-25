import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import paths
from model import Classifier_I
from model import SwinTransformer_Unet_Luma
from model96 import SwinTransformer_Unet_Luma96


CLASSIFIER_I_GRID_SIZES = (
    (16, 16),
    (8, 8),
    (8, 4),
    (4, 8),
    (8, 2),
    (2, 8),
    (8, 1),
    (1, 8),
    (4, 2),
    (2, 4),
    (4, 1),
    (1, 4),
    (4, 4),
    (2, 2),
    (2, 1),
    (1, 2),
)


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_checkpoint_state_dict(checkpoint_path, device):
    source_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(source_dict, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in source_dict and isinstance(source_dict[key], dict):
                source_dict = source_dict[key]
                break
    return remove_prefix(source_dict, "module.")


def load_model_weights(model, checkpoint_path, device):
    source_dict = load_checkpoint_state_dict(checkpoint_path, device)
    dest_dict = model.state_dict()
    trained_dict = {
        key: value
        for key, value in source_dict.items()
        if key in dest_dict and value.shape == dest_dict[key].shape
    }
    missing = sorted(set(dest_dict.keys()) - set(trained_dict.keys()))
    unexpected = sorted(set(source_dict.keys()) - set(dest_dict.keys()))
    model.load_state_dict({**dest_dict, **trained_dict})
    return missing, unexpected


class ClassifierIExportWrapper(nn.Module):
    """TorchScript-friendly Classifier_I wrapper with explicit H/W branches."""

    def __init__(self, source):
        super().__init__()
        self.num_classes = int(source.num_classes)
        self.input_channel = int(source.input_channel)
        self.mask_value = float(source.mask_value)

        self.branch_16x16 = source.branches["16x16"]
        self.branch_8x8 = source.branches["8x8"]
        self.branch_8x4 = source.branches["8x4"]
        self.branch_4x8 = source.branches["4x8"]
        self.branch_8x2 = source.branches["8x2"]
        self.branch_2x8 = source.branches["2x8"]
        self.branch_8x1 = source.branches["8x1"]
        self.branch_1x8 = source.branches["1x8"]
        self.branch_4x2 = source.branches["4x2"]
        self.branch_2x4 = source.branches["2x4"]
        self.branch_4x1 = source.branches["4x1"]
        self.branch_1x4 = source.branches["1x4"]
        self.branch_4x4 = source.branches["4x4"]
        self.branch_2x2 = source.branches["2x2"]
        self.branch_2x1 = source.branches["2x1"]
        self.branch_1x2 = source.branches["1x2"]

        for grid_h, grid_w in CLASSIFIER_I_GRID_SIZES + ((1, 1),):
            key = "{}x{}".format(grid_h, grid_w)
            self.register_buffer("mask_" + key, getattr(source, "mask_" + key).clone())

    def _apply_mask(self, logits, mask):
        mask = mask.to(device=logits.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(logits)
        return logits.masked_fill(~mask, self.mask_value)

    def forward(
        self,
        x: torch.Tensor,
        dynamic_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> torch.Tensor:
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        if C != self.input_channel:
            raise RuntimeError("Classifier_I expects 2 input channels")

        if H == 16 and W == 16:
            logits = self.branch_16x16(x.reshape(B, -1))
            static_mask = self.mask_16x16
        elif H == 8 and W == 8:
            logits = self.branch_8x8(x.reshape(B, -1))
            static_mask = self.mask_8x8
        elif H == 8 and W == 4:
            logits = self.branch_8x4(x.reshape(B, -1))
            static_mask = self.mask_8x4
        elif H == 4 and W == 8:
            logits = self.branch_4x8(x.reshape(B, -1))
            static_mask = self.mask_4x8
        elif H == 8 and W == 2:
            logits = self.branch_8x2(x.reshape(B, -1))
            static_mask = self.mask_8x2
        elif H == 2 and W == 8:
            logits = self.branch_2x8(x.reshape(B, -1))
            static_mask = self.mask_2x8
        elif H == 8 and W == 1:
            logits = self.branch_8x1(x.reshape(B, -1))
            static_mask = self.mask_8x1
        elif H == 1 and W == 8:
            logits = self.branch_1x8(x.reshape(B, -1))
            static_mask = self.mask_1x8
        elif H == 4 and W == 2:
            logits = self.branch_4x2(x.reshape(B, -1))
            static_mask = self.mask_4x2
        elif H == 2 and W == 4:
            logits = self.branch_2x4(x.reshape(B, -1))
            static_mask = self.mask_2x4
        elif H == 4 and W == 1:
            logits = self.branch_4x1(x.reshape(B, -1))
            static_mask = self.mask_4x1
        elif H == 1 and W == 4:
            logits = self.branch_1x4(x.reshape(B, -1))
            static_mask = self.mask_1x4
        elif H == 4 and W == 4:
            logits = self.branch_4x4(x.reshape(B, -1))
            static_mask = self.mask_4x4
        elif H == 2 and W == 2:
            logits = self.branch_2x2(x.reshape(B, -1))
            static_mask = self.mask_2x2
        elif H == 2 and W == 1:
            logits = self.branch_2x1(x.reshape(B, -1))
            static_mask = self.mask_2x1
        elif H == 1 and W == 2:
            logits = self.branch_1x2(x.reshape(B, -1))
            static_mask = self.mask_1x2
        elif H == 1 and W == 1:
            logits = x.new_full((B, self.num_classes), self.mask_value)
            logits[:, 0] = 0
            static_mask = self.mask_1x1
        else:
            raise RuntimeError("Unsupported Classifier_I grid ROI")

        logits = self._apply_mask(logits, static_mask)
        if dynamic_mask is not None:
            logits = self._apply_mask(logits, dynamic_mask.to(dtype=torch.bool))
        if return_probs:
            return F.softmax(logits, dim=1)
        return logits


def export_classifier_i_script(checkpoint_path, output_path, device):
    model = Classifier_I()
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()
    wrapper = ClassifierIExportWrapper(model).eval()
    scripted = torch.jit.script(wrapper)
    scripted.save(str(output_path))
    return missing, unexpected


def export_classifier_i_traces(checkpoint_path, output_dir, device):
    model = Classifier_I()
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for grid_h, grid_w in CLASSIFIER_I_GRID_SIZES:
        dummy_input = torch.randn(1, 2, grid_h, grid_w, device=device)
        traced = torch.jit.trace(model, dummy_input, strict=False)
        traced.save(str(output_dir / "classifier_i_{}x{}.pt".format(grid_h, grid_w)))
    return missing, unexpected


def export_swin_luma(checkpoint_path, output_path, device):
    model = SwinTransformer_Unet_Luma()
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    dummy_qp = torch.randn(1, 1, device=device)
    traced = torch.jit.trace(model, (dummy_input, dummy_qp), strict=False)
    traced.save(str(output_path))
    return missing, unexpected


def export_swin_luma96(checkpoint_path, output_path, device, use_context_mask):
    model = SwinTransformer_Unet_Luma96(use_context_mask=use_context_mask)
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()
    dummy_input = torch.randn(1, 1, 96, 96, device=device)
    dummy_qp = torch.randn(1, 1, device=device)
    traced = torch.jit.trace(model, (dummy_input, dummy_qp), strict=False)
    traced.save(str(output_path))
    return missing, unexpected


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained checkpoints for C++ deployment.")
    parser.add_argument("--task", choices=("classifier_i_script", "classifier_i_traces", "swin_luma", "swin_luma96"), required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint under network/checkpoints.")
    parser.add_argument("--output", required=True, help="Output .pt path, or output directory for classifier_i_traces.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--useContextMask", action="store_true", help="Use the context-mask variant for SwinTransformer_Unet_Luma96.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.is_absolute():
        checkpoint_path = paths.project_root() / checkpoint_path
    if not output_path.is_absolute():
        output_path = paths.project_root() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "classifier_i_script":
        missing, unexpected = export_classifier_i_script(checkpoint_path, output_path, device)
    elif args.task == "classifier_i_traces":
        missing, unexpected = export_classifier_i_traces(checkpoint_path, output_path, device)
    elif args.task == "swin_luma":
        missing, unexpected = export_swin_luma(checkpoint_path, output_path, device)
    else:
        missing, unexpected = export_swin_luma96(checkpoint_path, output_path, device, args.useContextMask)

    print("Saved:", output_path)
    if missing:
        print("Missing checkpoint tensors:", len(missing))
    if unexpected:
        print("Unexpected checkpoint tensors:", len(unexpected))
