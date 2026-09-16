import argparse
import json
from pathlib import Path

import torch

import paths
from model import Classifier_I
from model import SwinTransformer_Unet


CLASSIFIER_I_GRID_SIZES = (
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


def serve_netron(model_path, host, port):
    try:
        import netron
    except ImportError as exc:
        raise ImportError(
            "netron is required for --viewNetron. "
            "Install it in the FastPartitionVTM environment first."
        ) from exc
    address = (host, port)
    print(f"Netron: http://{host}:{port}")
    print("Press Ctrl+C to stop the Netron server.")
    netron.start(str(model_path), address=address, browse=False)
    netron.wait()


def tensor_to_list(tensor):
    return tensor.detach().cpu().float().tolist()


def export_classifier_json(checkpoint_path, output_path, device):
    model = Classifier_I()
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()

    branches = []
    for grid_h, grid_w in CLASSIFIER_I_GRID_SIZES:
        key = "{}x{}".format(grid_h, grid_w)
        branch = model.branches[key]
        fc1 = branch[0]
        fc2 = branch[2]
        branches.append({
            "name": key,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "input_dim": int(fc1.in_features),
            "hidden_dim": int(fc1.out_features),
            "fc1_weight": tensor_to_list(fc1.weight),
            "fc1_bias": tensor_to_list(fc1.bias),
            "fc2_weight": tensor_to_list(fc2.weight),
            "fc2_bias": tensor_to_list(fc2.bias),
            "static_mask": tensor_to_list(getattr(model, "mask_" + key).to(dtype=torch.int32)),
        })

    payload = {
        "format": "ClassifierI.NativeJSON",
        "version": 1,
        "num_classes": int(model.num_classes),
        "input_channel": int(model.input_channel),
        "mask_value": float(model.mask_value),
        "branches": branches,
        "branch_1x1": {
            "grid_h": 1,
            "grid_w": 1,
            "static_mask": tensor_to_list(model.mask_1x1.to(dtype=torch.int32)),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return missing, unexpected


def export_swin_luma(checkpoint_path, output_path, device, use_context_mask):
    model = SwinTransformer_Unet(use_context_mask=use_context_mask)
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()
    dummy_input = torch.randn(1, 1, 48, 48, device=device)
    dummy_qp = torch.randn(1, 1, device=device)
    traced = torch.jit.trace(model, (dummy_input, dummy_qp), strict=False)
    traced.save(str(output_path))
    return missing, unexpected


def parse_args():
    parser = argparse.ArgumentParser(description="Export 48x48 trained checkpoints for C++ deployment.")
    parser.add_argument(
        "--task",
        choices=("export_classifier_json", "swin_luma"),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint under network/checkpoints.")
    parser.add_argument("--output", required=True, help="Output .pt or .native.json path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--useContextMask", action="store_true", default=True, help="Use the context-mask variant for SwinTransformer_Unet.")
    parser.add_argument("--viewNetron", action="store_true", help="Start a Netron server for the exported .pt model.")
    parser.add_argument("--netronHost", default="127.0.0.1", help="Host address for --viewNetron.")
    parser.add_argument("--netronPort", type=int, default=8080, help="Port for --viewNetron.")
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
    if args.task == "export_classifier_json":
        missing, unexpected = export_classifier_json(checkpoint_path, output_path, device)
    elif args.task == "swin_luma":
        missing, unexpected = export_swin_luma(checkpoint_path, output_path, device, args.useContextMask)

    print("Saved:", output_path)
    if missing:
        print("Missing checkpoint tensors:", len(missing))
    if unexpected:
        print("Unexpected checkpoint tensors:", len(unexpected))
    if args.viewNetron:
        serve_netron(output_path, args.netronHost, args.netronPort)
