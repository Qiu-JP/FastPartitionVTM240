import argparse
from pathlib import Path

import numpy as np
import torch

import paths
from model import SwinTransformer_Unet_Luma96
from utils import (
    build_tensorboard_preview_background,
    gridmap_comparison_image,
    IdAlignedGridmapDataset,
)


def remove_prefix(state_dict, prefix):
    return {
        key.split(prefix, 1)[-1] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


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


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return paths.project_root() / path


def dataset_index_from_id(dataset, sample_id):
    try:
        loc = dataset.common_ids.get_loc(sample_id)
    except KeyError as exc:
        raise KeyError(f"Sample id not found in aligned input/gridmap ids: {sample_id}") from exc

    if isinstance(loc, slice):
        return loc.start
    if isinstance(loc, np.ndarray):
        matches = np.flatnonzero(loc) if loc.dtype == bool else loc
        if len(matches) == 0:
            raise KeyError(f"Sample id resolved to no dataset position: {sample_id}")
        return int(matches[0])
    return int(loc)


def select_sample(dataset, args):
    if args.sampleIndex is not None:
        sample_index = args.sampleIndex
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sampleIndex {sample_index} out of range [0, {len(dataset)})")
        return sample_index, dataset.common_ids[sample_index]

    if args.randomSample:
        rng = np.random.default_rng(args.seed)
        sample_index = int(rng.integers(0, len(dataset)))
        return sample_index, dataset.common_ids[sample_index]

    id_fields = (args.sequence, args.qp, args.frameID, args.ctuID)
    if all(value is not None for value in id_fields):
        sample_id = id_fields
        return dataset_index_from_id(dataset, sample_id), sample_id

    raise ValueError("Use --sampleIndex, --randomSample, or provide --sequence/--qp/--frameID/--ctuID.")


def save_visualization(image, output_name):
    from PIL import Image

    output_path = paths.ensure_dir(paths.network_root() / "figures") / Path(output_name).name
    Image.fromarray(image).save(output_path)
    return output_path


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)

    dataset = IdAlignedGridmapDataset(
        dataset_name=args.dataset,
        type=args.split,
        component=args.component,
    )
    sample_index, sample_id = select_sample(dataset, args)
    input_sample, qp_sample, label_gridmap = dataset[sample_index]

    model = SwinTransformer_Unet_Luma96(use_context_mask=args.useContextMask).to(device)
    missing, unexpected = load_model_weights(model, checkpoint_path, device)
    model.eval()

    pred_gridmap = model(
        input_sample.unsqueeze(0).to(device),
        qp_sample.unsqueeze(0).to(device),
    ).squeeze(0).cpu()

    background = build_tensorboard_preview_background(dataset, sample_index)
    image = gridmap_comparison_image(
        background=background,
        label_gridmap=label_gridmap.numpy(),
        pred_gridmap=pred_gridmap.numpy(),
        title=f"{args.dataset} {args.split} {sample_id} sample={sample_index}",
    )
    output_path = save_visualization(image, args.visualizeOutput)

    print("checkpoint:", checkpoint_path)
    print("dataset:", args.dataset, args.split, args.component)
    print("sample_id:", sample_id)
    print("sample_index:", sample_index)
    print("input_shape:", tuple(input_sample.shape))
    print("pred_shape:", tuple(pred_gridmap.shape))
    print("label_shape:", tuple(label_gridmap.shape))
    print("visualized:", output_path)
    if missing:
        print("missing_checkpoint_tensors:", len(missing))
    if unexpected:
        print("unexpected_checkpoint_tensors:", len(unexpected))


def parse_args():
    parser = argparse.ArgumentParser(description="Run 96x96 Swin gridmap inference for one dataset sample.")
    parser.add_argument("--checkpoint", required=True, help="Swin checkpoint path.")
    parser.add_argument("--dataset", default="DIV2K")
    parser.add_argument("--split", default="validating96")
    parser.add_argument("--component", choices=("Luma",), default="Luma")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--qp", type=int, default=None)
    parser.add_argument("--frameID", type=int, default=None)
    parser.add_argument("--ctuID", type=int, default=None)
    parser.add_argument("--sampleIndex", type=int, default=None)
    parser.add_argument("--randomSample", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--useContextMask", action="store_true")
    parser.add_argument("--visualizeOutput", default="inference96_gridmap.png", help="Preview image name under network/figures/.")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
