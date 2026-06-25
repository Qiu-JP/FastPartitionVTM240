import os
import argparse
import paths
import random
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from tqdm import tqdm

from utils import (
    train_one_epoch,
    evaluate,
    adjust_learning_rate,
    train_one_epoch_classifier,
    train_one_epoch_multi,
    evaluate_multi,
)
from model96 import SwinTransformer_Unet_Luma96 as model96
from model import Classifier_I as classifier_i


ID_COLUMNS = ["sequence_name", "qp", "frame_id", "ctu_id"]
CLASSIFIER_SUPPORTED_SIZES = {
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
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_file(args):
    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    os.makedirs(log_out_dir, exist_ok=True)
    log_path = os.path.join(log_out_dir, args.logFile)
    log_file = open(log_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("Log file:", log_path)
    return log_file


def setup_tensorboard(args, log_out_dir):
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is not installed in the FastPartitionVTM environment. "
            "Install it first, for example: "
            "/home/qiujunpeng.726/miniconda3/envs/FastPartitionVTM/bin/python -m pip install tensorboard"
        )
    tb_log_dir = args.tbLogDir
    if tb_log_dir is None:
        tb_log_dir = os.path.join(log_out_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print("TensorBoard log dir:", tb_log_dir)
    return writer


def split_dir_from_name(type):
    return type.lower()


def load_model_weights(model, checkpoint_path, device, model_name):
    if checkpoint_path is None:
        return
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{model_name} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    model.load_state_dict(checkpoint)
    print(f"Loaded {model_name} checkpoint:", checkpoint_path)


def is_supported_classifier_size(h, w):
    return (h, w) in CLASSIFIER_SUPPORTED_SIZES


def _full_vertical_line(gridmap, x, y0, h, threshold):
    if x < 0 or x >= gridmap.shape[-1]:
        return False
    return bool((gridmap[0, y0:y0 + h, x] > threshold).all().item())


def _full_horizontal_line(gridmap, y, x0, w, threshold):
    if y < 0 or y >= gridmap.shape[-2]:
        return False
    return bool((gridmap[1, y, x0:x0 + w] > threshold).all().item())


def infer_split_label_from_gridmap(gridmap, y0, x0, h, w, threshold=0.5):
    if h <= 1 and w <= 1:
        return 0

    can_qt = h == w and h >= 4
    if can_qt:
        has_v_mid = _full_vertical_line(gridmap, x0 + w // 2 - 1, y0, h, threshold)
        has_h_mid = _full_horizontal_line(gridmap, y0 + h // 2 - 1, x0, w, threshold)
        if has_v_mid and has_h_mid:
            return 1

    if h >= 4:
        has_h_q1 = _full_horizontal_line(gridmap, y0 + h // 4 - 1, x0, w, threshold)
        has_h_q3 = _full_horizontal_line(gridmap, y0 + (3 * h) // 4 - 1, x0, w, threshold)
        if has_h_q1 and has_h_q3:
            return 4

    if w >= 4:
        has_v_q1 = _full_vertical_line(gridmap, x0 + w // 4 - 1, y0, h, threshold)
        has_v_q3 = _full_vertical_line(gridmap, x0 + (3 * w) // 4 - 1, y0, h, threshold)
        if has_v_q1 and has_v_q3:
            return 5

    if h >= 2 and _full_horizontal_line(gridmap, y0 + h // 2 - 1, x0, w, threshold):
        return 2

    if w >= 2 and _full_vertical_line(gridmap, x0 + w // 2 - 1, y0, h, threshold):
        return 3

    return 0


def split_children(y0, x0, h, w, label):
    if label == 1:
        h2 = h // 2
        w2 = w // 2
        return [
            (y0, x0, h2, w2),
            (y0, x0 + w2, h2, w2),
            (y0 + h2, x0, h2, w2),
            (y0 + h2, x0 + w2, h2, w2),
        ]
    if label == 2:
        h2 = h // 2
        return [(y0, x0, h2, w), (y0 + h2, x0, h2, w)]
    if label == 3:
        w2 = w // 2
        return [(y0, x0, h, w2), (y0, x0 + w2, h, w2)]
    if label == 4:
        h1 = h // 4
        h2 = h // 2
        return [(y0, x0, h1, w), (y0 + h1, x0, h2, w), (y0 + h1 + h2, x0, h1, w)]
    if label == 5:
        w1 = w // 4
        w2 = w // 2
        return [(y0, x0, h, w1), (y0, x0 + w1, h, w2), (y0, x0 + w1 + w2, h, w1)]
    return []


def collect_classifier_nodes(target_gridmap, threshold=0.5, max_nodes=0):
    _, root_h, root_w = target_gridmap.shape
    stack = [(0, 0, root_h, root_w)]
    nodes = []

    while stack:
        y0, x0, h, w = stack.pop()
        if not is_supported_classifier_size(h, w):
            continue

        label = infer_split_label_from_gridmap(target_gridmap, y0, x0, h, w, threshold)
        nodes.append((y0, x0, h, w, label))
        if max_nodes > 0 and len(nodes) >= max_nodes:
            break

        for child in reversed(split_children(y0, x0, h, w, label)):
            cy, cx, ch, cw = child
            if ch >= 1 and cw >= 1:
                stack.append((cy, cx, ch, cw))

    return nodes


def classifier_tree_loss(classifier, pred_gridmap, target_gridmap, ce_loss, threshold=0.5, max_nodes=0):
    node_batches = {}
    total_nodes = 0

    for batch_idx in range(target_gridmap.shape[0]):
        nodes = collect_classifier_nodes(
            target_gridmap[batch_idx].detach(),
            threshold=threshold,
            max_nodes=max_nodes,
        )
        for y0, x0, h, w, label in nodes:
            key = (h, w)
            if key not in node_batches:
                node_batches[key] = {"roi": [], "label": []}
            node_batches[key]["roi"].append(pred_gridmap[batch_idx:batch_idx + 1, :, y0:y0 + h, x0:x0 + w])
            node_batches[key]["label"].append(label)
            total_nodes += 1

    if total_nodes == 0:
        zero = pred_gridmap.sum() * 0.0
        return zero, 0.0, 0

    losses = []
    correct = 0
    for key, payload in node_batches.items():
        roi = torch.cat(payload["roi"], dim=0)
        labels = torch.tensor(payload["label"], dtype=torch.long, device=pred_gridmap.device)
        logits = classifier(roi)
        losses.append(ce_loss(logits, labels))
        correct += torch.sum(torch.argmax(logits, dim=1) == labels).item()

    return torch.stack(losses).mean(), correct / float(total_nodes), total_nodes


def classifier_node_loss(classifier, pred_gridmap, node_batch, ce_loss, max_nodes=0, return_size_stats=False):
    node_batches = {}
    total_nodes = 0

    for batch_idx, nodes in enumerate(node_batch):
        if torch.is_tensor(nodes):
            nodes_iter = nodes.tolist()
        else:
            nodes_iter = nodes
        for node in nodes_iter:
            grid_y, grid_x, grid_h, grid_w, label = [int(v) for v in node]
            if max_nodes > 0 and total_nodes >= max_nodes * pred_gridmap.shape[0]:
                break
            if not is_supported_classifier_size(grid_h, grid_w):
                continue
            key = (grid_h, grid_w)
            if key not in node_batches:
                node_batches[key] = {"roi": [], "label": []}
            node_batches[key]["roi"].append(
                pred_gridmap[
                    batch_idx:batch_idx + 1,
                    :,
                    grid_y:grid_y + grid_h,
                    grid_x:grid_x + grid_w,
                ]
            )
            node_batches[key]["label"].append(label)
            total_nodes += 1

    if total_nodes == 0:
        zero = pred_gridmap.sum() * 0.0
        return zero, 0.0, 0

    losses = []
    correct = 0
    size_stats = {}
    for key, payload in node_batches.items():
        roi = torch.cat(payload["roi"], dim=0)
        labels = torch.tensor(payload["label"], dtype=torch.long, device=pred_gridmap.device)
        logits = classifier(roi)
        size_loss = ce_loss(logits, labels)
        losses.append(size_loss)
        size_correct = torch.sum(torch.argmax(logits, dim=1) == labels).item()
        correct += size_correct
        if return_size_stats:
            size_stats[key] = {
                "loss": float(size_loss.detach().item()),
                "correct": int(size_correct),
                "total": int(labels.numel()),
            }

    loss = torch.stack(losses).mean()
    acc = correct / float(total_nodes)
    if return_size_stats:
        return loss, acc, total_nodes, size_stats
    return loss, acc, total_nodes


def collate_gridmap_with_nodes(batch):
    inputs, qps, gridmaps, nodes = zip(*batch)
    return (
        torch.stack(inputs, dim=0),
        torch.stack(qps, dim=0),
        torch.stack(gridmaps, dim=0),
        list(nodes),
    )


def update_size_stats(total_stats, batch_stats):
    for key, stats in batch_stats.items():
        if key not in total_stats:
            total_stats[key] = {"loss_sum": 0.0, "batches": 0, "correct": 0, "total": 0}
        total_stats[key]["loss_sum"] += stats["loss"]
        total_stats[key]["batches"] += 1
        total_stats[key]["correct"] += stats["correct"]
        total_stats[key]["total"] += stats["total"]


def format_size_stats(size_stats):
    parts = []
    for key in sorted(size_stats):
        stats = size_stats[key]
        total = stats["total"]
        batches = max(stats["batches"], 1)
        acc = stats["correct"] / float(total) if total > 0 else 0.0
        loss = stats["loss_sum"] / float(batches)
        parts.append("{}x{}:loss={:.6f},acc={:.6f},n={}".format(key[0], key[1], loss, acc, total))
    return "; ".join(parts)


class IdAlignedGridmapDataset(Dataset):
    def __init__(self, dataset_name, type, component="Luma", min_qp=0, max_qp=51):
        self.dataset_name = dataset_name
        self.type = type
        self.component = component
        self.min_qp = min_qp
        self.max_qp = max_qp
        if max_qp <= min_qp:
            raise ValueError(f"max_qp must be larger than min_qp, got {min_qp} and {max_qp}")
        split_dir = split_dir_from_name(type)
        dataset_dir = paths.dataset_root() / dataset_name / split_dir
        self.input_path = dataset_dir / f"{component}_Input.pkl"
        self.gridmap_path = dataset_dir / f"{component}_Gridmap.pkl"
        self.input_npy_path = dataset_dir / f"{component}_Input.npy"
        self.gridmap_npy_path = dataset_dir / f"{component}_Gridmap.npy"
        self.ids_path = dataset_dir / f"{component}_Ids.pkl"

        if self.input_npy_path.exists() and self.gridmap_npy_path.exists() and self.ids_path.exists():
            print(
                f"{component} {type}: using mmap arrays "
                f"{self.input_npy_path.name}, {self.gridmap_npy_path.name}"
            )
            self.input_array = np.load(self.input_npy_path, mmap_mode="r")
            self.gridmap_array = np.load(self.gridmap_npy_path, mmap_mode="r")
            ids = pd.read_pickle(self.ids_path)
            self.input_ids = ids
            self.gridmap_ids = ids
        else:
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input pkl not found: {self.input_path}")
            if not self.gridmap_path.exists():
                raise FileNotFoundError(f"Gridmap pkl not found: {self.gridmap_path}")
            print(
                f"{component} {type}: mmap npy files not found, falling back to pkl. "
                "This loads the full dataset into CPU memory. "
                "Run export_dataset_npy.py first to reduce training memory use."
            )
            input_payload = pd.read_pickle(self.input_path)
            gridmap_payload = pd.read_pickle(self.gridmap_path)
            self.input_array = input_payload["input"]
            self.gridmap_array = gridmap_payload["gridmap"]
            self.input_ids = input_payload["ids"]
            self.gridmap_ids = gridmap_payload["ids"]

        if list(self.input_ids.index.names) != ID_COLUMNS:
            raise RuntimeError(
                f"input ids must be indexed by {ID_COLUMNS}. "
                "Regenerate the pkl files with the current createDataset.py."
            )
        if list(self.gridmap_ids.index.names) != ID_COLUMNS:
            raise RuntimeError(
                f"gridmap ids must be indexed by {ID_COLUMNS}. "
                "Regenerate the pkl files with the current createDataset.py."
            )

        common_ids = self.input_ids.index.intersection(self.gridmap_ids.index, sort=True)
        if common_ids.empty:
            raise RuntimeError(
                f"No common ids between {self.input_path} and {self.gridmap_path}"
            )
        self.common_ids = common_ids
        self.input_positions = self.input_ids.loc[common_ids, "sample_index"].to_numpy(dtype=np.int64)
        self.gridmap_positions = self.gridmap_ids.loc[common_ids, "sample_index"].to_numpy(dtype=np.int64)
        qp_values = common_ids.get_level_values("qp").to_numpy(dtype=np.float32)
        self.qp_values = (qp_values - float(min_qp)) / float(max_qp - min_qp)
        missing_input = len(self.gridmap_ids) - len(common_ids)
        missing_gridmap = len(self.input_ids) - len(common_ids)
        print(
            f"{component} {type}: {len(common_ids):,} aligned samples "
            f"from {self.input_path.name} and {self.gridmap_path.name} "
            f"(missing input: {missing_input:,}, missing gridmap: {missing_gridmap:,})"
        )
        print(
            f"{component} {type}: input shape {self.input_array.shape}, "
            f"gridmap shape {self.gridmap_array.shape}"
        )
        print(
            f"{component} {type}: normalized QP range "
            f"{self.qp_values.min():.4f} to {self.qp_values.max():.4f} "
            f"(min_qp={min_qp}, max_qp={max_qp})"
        )

    def __len__(self):
        return len(self.input_positions)

    def __getitem__(self, idx):
        input_idx = self.input_positions[idx]
        gridmap_idx = self.gridmap_positions[idx]
        input_sample = torch.from_numpy(self.input_array[input_idx].copy()).float()
        qp_sample = torch.tensor([self.qp_values[idx]], dtype=torch.float32)
        gridmap_sample = torch.from_numpy(self.gridmap_array[gridmap_idx].copy()).float()
        return input_sample, qp_sample, gridmap_sample


class IdAlignedGridmapCuTreeDataset(IdAlignedGridmapDataset):
    def __init__(self, dataset_name, type, component="Luma", min_qp=0, max_qp=51):
        super().__init__(
            dataset_name=dataset_name,
            type=type,
            component=component,
            min_qp=min_qp,
            max_qp=max_qp,
        )
        split_dir = split_dir_from_name(type)
        dataset_dir = paths.dataset_root() / dataset_name / split_dir
        self.cu_tree_path = dataset_dir / f"{component}_CU_Tree.pkl"
        if not self.cu_tree_path.exists():
            raise FileNotFoundError(
                f"CU tree labels not found: {self.cu_tree_path}. "
                "Generate them first with createDataset96.py --action cu-tree."
            )

        payload = pd.read_pickle(self.cu_tree_path)
        samples = payload["samples"]
        if list(samples.index.names) != ID_COLUMNS:
            raise RuntimeError(f"CU tree samples must be indexed by {ID_COLUMNS}")
        if payload.get("format") != "cu_tree_binary_v1":
            raise RuntimeError(
                f"Unsupported CU tree label format in {self.cu_tree_path}. "
                "Regenerate with the current createDataset96.py --action cu-tree."
            )

        common_index = self.common_ids
        missing = common_index.difference(samples.index)
        if not missing.empty:
            raise RuntimeError(
                f"CU tree labels are missing {len(missing):,} aligned samples. "
                f"First missing key: {missing[0]}"
            )

        tree_sample_indices = samples.loc[common_index, "sample_index"].to_numpy(dtype=np.int64)
        self.tree_sample_indices = tree_sample_indices
        self.node_offsets = payload["offsets"]
        nodes_path = self.cu_tree_path.with_name(payload["nodes_file"])
        if not nodes_path.exists():
            raise FileNotFoundError(f"CU tree node array not found: {nodes_path}")
        self.node_memmap = np.memmap(
            nodes_path,
            mode="r",
            dtype=np.dtype(payload["node_dtype"]),
            shape=tuple(payload["node_shape"]),
        )
        print(
            f"{component} {type}: loaded CU tree labels from {self.cu_tree_path.name}, "
            f"{len(self.tree_sample_indices):,} aligned samples, "
            f"{self.node_memmap.shape[0]:,} nodes"
        )

    def __getitem__(self, idx):
        input_sample, qp_sample, gridmap_sample = super().__getitem__(idx)
        sample_index = int(self.tree_sample_indices[idx])
        start = int(self.node_offsets[sample_index])
        end = int(self.node_offsets[sample_index + 1])
        node_sample = torch.from_numpy(np.asarray(self.node_memmap[start:end]).copy()).long()
        return input_sample, qp_sample, gridmap_sample, node_sample


def train_SwinTransU(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Net = model96(use_context_mask=args.useContextMask)
    Net = Net.to(device)
    print("Swin context mask:", args.useContextMask)

    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)

    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    if not os.path.exists(ckpt_out_dir):
        os.makedirs(ckpt_out_dir)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_dir = os.path.join(log_out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_num, epoch_loss, epoch_accu, val_loss, val_accu\n"
        f.write(s)
        for s in [args.lr,args.dr, args.batchSize]:
            f.write(str(s))
            f.write(',')
        f.write('\n')

    print("Creating data loader...")
    train_dataset = IdAlignedGridmapDataset(
        dataset_name=args.dataset,
        type=args.trainSplit,
        component=args.component,

    )
    train_dataLoader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=True)
    if args.noVal:
        val_dataLoader = None
        print("Validation disabled by --noVal.")
    else:
        val_dataset = IdAlignedGridmapDataset(
            dataset_name=args.dataset,
            type=args.valSplit,
            component=args.component,
        )
        val_dataLoader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=False)

    pg = [p for p in Net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    print('Start Training ...')
    for epoch in range(args.epoch):
        #train
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)

        train_loss, train_acc = train_one_epoch_multi(model=Net,
                                                      optimizer=optimizer,
                                                      data_loader=train_dataLoader,
                                                      device=device,
                                                      epoch=epoch,
                                                      lossFunction="BCE")

        # validate
        if val_dataLoader is None:
            val_loss, val_acc = 0.0, 0.0
        else:
            val_loss, val_acc = evaluate_multi(model=Net,
                                               data_loader=val_dataLoader,
                                               device=device,
                                               epoch=epoch,
                                               lossFunction="BCE")

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Accu/train", train_acc, epoch)
            tb_writer.add_scalar("Accu/val", val_acc, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-{}.pth".format(epoch)))

        print('***********************************************************************'
              '***********************************************************************')
        print("Epoch: %d  Loss: %.6f " % (epoch, train_loss))
        print("Val: Loss: %.6f  Acc: %.6f" % (val_loss, val_acc))
        #print("Test: Loss: %.6f  Acc: %.6f" % (test_out_info_list[0], test_out_info_list[1]))
        print('***********************************************************************'
              '***********************************************************************')

        with open(log_dir, 'a') as f:
            for s in [epoch,train_loss, train_acc, val_loss, val_acc]:
                f.write(str(s))
                f.write(',')
            f.write('\n')
        
        print('Epoch ' + str(epoch) + ' done.')

    torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


def train_Swin_with_FrozenClassifier(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Net = model96(use_context_mask=args.useContextMask).to(device)
    Classifier = classifier_i().to(device)
    print("Swin context mask:", args.useContextMask)

    load_model_weights(Net, args.swinCkpt, device, "Swin")
    load_model_weights(Classifier, args.classifierCkpt, device, "Classifier_I")

    Classifier.eval()
    for p in Classifier.parameters():
        p.requires_grad = False

    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)

    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    if not os.path.exists(ckpt_out_dir):
        os.makedirs(ckpt_out_dir)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_dir = os.path.join(log_out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_num, epoch_loss, grid_loss, cls_loss, grid_accu, cls_accu, avg_cls_nodes, val_loss, val_grid_loss, val_cls_loss, val_grid_accu, val_cls_accu, val_avg_cls_nodes\n"
        f.write(s)
        for s in [args.lr, args.dr, args.batchSize, args.gridLossWeight, args.clsLossWeight]:
            f.write(str(s))
            f.write(',')
        f.write('\n')

    print("Creating data loader...")
    train_dataset = IdAlignedGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.trainSplit,
        component=args.component,
    )
    train_dataLoader = DataLoader(
        dataset=train_dataset,
        num_workers=2,
        batch_size=args.batchSize,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_gridmap_with_nodes,
    )
    if args.noVal:
        val_dataLoader = None
        print("Validation disabled by --noVal.")
    else:
        val_dataset = IdAlignedGridmapCuTreeDataset(
            dataset_name=args.dataset,
            type=args.valSplit,
            component=args.component,
        )
        val_dataLoader = DataLoader(
            dataset=val_dataset,
            num_workers=2,
            batch_size=args.batchSize,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_gridmap_with_nodes,
        )

    pg = [p for p in Net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    grid_loss_fn = nn.BCELoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    def run_epoch(data_loader, epoch, training):
        if training:
            Net.train()
        else:
            Net.eval()

        accu_loss = torch.zeros(1).to(device)
        accu_grid_loss = torch.zeros(1).to(device)
        accu_cls_loss = torch.zeros(1).to(device)
        accu_grid_acc = torch.zeros(1).to(device)
        accu_cls_acc = torch.zeros(1).to(device)
        total_cls_nodes = 0

        data_loader = tqdm(data_loader, file=sys.stdout)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, data in enumerate(data_loader):
                input_batch, qp_batch, gridmap_batch, node_batch = data

                input_batch = input_batch.to(device)
                qp_batch = qp_batch.to(device)
                gridmap_batch = gridmap_batch.to(device)

                pred_gridmap = Net(input_batch, qp_batch)
                grid_loss = grid_loss_fn(pred_gridmap, gridmap_batch)
                cls_loss, cls_acc, cls_nodes = classifier_node_loss(
                    classifier=Classifier,
                    pred_gridmap=pred_gridmap,
                    node_batch=node_batch,
                    ce_loss=cls_loss_fn,
                    max_nodes=args.maxClassifierNodes,
                )
                loss = args.gridLossWeight * grid_loss + args.clsLossWeight * cls_loss

                grid_acc = torch.sum(abs(pred_gridmap - gridmap_batch) <= 1e-1).item() / float(pred_gridmap.numel())

                if training:
                    loss.backward()
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss)
                        sys.exit(1)
                    optimizer.step()
                    optimizer.zero_grad()

                accu_loss += loss.detach()
                accu_grid_loss += grid_loss.detach()
                accu_cls_loss += cls_loss.detach()
                accu_grid_acc += grid_acc
                accu_cls_acc += cls_acc
                total_cls_nodes += cls_nodes

                phase = "train" if training else "valid"
                data_loader.desc = (
                    "[{} epoch {}] loss: {:.6f}, grid: {:.6f}, cls: {:.6f}, "
                    "grid_acc: {:.6f}, cls_acc: {:.6f}, cls_nodes: {:.2f}"
                ).format(
                    phase,
                    epoch,
                    accu_loss.item() / (step + 1),
                    accu_grid_loss.item() / (step + 1),
                    accu_cls_loss.item() / (step + 1),
                    accu_grid_acc.item() / (step + 1),
                    accu_cls_acc.item() / (step + 1),
                    total_cls_nodes / float(step + 1),
                )

        step_count = step + 1
        return (
            accu_loss.item() / step_count,
            accu_grid_loss.item() / step_count,
            accu_cls_loss.item() / step_count,
            accu_grid_acc.item() / step_count,
            accu_cls_acc.item() / step_count,
            total_cls_nodes / float(step_count),
        )

    print('Start Frozen-Classifier Swin Fine-tuning ...')
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)

        train_metrics = run_epoch(train_dataLoader, epoch, training=True)
        if val_dataLoader is None:
            val_metrics = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            val_metrics = run_epoch(val_dataLoader, epoch, training=False)

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train_total", train_metrics[0], epoch)
            tb_writer.add_scalar("Loss/train_grid", train_metrics[1], epoch)
            tb_writer.add_scalar("Loss/train_cls", train_metrics[2], epoch)
            tb_writer.add_scalar("Accu/train_grid", train_metrics[3], epoch)
            tb_writer.add_scalar("Accu/train_cls", train_metrics[4], epoch)
            tb_writer.add_scalar("Loss/val_total", val_metrics[0], epoch)
            tb_writer.add_scalar("Loss/val_grid", val_metrics[1], epoch)
            tb_writer.add_scalar("Loss/val_cls", val_metrics[2], epoch)
            tb_writer.add_scalar("Accu/val_grid", val_metrics[3], epoch)
            tb_writer.add_scalar("Accu/val_cls", val_metrics[4], epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-{}.pth".format(epoch)))

        print('***********************************************************************'
              '***********************************************************************')
        print(
            "Epoch: {} Loss: {:.6f} Grid: {:.6f} Cls: {:.6f} GridAcc: {:.6f} ClsAcc: {:.6f}".format(
                epoch,
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[3],
                train_metrics[4],
            )
        )
        print(
            "Val: Loss: {:.6f} Grid: {:.6f} Cls: {:.6f} GridAcc: {:.6f} ClsAcc: {:.6f}".format(
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                val_metrics[3],
                val_metrics[4],
            )
        )
        print('***********************************************************************'
              '***********************************************************************')

        with open(log_dir, 'a') as f:
            for s in [epoch, *train_metrics, *val_metrics]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        print('Epoch ' + str(epoch) + ' done.')

    torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


def calibrate_Classifier_from_Swin(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Net = model96(use_context_mask=args.useContextMask).to(device)
    Classifier = classifier_i().to(device)
    print("Swin context mask:", args.useContextMask)

    load_model_weights(Net, args.swinCkpt, device, "Swin")
    load_model_weights(Classifier, args.classifierCkpt, device, "Classifier_I")

    Net.eval()
    for p in Net.parameters():
        p.requires_grad = False

    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)

    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    if not os.path.exists(ckpt_out_dir):
        os.makedirs(ckpt_out_dir)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_dir = os.path.join(log_out_dir, 'loss.txt')
    size_log_dir = os.path.join(log_out_dir, 'size_acc.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_num, cls_loss, cls_accu, grid_accu, avg_cls_nodes, val_cls_loss, val_cls_accu, val_grid_accu, val_avg_cls_nodes\n"
        f.write(s)
        for s in [args.lr, args.dr, args.batchSize]:
            f.write(str(s))
            f.write(',')
        f.write('\n')
    with open(size_log_dir, 'a') as f:
        f.write("epoch,phase,size_stats\n")

    print("Creating data loader...")
    train_dataset = IdAlignedGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.trainSplit,
        component=args.component,
    )
    train_dataLoader = DataLoader(
        dataset=train_dataset,
        num_workers=2,
        batch_size=args.batchSize,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_gridmap_with_nodes,
    )
    if args.noVal:
        val_dataLoader = None
        print("Validation disabled by --noVal.")
    else:
        val_dataset = IdAlignedGridmapCuTreeDataset(
            dataset_name=args.dataset,
            type=args.valSplit,
            component=args.component,
        )
        val_dataLoader = DataLoader(
            dataset=val_dataset,
            num_workers=2,
            batch_size=args.batchSize,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_gridmap_with_nodes,
        )

    pg = [p for p in Classifier.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    cls_loss_fn = nn.CrossEntropyLoss()

    def run_epoch(data_loader, epoch, training):
        if training:
            Classifier.train()
        else:
            Classifier.eval()

        accu_cls_loss = torch.zeros(1).to(device)
        accu_cls_acc = torch.zeros(1).to(device)
        accu_grid_acc = torch.zeros(1).to(device)
        total_cls_nodes = 0
        epoch_size_stats = {}

        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            input_batch, qp_batch, gridmap_batch, node_batch = data

            input_batch = input_batch.to(device)
            qp_batch = qp_batch.to(device)
            gridmap_batch = gridmap_batch.to(device)

            with torch.no_grad():
                pred_gridmap = Net(input_batch, qp_batch)

            if training:
                cls_loss, cls_acc, cls_nodes, size_stats = classifier_node_loss(
                    classifier=Classifier,
                    pred_gridmap=pred_gridmap.detach(),
                    node_batch=node_batch,
                    ce_loss=cls_loss_fn,
                    max_nodes=args.maxClassifierNodes,
                    return_size_stats=True,
                )
                update_size_stats(epoch_size_stats, size_stats)
                cls_loss.backward()
                if not torch.isfinite(cls_loss):
                    print('WARNING: non-finite loss, ending training ', cls_loss)
                    sys.exit(1)
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    cls_loss, cls_acc, cls_nodes, size_stats = classifier_node_loss(
                        classifier=Classifier,
                        pred_gridmap=pred_gridmap,
                        node_batch=node_batch,
                        ce_loss=cls_loss_fn,
                        max_nodes=args.maxClassifierNodes,
                        return_size_stats=True,
                    )
                    update_size_stats(epoch_size_stats, size_stats)

            grid_acc = torch.sum(abs(pred_gridmap - gridmap_batch) <= 1e-1).item() / float(pred_gridmap.numel())

            accu_cls_loss += cls_loss.detach()
            accu_cls_acc += cls_acc
            accu_grid_acc += grid_acc
            total_cls_nodes += cls_nodes

            phase = "train" if training else "valid"
            data_loader.desc = (
                "[{} epoch {}] cls_loss: {:.6f}, cls_acc: {:.6f}, "
                "grid_acc: {:.6f}, cls_nodes: {:.2f}"
            ).format(
                phase,
                epoch,
                accu_cls_loss.item() / (step + 1),
                accu_cls_acc.item() / (step + 1),
                accu_grid_acc.item() / (step + 1),
                total_cls_nodes / float(step + 1),
            )

        step_count = step + 1
        return (
            accu_cls_loss.item() / step_count,
            accu_cls_acc.item() / step_count,
            accu_grid_acc.item() / step_count,
            total_cls_nodes / float(step_count),
            epoch_size_stats,
        )

    print('Start Classifier Calibration from Swin outputs ...')
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)

        train_metrics = run_epoch(train_dataLoader, epoch, training=True)
        if val_dataLoader is None:
            val_metrics = (0.0, 0.0, 0.0, 0.0, {})
        else:
            val_metrics = run_epoch(val_dataLoader, epoch, training=False)

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train_cls", train_metrics[0], epoch)
            tb_writer.add_scalar("Accu/train_cls", train_metrics[1], epoch)
            tb_writer.add_scalar("Accu/train_grid", train_metrics[2], epoch)
            tb_writer.add_scalar("Loss/val_cls", val_metrics[0], epoch)
            tb_writer.add_scalar("Accu/val_cls", val_metrics[1], epoch)
            tb_writer.add_scalar("Accu/val_grid", val_metrics[2], epoch)
            for key, stats in train_metrics[4].items():
                total = stats["total"]
                if total > 0:
                    tb_writer.add_scalar("AccuBySize/train_{}x{}".format(key[0], key[1]), stats["correct"] / float(total), epoch)
            for key, stats in val_metrics[4].items():
                total = stats["total"]
                if total > 0:
                    tb_writer.add_scalar("AccuBySize/val_{}x{}".format(key[0], key[1]), stats["correct"] / float(total), epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(Classifier.state_dict(), os.path.join(ckpt_out_dir, "model-{}.pth".format(epoch)))

        print('***********************************************************************'
              '***********************************************************************')
        print(
            "Epoch: {} ClsLoss: {:.6f} ClsAcc: {:.6f} GridAcc: {:.6f} Nodes: {:.2f}".format(
                epoch,
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[3],
            )
        )
        print(
            "Val: ClsLoss: {:.6f} ClsAcc: {:.6f} GridAcc: {:.6f} Nodes: {:.2f}".format(
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                val_metrics[3],
            )
        )
        print('***********************************************************************'
              '***********************************************************************')

        with open(log_dir, 'a') as f:
            for s in [epoch, *train_metrics[:4], *val_metrics[:4]]:
                f.write(str(s))
                f.write(',')
            f.write('\n')
        with open(size_log_dir, 'a') as f:
            f.write("{},train,{}\n".format(epoch, format_size_stats(train_metrics[4])))
            if val_metrics[4]:
                f.write("{},val,{}\n".format(epoch, format_size_stats(val_metrics[4])))

        print('Epoch ' + str(epoch) + ' done.')

    torch.save(Classifier.state_dict(), os.path.join(ckpt_out_dir, "model-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


def pretrain_Classifier(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    Net = classifier_i()

    Net = Net.to(device)

    path = paths.dataset_root() / 'Classifier_I' / 'pretrain_logical'
    
    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)

    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    if not os.path.exists(ckpt_out_dir):
        os.makedirs(ckpt_out_dir)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_dir = os.path.join(log_out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_num, epoch_loss, epoch_accu\n"
        f.write(s)
        for s in [args.lr,args.dr, args.batchSize]:
            f.write(str(s))
            f.write(',')
        f.write('\n')

    print("Creating data loader...")
    train_dataLoaders = []
    grid_size_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
    for grid_size_dir in grid_size_dirs:
        input_train_path = grid_size_dir / 'gridmap.npy'
        label_train_path = grid_size_dir / 'label.npy'
        if not input_train_path.exists() or not label_train_path.exists():
            continue

        input_train_batch = torch.FloatTensor(np.load(input_train_path))
        label_train_batch = torch.LongTensor(np.load(label_train_path))
        train_dataset = TensorDataset(input_train_batch, label_train_batch)
        train_dataLoader = DataLoader(dataset=train_dataset,
                                      num_workers=0,
                                      batch_size=args.batchSize,
                                      pin_memory=True,
                                      shuffle=True)
        train_dataLoaders.append((grid_size_dir.name, train_dataLoader))
        print(grid_size_dir.name, 'input_batch.shape:', input_train_batch.shape, 'label_batch.shape:', label_train_batch.shape)

    if len(train_dataLoaders) == 0:
        raise RuntimeError("No Classifier_I pretrain data found in {}".format(path))

    pg = [p for p in Net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    print('Start Training ...')
    for epoch in range(args.epoch):
        #train
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)

        random.shuffle(train_dataLoaders)
        train_loss, train_acc = train_one_epoch_classifier(model=Net,
                                                           optimizer=optimizer,
                                                           data_loaders=train_dataLoaders,
                                                           device=device,
                                                           epoch=epoch,
                                                           lossFunction="CE")

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Accu/train", train_acc, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-{}.pth".format(epoch)))

        print('***********************************************************************'
              '***********************************************************************')
        print("Epoch: %d  Loss: %.6f " % (epoch, train_loss))
        print('***********************************************************************'
              '***********************************************************************')

        with open(log_dir, 'a') as f:
            for s in [epoch,train_loss, train_acc]:
                f.write(str(s))
                f.write(',')
            f.write('\n')
        
        print('Epoch ' + str(epoch) + ' done.')

    torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "model-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--outDir', type=str, default='classifier_i_pretrain')
    parser.add_argument('--task', type=str, default='swin_luma',
                        choices=[
                            'classifier_pretrain_logical',
                            'swin_luma',
                            'classifier_calibrate_from_swin',
                            'swin_luma_frozen_classifier',
                        ])
    parser.add_argument('--logFile', type=str, default='train.log')
    parser.add_argument('--dr', default=20, type=int, help='decay rate of lr')
    parser.add_argument('--dataset', type=str, default='DIV2K')
    parser.add_argument('--trainSplit', type=str, default='training')
    parser.add_argument('--valSplit', type=str, default='validating')
    parser.add_argument('--component', type=str, choices=['Luma'], default='Luma')
    parser.add_argument('--tbLogDir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--noVal', action='store_true', help='Skip validation when no validation split is available')
    parser.add_argument('--useContextMask', action='store_true', help='Enable 96x96 context attention mask in SwinTransformer_Unet_Luma96')
    parser.add_argument('--swinCkpt', type=str, default=None, help='Optional Swin checkpoint to resume from')
    parser.add_argument('--classifierCkpt', type=str, default=None, help='Classifier_I checkpoint for frozen-classifier fine-tuning')
    parser.add_argument('--gridLossWeight', type=float, default=1.0, help='Weight for gridmap BCE loss')
    parser.add_argument('--clsLossWeight', type=float, default=0.1, help='Weight for frozen classifier CE loss')
    parser.add_argument('--treeThreshold', type=float, default=0.5, help='Threshold used to parse target gridmap split tree')
    parser.add_argument('--maxClassifierNodes', type=int, default=0, help='Maximum classifier nodes per sample, 0 means no limit')

    args = parser.parse_args()

    setup_log_file(args)

    if args.task == 'swin_luma':
        train_SwinTransU(args)
    elif args.task == 'classifier_pretrain_logical':
        pretrain_Classifier(args)
    elif args.task == 'classifier_calibrate_from_swin':
        if args.swinCkpt is None:
            raise RuntimeError("--swinCkpt is required for classifier_calibrate_from_swin")
        calibrate_Classifier_from_Swin(args)
    elif args.task == 'swin_luma_frozen_classifier':
        if args.classifierCkpt is None:
            raise RuntimeError("--classifierCkpt is required for swin_luma_frozen_classifier")
        train_Swin_with_FrozenClassifier(args)


#python ./train.py --outDir ./output/ --lr 5e-4 --dr 30 --batchSize 200 --device cuda:0 --jobID 0000 --epoch 200 2>&1 | tee ./output/test.txt
#CUDA_VISIBLE_DEVICES=0 nohup python train.py --outDir ./Train_Loss/ --lr 2e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &
#nohup python train.py --outDir ./Train_Loss/ --lr 1e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &

#nohup python train.py --task classifier_pretrain_logical --outDir classifier_i_pretrain --jobID logical_v1 --epoch 50 --batchSize 128 --lr 1e-3 --dr 20 --device cuda:0 &
