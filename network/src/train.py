import os
import argparse
import paths
import random
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.cuda.amp import GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    adjust_learning_rate,
    collate_gridmap_with_nodes,
    evaluate,
    add_gridmap_prediction_image,
    IdAlignedGridmapCuTreeDataset,
    train_one_epoch,
    train_one_epoch_classifier,
    get_loss_function,
)
from model import SwinTransformer_Unet as model
from model import Classifier_I as classifier_i

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


def add_classifier_shape_scalars(tb_writer, phase, shape_stats, epoch):
    for shape_name, stats in shape_stats.items():
        tb_writer.add_scalar(f"Classifier/{phase}/{shape_name}/loss", stats["loss"], epoch)
        tb_writer.add_scalar(f"Classifier/{phase}/{shape_name}/acc", stats["acc"], epoch)


def classifier_shape_sort_key(shape_name):
    try:
        h_str, w_str = shape_name.split("x", 1)
        h = int(h_str)
        w = int(w_str)
        return h * w, h, w
    except Exception:
        return 0, 0, 0


def format_classifier_shape_acc(shape_stats):
    return format_classifier_shape_metric(shape_stats, "acc")


def format_classifier_shape_metric(shape_stats, metric_name):
    if not shape_stats:
        return "none"
    parts = []
    for shape_name, stats in sorted(shape_stats.items(), key=lambda item: classifier_shape_sort_key(item[0])):
        parts.append("{}={:.2f}%".format(shape_name, float(stats[metric_name]) * 100.0))
    return ", ".join(parts)


def format_prune_stats(prune_stats):
    if not prune_stats:
        return "none"
    parts = []
    for threshold, stats in sorted(prune_stats.items()):
        parts.append(
            "T={:.2f}:keep={:.2f}% false={:.2f}% reduction={:.2f}%".format(
                float(threshold),
                float(stats["keep_rate"]) * 100.0,
                float(stats["false_prune_rate"]) * 100.0,
                float(stats["candidate_reduction"]) * 100.0,
            )
        )
    return " | ".join(parts)


def write_epoch_summary(summary_path, epoch, stage_name, train_metrics, val_metrics, test_metrics=None):
    with open(summary_path, "a") as f:
        f.write("Epoch {} Stage {}\n".format(epoch, stage_name))
        f.write("Classifier shape acc train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "acc")))
        f.write("Classifier shape top2 train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "top2")))
        f.write("Classifier shape top3 train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "top3")))
        f.write("Classifier prune train: {}\n".format(format_prune_stats(train_metrics[9])))
        f.write("Classifier shape acc val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "acc")))
        f.write("Classifier shape top2 val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "top2")))
        f.write("Classifier shape top3 val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "top3")))
        f.write("Classifier prune val: {}\n".format(format_prune_stats(val_metrics[9])))
        if test_metrics is not None:
            f.write("Classifier shape acc test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "acc")))
            f.write("Classifier shape top2 test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "top2")))
            f.write("Classifier shape top3 test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "top3")))
            f.write("Classifier prune test: {}\n".format(format_prune_stats(test_metrics[9])))
        f.write("\n")


def parse_tb_image_sample_spec(spec):
    samples = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            name, index = item.split(':', 1)
        else:
            index = item
            name = f"sample_{len(samples)}"
        samples.append((name.strip(), int(index)))
    if not samples:
        raise ValueError("TensorBoard image sample spec is empty")
    return samples


def resolve_tb_image_samples(args):
    if args.tbImageSampleIndex is not None:
        sample = [("sample", args.tbImageSampleIndex)]
        return sample, sample
    return (
        parse_tb_image_sample_spec(args.tbTrainImageSamples),
        parse_tb_image_sample_spec(args.tbValImageSamples),
    )


def add_gridmap_sample_set(tb_writer, phase, model, dataset, samples, device, epoch):
    for sample_name, sample_index in samples:
        add_gridmap_prediction_image(
            tb_writer=tb_writer,
            tag=f"Gridmap/{phase}/{sample_name}",
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            device=device,
            epoch=epoch,
        )


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


def classifier_safety_threshold_table(args, device):
    if args.classifierSafetyThresholdPreset == "none":
        return None
    if args.classifierSafetyThresholdPreset != "table3_lambda2000":
        raise ValueError(
            "Unsupported classifier safety threshold preset: {}".format(
                args.classifierSafetyThresholdPreset
            )
        )
    default_tau = float(args.classifierSafetyTauLarge)
    table_by_pixel_hw = {
        (32, 32): [0.006, 0.147, 0.114, 0.121, 0.073, 0.085],
        (16, 16): [0.011, 0.170, 0.025, 0.031, 0.028, 0.036],
        (16, 32): [0.039, None, 0.114, 0.108, 0.078, 0.091],
        (8, 32): [0.042, None, 0.062, 0.042, None, 0.052],
        (8, 16): [-0.010, None, 0.042, 0.049, None, 0.037],
        (32, 16): [0.031, None, 0.107, 0.115, 0.088, 0.080],
        (32, 8): [0.034, None, 0.045, 0.068, 0.052, None],
        (16, 8): [-0.010, None, 0.045, 0.051, 0.043, None],
    }
    thresholds = {}
    for (pixel_h, pixel_w), values in table_by_pixel_hw.items():
        grid_h = pixel_h // 4
        grid_w = pixel_w // 4
        filled = [default_tau if value is None else float(value) for value in values]
        thresholds[(grid_h, grid_w)] = torch.tensor(filled, dtype=torch.float32, device=device)
    return thresholds


def train_SwinTransU(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fasttrain_enabled = bool(args.fasttrain)
    roi_extraction = "vectorized" if fasttrain_enabled else "legacy"
    classifier_roi_chunk_size = 0
    num_workers = 4 if fasttrain_enabled else 2
    progress_update_interval = 50 if fasttrain_enabled else 1
    amp_enabled = bool(fasttrain_enabled and device.type == "cuda")
    if fasttrain_enabled and not amp_enabled:
        print("Fast training requested but CUDA is unavailable; falling back to FP32.")

    Net = model(use_context_mask=args.useContextMask).to(device)
    Classifier = classifier_i().to(device)
    print("Swin context mask:", args.useContextMask)

    load_model_weights(Net, args.swinCkpt, device, "SwinTransformer_Unet")
    load_model_weights(Classifier, args.classifierCkpt, device, "Classifier_I")
    classifier_safety_thresholds = classifier_safety_threshold_table(args, device)

    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)

    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    if not os.path.exists(ckpt_out_dir):
        os.makedirs(ckpt_out_dir)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_dir = os.path.join(log_out_dir, 'loss.txt')
    summary_dir = os.path.join(log_out_dir, 'summary.txt')
    with open(log_dir, 'a') as f:
        s = (
            "epoch_num, stage, grid_weight, cls_weight, lr, epoch_loss, grid_loss, cls_loss, "
            "grid_precision, grid_recall, cls_accu, val_loss, val_grid_loss, val_cls_loss, "
            "val_grid_precision, val_grid_recall, val_cls_accu\n"
        )
        f.write(s)
        for s in [
            args.lr,
            args.dr,
            args.batchSize,
            args.stage1Lr if args.stage1Lr is not None else args.lr,
            args.stage2Lr if args.stage2Lr is not None else args.lr,
            args.gridLossType,
            args.jointStage1Epoch,
            args.stage1GridLossWeight,
            args.stage1ClsLossWeight,
            args.stage2GridLossWeight,
            args.stage2ClsLossWeight,
            args.gridPositiveWeight,
            args.gridNegativeWeight,
            args.gridL1Weight,
            args.fasttrain,
            roi_extraction,
            classifier_roi_chunk_size,
            num_workers,
            progress_update_interval,
            int(amp_enabled),
            args.classifierSafetyTauLarge,
            args.classifierSafetyLossWeight,
            args.classifierSafetyThresholdPreset,
            args.classifierSafetyThresholdOnlyShapes,
            args.checkpointInterval,
        ]:
            f.write(str(s))
            f.write(',')
        f.write('\n')

    print("Creating data loader...")
    train_dataset = IdAlignedGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.trainSplit,
        component=args.component,
    )
    loader_worker_args = {}
    if num_workers > 0:
        loader_worker_args.update(persistent_workers=True, prefetch_factor=2)
    train_dataLoader = DataLoader(
        dataset=train_dataset,
        num_workers=num_workers,
        batch_size=args.batchSize,
        pin_memory=device.type == "cuda",
        shuffle=True,
        collate_fn=collate_gridmap_with_nodes,
        **loader_worker_args,
    )
    val_dataset = IdAlignedGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.valSplit,
        component=args.component,
    )
    val_dataLoader = DataLoader(
        dataset=val_dataset,
        num_workers=num_workers,
        batch_size=args.batchSize,
        pin_memory=device.type == "cuda",
        shuffle=False,
        collate_fn=collate_gridmap_with_nodes,
        **loader_worker_args,
    )

    pg = [p for p in list(Net.parameters()) + list(Classifier.parameters()) if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    grad_scaler = GradScaler(enabled=amp_enabled)
    grid_loss_fn = get_loss_function(
        args.gridLossType,
        positive_weight=args.gridPositiveWeight,
        negative_weight=args.gridNegativeWeight,
        l1_weight=args.gridL1Weight,
    )
    if args.gridLossType == "WBCE":
        print(
            "Weighted BCE: positive_weight={}, negative_weight={}".format(
                args.gridPositiveWeight,
                args.gridNegativeWeight,
            )
        )
    elif args.gridLossType == "BCE_L1":
        print("BCE + L1: l1_weight={}".format(args.gridL1Weight))
    cls_loss_fn = nn.CrossEntropyLoss()
    print(
        "Fast training: {}, ROI extraction: {}, classifier nodes: all, "
        "ROI chunk size: {}, AMP: {}, DataLoader workers: {}".format(
            fasttrain_enabled,
            roi_extraction,
            classifier_roi_chunk_size if classifier_roi_chunk_size > 0 else "all",
            amp_enabled,
            num_workers,
        )
    )
    if args.classifierSafetyLossWeight > 0 and args.classifierSafetyTauLarge > 0:
        print(
            "Classifier all-shape safety loss: tau={}, weight={}".format(
                args.classifierSafetyTauLarge,
                args.classifierSafetyLossWeight,
            )
        )
    if classifier_safety_thresholds is not None:
        print(
            "Classifier safety threshold preset: {} (fallback tau={})".format(
                args.classifierSafetyThresholdPreset,
                args.classifierSafetyTauLarge,
            )
        )
    if args.classifierSafetyThresholdOnlyShapes:
        print("Classifier loss is restricted to shapes covered by the safety threshold preset.")
    tb_train_samples, tb_val_samples = resolve_tb_image_samples(args)

    def stage_config(epoch):
        if epoch < args.jointStage1Epoch:
            stage_lr = args.stage1Lr if args.stage1Lr is not None else args.lr
            return "stage1_grid_first", args.stage1GridLossWeight, args.stage1ClsLossWeight, stage_lr, epoch
        stage_lr = args.stage2Lr if args.stage2Lr is not None else args.lr
        return "stage2_joint", args.stage2GridLossWeight, args.stage2ClsLossWeight, stage_lr, epoch - args.jointStage1Epoch

    print('Start Joint Swin + Classifier Training ...')
    for epoch in range(args.epoch):
        stage_name, grid_weight, cls_weight, stage_lr, stage_epoch = stage_config(epoch)
        adjust_learning_rate(stage_lr, optimizer, stage_epoch, args.dr)

        train_metrics = train_one_epoch(
            swin_model=Net,
            classifier=Classifier,
            optimizer=optimizer,
            data_loader=train_dataLoader,
            device=device,
            epoch=epoch,
            grid_loss_fn=grid_loss_fn,
            cls_loss_fn=cls_loss_fn,
            grid_weight=grid_weight,
            cls_weight=cls_weight,
            stage_name=stage_name,
            roi_extraction=roi_extraction,
            classifier_roi_chunk_size=classifier_roi_chunk_size,
            progress_update_interval=progress_update_interval,
            amp_enabled=amp_enabled,
            grad_scaler=grad_scaler,
            classifier_safety_tau_large=args.classifierSafetyTauLarge,
            classifier_safety_loss_weight=args.classifierSafetyLossWeight,
            classifier_safety_thresholds=classifier_safety_thresholds,
            classifier_safety_threshold_only_shapes=bool(args.classifierSafetyThresholdOnlyShapes),
        )
        val_metrics = evaluate(
            swin_model=Net,
            classifier=Classifier,
            data_loader=val_dataLoader,
            device=device,
            epoch=epoch,
            grid_loss_fn=grid_loss_fn,
            cls_loss_fn=cls_loss_fn,
            grid_weight=grid_weight,
            cls_weight=cls_weight,
            stage_name=stage_name,
            roi_extraction=roi_extraction,
            classifier_roi_chunk_size=classifier_roi_chunk_size,
            progress_update_interval=progress_update_interval,
            amp_enabled=amp_enabled,
            classifier_safety_tau_large=args.classifierSafetyTauLarge,
            classifier_safety_loss_weight=args.classifierSafetyLossWeight,
            classifier_safety_thresholds=classifier_safety_thresholds,
            classifier_safety_threshold_only_shapes=bool(args.classifierSafetyThresholdOnlyShapes),
        )

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train_total", train_metrics[0], epoch)
            tb_writer.add_scalar("Loss/train_grid", train_metrics[1], epoch)
            tb_writer.add_scalar("Loss/train_cls", train_metrics[2], epoch)
            tb_writer.add_scalar("Grid/train_precision", train_metrics[3], epoch)
            tb_writer.add_scalar("Grid/train_recall", train_metrics[4], epoch)
            tb_writer.add_scalar("Accu/train_cls", train_metrics[5], epoch)
            tb_writer.add_scalar("Loss/val_total", val_metrics[0], epoch)
            tb_writer.add_scalar("Loss/val_grid", val_metrics[1], epoch)
            tb_writer.add_scalar("Loss/val_cls", val_metrics[2], epoch)
            tb_writer.add_scalar("Grid/val_precision", val_metrics[3], epoch)
            tb_writer.add_scalar("Grid/val_recall", val_metrics[4], epoch)
            tb_writer.add_scalar("Accu/val_cls", val_metrics[5], epoch)
            tb_writer.add_scalar("Weight/grid", grid_weight, epoch)
            tb_writer.add_scalar("Weight/cls", cls_weight, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            add_classifier_shape_scalars(tb_writer, "train", train_metrics[7], epoch)
            add_classifier_shape_scalars(tb_writer, "val", val_metrics[7], epoch)
            tb_writer.add_scalar("Memory/train_peak_allocated_mb", train_metrics[8], epoch)
            tb_writer.add_scalar("Memory/val_peak_allocated_mb", val_metrics[8], epoch)
            add_gridmap_sample_set(tb_writer, "train", Net, train_dataset, tb_train_samples, device, epoch)
            add_gridmap_sample_set(tb_writer, "val", Net, val_dataset, tb_val_samples, device, epoch)
            tb_writer.flush()

        if args.checkpointInterval > 0 and (epoch + 1) % args.checkpointInterval == 0:
            torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "swin-{}.pth".format(epoch)))
            torch.save(Classifier.state_dict(), os.path.join(ckpt_out_dir, "classifier-{}.pth".format(epoch)))

        print('***********************************************************************'
              '***********************************************************************')
        print(
            "Epoch: {} Stage: {} GridW: {:.6f} ClsW: {:.6f} "
            "Loss: {:.6f} Grid: {:.6f} Cls: {:.6f} GridPrecision: {:.6f} GridRecall: {:.6f} ClsAcc: {:.6f}".format(
                epoch,
                stage_name,
                grid_weight,
                cls_weight,
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[3],
                train_metrics[4],
                train_metrics[5],
            )
        )
        print(
            "Val: Loss: {:.6f} Grid: {:.6f} Cls: {:.6f} GridPrecision: {:.6f} GridRecall: {:.6f} ClsAcc: {:.6f}".format(
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                val_metrics[3],
                val_metrics[4],
                val_metrics[5],
            )
        )
        print("Classifier shape acc train: {}".format(format_classifier_shape_acc(train_metrics[7])))
        print("Classifier shape acc val: {}".format(format_classifier_shape_acc(val_metrics[7])))
        write_epoch_summary(summary_dir, epoch, stage_name, train_metrics, val_metrics)
        print('***********************************************************************'
              '***********************************************************************')

        with open(log_dir, 'a') as f:
            for s in [
                epoch,
                stage_name,
                grid_weight,
                cls_weight,
                optimizer.param_groups[0]["lr"],
                *train_metrics[:6],
                *val_metrics[:6],
            ]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        print('Epoch ' + str(epoch) + ' done.')

    torch.save(Net.state_dict(), os.path.join(ckpt_out_dir, "swin-final.pth"))
    torch.save(Classifier.state_dict(), os.path.join(ckpt_out_dir, "classifier-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


def pretrain_Classifier(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    Net = classifier_i()

    Net = Net.to(device)

    path = paths.dataset_root() / 'pretrain' / 'Classifier_I' / 'logical'

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
        train_loss, train_acc, shape_stats = train_one_epoch_classifier(model=Net,
                                                                        optimizer=optimizer,
                                                                        data_loaders=train_dataLoaders,
                                                                        device=device,
                                                                        epoch=epoch,
                                                                        lossFunction="CE")

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Accu/train", train_acc, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            add_classifier_shape_scalars(tb_writer, "pretrain", shape_stats, epoch)
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
    parser.add_argument('--outDir', type=str, default='swin_luma_joint')
    parser.add_argument('--task', type=str, default='swin_luma',
                        choices=['pretrain_classifier_logical', 'swin_luma'])
    parser.add_argument('--logFile', type=str, default='train.log')
    parser.add_argument('--dr', default=20, type=int, help='decay rate of lr')
    parser.add_argument('--dataset', type=str, default='DIV2K')
    parser.add_argument('--trainSplit', type=str, default='training')
    parser.add_argument('--valSplit', type=str, default='validating')
    parser.add_argument('--component', type=str, choices=['Luma'], default='Luma')
    parser.add_argument('--tbLogDir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--tbImageSampleIndex', type=int, default=None, help='Optional fixed sample index for both train and val TensorBoard gridmap images')
    parser.add_argument('--tbTrainImageSamples', type=str, default='simple:1989496,medium:819088,complex:320136', help='Comma-separated training TensorBoard gridmap samples, e.g. simple:0,medium:1,complex:2')
    parser.add_argument('--tbValImageSamples', type=str, default='simple:116602,medium:54490,complex:121714', help='Comma-separated validation TensorBoard gridmap samples, e.g. simple:0,medium:1,complex:2')
    parser.add_argument('--useContextMask', action='store_true', default=True, help='Enable context attention mask in SwinTransformer_Unet')
    parser.add_argument('--swinCkpt', type=str, default=None, help='Optional Swin checkpoint to resume from')
    parser.add_argument('--classifierCkpt', type=str, default=None, help='Optional Classifier_I checkpoint to resume from')
    parser.add_argument('--gridLossType', type=str, default='BCE', choices=['BCE', 'BCE_L1', 'WBCE', 'L1', 'HUBER', 'MSE'], help='Loss function for Swin gridmap supervision')
    parser.add_argument('--gridPositiveWeight', type=float, default=1.0, help='Positive-label weight used by WBCE')
    parser.add_argument('--gridNegativeWeight', type=float, default=1.0, help='Negative-label weight used by WBCE')
    parser.add_argument('--gridL1Weight', type=float, default=0.2, help='L1 coefficient used by BCE_L1')
    parser.add_argument(
        '--fasttrain',
        type=int,
        choices=[0, 1],
        default=0,
        help='Enable full-node vectorized ROI extraction, CUDA AMP, 4 DataLoader workers, and progress updates every 50 batches',
    )
    parser.add_argument('--jointStage1Epoch', type=int, default=50, help='Epoch threshold for joint training stage 1')
    parser.add_argument('--stage1Lr', type=float, default=None, help='Joint stage 1 base learning rate, default uses --lr')
    parser.add_argument('--stage2Lr', type=float, default=None, help='Joint stage 2 base learning rate, default uses --lr')
    parser.add_argument('--stage1GridLossWeight', type=float, default=1.0, help='Joint stage 1 gridmap loss weight')
    parser.add_argument('--stage1ClsLossWeight', type=float, default=0.02, help='Joint stage 1 classifier loss weight')
    parser.add_argument('--stage2GridLossWeight', type=float, default=0.5, help='Joint stage 2 gridmap loss weight')
    parser.add_argument('--stage2ClsLossWeight', type=float, default=1.0, help='Joint stage 2 classifier loss weight')
    parser.add_argument('--classifierSafetyTauLarge', type=float, default=0.0, help='True-class probability floor for large ROI safety loss; 0 disables it')
    parser.add_argument('--classifierSafetyLossWeight', type=float, default=0.0, help='Weight for large ROI true-class probability safety loss')
    parser.add_argument(
        '--classifierSafetyThresholdPreset',
        type=str,
        default='none',
        choices=['none', 'table3_lambda2000'],
        help='Optional per-shape/per-class true-label safety threshold preset; missing shapes fall back to --classifierSafetyTauLarge',
    )
    parser.add_argument(
        '--classifierSafetyThresholdOnlyShapes',
        type=int,
        choices=[0, 1],
        default=0,
        help='If 1, classifier CE/safety loss only uses ROI shapes included in --classifierSafetyThresholdPreset',
    )
    parser.add_argument('--checkpointInterval', type=int, default=10, help='Save Swin/classifier checkpoints every N epochs; 0 disables periodic saves')

    args = parser.parse_args()

    setup_log_file(args)

    if args.task == 'swin_luma':
        train_SwinTransU(args)
    elif args.task == 'pretrain_classifier_logical':
        pretrain_Classifier(args)


#python ./train.py --outDir ./output/ --lr 5e-4 --dr 30 --batchSize 200 --device cuda:0 --jobID 0000 --epoch 200 2>&1 | tee ./output/test.txt
#CUDA_VISIBLE_DEVICES=0 nohup python train.py --outDir ./Train_Loss/ --lr 2e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &
#nohup python train.py --outDir ./Train_Loss/ --lr 1e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &

#nohup python train.py --task pretrain_classifier_logical --outDir classifier_i_pretrain --jobID logical_v1 --epoch 50 --batchSize 128 --lr 1e-3 --dr 20 --device cuda:0 &
