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

from utils import (
    train_one_epoch,
    evaluate,
    adjust_learning_rate,
    train_one_epoch_classifier,
    train_one_epoch_multi,
    evaluate_multi,
)
from model import SwinTransformer_Unet_Luma as model64
from model import Classifier_I as classifier_i


ID_COLUMNS = ["sequence_name", "qp", "frame_id", "ctu_id"]


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

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input pkl not found: {self.input_path}")
        if not self.gridmap_path.exists():
            raise FileNotFoundError(f"Gridmap pkl not found: {self.gridmap_path}")

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


def train_SwinTransU(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Net = model64()
    Net = Net.to(device)

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
                        choices=['classifier_pretrain_logical', 'swin_luma'])
    parser.add_argument('--logFile', type=str, default='train.log')
    parser.add_argument('--dr', default=20, type=int, help='decay rate of lr')
    parser.add_argument('--dataset', type=str, default='DIV2K')
    parser.add_argument('--trainSplit', type=str, default='training')
    parser.add_argument('--valSplit', type=str, default='validating')
    parser.add_argument('--component', type=str, choices=['Luma'], default='Luma')
    parser.add_argument('--tbLogDir', type=str, default=None, help='TensorBoard log directory')

    args = parser.parse_args()

    setup_log_file(args)

    if args.task == 'swin_luma':
        train_SwinTransU(args)
    elif args.task == 'classifier_pretrain_logical':
        pretrain_Classifier(args)


#python ./train.py --outDir ./output/ --lr 5e-4 --dr 30 --batchSize 200 --device cuda:0 --jobID 0000 --epoch 200 2>&1 | tee ./output/test.txt
#CUDA_VISIBLE_DEVICES=0 nohup python train.py --outDir ./Train_Loss/ --lr 2e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &
#nohup python train.py --outDir ./Train_Loss/ --lr 1e-4 --dr 30 --batchSize 256 --epoch 300 --jobID 0000 --device cuda:0 > ./Train_Loss/0000/train_result.log 2>&1 &

#nohup python train.py --task classifier_pretrain_logical --outDir classifier_i_pretrain --jobID logical_v1 --epoch 50 --batchSize 128 --lr 1e-3 --dr 20 --device cuda:0 &
