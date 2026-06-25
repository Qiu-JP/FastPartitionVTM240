import sys
import os

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

BCE_loss = nn.BCELoss()
Huber_loss = nn.SmoothL1Loss()
L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
CE_loss = nn.CrossEntropyLoss()
delta = 1e-1

LOSS_FUNCTIONS = {
    "BCE": BCE_loss,
    "HUBER": Huber_loss,
    "L1": L1_loss,
    "MSE": MSE_loss,
    "CE": CE_loss,
}


def get_loss_function(loss_name):
    try:
        return LOSS_FUNCTIONS[loss_name.upper()]
    except KeyError:
        raise ValueError(
            "Unsupported lossFunction '{}'. Available options: {}".format(
                loss_name, ", ".join(sorted(LOSS_FUNCTIONS.keys()))
            )
        )

def train_one_epoch(model, optimizer, data_loader, device, epoch, lossFunction = "L1"):
    model.train()

    loss_function = get_loss_function(lossFunction)

    optimizer.zero_grad()

    accu_list = []
    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        input_batch, gridmap_batch = data

        input_batch = input_batch.to(device)
        gridmap_batch = gridmap_batch.to(device)
        #print(input_batch.device)

        gridmap_output_batch = model(input_batch)

        #gridmap_accuracy = torch.sum(torch.round(gridmap_output_batch) == gridmap_batch).item() / float(gridmap_output_batch.numel())
        gridmap_accuracy = torch.sum(abs(gridmap_output_batch -  gridmap_batch) <= delta).item() / float(gridmap_output_batch.numel())
        #accu_list.append(gridmap_accuracy)
        accu += gridmap_accuracy

        loss = loss_function(gridmap_output_batch, gridmap_batch)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.6f}, acc: {:.6f}".format(epoch,accu_loss.item() / (step + 1),accu.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu.item() / (step + 1)

def train_one_epoch_classifier(model, optimizer, data_loaders, device, epoch, lossFunction = "CE"):
    model.train()

    loss_function = get_loss_function(lossFunction)

    optimizer.zero_grad()

    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    total_steps = 0

    for loader_name, data_loader in data_loaders:
        data_loader = tqdm(data_loader, file=sys.stdout)

        for step, data in enumerate(data_loader):
            input_batch, label_batch = data

            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device).long()

            logits = model(input_batch)
            loss = loss_function(logits, label_batch)
            loss.backward()

            batch_acc = torch.sum(torch.argmax(logits, dim=1) == label_batch).item() / float(label_batch.numel())
            accu += batch_acc
            accu_loss += loss.detach()
            total_steps += 1

            data_loader.desc = "[train epoch {}][{}] loss: {:.6f}, acc: {:.6f}".format(
                epoch,
                loader_name,
                accu_loss.item() / total_steps,
                accu.item() / total_steps,
            )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

    return accu_loss.item() / total_steps, accu.item() / total_steps

def adjust_learning_rate(lr, optimizer, epoch, decay_rate):
    adj_lr = lr * (0.5 ** (epoch // decay_rate))
    if adj_lr > 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = adj_lr


@torch.no_grad()
def evaluate(model,data_loader,device,epoch, lossFunction = "L1"):
    model.eval()

    loss_function = get_loss_function(lossFunction)

    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    accu_list = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        input_batch, gridmap_batch = data
        gridmap_output_batch = model(input_batch.to(device))
        gridmap_batch = gridmap_batch.to(device)
        
        #gridmap_accuracy = torch.sum(torch.round(gridmap_output_batch) == gridmap_batch).item() / float(gridmap_output_batch.numel())
        gridmap_accuracy = torch.sum(abs(gridmap_output_batch -  gridmap_batch) <= delta).item() / float(gridmap_output_batch.numel())
        #accu_list.append(gridmap_accuracy)
        accu += gridmap_accuracy

        loss = loss_function(gridmap_output_batch, gridmap_batch)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.6f}, acc: {:.6f}".format(epoch, accu_loss.item() / (step + 1), accu.item() / (step + 1))

    return accu_loss.item() / (step + 1), accu.item() / (step + 1)

def train_one_epoch_multi(model, optimizer, data_loader, device, epoch, lossFunction = "L1"):
    model.train()

    loss_function = get_loss_function(lossFunction)

    optimizer.zero_grad()

    accu_list = []
    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        input_batch, qp_batch ,gridmap_batch = data

        input_batch = input_batch.to(device)
        qp_batch = qp_batch.to(device)
        gridmap_batch = gridmap_batch.to(device)

        gridmap_output_batch = model(input_batch,qp_batch)

        gridmap_accuracy = torch.sum(abs(gridmap_output_batch -  gridmap_batch) <= delta).item() / float(gridmap_output_batch.numel())

        accu += gridmap_accuracy

        loss = loss_function(gridmap_output_batch, gridmap_batch)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.6f}, acc: {:.6f}".format(epoch,accu_loss.item() / (step + 1),accu.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu.item() / (step + 1)

@torch.no_grad()
def evaluate_multi(model,data_loader,device,epoch, lossFunction = "L1"):
    model.eval()

    loss_function = get_loss_function(lossFunction)

    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    accu_list = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        input_batch, qp_batch, gridmap_batch = data
        gridmap_output_batch = model(input_batch.to(device),qp_batch.to(device))
        gridmap_batch = gridmap_batch.to(device)
        
        gridmap_accuracy = torch.sum(abs(gridmap_output_batch -  gridmap_batch) <= delta).item() / float(gridmap_output_batch.numel())
        accu += gridmap_accuracy

        loss = loss_function(gridmap_output_batch, gridmap_batch)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.6f}, acc: {:.6f}".format(epoch, accu_loss.item() / (step + 1), accu.item() / (step + 1))

    return accu_loss.item() / (step + 1), accu.item() / (step + 1)

class SwinTransUDataset(Dataset):
    def __init__(self, data_type, path, qp_list, min_qp=0, max_qp=51):
        """
        初始化数据集
        
        参数:
            data_type (str): 'Training' 或 'Validating' or 'Testing'
            path (str): 数据根目录
            qp_list (list): QP值列表
            min_qp, max_qp: 用于标准化QP值
        """
        self.data_type = data_type
        self.path = path
        self.qp_list = qp_list
        self.min_qp = min_qp
        self.max_qp = max_qp
        
        # 标准化QP值
        self.qp_standardized_list = [(x - min_qp) / (max_qp - min_qp) for x in qp_list]
        
        # 加载YUV数据的形状（不加载完整数据）
        yuv_path = os.path.join(path, 'YUV96', data_type, f'{data_type}_Y_Block96.npy')
        with open(yuv_path, 'rb') as f:
            # 只读取numpy文件的形状信息
            self.yuv_shape = np.lib.format.read_array(f).shape
        self.single_qp_count = self.yuv_shape[0]  # 单个QP的样本数量
        self.total_count = self.single_qp_count * len(qp_list)  # 总样本数量
        
        # 存储网格图文件路径，避免重复计算
        self.gridmap_paths = [
            os.path.join(path, 'Gridmap', data_type, f'{data_type}_Luma_QP_{qp}_Gridmap16.npy')
            for qp in qp_list
        ]

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        # 确定当前索引对应的QP索引和内部索引
        qp_idx = idx // self.single_qp_count
        inner_idx = idx % self.single_qp_count
        qp = self.qp_list[qp_idx]
        qp_standardized = self.qp_standardized_list[qp_idx]
        
        # 加载YUV数据（只加载需要的样本）
        yuv_path = os.path.join(self.path, 'YUV96', self.data_type, f'{self.data_type}_Y_Block96.npy')
        yuv_data = np.load(yuv_path, mmap_mode='r')  # 使用内存映射，不加载整个文件
        yuv_sample = torch.FloatTensor(yuv_data[inner_idx].copy())
        
        # 准备QP数据
        qp_sample = torch.FloatTensor([qp_standardized])
        
        # 加载网格图数据（只加载需要的样本）
        gridmap_data = np.load(self.gridmap_paths[qp_idx], mmap_mode='r')  # 使用内存映射
        gridmap_sample = torch.FloatTensor(gridmap_data[inner_idx].copy())
        
        return yuv_sample, qp_sample, gridmap_sample
    
class DynamicShapeDataset(Dataset):
    def __init__(self, data_list, labels=None):
        """
        初始化动态形状数据集
        :param data_list: 包含不同形状数据的列表，如 [data1, data2, data3]
        :param labels: 对应的标签列表，如果为None则生成随机标签
        """
        self.data_list = data_list
        # 如果没有提供标签，生成随机标签（0-2三类）
        self.labels = labels if labels is not None else [np.random.randint(0, 3) for _ in range(len(data_list))]
        
        # 打印数据集信息
        print(f"数据集包含 {len(self.data_list)} 个样本")
        for i, data in enumerate(self.data_list):
            print(f"样本 {i+1} 形状: {data.shape}, 标签: {self.labels[i]}")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 直接返回原始数据（不做任何形状修改）和对应的标签
        data = self.data_list[idx]
        label = self.labels[idx]
        
        # 转换为Tensor，保持原始形状
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
