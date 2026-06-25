import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import paths

from model import SwinTransformer_Unet_Luma as model_single_qp
from model import Classifier as classifier

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain_model(current_model, pretrain_model,device):
    source_dict = torch.load(pretrain_model,map_location=device)
    if "state_dict" in source_dict.keys():
        source_dict = remove_prefix(source_dict['state_dict'], 'module.')
    else:
        source_dict = remove_prefix(source_dict, 'module.')
    dest_dict = current_model.state_dict()
    trained_dict = {k: v for k, v in source_dict.items() if
                    k in dest_dict and source_dict[k].shape == dest_dict[k].shape}
    dest_dict.update(trained_dict)
    current_model.load_state_dict(dest_dict)
    # for k, v in trained_dict.items():
    #     print(k)
    return current_model

@torch.no_grad()
def inference_pre(dataloader, Net, device):  # for overall inference
    total_qt_out_batch = torch.zeros((1, 2, 16, 16))
    accu = torch.zeros(1).to(device)
    accu1 = torch.zeros(1).to(device)

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_batch,input_label = data
            input_batch = input_batch.to(device)
            input_label = input_label.to(device)
            gridmap_output_batch = Net(input_batch)

            gridmap_accuracy = torch.sum(abs(gridmap_output_batch -  input_label) <= 1e-1).item() / float(gridmap_output_batch.numel())
            gridmap_accuracy1 = torch.sum(torch.round(gridmap_output_batch) == input_label).item() / float(gridmap_output_batch.numel())
            accu += gridmap_accuracy
            accu1 += gridmap_accuracy1
            total_accu = accu.item() / (step + 1)
            total_accu1 = accu1.item() / (step + 1)

            total_qt_out_batch = torch.cat([total_qt_out_batch, gridmap_output_batch.cpu()], 0)
            if step % 100 == 0:
                print("Number of finished blocks: ", total_qt_out_batch.shape[0])
            # del input_batch, gridmap_output_batch, bt_out_batch
        total_qt_out_batch = total_qt_out_batch[1:]
        print("Total accuracy:",total_accu)
        print("Total accuracy1:",total_accu1)

    return total_qt_out_batch


@torch.no_grad()
def inference_VVC_seqs(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ********************************** Load Input Blocks *************************************
    #block_y = np.load("./Dataset/YUV/Validating/Block_Y/0900_Y_Block64.npy")
    #block_y = np.load("./Dataset/YUV/Training/Block_Y/0100_Y_Block64.npy")
    block_y = np.load("./Dataset/YUV/Testing/Block_Y/RaceHorsesC_Y_Block64.npy")
    #gridmap = np.load("./Dataset/Gridmap/Training/qp_32/0100_Luma_qp32_gridmap.npy")
    gridmap = np.load("./Dataset/Gridmap/Validating/qp_32/0900_Luma_qp32_gridmap.npy")
    
    comp = "LU"
    input_batch = torch.FloatTensor(np.expand_dims(block_y, 1))
    input_label = torch.FloatTensor(gridmap)
    print('input_batch.shape:', input_batch.shape)
    print('input_label.shape:', input_label.shape)
    input_batch = input_batch[0:558,:,:,:]
    print("Creating inference data loader...")
    dataset = TensorDataset(input_batch,input_label)
    test_loader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=False)

    qp = 32
    # ********************************** Load Models *************************************
    start_time = time.time()
    Net = model_single_qp()

    net_path = "./Train_Loss/0002_L1_qp32/model-299.pth"
    Net = load_pretrain_model(Net, net_path,device)
    Net = Net.to(device)
    # ********************************** Network Inference *************************************
    gridmap_output_batch = inference_pre(test_loader, Net, device)
    print(gridmap_output_batch.shape)

    # ********************************** Post Process ************************************
    #gridmap_output_batch = torch.FloatTensor(gridmap_output_batch).cuda()  # b*1*4*4

    save_path = os.path.join(save_dir, "test.npy")
    print("Save:", save_path)
    np.save(save_path, gridmap_output_batch)
 
    #with open(save_path, 'w') as f:
    #    for batch_idx in range(gridmap_output_batch.size(0)):
    #        current_batch_4x4 = gridmap_output_batch[batch_idx,0,:,:]
    #        for row_idx in range(current_batch_4x4.size(0)):
    #            row_str = ' '.join(map(str, current_batch_4x4[row_idx, :]))
    #            f.write(row_str + '\n')

@torch.no_grad()
def inference_Classifier(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ********************************** Load Input Blocks *************************************
    gridmap = np.load("./Dataset/Gridmap/Validating/qp_32/0900_Luma_qp32_gridmap.npy")
    
    comp = "LU"
    input_batch = torch.FloatTensor(gridmap)
    print('input_batch.shape:', input_batch.shape)

    print("Creating inference data loader...")
    dataset = TensorDataset(input_batch)
    infer_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=args.batchSize, pin_memory=True, shuffle=False)

    qp = 32
    # ********************************** Load Models *************************************
    start_time = time.time()
    Net = classifier()

    net_path = "./Train_Loss/0000/model-29.pth"
    Net = load_pretrain_model(Net, net_path)
    Net = Net.to(device)
    # ********************************** Network Inference *************************************
    total_prediction_batch = torch.zeros((1,6))
    with torch.no_grad():
        for step, data in enumerate(infer_loader):
            input, = data
            print(f"Input type: {type(input)}")
            input = input.to(device)
            prediction_output_batch = Net(input)

            total_prediction_batch = torch.cat([total_prediction_batch, prediction_output_batch.cpu()], 0)
            if step % 100 == 0:
                print("Number of finished blocks: ", total_prediction_batch.shape[0])

        total_prediction_batch = total_prediction_batch[1:]
        total_prediction_batch = F.softmax(total_prediction_batch,dim = 1)

    # ********************************** Post Process ************************************

    save_path = os.path.join(save_dir, "test.npy")
    print("Save:", save_path)
    np.save(save_path, total_prediction_batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default=str(paths.output_root()))
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--startSeqID', default=0, type=int, help='QP start ID')
    parser.add_argument('--seqNum', default=22, type=int, help='test QP number')

    args = parser.parse_args()

    start_time = time.time()
    inference_VVC_seqs(args)
    #inference_Classifier(args)
    infe_time = time.time() - start_time
    print('Total inference time:', infe_time)

    '''
        python inference.py --batchSize 256 --outDir ./output/ --jobID 0002
    '''
