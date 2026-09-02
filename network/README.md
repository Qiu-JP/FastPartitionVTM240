# network

`network/` 是 FastPartitionVTM 的神经网络工作区，负责把标准 VTM 导出的划分信息转换为训练数据集，并完成模型训练、单样本推理、TorchScript 导出和实验结果查看。

Luma 输入为 `1x48x48`，gridmap 标签为 `2x8x8`；Chroma 输入为 `2x32x32`，gridmap 标签为 `2x4x4`。`Classifier_I` 在局部 gridmap ROI 上预测 CU 划分类型，类别顺序为 `[NO_SPLIT, QT, BTH, BTV, TTH, TTV]`。

当前主线数据尺寸为：Luma block `32x32`、输入 `48x48`；Chroma block `16x16`、输入 `32x32`。两者均由 `createDataset.py` 直接生成，不需要额外的合并脚本。

## 目录结构

```text
network/
  src/
  script/
  checkpoints/
  output/
  figures/
```

| 路径 | 内容和功能 | 
| --- | --- |
| `src/` | 源码：路径管理、数据集生成、模型定义、训练、推理、导出和工具函数。 |
| `script/` | 临时脚本。 |
| `checkpoints/` | 训练 checkpoint 和导出的 TorchScript 模型。 |
| `output/` | 训练日志、loss、TensorBoard 事件文件和实验输出。 |
| `figures/` | 数据集预览、推理可视化图和文档配图。 |

原始视频、VTM 划分文本和生成后的训练数据不放在 `network/`，统一由 `data/` 管理。

## 主线流程

1. `ref_model/script/` 运行标准 VTM，生成标准划分文本：
   - `data/partition/<dataset>/<split>/Luma_Partition_Info.txt`
   - `data/partition/<dataset>/<split>/Chroma_Partition_Info.txt`
2. `network/src/createDataset.py` 读取 `data/video/`、`data/partition/` 和序列清单，生成训练数据：
   - `Luma_Input.npy` / `Luma_Input.pkl`
   - `Luma_Gridmap.npy` / `Luma_Gridmap.pkl`
   - `Luma_CU_Tree.npy` / `Luma_CU_Tree.pkl`
   - `Chroma_Input.npy` / `Chroma_Input.pkl`
   - `Chroma_Gridmap.npy` / `Chroma_Gridmap.pkl`
   - `Chroma_CU_Tree.npy` / `Chroma_CU_Tree.pkl`
3. `network/src/createDataset.py --action classifier-pretrain` 生成 `Classifier_I` 逻辑预训练数据。
4. `network/src/train.py` 训练 `Classifier_I`，或联合训练 Swin gridmap 网络与 `Classifier_I`。
5. checkpoint 写入 `network/checkpoints/<outDir>/<jobID>/`，日志和 TensorBoard 写入 `network/output/<outDir>/<jobID>/`。
6. `network/src/inference.py` 加载 checkpoint，对一个样本推理并保存 label/prediction 对比图。
7. `network/src/saveModel.py` 将 `.pth` 导出为 C++ 可加载的 TorchScript `.pt`，可使用 Netron 查看结构。

## `src/` 文件说明

| 文件 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `paths.py` | 统一管理项目路径和数据集路径。 | 项目根目录、dataset 名称、split 名称。 | `Path` 对象。 |
| `createDataset.py` | 32x32 Luma 和 16x16 Chroma 数据集生成与预览。 | VTM 划分文本、YUV 视频、序列清单；或内置逻辑划分规则。 | `data/dataset/<dataset>/<split>/` 下的 input/gridmap/CU tree；或 `network/figures/` 下预览图。 |
| `model.py` | 48x48 输入、32x32 目标的 Swin gridmap 网络和 `Classifier_I` 定义。 | 训练/推理脚本传入的张量。 | gridmap 预测、分类 logits/probability。 |
| `train.py` | 32x32 主线训练入口。 | 数据集 `.npy/.pkl`、可选预训练 checkpoint。 | `.pth` checkpoint、日志、loss、TensorBoard。 |
| `inference.py` | 32x32 单样本推理与可视化。 | Swin checkpoint、数据集样本 index 或样本 id。 | `network/figures/<name>.png`。 |
| `saveModel.py` | 32x32 模型导出。 | `.pth` checkpoint。 | TorchScript `.pt`。 |
| `utils.py` | Dataset、样本对齐、CU tree 读取、loss、训练/验证循环、可视化辅助函数。 | 数据集文件、模型输出和标签。 | batch、loss、指标、TensorBoard 图像。 |

## 数据集生成

生成 training 数据：

```bash
python network/src/createDataset.py \
  --data-type 1 \
  --dataset DIV2K \
  --sequence-list ref_model/script/Training_Sequences_DIV2K.txt \
  --component both \
  --action gridmap-input-cu-tree
```

生成 validating 数据：

```bash
python network/src/createDataset.py \
  --data-type 3 \
  --dataset DIV2K \
  --sequence-list ref_model/script/Validating_Sequences_DIV2K.txt \
  --component both \
  --action gridmap-input-cu-tree
```

生成 `Classifier_I` 逻辑预训练数据：

```bash
python network/src/createDataset.py \
  --action classifier-pretrain
```

预览数据集中的一个样本：

```bash
python network/src/createDataset.py \
  --action preview \
  --data-type 1 \
  --dataset DIV2K \
  --component luma \
  --show-sample-index 819088 \
  --show-output div2k_training_luma_819088.png
```

也可以用完整样本 id 指定预览对象：

```bash
python network/src/createDataset.py \
  --action preview \
  --data-type 1 \
  --dataset DIV2K \
  --component luma \
  --show-sequence 0367 \
  --show-qp 27 \
  --show-frame-id 0 \
  --show-ctu-id 400 \
  --show-output div2k_training_luma_0367_qp27_f0_ctu400.png
```

`createDataset.py` 的主要参数：

| 参数 | 作用 |
| --- | --- |
| `--data-type` | `1` 表示 training，`2` 表示 testing，`3` 表示 validating。 |
| `--dataset` | 数据集名称；默认由 `--data-type` 推导。 |
| `--sequence-list` | 序列清单路径，用于从 YUV 读取输入块。 |
| `--component` | `luma`、`chroma` 或 `both`。 |
| `--action` | `gridmap`、`input`、`cu-tree`、`gridmap-input-cu-tree`、`preview`、`classifier-pretrain` 等。 |
| `--show-sample-index` | 预览时按当前数据集 sample index 选样本。 |
| `--show-sequence/--show-qp/--show-frame-id/--show-ctu-id` | 预览时按完整样本 id 选样本。 |

## 模型训练

预训练 `Classifier_I`：

```bash
python network/src/train.py \
  --task pretrain_classifier_logical \
  --outDir classifier_i_pretrain \
  --jobID logical \
  --epoch 50 \
  --batchSize 128 \
  --lr 1e-3 \
  --device cuda:0
```

联合训练 Swin gridmap 网络和 `Classifier_I`：

```bash
python network/src/train.py \
  --task swin_luma \
  --dataset DIV2K \
  --trainSplit training \
  --valSplit validating \
  --outDir swin_luma64_joint \
  --jobID exp001 \
  --epoch 50 \
  --batchSize 256 \
  --lr 1e-4 \
  --dr 20 \
  --jointStage1Epoch 25 \
  --stage1Lr 1e-4 \
  --stage2Lr 1e-4 \
  --stage1GridLossWeight 1.0 \
  --stage1ClsLossWeight 0.02 \
  --stage2GridLossWeight 0.5 \
  --stage2ClsLossWeight 1.0 \
  --gridLossType BCE \
  --classifierCkpt network/checkpoints/classifier_i_pretrain/logical/model-final.pth \
  --device cuda:0
```

训练输出：

```text
network/checkpoints/<outDir>/<jobID>/
  swin-<epoch>.pth
  classifier-<epoch>.pth
  swin-final.pth
  classifier-final.pth

network/output/<outDir>/<jobID>/
  train.log
  loss.txt
  tensorboard/
```

联合训练中常调参数：

| 参数 | 作用 |
| --- | --- |
| `--epoch` | 总训练轮数。 |
| `--batchSize` | mini-batch 大小，显存不足时优先减小。 |
| `--lr` | 默认基础学习率。 |
| `--dr` | 学习率衰减间隔。 |
| `--jointStage1Epoch` | 第一阶段 epoch 数；之后进入第二阶段。 |
| `--stage1Lr` / `--stage2Lr` | 两阶段学习率。 |
| `--stage1GridLossWeight` / `--stage2GridLossWeight` | 两阶段 gridmap loss 权重。 |
| `--stage1ClsLossWeight` / `--stage2ClsLossWeight` | 两阶段 classifier loss 权重。 |
| `--gridLossType` | gridmap loss 类型：`BCE`、`BCE_L1`、`WBCE`、`L1`、`HUBER`、`MSE`。 |
| `--fasttrain` | `1` 启用推荐快速训练配置：全部节点、向量化 ROI、CUDA AMP、4 个 DataLoader workers、每 50 batch 更新进度；不会进行节点采样。`0` 使用原有的 legacy FP32 路径。 |
| `--swinCkpt` | 可选 Swin checkpoint。 |
| `--classifierCkpt` | 可选 `Classifier_I` checkpoint。 |
| `--tbTrainImageSamples` | TensorBoard training 图像样本，默认 `simple:1989496,medium:819088,complex:320136`。 |
| `--tbValImageSamples` | TensorBoard validation 图像样本，默认 `simple:116602,medium:54490,complex:121714`。 |
| `--tbImageSampleIndex` | 可选覆盖参数；若提供，training 和 validation 都只写这一个样本。 |

推荐快速训练配置：

```bash
--batchSize 256 \
--fasttrain 1
```

`--fasttrain 1` 内部固定使用全节点向量化 ROI、无分块、4 个 DataLoader workers、
每 50 batch 更新一次 tqdm，并为 Swin 主干启用 CUDA AMP。Classifier 与 BCE/grid loss
仍保持 FP32。训练过程中不会采样或丢弃 Classifier 节点。

训练和验证结束时会打印峰值 CUDA allocated memory，并写入 TensorBoard 的
`Memory/train_peak_allocated_mb` 与 `Memory/val_peak_allocated_mb`。

## TensorBoard

TensorBoard 默认写入：

```text
network/output/<outDir>/<jobID>/tensorboard/
```

启动方式：

```bash
tensorboard \
  --logdir network/output/<outDir>/<jobID>/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

联合训练会记录总 loss、gridmap loss、classifier loss、gridmap precision/recall、classifier accuracy、loss 权重和学习率；同时按 `Classifier_I` 的 ROI 形状记录：

```text
Classifier/train/<HxW>/loss
Classifier/train/<HxW>/acc
Classifier/val/<HxW>/loss
Classifier/val/<HxW>/acc
```

每个 epoch 还会写入固定样本的 gridmap 对比图：

```text
Gridmap/train/simple
Gridmap/train/medium
Gridmap/train/complex
Gridmap/val/simple
Gridmap/val/medium
Gridmap/val/complex
```

每张图包含 label 划分图、label gridmap 数值矩阵、prediction 划分图和 prediction gridmap 数值矩阵。

## 单样本推理

`inference.py` 用于快速观察一个样本上的 gridmap 预测结果。

按 sample index 推理：

```bash
python network/src/inference.py \
  --checkpoint network/checkpoints/swin_luma64_joint/exp001/swin-final.pth \
  --dataset DIV2K \
  --split validating \
  --sampleIndex 116602 \
  --visualizeOutput inference_sample.png
```

按完整样本 id 推理：

```bash
python network/src/inference.py \
  --checkpoint network/checkpoints/swin_luma64_joint/exp001/swin-final.pth \
  --dataset DIV2K \
  --split training \
  --sequence 0367 \
  --qp 27 \
  --frameID 0 \
  --ctuID 400 \
  --visualizeOutput inference_0367_qp27_f0_ctu400.png
```

输出图像写入：

```text
network/figures/<visualizeOutput>
```

## 模型导出

导出 Swin TorchScript：

```bash
python network/src/saveModel.py \
  --task swin_luma \
  --checkpoint network/checkpoints/swin_luma64_joint/exp001/swin-final.pth \
  --device cpu
```

默认导出到 checkpoint 同目录，文件名为 `swin-final.pt`。

导出 `Classifier_I` native JSON：

```bash
python network/src/saveModel.py \
  --task export_classifier_json \
  --checkpoint network/checkpoints/classifier_i_pretrain/logical/model-final.pth \
  --output network/checkpoints/classifier_i_pretrain/logical/model-final.native.json \
  --device cpu
```

VTM 通过 `--FastPartitionClassifierModel=<path>` 直接加载导出的 `.native.json` 文件。

导出完成后，使用独立命令启动 Netron 查看 `.pt` 结构：

```bash
netron network/checkpoints/swin_luma64_joint/exp001/swin-final.pt \
  --host 127.0.0.1 \
  --port 8080
```
