# network

`network/` 是 FastPartitionVTM 的神经网络工作区，负责把标准 VTM 导出的划分信息转换为训练数据集，并完成模型训练、单样本推理、ONNX 导出、量化和实验结果查看。

Luma 输入为 `1x48x48`，gridmap 标签为 `2x8x8`；Chroma 输入为 `2x32x32`，gridmap 标签为 `2x4x4`。`Classifier_I` 在局部 gridmap ROI 上预测 CU 划分类型，类别顺序为 `[NO_SPLIT, QT, BTH, BTV, TTH, TTV]`。

当前训练与 VTM 部署主线为 Luma block `32x32`、输入 `48x48`。数据工具另支持 Chroma block `16x16`、输入 `32x32`，用于辅助数据和预览，不代表部署了色度网络。

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
| `checkpoints/` | 训练 checkpoint 和导出的 ONNX 模型。 |
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
3. 准备与 CU tree 节点对齐的 RD cost 数据；现有缓存直接复用。
4. `network/src/train.py` 使用 gridmap、分类标签与 RD cost 联合训练 Swin 和 `Classifier_I`。
5. checkpoint 写入 `network/checkpoints/<outDir>/<jobID>/`，日志和 TensorBoard 写入 `network/output/<outDir>/<jobID>/`。
6. `network/src/inference.py` 加载 checkpoint，对一个样本推理并保存 label/prediction 对比图。
7. `network/src/saveModel.py --task onnx_bundle` 将 `.pth` 导出为 FP32 ONNX；同文件的 `--task quantize_onnx_bundle` 可进一步生成部分 INT8 的部署版本。当前 VTM 主线使用 ONNX Runtime，旧 TorchScript/native JSON 入口仅保留兼容。

## `src/` 文件说明

| 文件 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `paths.py` | 统一管理项目路径和数据集路径。 | 项目根目录、dataset 名称、split 名称。 | `Path` 对象。 |
| `createDataset.py` | 32x32 Luma 和 16x16 Chroma 数据集生成与预览。 | VTM 划分文本、YUV 视频、序列清单；或内置逻辑划分规则。 | `data/dataset/<dataset>/<split>/` 下的 input/gridmap/CU tree；或 `network/figures/` 下预览图。 |
| `model.py` | 48x48 输入、32x32 目标的 Swin gridmap 网络和 `Classifier_I` 定义。 | 训练/推理脚本传入的张量。 | gridmap 预测、分类 logits/probability。 |
| `train.py` | 32x32 RD cost 联合训练唯一入口。 | 数据集 `.npy/.pkl`、可选预训练 checkpoint。 | `.pth` checkpoint、日志、loss、TensorBoard。 |
| `inference.py` | 32x32 单样本推理与可视化。 | Swin checkpoint、数据集样本 index 或样本 id。 | `network/figures/<name>.png`。 |
| `saveModel.py` | 模型导出与 ONNX 量化。 | `.pth` checkpoint 或 FP32 ONNX bundle。 | FP32 ONNX、部分 INT8 ONNX；另保留旧导出接口。 |
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
| `--source-dataset`、`--output-split` | 原始数据集名称、输出 split；例如源 CUSTOM，输出 dataset 为 CUSTOM_32。 |

## 模型训练

唯一入口为 `network/src/train.py`，原 `train2.py` 的 RD cost 训练和必要辅助函数已合并到这里。
旧普通训练及独立逻辑分类器预训练入口已删除。逻辑数据生成工具仍保留，但不是当前主线必需步骤。

以下命令从项目根目录、在 FastPartitionVTM Python 环境执行。训练需要 PyTorch，导出需要 `onnx` 和 `onnxruntime`。

### 数据和尺寸

当前主线用 CUSTOM_32 的 training/validating，VVC 测试帧不参与训练。
每个 split 的 `data/dataset/CUSTOM_32/<split>/` 包含 Luma Input、Gridmap、CU Tree 的
`.npy/.pkl` 文件，以及 `Luma_CU_RDCost.npy/.pkl`。
RD 数据按 CU tree 节点顺序对齐，读取时校验节点数和 offsets。
相对 RD 差为 `(candidate_cost - best_cost) / max(abs(best_cost), 1e-12)`，未完成候选由独立标记区分。
训练直接读取各 split 内的 `Luma_CU_RDCost.npy/.pkl`，不在训练入口解析原始 RD dump。

目标是 48×48 输入右下角的原图 32×32，即 `[16:48,16:48]`，Swin 输出为 `N×2×8×8`。
原图尺寸按宽×高描述，分类头和 tensor 空间尺寸按高×宽：原图 32×16 对应 grid 4×8。
预览背景也按右下角目标区域裁剪。

现有 RD cache 直接复用。确需生成时，training 使用下面命令；validating 改为 `--data-type 3`、验证集 RD 目录和验证序列清单：

```bash
python network/src/createDataset.py \
  --data-type 1 --dataset CUSTOM_32 --component luma --action rdocost \
  --rdo-root data/rdo_cost/CUSTOM/training \
  --rdo-sequence-list ref_model/script/Training_Sequences_CUSTOM.txt
```

RD 数据准备统一使用 `createDataset.py --action rdocost`；`train.py` 仅负责训练及训练期间的验证。
前面的 DIV2K 数据生成命令是接口示例，不是当前主线训练数据来源。

### 主线命令与损失

```bash
python network/src/train.py \
  --dataset CUSTOM_32 --trainSplit training --valSplit validating \
  --outDir swin_luma32_custom_rd_delta_safety --jobID custom_rd_new_run \
  --epoch 45 --batchSize 256 --lr 1e-4 --dr 20 \
  --jointStage1Epoch 15 \
  --stage1GridLossWeight 1 --stage1ClsLossWeight 0.05 \
  --stage2GridLossWeight 0.5 --stage2ClsLossWeight 1 \
  --gridLossType BCE --rdLossType delta \
  --hardCeWeight 1 --rdPenaltyWeight 1 --rdDeltaClamp 0.2 \
  --rdSafetyWeight 0 --rdTopKWeight 0 \
  --classifierSafetyTauLarge 0.1 --classifierSafetyLossWeight 0.5 \
  --classifierSafetyThresholdPreset table3_lambda2000 \
  --classifierSafetyThresholdOnlyShapes 0 \
  --useContextMask --fasttrain 1 --numWorkers 0 \
  --checkpointInterval 1 --device cuda:0
```

这些训练参数已设为默认配置。新实验使用新的 `--jobID`，避免覆盖历史权重。
不传 `--swinCkpt`、`--classifierCkpt` 时从头训练。
前 15 轮侧重 gridmap，后 30 轮调整两部分权重；分类损失为硬标签 CE + RD delta 惩罚 + 0.5 倍真实类别安全损失。

| 参数 | 作用 |
| --- | --- |
| `--rdLossType soft` | 切换到 RD soft-target 损失；`--rdSoftWeight` 仅此模式生效。 |
| `--rdSafetyWeight` | RD 安全损失权重，主线为 0；不同于真实类别安全损失。 |
| `--rdTopKWeight` | 可选 Top-K 排序损失，默认 0；不是 VTM 剪枝策略开关。 |
| `--freezeSwin 1` | 冻结 Swin，仅微调分类器。 |
| `--no-useContextMask` | 关闭默认启用的 context mask，供消融使用。 |
| `--fasttrain 1` | CUDA 上启用 AMP。 |
| `--numWorkers` | DataLoader 进程数，默认 0。 |
| `--evalDataset VVC_CTC --evalSplit testing` | 增加测试指标记录，不参与反向传播；默认不启用。 |

训练安全阈值表不等于部署时的四档阈值。部署阈值须使用对应 ONNX 版本估计。

### 输出、恢复与现用权重

每轮 checkpoint 包括 `swin-epochNNN.pth`、`classifier-epochNNN.pth`、`checkpoint-epochNNN.pth`，
保存在 `network/checkpoints/<outDir>/<jobID>/`，结束时另存对应 `*-final.pth`。
日志目录为 `network/output/<outDir>/<jobID>/`，包含 `train.log`、`loss.txt`、`summary.txt` 和 `tensorboard/`。

`--resume <完整 checkpoint.pth>` 恢复模型、优化器和已完成轮数，仍需提供原实验训练参数；
参数不会自动从 checkpoint 恢复。`--epoch` 是目标总轮数。
`--swinCkpt`、`--classifierCkpt` 仅加载网络权重，不恢复优化器。

本地保存的主线训练目录是
`network/checkpoints/swin_luma32_custom_rd_delta_safety/custom_stage15_stage2_30_table3_safety_bs256_ep45/`，
共训练 45 轮，依据 CUSTOM 验证损失选择第 18 轮。这些 `.pth` 和日志不随仓库提供；下述推理/导出命令需要本地已有 checkpoint。仓库内可直接部署的是 `checkpoints/onnx/final/`。
历史日志中的 `train2.py` 指合并前入口；本次整理未重新训练或修改已有权重。

```bash
tensorboard --logdir network/output/<outDir>/<jobID>/tensorboard --host 127.0.0.1 --port 6006
```

## 单样本推理

`inference.py` 用于快速观察一个样本上的 gridmap 预测结果。

按 sample index 推理：

```bash
python network/src/inference.py \
  --checkpoint network/checkpoints/swin_luma32_custom_rd_delta_safety/custom_stage15_stage2_30_table3_safety_bs256_ep45/swin-epoch018.pth \
  --dataset CUSTOM_32 --split validating --sampleIndex 0 \
  --visualizeOutput custom_preview.png
```

也可用 `--sequence`、`--qp`、`--frameID`、`--ctuID` 指定完整样本 ID。

输出图像写入：

```text
network/figures/<visualizeOutput>
```

## 模型导出

仓库提供的固定部署模型位于 `checkpoints/onnx/final/`，来自 CUSTOM_32 的主线 RD-cost 联合训练 epoch 18。目录包含 `swin.onnx`、32×32及以下形状的 `classifier_*.onnx`；Swin 为部分动态 INT8，classifier 为 FP32。VTM 评估默认加载这一目录。其他训练 checkpoint 和导出模型保留在本地，不提交。

当前Classifier仅保留原图32×32及以下的分类头，已去掉grid 16×16（原图64×64）旧头。历史checkpoint加载时仅跳过该旧头的参数，其余参数仍严格校验；RD联合训练恢复时同步移除对应优化器状态。新ONNX导出不含 `classifier_16x16.onnx`，需使用已同步精简加载列表的新编译VTM。现有冻结实验bundle和旧二进制保持原样，不能把新bundle直接交给仍强制加载旧头的历史二进制。

当前主线统一由 `src/saveModel.py` 负责：

```bash
# .pth -> FP32 ONNX；--checkpoint 为包含同轮 Swin/classifier 权重的目录。
python network/src/saveModel.py --task onnx_bundle \
  --checkpoint network/checkpoints/swin_luma32_custom_rd_delta_safety/custom_stage15_stage2_30_table3_safety_bs256_ep45 \
  --epoch 18 --output /tmp/custom_ep018_export

# FP32 ONNX -> 部分 INT8 ONNX；--checkpoint 在此任务中为 FP32 ONNX 目录。
python network/src/saveModel.py --task quantize_onnx_bundle \
  --checkpoint /tmp/custom_ep018_export --output /tmp/custom_ep018_int8_export
```

量化任务调用 `quantize_onnx_bundle(source_dir, output_dir)`，只量化 Swin 中具有常量权重的 MatMul（动态量化、逐通道 QInt8），Classifier保持FP32。直接读取源目录的 ONNX 文件，检查模型格式，拒绝覆盖已有量化目录。FP32 导出与 INT8 量化均只输出 ONNX 文件，不生成或读取模型 manifest。它不读取 `.pth` 或重新训练。上例是重新导出的示例路径，现有正式bundle无需重新生成；实际运行需使用项目的FastPartitionVTM环境。

VTM 主线通过 ONNX Runtime CPU 加载 bundle，使用绝对路径配置：

```text
--FastPartitionSwinModel=/absolute/path/to/bundle/swin.onnx
--FastPartitionClassifierModel=/absolute/path/to/bundle
--FastPartitionLumaModelScale=32
```

FP32 bundle 也可直接部署，阈值另行配置。
`saveModel.py` 仍保留 `swin_luma`（TorchScript）和 `export_classifier_json` 的历史导出接口；当前部署流程使用 ONNX。
