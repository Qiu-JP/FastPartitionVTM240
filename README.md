# FastPartitionVTM

基于 VTM 24.0 的 VVC intra 快速划分。Swin Transformer 提取特征，各形状 classifier 预测划分概率，再按原图尺寸和划分类别的独立阈值剪枝，最终候选由 VTM 的 RD 搜索决定。

## 目录

| 目录 | 内容与使用说明 |
| --- | --- |
| [ref_model/](ref_model/README.md) | 标准 anchor、划分与 RD cost 导出程序，以及训练数据生成脚本。 |
| [data/](data/README.md) | 原始视频、划分标签、RD cost 和训练数据；数据文件不随仓库提供。 |
| [network/](network/README.md) | 数据集转换、RD 联合训练、推理、ONNX 导出与量化。 |
| [VVCSoftware_VTM/](VVCSoftware_VTM/README.md) | 完整 VTM 源码、快速划分、阈值搜索与编码评估。 |
| [deps/](deps/README.md) | 随仓库提供的 Linux x86-64 ONNX Runtime CPU C/C++ 依赖。 |

本地 `docs/` 保存实验记录和工作笔记，不随仓库提交。

## 模型与尺寸

当前主线仅部署亮度模型，尺寸统一按原图宽×高描述：

| 项目 | 尺寸或含义 |
| --- | --- |
| 目标块 | 原图 32×32 像素。 |
| 网络输入 | 含上方、左方邻域的 48×48，目标位于右下角 `[16:48,16:48]`。 |
| Swin 输出 | 每样本 `2×8×8` grid map；一个网格单元对应原图 4×4 像素。 |
| 分类器 ROI | tensor 空间顺序为高×宽；例如原图 32×16 对应 grid 4×8。 |
| 类别顺序 | `NO_SPLIT, QT, BTH, BTV, TTH, TTV`。 |
| VTM 推理 | 每个 128×128 CTU 提前批量处理 16 个目标块，递归搜索按需使用缓存特征。 |

固定部署权重位于 `network/checkpoints/onnx/final/`，包括 `swin.onnx` 和 `classifier_*.onnx`。
该版本来自 CUSTOM_32 的 epoch 18，Swin 部分动态 INT8 量化，分类器 FP32。训练使用 PyTorch，VTM 部署使用 ONNX Runtime CPU，不需要 LibTorch、CUDA 或模型 manifest。

## 直接评估

在兼容的 Linux x86-64 环境，可使用 `VVCSoftware_VTM/script/bin/` 的预编译程序。
准备 `ref_model/script/Testing_Sequences_VVC.txt` 对应的 YUV，放入 `data/video/VVC_CTC/`，并在 Python 环境安装评估依赖后运行：

```bash
python VVCSoftware_VTM/script/evaluate.py \
  --bundle network/checkpoints/onnx/final \
  --thresholds VVCSoftware_VTM/script/thresholds_fast.cfg \
  --workdir VVCSoftware_VTM/script/output/fast_run
```

默认评估首帧及 QP 22/27/32/37，串行比较 `ref_model/bin/vtm240/` anchor 和快速划分程序，输出 BD-rate、time saving 和 4×4 compute saving。
阈值需显式指定。构建指令、依赖、其他参数及四档命令见 [VTM 说明](VVCSoftware_VTM/README.md)。

## 从数据到部署

1. 使用 `ref_model/script/gencfg.sh` 和 `run.sh` 生成划分标签；RD 训练需选择 `dumpRDCost` 编解码器。
2. 使用 `network/src/createDataset.py` 生成 Input、Gridmap、CU Tree 和对齐的 RD cost `.npy/.pkl`。
3. 使用唯一训练入口 `network/src/train.py` 联合训练 Swin 与分类器。
4. 使用 `network/src/saveModel.py` 将 `.pth` 导出为 FP32 ONNX，按需进一步量化为 INT8。
5. 使用 `collect_threshold_probabilities.py` 从数据集与模型生成概率缓存，再用 `ThSearch_Ratio.py` 或 `ThSearch_RdoCost.py` 搜索阈值；也支持直接推理后搜索。无需 CTU recipe。
6. 将生成的 `.cfg` 传给 `evaluate.py` 实测。数据集顺序组批的动态 INT8 概率不保证与 VTM CTU 组批逐项一致，离线搜索结果不能代替编码评估。
