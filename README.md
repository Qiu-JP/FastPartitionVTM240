# FastPartitionVTM

FastPartitionVTM 是一个面向 **VVC 标准参考软件 VTM 24.0** 的神经网络快速划分研究与开发仓库。

本仓库按职责拆分为几个相互独立的部分：

```text
ref_model/        标准 VTM 基线与划分标签生成
data/             视频、生成的 cfg、划分标签、训练数据集和日志
network/          模型代码、数据转换、训练、推理和模型导出
VVCSoftware_VTM/  实验性 VTM 集成与编码器侧验证
docs/             原理说明、模型细节和集成文档
deps/             ONNX Runtime CPU C/C++ 依赖
```

## 目录入口

| 目录 | 作用 | 继续阅读 |
| --- | --- | --- |
| `data/` | 存放外部输入数据与实验生成数据，主要被 `ref_model/` 和 `network/` 使用。 | [data/README.md](data/README.md) |
| `network/` | 存放 Python 模型定义、数据集转换、训练、推理、可视化和模型导出工具。 | [network/README.md](network/README.md) |
| `ref_model/` | 存放最小化标准 VTM 24.0 可执行基线、cfg 模板、序列清单和标签生成脚本。 | [ref_model/README.md](ref_model/README.md) |
| `VVCSoftware_VTM/` | 用于快速划分方法接入和验证的 VTM 主开发目录。 | [VVCSoftware_VTM/README.md](VVCSoftware_VTM/README.md) |
| `docs/` | 存放更详细的数据流、模型设计、分类器说明、VTM 集成和实现细节。 | [docs/README.md](docs/README.md) |

## 主流程

本仓库的完整使用流程可以按“标准标签生成 -> 数据集创建 -> 模型训练 -> 模型导出 -> VTM 集成验证”理解。

1. 使用 `ref_model/` 生成标准 VTM 划分标签

   `ref_model/` 保存标准 VTM 24.0 的最小可运行基线和标签生成脚本。使用时先根据序列清单和 cfg 模板生成逐序列编码配置，再调用 `dumpPartition` 版本的 VTM 编码器/解码器运行标准编码流程。该步骤的输入是 `data/video/` 中的原始 YUV 序列和 `ref_model/script/` 中的序列清单；输出主要为标准 VTM 编码得到的 CU 划分文本，保存到 `data/partition/` 下，运行日志与临时产物分别保存到 `data/logs/` 和 `data/codec_run/` 下。

2. 使用 `network/src/createDataset.py` 创建网络训练数据集

   标准划分文本生成后，进入 `network/` 侧的数据处理流程。`createDataset.py` 会把 `data/video/` 中的原始 YUV 像素和 `data/partition/` 中的 VTM 划分记录整理成两个网络需要的训练数据：一部分是 Swin gridmap 预测网络使用的 Luma/Chroma input 及其 gridmap 标签，其中 Luma 为 64x64 input 和 2 通道、16x16 gridmap，Chroma 为 32x32 input 和 2 通道、8x8 gridmap；另一部分是 `Classifier_I` 使用的局部 gridmap ROI 及其对应的 CU 划分类型标签。生成 input 时需要根据序列清单读取原始 YUV 的文件名、宽高和帧数等元信息；生成 gridmap 和划分类型标签时主要依赖标准 VTM 导出的划分记录。最终数据统一保存到 `data/dataset/`，供后续训练、验证和推理读取。

3. 使用 `network/src/train.py` 训练划分预测模型

   当前网络侧流程主要围绕 64x64 亮度块的 gridmap 预测和 CU 划分分类。`SwinTransformer_Unet` 预测 2 通道、16x16 的 gridmap；`Classifier_I` 将局部 gridmap ROI 映射为划分类别：

```text
[NO_SPLIT, QT, BTH, BTV, TTH, TTV]
```

   训练入口位于 `network/src/train.py`。训练脚本从 `data/dataset/` 读取数据，将训练日志、loss 记录和可视化/评估输出写入 `network/output/<outDir>/<jobID>/`，将模型权重保存到 `network/checkpoints/<outDir>/<jobID>/`。

4. 导出 VTM 可加载的模型文件

   训练得到的 `.pth` checkpoint 是 PyTorch 训练权重，主要用于继续训练、推理测试或模型分析。若要在 VTM C++ 侧加载模型，需要使用 `network/src/saveModel.py` 将 checkpoint 导出为 TorchScript `.pt` 文件。导出的 `.pt` 模型通常继续保存在对应的 `network/checkpoints/<outDir>/<jobID>/` 目录中，作为后续 VTM 集成的部署输入。

5. 在 `VVCSoftware_VTM/` 中加载模型并执行快速划分

   `VVCSoftware_VTM/` 是实验性 VTM 主开发目录。集成流程中，VTM 编码器通过 libtorch 加载 `network/checkpoints/` 中导出的 TorchScript 模型，在编码过程中对当前 CTU/CU 提取亮度输入，先由 Swin 网络预测 gridmap，再由 `Classifier_I` 给出当前 CU 的划分模式概率。VTM 侧仍以 RDO 流程为主，网络输出用于辅助判断进行剪枝，根据阈值决策模式跳过部分候选划分模式。最终使用 `VVCSoftware_VTM/script/` 中的评测脚本对 anchor 和快速划分版本进行编码时间、码率失真和 BD-rate 等指标对比。
