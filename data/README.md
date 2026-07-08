# data

`data/` 用于存放原始数据、标准 VTM 导出的划分数据、网络训练数据集，以及数据生成过程中的配置、日志和临时产物。

整体数据流为：

```text
原始 YUV 序列
  -> ref_model/script/gencfg.sh 生成 VTM cfg
  -> ref_model/script/run.sh 运行标准 VTM 并导出 CU 划分文本
  -> network/src/createDataset.py 生成网络训练数据集
  -> network/src/train.py 读取数据集训练模型
```

## 目录结构

```text
data/
  video/
  CodecTrainCfg/
  partition/
  dataset/
  codec_run/
  logs/
```

| 目录 | 内容 | 数据来源 | 功能 |
| --- | --- | --- | --- |
| `video/` | 原始 YUV 视频序列，按数据集分目录存放，例如 `DIV2K/`、`HEVC_CTC/`。 | 需要提前准备；DIV2K 可由图像数据转换为 YUV，HEVC/VVC CTC 通常来自外部测试序列。 | 作为标准 VTM 编码输入，也用于网络输入的数据集构建。 |
| `CodecTrainCfg/` | 按数据集和 QP 保存逐序列 VTM 编码配置，例如 `DIV2K/qp_22/0001_intra_vtm.cfg`。 | 由 `ref_model/script/gencfg.sh` 根据序列清单和 cfg 模板生成。 | 供 `ref_model/script/run.sh` 调用标准 VTM 编码器生成标准划分数据时读取。 |
| `partition/` | 标准 VTM 编码/解码后导出的 CU 划分文本，按数据集和划分集合组织，例如 `DIV2K/training/Luma_Partition_Info.txt`、`DIV2K/validating/Chroma_Partition_Info.txt`。 | 由 `ref_model/script/run.sh` 调用 `dumpPartition` 版本 VTM 生成。 | 作为编码器导出的待处理标准划分数据，用于生成 gridmap 标签和 CU 划分类型标签。 |
| `dataset/` | 网络可直接读取的训练、验证、测试和预训练数据集。 | 主要由 `network/src/createDataset.py` 根据 `video/` 和 `partition/` 生成；分类器预训练数据由 `network/src/createDataset.py --action classifier-pretrain` 生成。 | 主要供 `network/src/train.py` 使用。 |
| `codec_run/` | 编码/解码过程中的临时工作目录，包含 bitstream、重建 YUV、解码 YUV 等中间文件。 | 由 `ref_model/script/run.sh` 运行时产生。 | 临时存储文件。 |
| `logs/` | 标准 VTM 编码/解码日志和校验记录。 | 由 `ref_model/script/run.sh` 写入。 | 用于检查编码是否成功、解码是否一致，以及定位数据生成问题。 |

## 主要数据说明

### `video/`

该目录放置原始输入序列。每个数据集单独建子目录，子目录中的文件名应与序列清单中的文件名一致。

典型结构：

```text
data/video/
  DIV2K/
    0001_2040x1404.yuv
    0002_2040x1848.yuv
  HEVC_CTC/
    ...
```

### `CodecTrainCfg/`

该目录保存逐序列 VTM cfg，由 `ref_model/script/gencfg.sh` 自动生成。

典型结构：

```text
data/CodecTrainCfg/
  DIV2K/
    qp_22/
      0001_intra_vtm.cfg
      0002_intra_vtm.cfg
```

这些 cfg 描述每条序列的输入路径、分辨率、帧数、帧率、QP 和 bitstream 输出名，用于后续标准编码和划分标签导出。

### `partition/`

该目录保存标准 VTM 导出的划分标签文本，是后续监督学习标签的来源。它由 `ref_model/script/run.sh` 调用 `ref_model/bin/dumpPartition/` 下的 VTM 可执行文件生成。

典型结构：

```text
data/partition/
  DIV2K/
    training/
      Luma_Partition_Info.txt
      Chroma_Partition_Info.txt
    validating/
      Luma_Partition_Info.txt
      Chroma_Partition_Info.txt
```

其中亮度划分记录用于生成 64x64 亮度块对应的 gridmap 标签，以及 `Classifier_I` 所需的 CU 划分类型标签；色度划分记录用于生成与该 LCU 对齐的色度输入、gridmap 标签和 CU tree 标签。

### `dataset/`

该目录保存网络训练阶段真正读取的数据。

对于 64x64 的主线，`network/src/createDataset.py` 会基于 `video/` 和 `partition/` 在真实数据集目录下生成 Luma 与 Chroma 两组数据：

- 64x64 亮度 input；
- 32x32 色度 input；
- 亮度对应的 2 通道、16x16 gridmap 标签；
- 色度对应的 2 通道、8x8 gridmap 标签。

以 `data/dataset/DIV2K/training/` 为例，gridmap 预测网络的数据集如下：

```text
Luma_Input.npy      # 亮度输入数组，形状通常为 [N, 1, 64, 64]
Luma_Input.pkl      # input metadata，包含样本 id、数组文件名、shape、dtype 等信息
Luma_Gridmap.npy    # gridmap 标签数组，形状通常为 [N, 2, 16, 16]
Luma_Gridmap.pkl    # gridmap metadata，包含样本 id、数组文件名、shape、dtype 等信息
Chroma_Input.npy    # 色度输入数组，形状通常为 [N, 2, 32, 32]
Chroma_Input.pkl
Chroma_Gridmap.npy  # 色度 gridmap 标签数组，形状通常为 [N, 2, 8, 8]
Chroma_Gridmap.pkl
```

同一分量的 `Input.pkl`、`Gridmap.pkl` 和 `CU_Tree.pkl` 中各自保存一份样本 id。训练读取时会根据 `sequence_name, qp, frame_id, ctu_id` 作为唯一 id 索引对齐输入数据、gridmap 标签和 CU tree 标签。

`Classifier_I` 需要学习“局部 gridmap ROI -> 当前 CU 划分类型”的映射。它的数据集分为两类：

- 真实数据的 CU tree 节点标签：来自 VTM 实际划分记录，用于让分类器贴近真实编码分布；
- `pretrain/Classifier_I/logical/`：预训练数据，用人工构造的标准划分结构让分类器先学会基本划分模式，例如 NO_SPLIT、QT、BT、TT 的标准边界形态，由 `network/src/createDataset.py --action classifier-pretrain` 按规则生成。脚本会遍历 `Classifier_I` 支持的所有 gridmap ROI 尺寸，并为该尺寸下合法的划分类型构建标准gridmap样本和标签。

CU tree 节点的数据集如下：

```text
Luma_CU_Tree.npy    # CU tree 节点数组，形状通常为 [M, 5]
Luma_CU_Tree.pkl    # CU tree metadata，记录样本 id、节点范围、数组文件名、shape、dtype、类别顺序等信息
Chroma_CU_Tree.npy
Chroma_CU_Tree.pkl
```

这两个文件采用和 `Luma_Input`、`Luma_Gridmap` 相同的“数据数组 + 映射表”组织形式。`Luma_CU_Tree.npy` 是所有样本的节点顺序拼接结果，每一行表示一个分类器训练节点，列含义由 `Luma_CU_Tree.pkl` 中的 `node_columns` 描述：

```text
grid_y, grid_x, grid_h, grid_w, label
```

其中 `grid_y, grid_x, grid_h, grid_w` 表示该节点在当前分量 gridmap 上对应的 ROI 位置和尺寸，`label` 表示该 ROI 的标准划分类型。`CU_Tree.pkl` 中的 `samples` 负责把 `sequence_name, qp, frame_id, ctu_id` 映射到样本编号，并记录该样本在 `CU_Tree.npy` 中对应的节点范围。训练时先根据样本 id 对齐同一分量的 `Input`、`Gridmap` 和 `CU_Tree`，再从预测或标签 gridmap 中裁剪节点 ROI 送入 `Classifier_I`。

典型结构：

```text
data/dataset/
  DIV2K/
    training/
      Luma_Input.npy
      Luma_Input.pkl
      Luma_Gridmap.npy
      Luma_Gridmap.pkl
      Luma_CU_Tree.npy
      Luma_CU_Tree.pkl
      Chroma_Input.npy
      Chroma_Input.pkl
      Chroma_Gridmap.npy
      Chroma_Gridmap.pkl
      Chroma_CU_Tree.npy
      Chroma_CU_Tree.pkl
    validating/
      Luma_Input.npy
      Luma_Input.pkl
      Luma_Gridmap.npy
      Luma_Gridmap.pkl
      Luma_CU_Tree.npy
      Luma_CU_Tree.pkl
      Chroma_Input.npy
      Chroma_Input.pkl
      Chroma_Gridmap.npy
      Chroma_Gridmap.pkl
      Chroma_CU_Tree.npy
      Chroma_CU_Tree.pkl
  pretrain/
    Classifier_I/
      logical/
        16x16/
          gridmap.npy
          label.npy
        8x8/
          gridmap.npy
          label.npy
        8x4/
          gridmap.npy
          label.npy
        ...
```

其中 `DIV2K/training/` 与 `DIV2K/validating/` 是从真实视频和标准 VTM 划分结果转换出的训练/验证数据；`pretrain/Classifier_I/logical/<HxW>/` 是分类器逻辑预训练数据，每个尺寸目录下保存该 ROI 尺寸对应的标准 gridmap 样本和划分类型标签。

### `codec_run/`

该目录是标准编码流程的临时工作区。`ref_model/script/run.sh` 运行时会在这里放置 bitstream、重建 YUV、解码 YUV 等中间文件，编码结束后会被自动清理。

### `logs/`

该目录保存标准 VTM 编码/解码日志。数据生成失败、MD5 校验异常或某条序列处理失败时，优先检查这里的日志。

## 典型生成顺序

1. 提前准备原始 YUV 序列，放入 `data/video/<dataset>/`。
2. 使用 `ref_model/script/gencfg.sh` 生成逐序列 VTM cfg，写入 `data/CodecTrainCfg/`。
3. 使用 `ref_model/script/run.sh` 运行标准 VTM，导出划分文本到 `data/partition/`，同时写入 `data/logs/` 和 `data/codec_run/`。
4. 使用 `network/src/createDataset.py` 将原始像素和标准划分数据转换为网络训练数据，写入 `data/dataset/`。
5. 使用 `network/src/train.py` 从 `data/dataset/` 读取数据并训练模型。
