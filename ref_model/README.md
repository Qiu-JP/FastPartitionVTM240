# ref_model

`ref_model/` 主要用于生成 intra 快速划分模型所需的训练监督数据：根据 YUV 序列和编码配置运行 VTM，导出划分结构及可选的候选 RD cost。同时提供原始 anchor 编解码器和编解码一致性校验工具。

| 子目录 | 功能 |
| --- | --- |
| `bin/` | 保存原始参考、划分导出、RD cost 导出三组编解码器。 |
| `cfg/` | 保存 intra 编码配置和序列参数模板。 |
| `script/` | 保存序列清单，以及清单生成、配置生成、数据导出和编解码校验脚本。 |

## 目录结构

```text
ref_model/
  bin/
    vtm240/
      EncoderAppStatic
      DecoderAppStatic
      DecoderAnalyserAppStatic
    dumpPartition/
      EncoderAppStatic
      DecoderAppStatic
    dumpRDCost/
      EncoderAppStatic
      DecoderAppStatic
  cfg/
    sequence.cfg
    encoder_intra_vtm.cfg
  script/
    Training_Sequences_DIV2K.txt
    Validating_Sequences_DIV2K.txt
    Training_Sequences_CUSTOM.txt
    Validating_Sequences_CUSTOM.txt
    Testing_Sequences_HEVC.txt
    Testing_Sequences_VVC.txt
    gentxt.sh
    gencfg.sh
    run.sh
    roundtrip.sh
```

- `bin/vtm240/`：VTM 24.0 参考编解码器，用于原始 anchor 和 roundtrip，输出只读 compute 计数，不导出划分或 RD cost。
- `bin/dumpPartition/`：用于生成训练划分标签的一组 VTM 可执行文件。其中 `EncoderAppStatic` 负责编码生成 bitstream，`DecoderAppStatic` 解码 bitstream，并读取划分导出环境变量生成 `Luma_Partition_Info.txt` 和 `Chroma_Partition_Info.txt`。
- `bin/dumpRDCost/`：编码器额外导出候选 RD cost，解码器导出划分结构。
- `cfg/sequence.cfg`：逐序列配置头模板，包含 `InputFile`、分辨率、帧数、帧率等字段。
- `cfg/encoder_intra_vtm.cfg`：VTM intra 编码主模板。生成逐序列 cfg 时，`gencfg.sh` 会先拼接 `sequence.cfg` 和该模板，再替换序列参数。
- `script/*_Sequences_*.txt`：序列清单，格式为 `sequence_name,file_name,width,height,frames,fps`。
- `script/gentxt.sh`：通过 FastPartitionVTM 环境的 Python/Pillow 读取 PNG 尺寸，生成序列清单；不负责把 PNG 转为 YUV。
- `script/gencfg.sh`：生成逐序列编码 cfg。
- `script/run.sh`：按序列清单批量编解码，默认导出划分标签；指定 `dumpRDCost` 编解码器时额外导出 RD cost。
- `script/roundtrip.sh`：调用 `bin/vtm240/` 中的标准编解码器做编码、解码和重建一致性校验。

三组编解码器均关闭快速划分。以下命令从项目根目录执行。

## 按数据来源选择流程

`gentxt.sh` 用于 DIV2K 这类 PNG 图像数据集，只读取图像尺寸并生成序列清单，
不负责 PNG → YUV 转换，也不是 VVC 专用工具。

- **PNG 图像数据集**：另行将 PNG 转换为 YUV → 用 `gentxt.sh` 生成对应清单 → 用 `gencfg.sh` 生成配置 → 用 `run.sh` 编解码并导出训练数据。
- **CUSTOM / VVC 序列**：已有 YUV 和序列清单 → 直接用 `gencfg.sh` 生成配置 → 用 `run.sh` 编解码并导出训练数据，无需运行 `gentxt.sh`。

例如，CUSTOM 训练集：

```bash
bash ref_model/script/gencfg.sh --dataset CUSTOM --qp 22 --type train
bash ref_model/script/run.sh --dataset CUSTOM --qp 22 --type train
```

VVC 测试序列：

```bash
bash ref_model/script/gencfg.sh --dataset VVC_CTC --qp 22 --type test
bash ref_model/script/run.sh --dataset VVC_CTC --qp 22 --type test
```

两者分别使用 `Training_Sequences_CUSTOM.txt` 和 `Testing_Sequences_VVC.txt`。
若需额外导出 RD cost，在 `run.sh` 中指定 `dumpRDCost` 编解码器，命令见下文。

## run.sh 与 roundtrip.sh

| 项目 | `run.sh` | `roundtrip.sh` |
| --- | --- | --- |
| 主要用途 | 批量生成划分标签及可选 RD cost 数据。 | 对指定输入做标准编码、解码和重建一致性校验。 |
| 输入 | 序列清单和 `gencfg.sh` 已生成的逐序列配置。 | `--input`、`--cfg`、宽高、帧数、帧率、QP 等参数。 |
| 编解码器 | 默认 `dumpPartition`；可通过 `--encoder` / `--decoder` 改为 `dumpRDCost`。 | 固定为 `bin/vtm240/`。 |
| 是否需要 gencfg | 标准流程先运行 `gencfg.sh`，也可提供已有配置目录。 | 不需要；直接指定主 cfg，另可传 `--seq-cfg`。 |
| 校验 | 比较编码重建和解码输出的 MD5。 | 用 `cmp` 逐字节比较重建和解码输出，同时记录 MD5。 |
| 输出保留 | 保留划分数据、可选 RD cost 和日志；校验成功后删除临时码流及 YUV。 | 保留码流、编码重建、解码 YUV 和日志，便于检查。 |

两者是按用途选择的独立入口，不需要先运行一个再运行另一个。
`run.sh --skip-decode` 仅执行编码，跳过解码、划分导出和重建校验。

## 生成标准划分数据

标准划分数据生成流程分两步：先生成每条序列的 VTM cfg，再运行 `dumpPartition` 编解码流程；其中 decoder 负责导出 partition 信息。

### 1. 生成编码配置

示例：生成 DIV2K 训练集 QP 22 的 cfg：

```bash
bash ref_model/script/gencfg.sh \
  --dataset DIV2K \
  --qp 22 \
  --type train
```

示例：生成 DIV2K 验证集 QP 22 的 cfg：

```bash
bash ref_model/script/gencfg.sh \
  --dataset DIV2K \
  --qp 22 \
  --type valid
```

示例：生成 HEVC 测试集 QP 32 的 cfg：

```bash
bash ref_model/script/gencfg.sh \
  --dataset HEVC \
  --qp 32 \
  --type test
```

默认输入：

- 序列清单：`ref_model/script/<Split>_Sequences_<dataset>.txt`；VVC_CTC、HEVC_CTC 分别使用名称为 VVC、HEVC 的清单。
- 序列头模板：`ref_model/cfg/sequence.cfg`
- 编码主模板：`ref_model/cfg/encoder_intra_vtm.cfg`
- 视频目录：`data/video/<dataset>/`；VVC/VVC_CTC 对应 `data/video/VVC_CTC/`，HEVC/HEVC_CTC 对应 `data/video/HEVC_CTC/`。可用 `--video-root` 覆盖。

默认输出：

```text
data/CodecTrainCfg/<dataset>/qp_<qp>/
```

`gencfg.sh` 会先拼接 `sequence.cfg` 与 `encoder_intra_vtm.cfg`，然后替换 `InputFile`、`FramesToBeEncoded`、`FrameRate`、`SourceWidth`、`SourceHeight`、`BitstreamFile`、`QP`，并设置 `TemporalSubsampleRatio`：CUSTOM、VVC、VVC_CTC 为 `1`，其他数据集为 `20`。

### 2. 运行编码并导出划分

示例：运行 DIV2K 训练集 QP 22：

```bash
bash ref_model/script/run.sh \
  --dataset DIV2K \
  --qp 22 \
  --type train
```

示例：运行 DIV2K 验证集 QP 22：

```bash
bash ref_model/script/run.sh \
  --dataset DIV2K \
  --qp 22 \
  --type valid
```

示例：运行 HEVC 测试集 QP 32：

```bash
bash ref_model/script/run.sh \
  --dataset HEVC \
  --qp 32 \
  --type test
```

`run.sh` 默认使用：

```text
ref_model/bin/dumpPartition/EncoderAppStatic
ref_model/bin/dumpPartition/DecoderAppStatic
```

默认读取 cfg：

```text
data/CodecTrainCfg/<dataset>/qp_<qp>/
```

默认输出：

```text
data/partition/<dataset>/<split>/
data/logs/<dataset>/qp_<qp>/
data/codec_run/<dataset>/qp_<qp>/
```

其中划分导出路径通过以下环境变量传给 `DecoderAppStatic`：

```bash
FASTPARTITION_DEPTH_DIR=data/partition/<dataset>/<split>
FASTPARTITION_DEPTH_PREFIX=<sequence_name>
```

编码和解码 MD5 校验通过后，`run.sh` 会删除临时 bitstream、重建 YUV 和解码 YUV，仅保留划分信息与日志。

生成配置前须准备清单对应的 YUV 文件。VVC/VVC_CTC 的输入位深从
`VVCSoftware_VTM/cfg/per-sequence/<sequence_name>.cfg` 读取；其他数据集使用序列模板中的位深。
可用 `--input-bitdepth 8|10|12|16` 显式覆盖。默认序列模板不固定 Level 限制。

## 导出 RD cost

先使用 `gencfg.sh` 生成配置，再通过 `--encoder`、`--decoder` 选择 `dumpRDCost`：

```bash
bash ref_model/script/run.sh \
  --dataset DIV2K --qp 22 --type train \
  --encoder ref_model/bin/dumpRDCost/EncoderAppStatic \
  --decoder ref_model/bin/dumpRDCost/DecoderAppStatic
```

编码器通过 `VTM_RDO_COST_FILE` 指定的文件导出 RD cost，脚本自动设置为：

```text
data/rdo_cost/<dataset>/<split>/<sequence_name>_QP<qp>.tsv
```

解码器同时导出划分结构，路径与仅导出划分时相同。

## 数据生成脚本接口

| 脚本 | 参数 | 用途 |
| --- | --- | --- |
| `gencfg.sh` / `run.sh` | `--dataset`、`--qp`、`--type train\|valid\|test` | 选择数据集、QP 和数据划分。 |
| `gencfg.sh` / `run.sh` | `--sequence-list` | 显式指定序列清单。 |
| `gencfg.sh` | `--template`、`--sequence-template` | 编码配置和序列头模板。 |
| `gencfg.sh` | `--video-root`、`--output-root` | 视频根目录和生成配置目录。 |
| `gencfg.sh` | `--input-bitdepth` | 显式设置输入位深，优先于序列元数据与模板。 |
| `run.sh` | `--cfg-root` | 读取的逐序列配置目录。 |
| `run.sh` | `--partition-root`、`--rdo-cost-root` | 划分与 RD cost 输出目录。 |
| `run.sh` | `--log-root`、`--work-root` | 日志和编码中间文件目录。 |
| `run.sh` | `--encoder`、`--decoder` | 指定编解码器。 |
| `run.sh` | `--skip-decode` | 仅编码，跳过解码、划分导出和重建校验。 |

生成序列清单（先激活 FastPartitionVTM 环境，提供 Python 与 Pillow）：

```bash
bash ref_model/script/gentxt.sh \
  --input-dir /path/to/images \
  --output-file ref_model/script/Training_Sequences_DIV2K.txt \
  --frames 1 --fps 1
```

`gentxt.sh` 接受 `.png`/`.PNG` 文件，默认每个图像对应一帧，生成六列清单：
`sequence_name,file_name,width,height,frames,fps`。YUV 文件命名为 `<name>_<width>x<height>.yuv`，
须提前另行转换并放入视频目录；设置 `--frames` 不会自动复制或生成视频帧。
空目录、损坏图像、重复名称或不适合 YUV420 的奇数尺寸会报错，原清单不被覆盖。
也可通过 `FASTPARTITION_PYTHON=/path/to/FastPartitionVTM/bin/python` 指定解释器。
各脚本用 `--help` 查看完整接口。

## Roundtrip 校验

`roundtrip.sh` 用于验证 `bin/vtm240/` 中标准 VTM 编码器和解码器的一致性。它会编码输入 YUV，解码生成的 bitstream，并比较编码器重建 YUV 与解码器输出 YUV 是否完全一致。

编解码器固定为 `bin/vtm240/`；主配置由必填参数 `--cfg` 指定。

示例：

```bash
bash ref_model/script/roundtrip.sh \
  --input ~/qiujp/HEVC_CTC_YUV/416x240/BasketballPass_416x240_50.yuv \
  --cfg ref_model/cfg/encoder_intra_vtm.cfg \
  --width 416 \
  --height 240 \
  --frames 10 \
  --framerate 50 \
  --qp 32
```

可选参数：

- `--seq-cfg`：额外传入序列级 cfg。
- `--bitdepth`：显式指定输入 bit depth。
- `--chroma-format`：显式指定输入 chroma format。
- `--workdir`：指定输出工作目录，默认写入 `ref_model/script/output/<timestamp>/`。

该脚本只用于标准基线一致性验证，不用于导出训练划分标签。
