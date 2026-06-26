# ref_model

`ref_model/` 用于保存 VTM 24.0 参考执行模型、标准配置和基于参考编码器的数据生成脚本。该目录的重点是标准行为复现、基线验证和训练划分标签生成，不承载神经网络模型训练代码。

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
  cfg/
    sequence.cfg
    encoder_intra_vtm.cfg
    encoder_lowdelay_P_vtm.cfg
    encoder_lowdelay_vtm.cfg
    encoder_randomaccess_vtm.cfg
  script/
    Training_Sequences_DIV2K.txt
    Validating_Sequences_DIV2K.txt
    Testing_Sequences_HEVC.txt
    Testing_Sequences_VVC.txt
    gentxt.sh
    gencfg.sh
    run.sh
    roundtrip.sh
```

- `bin/vtm240/`：标准、未改动的 VTM 24.0 可执行文件，用于标准基线运行和一致性校验。
- `bin/dumpPartition/`：用于生成训练划分标签的一组 VTM 可执行文件。其中 `EncoderAppStatic` 负责编码生成 bitstream，`DecoderAppStatic` 解码 bitstream，并读取划分导出环境变量生成 `Luma_Partition_Info.txt` 和 `Chroma_Partition_Info.txt`。
- `cfg/sequence.cfg`：逐序列配置头模板，包含 `InputFile`、分辨率、帧数、帧率等字段。
- `cfg/encoder_intra_vtm.cfg`：VTM intra 编码主模板。生成逐序列 cfg 时，`gencfg.sh` 会先拼接 `sequence.cfg` 和该模板，再替换序列参数。
- `script/*_Sequences_*.txt`：序列清单，格式为 `sequence_name,file_name,width,height,frames,fps`。
- `script/gencfg.sh`：生成逐序列编码 cfg。
- `script/run.sh`：调用 `bin/dumpPartition/` 中的编解码器生成划分数据，用作网络训练标签。
- `script/roundtrip.sh`：调用 `bin/vtm240/` 中的标准编解码器做编码、解码和重建一致性校验。

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

- 序列清单：`ref_model/script/<Split>_Sequences_<dataset>.txt`
- 序列头模板：`ref_model/cfg/sequence.cfg`
- 编码主模板：`ref_model/cfg/encoder_intra_vtm.cfg`
- 视频目录：`data/video/<dataset>/`

默认输出：

```text
data/CodecTrainCfg/<dataset>/qp_<qp>/
```

`gencfg.sh` 会先拼接 `sequence.cfg` 与 `encoder_intra_vtm.cfg`，然后替换 `InputFile`、`FramesToBeEncoded`、`FrameRate`、`SourceWidth`、`SourceHeight`、`BitstreamFile`、`QP`，并将生成 cfg 中的 `TemporalSubsampleRatio` 设为 `20`。

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

## Roundtrip 校验

`roundtrip.sh` 用于验证 `bin/vtm240/` 中标准 VTM 编码器和解码器的一致性。它会编码输入 YUV，解码生成的 bitstream，并比较编码器重建 YUV 与解码器输出 YUV 是否完全一致。

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
