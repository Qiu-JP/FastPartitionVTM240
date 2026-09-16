# VVCSoftware_VTM

VTM 24.0 完整源码及基于 ONNX 的 intra 快速划分。以下命令均从项目根目录（包含 `network/`、`ref_model/` 的目录）执行。

## 1. 编译与运行环境

需要 CMake 和支持 C++17 的编译器。源码位于 `source/`，配置位于 `cfg/`；构建文件与程序输出到 `build/`，该目录不提交仓库。所有配置共用这一构建目录，直接构建整个 VTM 工程。

### 原生 VTM 编译

在 `VVCSoftware_VTM/source/Lib/CommonLib/TypeDef.h` 中设置：

```cpp
#define FastPartition 0
#define Save_Depth    0
#define DumpRdoCost   0
#define RDOStats      1
```

`RDOStats=1` 保留只读 compute 计数，供评估使用；不需要统计时可设为 0。关闭 `FastPartition` 后不需要 ONNX Runtime，也不提供快速划分模型参数。

```bash
cmake -S VVCSoftware_VTM -B VVCSoftware_VTM/build \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SEARCH_OPENSSL=OFF
cmake --build VVCSoftware_VTM/build -j 6
```

程序输出到 `build/bin/`。评估默认使用仓库 `ref_model/bin/vtm240/` 中带 compute 计数的 anchor，无需每次重新编译。

### 启用 ONNX 快速划分的编译

将 `FastPartition` 设为 1，评估需保留 `RDOStats=1`。`Save_Depth` 和 `DumpRdoCost` 控制划分与 RD cost 导出，普通编码评估不需要开启。

仓库 `deps/onnxruntime-linux-x64-1.30.0/` 已附带 Linux x86-64 CPU 的头文件、动态库和许可证，CMake 默认从该目录查找，无需另外下载。

```bash
cmake -S VVCSoftware_VTM -B VVCSoftware_VTM/build \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SEARCH_OPENSSL=OFF
cmake --build VVCSoftware_VTM/build -j 6
```

其他平台或自定义安装可从 [ONNX Runtime 发布页](https://github.com/microsoft/onnxruntime/releases) 获取对应 C/C++ 包，配置时增加
`-DFAST_PARTITION_ONNXRUNTIME_ROOT=/absolute/path/to/package`。
已有构建如果缓存了旧的外部依赖路径，重新配置时加 `-U 'ONNXRUNTIME_*' -U FAST_PARTITION_ONNXRUNTIME_ROOT`，即可恢复仓库默认路径。

输出为 `build/bin/EncoderApp` 和 `build/bin/DecoderApp`。切换配置时修改同一份 `TypeDef.h`，然后在同一个 `build/` 目录重新运行 CMake 和完整构建命令；新程序覆盖该目录中上一次的构建结果。不需要 LibTorch 或 CUDA；Python 的 `onnxruntime` 包不能代替 C++ 包的 `include/`、`lib/`。

### 使用仓库提供的程序

`script/bin/EncoderApp`、`script/bin/DecoderApp` 是已编译的 Linux x86-64 快速划分程序，供直接评估使用。
保持仓库目录结构时，程序通过相对路径找到 `deps/` 中的 ONNX Runtime，无需设置 `LD_LIBRARY_PATH`。
仍需兼容的 Linux 系统运行库；其他平台请自行编译。若只复制可执行文件到其他位置，需同时提供动态库并设置：

```bash
export LD_LIBRARY_PATH=/absolute/path/to/onnxruntime/lib:${LD_LIBRARY_PATH:-}
```

自行编译后，可通过评估参数指定 `build/bin/` 的程序，或更新内置程序：

```bash
cp VVCSoftware_VTM/build/bin/EncoderApp VVCSoftware_VTM/script/bin/
cp VVCSoftware_VTM/build/bin/DecoderApp VVCSoftware_VTM/script/bin/
```

Python 命令在 FastPartitionVTM 环境内执行，需要 NumPy、Pandas、PyTorch 和 CPU ONNX Runtime：

```bash
python -m pip install numpy pandas torch onnxruntime==1.30.0
```

## 2. 快速划分输入、输出与接口

模型由 `network/src/saveModel.py` 从训练权重导出，部署目录包含 `swin.onnx` 和各形状的 `classifier_*.onnx`，运行不需要模型 manifest。FP32 或 INT8 均可使用，阈值必须针对实际部署版本搜索。

- 原图目标块为 **32×32 像素**，附带上方、左方邻域的输入为 **48×48**；目标块在输入右下角，偏移为 16 像素。
- VTM 自动准备像素、边界填充和 QP/51；每个 128×128 CTU 的 16 个输入块按光栅顺序一起推理。用户提供 YUV 和序列配置即可。
- Swin 输出每块 `[2,8,8]` grid map；4×4 原图像素对应一个网格单元。分类器读取当前 CU 对应的 ROI，输出六类概率。
- 类别顺序固定为 `NO_SPLIT, QT, BTH, BTV, TTH, TTV`。配置尺寸为原图 **宽×高**；例如 32×16 对应 tensor 空间高×宽 4×8。

编译开启宏只表示具备此功能；运行时同时提供模型和阈值才启用：

```bash
VVCSoftware_VTM/script/bin/EncoderApp \
  -c ref_model/cfg/encoder_intra_vtm.cfg \
  -c VVCSoftware_VTM/cfg/per-sequence/BQMall.cfg \
  -i /path/to/BQMall_832x480_60.yuv -b /tmp/test.vvc -o /tmp/recon.yuv \
  --FramesToBeEncoded=1 --QP=32 \
  --FastPartitionSwinModel=network/checkpoints/onnx/final/swin.onnx \
  --FastPartitionClassifierModel=network/checkpoints/onnx/final \
  --FastPartitionLumaModelScale=32 \
  -c VVCSoftware_VTM/script/thresholds_fast.cfg
```

编码输出为 VVC 码流及重建 YUV。阈值 cfg 只包含 `FastPartitionThBySize`，格式为 `32x32:[六个阈值];32x16:[六个阈值];...`。四档配置为 `thresholds_{quality,balanced,fast,faster}.cfg`。

概率大于等于对应阈值则保留该合法候选，最终由 VTM 做 RD 选择。未配置尺寸、全部合法候选被拒绝或未实际剪枝时使用原生搜索。模型与阈值均不提供时也使用原生搜索，只提供部分参数会报错。主线只使用逐尺寸逐类别阈值。

## 3. 编解码评估

直接运行 `evaluate.py`，它自动调用 anchor 和快速划分编解码器，校验重建，并计算指标：

```bash
python VVCSoftware_VTM/script/evaluate.py \
  --bundle network/checkpoints/onnx/final \
  --thresholds VVCSoftware_VTM/script/thresholds_fast.cfg \
  --workdir VVCSoftware_VTM/script/output/fast_run
```

默认读取 `ref_model/script/Testing_Sequences_VVC.txt` 中全部序列，视频目录为 `data/video/VVC_CTC/`，逐序列配置为 `cfg/per-sequence/<name>.cfg`；每序列编码首帧，QP 为 22、27、32、37。程序按序列、QP 串行执行 anchor 编解码和快速划分编解码，避免并行编码干扰耗时。

配置读取顺序为：基础编码 cfg → 当前序列 cfg → 命令行指定的输入、分辨率、帧数、帧率和 QP。
快速划分编码器额外读取 `--thresholds` 指定的阈值 cfg，并根据 `--bundle` 设置 Swin 与 classifier 路径；
anchor 不接收阈值或模型参数。`--thresholds` 必填，不会自动选择 preset。

仅评估一个序列，并显式指定配置、序列清单和新编译的程序：

```bash
python VVCSoftware_VTM/script/evaluate.py \
  --bundle network/checkpoints/onnx/final \
  --thresholds VVCSoftware_VTM/script/thresholds_fast.cfg \
  --main-cfg ref_model/cfg/encoder_intra_vtm.cfg \
  --sequence-cfg-root VVCSoftware_VTM/cfg/per-sequence \
  --seq-list ref_model/script/Testing_Sequences_VVC.txt \
  --yuv-root data/video/VVC_CTC \
  --only-seq BQMall --qps 22 27 32 37 --max-frames 1 \
  --test-bin VVCSoftware_VTM/build/bin/EncoderApp \
  --test-decoder VVCSoftware_VTM/build/bin/DecoderApp \
  --workdir VVCSoftware_VTM/script/output/bqmall_fast
```

依次评估全部序列的四档阈值（串行执行，不要在后台并行启动）：

```bash
for preset in quality balanced fast faster; do
  python VVCSoftware_VTM/script/evaluate.py \
    --bundle network/checkpoints/onnx/final \
    --thresholds "VVCSoftware_VTM/script/thresholds_${preset}.cfg" \
    --workdir "VVCSoftware_VTM/script/output/${preset}_run" || break
done
```

每档分别输出自己的 `summary.csv`。测试搜索得到的新阈值时，将 `--thresholds` 换成搜索输出的 cfg 路径即可。

| 参数 | 用途 |
| --- | --- |
| `--bundle`、`--thresholds` | 模型默认 `network/checkpoints/onnx/final`；逐尺寸阈值 cfg 必填。 |
| `--test-bin`、`--test-decoder` | 默认 `script/bin/EncoderApp`、`script/bin/DecoderApp`；可指定新编译的 `build/bin/` 程序。 |
| `--anchor-bin`、`--anchor-decoder` | 默认 `ref_model/bin/vtm240/EncoderAppStatic`、`DecoderAppStatic`。 |
| `--seq-list`、`--yuv-root` | 序列清单和 YUV 目录。清单每行为 `名称,文件名,宽,高,帧数,帧率`。 |
| `--sequence-cfg-root`、`--main-cfg` | 逐序列配置目录、基础编码 cfg；后者默认 `ref_model/cfg/encoder_intra_vtm.cfg`。 |
| `--only-seq BQMall` | 仅测试一个序列。 |
| `--qps 22 27 32 37`、`--max-frames 1` | QP 列表、每序列最多编码帧数。 |
| `--workdir` | 新的输出目录，默认 `script/output/<时间戳>`；不能覆盖已有目录。 |
| `--keep-logs` | 保留编码和解码日志；默认整次评估成功后删除，失败时保留。 |
| `--keep-yuv` | 保留重建与解码 YUV；默认校验一致后删除。 |

输出保存在指定目录：

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 每序列及 Average 的 BD-rate Y/U/V、time saving、亮度/总 compute saving，单位均为百分比。 |
| `curves.json` | 每序列、QP 的 anchor/test 码率、PSNR、elapsed time、compute 计数和码流/重建校验值。 |
| `manifest.json` | 完成状态、实际命令、参数、模型与程序等文件校验值。 |
| `<序列>/QP<值>/<anchor或test>/stream.vvc` | 编码码流；解码器开启 dump 时还会生成对应的划分记录。 |

compute 累计实际测试模式的面积，以 4×4 为单位，包含各层搜索；不改变现有统计口径。time/compute saving 为逐 QP 的 `100 × (1 − test/anchor)` 均值，再对序列等权平均。BD-rate 使用 PCHIP；不足四个 QP、PSNR 重复或无共同区间时留空。anchor 必须有 `RDOStats=1`，缺少计数会报错。

四档各运行一次，分别指定阈值与输出目录。需要重新汇总已有数据时：

```bash
python VVCSoftware_VTM/script/getBdRate.py \
  VVCSoftware_VTM/script/output/fast_run/curves.json \
  --output VVCSoftware_VTM/script/output/fast_run/summary.csv
```

评估入口统一为 `evaluate.py`，不会自动编译。

## 4. 阈值搜索全流程

整个流程的外部输入只有数据集的 `.npy/.pkl` 和 ONNX 模型，不需要 recipe 或原始 YUV。
推荐先推理生成概率缓存，再复用缓存做百分位或 RD-cost 搜索。数据由 `network/src/createDataset.py` 生成。

### 输入文件

通过 `--dataset-dir` 指定包含以下文件的目录：

| 文件 | 用途 |
| --- | --- |
| `Luma_Input.npy`、`Luma_Input.pkl` | `[N,1,48,48]` 像素输入，以及样本身份、QP 等信息。 |
| `Luma_CU_Tree.npy`、`Luma_CU_Tree.pkl` | 各样本的 CU 节点、ROI 坐标、类别标签及节点 offsets。 |
| `Luma_CU_RDCost.npy`、`Luma_CU_RDCost.pkl` | RD-cost 搜索额外需要的逐节点 RD 数据；百分位搜索不需要。 |

`--bundle` 指定包含 `swin.onnx` 和各形状 classifier 的模型目录，支持 FP32/INT8 ONNX。
脚本检查样本顺序和节点索引，按输入样本执行 Swin，裁剪各节点 grid ROI 后执行 classifier，随后搜索阈值。
默认 `--batch-size 16`，按 NPY 中的样本顺序组批，最后一批使用实际剩余数量，不补造样本或标签。

这是对给定数据集的独立离线搜索，不要求重建完整 CTU。动态 INT8 的预测可能受同批样本影响；
这里的数据集组批不保证与 VTM 的完整 CTU 组批相同，生成的阈值需要通过实际编码评估，不能声称概率逐项与部署一致。
搜索不修改 VTM 的 batch=16 或特征缓存流程。

### 生成一次概率缓存

```bash
python VVCSoftware_VTM/script/collect_threshold_probabilities.py \
  --dataset-dir data/dataset/CUSTOM_32/validating \
  --bundle network/checkpoints/onnx/final \
  --out-dir VVCSoftware_VTM/script/output/probabilities
```

输出 `probabilities.npy` 和 `manifest.json`，记录节点概率、模型/数据校验值以及 batch 参数。
如果后续要做 RD-cost 搜索，采集时数据目录内应包含 RD 的 `.npy/.pkl`，以便缓存绑定它们的身份。
相同模型、数据、组批参数下可重复使用缓存；修改这些内容后重新采集。搜索时仍需原始数据文件和模型路径有效，以核验身份、读取标签及 RD cost。

### 百分位搜索

```bash
python VVCSoftware_VTM/script/ThSearch_Ratio.py \
  --probability-cache VVCSoftware_VTM/script/output/probabilities \
  --ratio-denominator all-nodes --ratio 0.02 \
  --out-dir VVCSoftware_VTM/script/output/ratio_002
```

只输出 `thresholds_ratio.cfg`，保存逐尺寸逐类别阈值。
改变 `--ratio` 并使用不同输出目录可得到多组候选阈值；该参数不是实际 compute/time saving。

### RD-cost 搜索

```bash
python VVCSoftware_VTM/script/ThSearch_RdoCost.py \
  --probability-cache VVCSoftware_VTM/script/output/probabilities \
  --lambdas 0.001,0.01,0.1 --regret-weighting uniform \
  --out-dir VVCSoftware_VTM/script/output/rd_search
```

一次推理后对多个 lambda 分别搜索，每个 lambda 只输出一个 `thresholds_cache_lambda_*.cfg`。
RD-cost 搜索结合相对 RD 损失与计算量代理，不补造缺失 RD cost。代理目标不等于最终 BD-rate 或实测耗时。
校准可用训练集或验证集；报告泛化性能时不要将测试序列混入校准数据。

### 编码验证

将输出 cfg 传给第 3 部分的 `evaluate.py --thresholds`，使用搜索时的同一 ONNX bundle。
每组阈值使用独立评估目录，根据实测 BD-rate、compute saving 和 time saving 选择档位。
更换模型或量化版本后重新搜索，不覆盖此前的正式结果。

也可不保存概率缓存：将上述搜索命令的 `--probability-cache` 替换为
`--dataset-dir data/dataset/CUSTOM_32/validating --bundle network/checkpoints/onnx/final`，
搜索脚本会先推理再搜索。两种输入方式不能同时指定；多次调参推荐使用缓存。

各脚本通过 `--help` 查看参数。`script/encoder_intra_pruneFastSplit*.cfg` 仅保留为后续消融资料，不作为主线运行配置。
