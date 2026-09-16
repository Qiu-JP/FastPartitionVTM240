# data

本目录存放原始视频、划分标签、候选 RD cost、网络数据集及数据生成产物。数据文件不提交仓库；使用前自行准备。

## 目录与来源

| 目录 | 内容 | 生成或使用方式 |
| --- | --- | --- |
| `video/<dataset>/` | 原始 YUV，例如 CUSTOM、VVC_CTC。 | 编码和网络输入准备；文件名须与序列清单一致。 |
| `CodecTrainCfg/<dataset>/qp_<qp>/` | 每个序列的编码 cfg。 | `ref_model/script/gencfg.sh` 生成。 |
| `partition/<dataset>/<split>/` | `Luma_Partition_Info.txt`、`Chroma_Partition_Info.txt`。 | `dumpPartition` 或 `dumpRDCost` 解码器导出最终划分。 |
| `rdo_cost/<dataset>/<split>/` | `<sequence>_QP<qp>.tsv`。 | `dumpRDCost` 编码器记录搜索中的候选 RD cost。 |
| `dataset/<dataset>/<split>/` | 网络使用的 `.npy/.pkl`。 | `network/src/createDataset.py` 生成。 |
| `codec_run/` | 码流、重建与解码 YUV 等临时文件。 | 数据生成脚本校验成功后清理；roundtrip 可保留结果。 |
| `logs/` | 编解码日志与校验记录。 | 数据生成脚本写入。 |

划分文本和 RD cost 是不同的数据：最终解码划分不能恢复编码搜索中所有候选的 RD cost。

## 当前 Luma32 数据格式

主线训练使用 `data/dataset/CUSTOM_32/training/` 和 `validating/`。
原图目标为 32×32，输入含上方、左方邻域，共 48×48；目标位于输入右下角。

| 数组 | 内容和形状 |
| --- | --- |
| `Luma_Input.npy` | `[N,1,48,48]` 像素输入。 |
| `Luma_Gridmap.npy` | `[N,2,8,8]` 网格标签。 |
| `Luma_CU_Tree.npy` | `[M,5]`，列为 `grid_y, grid_x, grid_h, grid_w, label`。 |
| `Luma_CU_RDCost.npy` | `[M]` 结构化数组，包含 `rd_delta[6]`、`completed[6]`、`valid`。 |

各数组配有同名 `.pkl` 元数据，记录数组格式、样本映射或节点 offsets。六类顺序固定为
`NO_SPLIT, QT, BTH, BTV, TTH, TTV`；未完成候选通过 `completed` 区分，不补造 RD 值。

Luma32 样本 ID 为 `sequence_name, qp, frame_id, ctu_id, sub_block_id`，另记录 `block_x/block_y`。
`CU_Tree.pkl` 的 `offsets` 将每个样本映射到节点区间；RD 数组严格使用相同节点顺序。
原图尺寸按宽×高，grid/tensor 按高×宽；原图 32×16 对应 grid 4×8。

当前数据生成器从原生划分中发生 QT 的 64×64 区域提取有标签的 32×32 子块，
因此数据集不保证为每个 CTU 保存全部 16 个位置。这里的 64×64 是标签来源的父块，不是模型目标大小。

数据工具也支持生成色度输入 `[N,2,32,32]`、目标原图 16×16、gridmap `[N,2,4,4]`，用于辅助数据与预览；当前训练和 VTM 快速划分主线使用亮度模型。
逻辑分类器数据生成工具仍保留，但不是当前联合训练的必需步骤。

## 生成流程

以下命令在项目根目录、相应 Python 环境中执行；已有数据可直接复用。

1. 准备 YUV 和 `ref_model/script/*_Sequences_CUSTOM.txt`。
2. 用 `gencfg.sh` 生成配置，再选择 `dumpRDCost` 程序运行 `ref_model/script/run.sh`，取得划分与 RD dump；具体命令见 [参考程序说明](../ref_model/README.md)。
3. 生成 Luma32 数据，例如训练集：

```bash
python network/src/createDataset.py \
  --data-type 1 --source-dataset CUSTOM --dataset CUSTOM_32 \
  --sequence-list ref_model/script/Training_Sequences_CUSTOM.txt \
  --component luma --action gridmap-input-cu-tree

python network/src/createDataset.py \
  --data-type 1 --dataset CUSTOM_32 --component luma --action rdocost \
  --rdo-root data/rdo_cost/CUSTOM/training \
  --rdo-sequence-list ref_model/script/Training_Sequences_CUSTOM.txt
```

验证集使用 `--data-type 3`、验证序列清单及 `data/rdo_cost/CUSTOM/validating`。
生成完成后，训练读取 Input、Gridmap、CU Tree 和 RD cost；百分位阈值搜索读取 Input、CU Tree，RD-cost 搜索额外读取 RD cost。概率缓存是搜索中间产物，无需额外 recipe。
