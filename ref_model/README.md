# ref_model

该目录用于存放标准、未改动的 VTM 24.0 参考执行模型的最小化基线副本。

## 当前目录约定

- `build/`：存放最小可运行的标准 VTM 24.0 基线可执行文件集合，默认仅保留运行所需的核心二进制。包含内容如下：
    - `bin/EncoderAppStatic`：标准 VTM 24.0 编码器
    - `bin/DecoderAppStatic`：标准 VTM 24.0 解码器
    - `bin/DecoderAnalyserAppStatic`：标准 VTM 24.0 解码分析器
- `cfg/`：存放编码与测试所需的配置文件。
- `script/`：存放运行脚本；当前包含 `roundtrip.sh`，用于完成编码、解码和重建一致性校验。
