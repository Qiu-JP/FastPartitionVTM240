# ref_model/build

该目录仅保留可直接运行的最小标准 VTM 24.0 可执行文件集合，不保留源码、不保留完整 CMake 构建中间文件，也不保留非必要工具。

## 当前保留内容

- `bin/EncoderAppStatic`：标准 VTM 24.0 编码器
- `bin/DecoderAppStatic`：标准 VTM 24.0 解码器
- `bin/DecoderAnalyserAppStatic`：标准 VTM 24.0 解码分析器

## 说明

- 当前保留的是静态版本可执行文件，便于直接运行并减少对额外构建目录内容的依赖。
- 若后续基线测试还需要其他官方工具，可按需再补充，但默认保持最小化。
