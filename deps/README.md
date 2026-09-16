# deps

`onnxruntime-linux-x64-1.30.0/` 包含 ONNX Runtime 1.30.0 的 Linux x86-64 CPU C/C++ 依赖（约 30 MB）。

- `include/`：VTM 编译使用的头文件。
- `lib/`：链接及运行时动态库，保留库文件的符号链接。
- `LICENSE`、`ThirdPartyNotices.txt`：许可证与第三方声明。
- `VERSION_NUMBER`、`GIT_COMMIT_ID`：版本信息。

VTM 的 CMake 默认从此目录查找，无需单独安装 C/C++ 包。保持仓库目录结构时，`VVCSoftware_VTM/build/bin/` 和 `script/bin/` 的程序通过相对路径加载本目录动态库。
其他系统或架构需下载对应包，并用 `-DFAST_PARTITION_ONNXRUNTIME_ROOT=/path/to/package` 指定。
Python 阈值搜索使用的 `onnxruntime` 包仍需在 Python 环境安装。

上游：https://github.com/microsoft/onnxruntime
