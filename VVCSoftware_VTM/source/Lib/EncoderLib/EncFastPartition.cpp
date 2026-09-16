/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#include "EncFastPartition.h"

#if FastPartition

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{
struct FastPartitionInferStats
{
  uint64_t swinCalls = 0;
  uint64_t classifierCalls = 0;
  double swinSeconds = 0.0;
  double classifierSeconds = 0.0;

  ~FastPartitionInferStats()
  {
    if (swinCalls == 0 && classifierCalls == 0)
    {
      return;
    }
    std::fprintf(stderr,
                 "[FastPartitionInferStats] swinCalls=%llu swinSeconds=%.6f classifierCalls=%llu classifierSeconds=%.6f totalInferSeconds=%.6f\n",
                 (unsigned long long)swinCalls,
                 swinSeconds,
                 (unsigned long long)classifierCalls,
                 classifierSeconds,
                 swinSeconds + classifierSeconds);
  }
};

FastPartitionInferStats g_fastPartitionInferStats;

double elapsedSeconds(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double>(end - start).count();
}


Ort::Env& onnxEnv()
{
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastPartition");
  return env;
}

void dumpTensorOnce(const std::string& name, const float* data, size_t count)
{
  const char* prefix = std::getenv("FASTPARTITION_ONNX_DUMP");
  if (prefix == nullptr || prefix[0] == '\0') return;
  static std::set<std::string> written;
  if (!written.insert(name).second) return;
  const std::string path = std::string(prefix) + "." + name + ".bin";
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) THROW("Cannot open FastPartition ONNX audit dump");
  const size_t saved = std::fwrite(data, sizeof(float), count, file);
  std::fclose(file);
  if (saved != count) THROW("Incomplete FastPartition ONNX audit dump");
}

Ort::SessionOptions sessionOptions()
{
  Ort::SessionOptions options;
  int threads = 1;
  if (const char* value = std::getenv("FASTPARTITION_ONNX_THREADS"))
  {
    threads = std::max(1, std::atoi(value));
  }
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

void checkTensor(const Ort::Value& value, const std::vector<int64_t>& expected)
{
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != expected)
  {
    THROW("FastPartition ONNX output dtype or NCHW shape mismatch");
  }
}

struct OnnxSwin
{
  std::unique_ptr<Ort::Session> session;
  std::vector<float> input, qpInput;
  std::vector<int64_t> inputShape, qpShape, outputShape;

  void init(const std::string& path, int batch, int channels, int size, int gridSize)
  {
    if (session) return;
    if (path.empty()) THROW("FastPartition requires an ONNX Swin model path");
    auto options = sessionOptions();
    session.reset(new Ort::Session(onnxEnv(), path.c_str(), options));
    if (session->GetInputCount() != 2 || session->GetOutputCount() != 1)
      THROW("FastPartition Swin ONNX expects input, qp -> gridmap");
    inputShape = {batch, channels, size, size};
    qpShape = {batch, 1};
    outputShape = {batch, 2, gridSize, gridSize};
    input.resize(size_t(batch * channels * size * size));
    qpInput.resize(size_t(batch));
    for (int i = 0; i < 3; ++i) (void)run();
    std::fprintf(stderr, "[FastPartitionONNX] runtime=%s model=%s batch=%d input=%dx%dx%d grid=%d\n",
                 OrtGetApiBase()->GetVersionString(), path.c_str(), batch, channels, size, size, gridSize);
  }

  std::vector<Ort::Value> run()
  {
    if (!session) THROW("FastPartition ONNX Swin not initialized");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 2> tensors = {
      Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), inputShape.data(), inputShape.size()),
      Ort::Value::CreateTensor<float>(memory, qpInput.data(), qpInput.size(), qpShape.data(), qpShape.size())
    };
    const char* names[] = {"input", "qp"};
    const char* outputs[] = {"gridmap"};
    auto result = session->Run(Ort::RunOptions{nullptr}, names, tensors.data(), 2, outputs, 1);
    checkTensor(result[0], outputShape);
    if (qpInput[0] > 0.0f)
    {
      const std::string tag = "swin" + std::to_string(inputShape[1]) + "x" + std::to_string(inputShape[2]);
      dumpTensorOnce(tag + ".input", input.data(), input.size());
      dumpTensorOnce(tag + ".qp", qpInput.data(), qpInput.size());
      dumpTensorOnce(tag + ".gridmap", result[0].GetTensorData<float>(),
                     size_t(outputShape[0]*outputShape[1]*outputShape[2]*outputShape[3]));
    }
    return result;
  }
};

template<class Inputs, class Maps, class Pixels>
void inferSwin(OnnxSwin& model, const Inputs& inputs, int qp, Maps& maps, Pixels pixels)
{
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    maps[i].valid = false;
    const auto& data = pixels(inputs[i]);
    std::copy(data.begin(), data.end(), model.input.begin() + i * data.size());
    model.qpInput[i] = float(qp) / 51.0f;
  }
  const auto start = std::chrono::steady_clock::now();
  auto result = model.run();
  g_fastPartitionInferStats.swinSeconds += elapsedSeconds(start, std::chrono::steady_clock::now());
  ++g_fastPartitionInferStats.swinCalls;
  const float* data = result[0].GetTensorData<float>();
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    maps[i].targetX = inputs[i].targetX;
    maps[i].targetY = inputs[i].targetY;
    std::copy_n(data + i * maps[i].values.size(), maps[i].values.size(), maps[i].values.begin());
    maps[i].valid = true;
  }
}
}

FILE* fastPartitionStatFile()
{
  static FILE* statFile = [] {
    const char* path = std::getenv("FASTPARTITION_STAT_LOG");
    return path != nullptr && path[0] != '\0' ? std::fopen(path, "a") : nullptr;
  }();
  return statFile;
}


struct EncFastPartitionLuma32SwinInfer::Impl : OnnxSwin {};

struct EncFastPartitionClassifierInfer::Impl
{
  std::map<std::pair<int,int>, std::unique_ptr<Ort::Session>> branches;

  void infer(int h, int w, std::vector<float>& input, std::array<float,6>& probabilities)
  {
    const auto branch = branches.find({h,w});
    if (branch == branches.end()) THROW("Missing FastPartition ONNX classifier shape");
    const std::array<int64_t,4> shape = {1,2,h,w};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* names[] = {"roi"};
    const char* outputs[] = {"probabilities"};
    auto result = branch->second->Run(Ort::RunOptions{nullptr}, names, &tensor, 1, outputs, 1);
    checkTensor(result[0], {1,6});
    const float* data = result[0].GetTensorData<float>();
    std::copy_n(data,6,probabilities.begin());
    if (std::any_of(input.begin(), input.end(), [](float x) { return x != 0.0f; }))
    {
      const std::string tag = "classifier" + std::to_string(h) + "x" + std::to_string(w);
      dumpTensorOnce(tag + ".roi", input.data(), input.size());
      dumpTensorOnce(tag + ".probabilities", data, 6);
    }
    float sum = 0.0f;
    for (float value : probabilities)
    {
      if (!std::isfinite(value) || value < 0.0f || value > 1.0001f)
        THROW("Invalid FastPartition ONNX classifier probabilities");
      sum += value;
    }
    if (std::abs(sum-1.0f) > 0.001f) THROW("FastPartition ONNX classifier probabilities must sum to one");
  }
};


EncFastPartitionLuma32SwinInfer::EncFastPartitionLuma32SwinInfer() : m_impl(new Impl) {}
EncFastPartitionLuma32SwinInfer::~EncFastPartitionLuma32SwinInfer() = default;
bool EncFastPartitionLuma32SwinInfer::isInitialized() const { return bool(m_impl->session); }
void EncFastPartitionLuma32SwinInfer::init(const std::string& path) { m_impl->init(path, 16, 1, 48, 8); }
void EncFastPartitionLuma32SwinInfer::inferCtu(const std::array<FastPartitionLuma32SwinInput, 16>& inputs,
                                     int qp, std::array<FastPartitionLuma32Gridmap, 16>& maps)
{
  inferSwin(*m_impl, inputs, qp, maps, [](const FastPartitionLuma32SwinInput& x) -> const auto& { return x.luma; });
}


EncFastPartitionClassifierInfer::EncFastPartitionClassifierInfer() : m_impl(new Impl) {}
EncFastPartitionClassifierInfer::~EncFastPartitionClassifierInfer() = default;
bool EncFastPartitionClassifierInfer::isInitialized() const { return !m_impl->branches.empty(); }
void EncFastPartitionClassifierInfer::init(const std::string& modelPath)
{
  if (isInitialized()) return;
  // modelPath is the exported bundle directory. Branch filenames use grid HxW.
  const std::array<std::pair<int,int>,15> sizes = {{{8,8},{8,4},{4,8},{8,2},{2,8},
    {8,1},{1,8},{4,2},{2,4},{4,1},{1,4},{4,4},{2,2},{2,1},{1,2}}};
  for (const auto& size : sizes)
  {
    const std::string path = modelPath + "/classifier_" + std::to_string(size.first)
                          + "x" + std::to_string(size.second) + ".onnx";
    auto options = sessionOptions();
    std::unique_ptr<Ort::Session> session(new Ort::Session(onnxEnv(), path.c_str(), options));
    if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
      THROW("FastPartition classifier ONNX expects roi -> probabilities");
    m_impl->branches.emplace(size, std::move(session));
    std::vector<float> input(size_t(2*size.first*size.second), 0.0f);
    std::array<float,6> output;
    m_impl->infer(size.first,size.second,input,output);
  }
}


bool EncFastPartitionClassifierInfer::inferCu(const FastPartitionLuma32CtuCache& ctuCache, int cuX, int cuY,
                                              int cuWidth, int cuHeight,
                                              std::array<float, 6>& splitProbabilities)
{
  if (!isInitialized() || !ctuCache.valid || cuWidth < 4 || cuHeight < 4 || cuWidth % 4 != 0 || cuHeight % 4 != 0)
  {
    return false;
  }
  const int gridWidth = cuWidth / 4;
  const int gridHeight = cuHeight / 4;
  const FastPartitionLuma32Gridmap* gridmap = nullptr;
  for (const auto& candidate : ctuCache.gridmaps)
  {
    if (candidate.valid && cuX >= candidate.targetX && cuY >= candidate.targetY
        && cuX + cuWidth <= candidate.targetX + 32 && cuY + cuHeight <= candidate.targetY + 32)
    {
      gridmap = &candidate;
      break;
    }
  }
  if (gridmap == nullptr)
  {
    return false;
  }
  const int gridX = (cuX - gridmap->targetX) / 4;
  const int gridY = (cuY - gridmap->targetY) / 4;
  if (gridX < 0 || gridY < 0 || gridX + gridWidth > 8 || gridY + gridHeight > 8)
  {
    return false;
  }
  const int roiArea = gridHeight * gridWidth;
  std::vector<float> inputData(size_t(2 * roiArea));
  for (int c = 0; c < 2; c++)
  {
    for (int y = 0; y < gridHeight; y++)
    {
      for (int x = 0; x < gridWidth; x++)
      {
        inputData[size_t(c * roiArea + y * gridWidth + x)] =
          gridmap->values[size_t(c * 8 * 8 + (gridY + y) * 8 + gridX + x)];
      }
    }
  }
  const auto inferStart = std::chrono::steady_clock::now();
  if (gridHeight == 1 && gridWidth == 1)
  {
    splitProbabilities = { { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };
  }
  else
  {
    m_impl->infer(gridHeight, gridWidth, inputData, splitProbabilities);
  }
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.classifierCalls++;
  g_fastPartitionInferStats.classifierSeconds += elapsedSeconds(inferStart, inferEnd);
  return true;
}


#endif
