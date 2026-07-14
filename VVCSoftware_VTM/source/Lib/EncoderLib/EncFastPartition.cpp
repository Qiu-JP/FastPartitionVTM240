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

#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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
}

FILE* fastPartitionStatFile()
{
  static FILE* statFile = [] {
    const char* path = std::getenv("FASTPARTITION_STAT_LOG");
    return path != nullptr && path[0] != '\0' ? std::fopen(path, "a") : nullptr;
  }();
  return statFile;
}

struct EncFastPartitionSwinInfer::Impl
{
  std::unique_ptr<torch::jit::script::Module> swinModule;
  torch::Tensor input;
  torch::Tensor qpTensor;
};

struct EncFastPartitionChromaSwinInfer::Impl
{
  std::unique_ptr<torch::jit::script::Module> swinModule;
  torch::Tensor input;
  torch::Tensor qpTensor;
};

struct EncFastPartitionClassifierInfer::Impl
{
  std::unique_ptr<torch::jit::script::Module> classifierModule;
};

EncFastPartitionSwinInfer::EncFastPartitionSwinInfer()
  : m_impl(new Impl)
{
}

EncFastPartitionSwinInfer::~EncFastPartitionSwinInfer() = default;

bool EncFastPartitionSwinInfer::isInitialized() const
{
  return bool(m_impl->swinModule);
}

void EncFastPartitionSwinInfer::init(const std::string& modelPath)
{
  if (m_impl->swinModule)
  {
    return;
  }

  if (modelPath.empty())
  {
    THROW("FastPartitionSwinModel must point to a SwinTransformer_Unet_Luma96 TorchScript model");
  }

  m_impl->swinModule.reset(new torch::jit::script::Module(torch::jit::load(modelPath, torch::Device(torch::kCPU))));
  m_impl->swinModule->eval();
  m_impl->input = torch::empty({ 4, 1, 96, 96 }, torch::kFloat32);
  m_impl->qpTensor = torch::empty({ 4, 1 }, torch::kFloat32);

  torch::NoGradGuard noGrad;
  m_impl->input.zero_();
  m_impl->qpTensor.zero_();
  for (int i = 0; i < 3; i++)
  {
    (void)m_impl->swinModule->forward({ m_impl->input, m_impl->qpTensor }).toTensor();
  }
}

EncFastPartitionChromaSwinInfer::EncFastPartitionChromaSwinInfer()
  : m_impl(new Impl)
{
}

EncFastPartitionChromaSwinInfer::~EncFastPartitionChromaSwinInfer() = default;

bool EncFastPartitionChromaSwinInfer::isInitialized() const
{
  return bool(m_impl->swinModule);
}

void EncFastPartitionChromaSwinInfer::init(const std::string& modelPath)
{
  if (m_impl->swinModule)
  {
    return;
  }

  if (modelPath.empty())
  {
    THROW("FastPartitionChromaSwinModel must point to a 2x48x48 chroma TorchScript model");
  }

  m_impl->swinModule.reset(new torch::jit::script::Module(torch::jit::load(modelPath, torch::Device(torch::kCPU))));
  m_impl->swinModule->eval();
  m_impl->input = torch::empty({ 4, 2, 48, 48 }, torch::kFloat32);
  m_impl->qpTensor = torch::empty({ 4, 1 }, torch::kFloat32);

  torch::NoGradGuard noGrad;
  m_impl->input.zero_();
  m_impl->qpTensor.zero_();
  for (int i = 0; i < 3; i++)
  {
    (void)m_impl->swinModule->forward({ m_impl->input, m_impl->qpTensor }).toTensor();
  }
}

void EncFastPartitionChromaSwinInfer::inferCtu(
  const std::array<FastPartitionChromaSwinInput, 4>& swinInputs,
  int qp,
  std::array<FastPartitionChromaGridmap32, 4>& gridmaps)
{
  if (!m_impl->swinModule)
  {
    THROW("FastPartition chroma Swin model has not been initialized");
  }

  for (auto& gridmap : gridmaps)
  {
    gridmap.valid = false;
  }

  torch::NoGradGuard noGrad;
  const float normalizedQp = float(qp) / 51.0f;
  float* inputData = m_impl->input.data_ptr<float>();
  float* qpData = m_impl->qpTensor.data_ptr<float>();

  for (size_t idx = 0; idx < swinInputs.size(); idx++)
  {
    const FastPartitionChromaSwinInput& swinInput = swinInputs[idx];
    std::copy(swinInput.chroma.begin(), swinInput.chroma.end(), inputData + idx * 2 * 48 * 48);
    qpData[idx] = normalizedQp;
  }

  const auto inferStart = std::chrono::steady_clock::now();
  auto output = m_impl->swinModule->forward({ m_impl->input, m_impl->qpTensor }).toTensor().contiguous();
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.swinCalls++;
  g_fastPartitionInferStats.swinSeconds += elapsedSeconds(inferStart, inferEnd);
  if (output.numel() != int64_t(gridmaps.size() * 2 * 8 * 8))
  {
    THROW("FastPartition chroma Swin output size mismatch");
  }
  const float* outputData = output.data_ptr<float>();

  for (size_t idx = 0; idx < gridmaps.size(); idx++)
  {
    const FastPartitionChromaSwinInput& swinInput = swinInputs[idx];
    FastPartitionChromaGridmap32& gridmap = gridmaps[idx];
    gridmap.targetX = swinInput.targetX;
    gridmap.targetY = swinInput.targetY;
    std::copy(outputData + idx * 2 * 8 * 8, outputData + (idx + 1) * 2 * 8 * 8, gridmap.values.begin());
    gridmap.valid = true;
  }
}

void EncFastPartitionSwinInfer::inferCtu(const std::array<FastPartitionSwinInput, 4>& swinInputs, int qp,
                                         std::array<FastPartitionGridmap64, 4>& gridmaps)
{
  if (!m_impl->swinModule)
  {
    THROW("FastPartition Swin model has not been initialized");
  }

  for (auto& gridmap: gridmaps)
  {
    gridmap.valid = false;
  }

  torch::NoGradGuard noGrad;
  const float normalizedQp = float(qp) / 51.0f;

  float* inputData = m_impl->input.data_ptr<float>();
  float* qpData = m_impl->qpTensor.data_ptr<float>();
  
  for (size_t idx = 0; idx < swinInputs.size(); idx++)
  {
    const FastPartitionSwinInput& swinInput = swinInputs[idx];
    std::copy(swinInput.luma.begin(), swinInput.luma.end(), inputData + idx * 96 * 96);
    qpData[idx] = normalizedQp;
  }

  const auto inferStart = std::chrono::steady_clock::now();
  auto output = m_impl->swinModule->forward({ m_impl->input, m_impl->qpTensor }).toTensor().contiguous();
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.swinCalls++;
  const double swinElapsed = elapsedSeconds(inferStart, inferEnd);
  g_fastPartitionInferStats.swinSeconds += swinElapsed;
  if (output.numel() != int64_t(gridmaps.size() * 2 * 16 * 16))
  {
    THROW("FastPartition Swin output size mismatch");
  }
  const float* outputData = output.data_ptr<float>();

  for (size_t idx = 0; idx < gridmaps.size(); idx++)
  {
    const FastPartitionSwinInput& swinInput = swinInputs[idx];
    FastPartitionGridmap64& gridmap = gridmaps[idx];
    gridmap.targetX = swinInput.targetX;
    gridmap.targetY = swinInput.targetY;
    std::copy(outputData + idx * 2 * 16 * 16, outputData + (idx + 1) * 2 * 16 * 16, gridmap.values.begin());
    gridmap.valid = true;
  }
}

EncFastPartitionClassifierInfer::EncFastPartitionClassifierInfer()
  : m_impl(new Impl)
{
}

EncFastPartitionClassifierInfer::~EncFastPartitionClassifierInfer() = default;

bool EncFastPartitionClassifierInfer::isInitialized() const
{
  return bool(m_impl->classifierModule);
}

void EncFastPartitionClassifierInfer::init(const std::string& modelPath)
{
  if (m_impl->classifierModule)
  {
    return;
  }

  if (modelPath.empty())
  {
    THROW("FastPartitionClassifierModel must point to a Classifier_I TorchScript model");
  }

  m_impl->classifierModule.reset(new torch::jit::script::Module(torch::jit::load(modelPath, torch::Device(torch::kCPU))));
  m_impl->classifierModule->eval();
}

bool EncFastPartitionClassifierInfer::inferCu(const FastPartitionCtuCache& ctuCache, int cuX, int cuY, int cuWidth,
                                              int cuHeight, std::array<float, 6>& splitProbabilities)
{
  if (!m_impl->classifierModule || !ctuCache.valid)
  {
    return false;
  }
  if (cuWidth < 4 || cuHeight < 4 || cuWidth % 4 != 0 || cuHeight % 4 != 0)
  {
    return false;
  }

  const int gridWidth = cuWidth / 4;
  const int gridHeight = cuHeight / 4;
  const FastPartitionGridmap64* gridmap = nullptr;
  for (const auto& candidate: ctuCache.gridmaps)
  {
    if (!candidate.valid)
    {
      continue;
    }
    if (cuX >= candidate.targetX && cuY >= candidate.targetY
        && cuX + cuWidth <= candidate.targetX + 64 && cuY + cuHeight <= candidate.targetY + 64)
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
  if (gridX < 0 || gridY < 0 || gridX + gridWidth > 16 || gridY + gridHeight > 16)
  {
    return false;
  }

  torch::NoGradGuard noGrad;
  auto input = torch::empty({ 1, 2, gridHeight, gridWidth }, torch::kFloat32);
  float* inputData = input.data_ptr<float>();
  const int roiArea = gridHeight * gridWidth;
  for (int c = 0; c < 2; c++)
  {
    for (int y = 0; y < gridHeight; y++)
    {
      for (int x = 0; x < gridWidth; x++)
      {
        inputData[c * roiArea + y * gridWidth + x] =
          gridmap->values[size_t(c * 16 * 16 + (gridY + y) * 16 + gridX + x)];
      }
    }
  }

  const auto inferStart = std::chrono::steady_clock::now();
  auto output = m_impl->classifierModule->forward({ input, c10::IValue(), true }).toTensor().contiguous();
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.classifierCalls++;
  const double classifierElapsed = elapsedSeconds(inferStart, inferEnd);
  g_fastPartitionInferStats.classifierSeconds += classifierElapsed;
  if (output.numel() != int64_t(splitProbabilities.size()))
  {
    THROW("FastPartition Classifier_I output size mismatch");
  }
  const float* outputData = output.data_ptr<float>();
  std::copy(outputData, outputData + splitProbabilities.size(), splitProbabilities.begin());
  return true;
}

bool EncFastPartitionClassifierInfer::inferCu(const FastPartitionChromaCtuCache& ctuCache, int cuX, int cuY,
                                              int cuWidth, int cuHeight,
                                              std::array<float, 6>& splitProbabilities)
{
  if (!m_impl->classifierModule || !ctuCache.valid)
  {
    return false;
  }
  if (cuWidth < 4 || cuHeight < 4 || cuWidth % 4 != 0 || cuHeight % 4 != 0)
  {
    return false;
  }

  const int gridWidth = cuWidth / 4;
  const int gridHeight = cuHeight / 4;
  const FastPartitionChromaGridmap32* gridmap = nullptr;
  for (const auto& candidate : ctuCache.gridmaps)
  {
    if (!candidate.valid)
    {
      continue;
    }
    if (cuX >= candidate.targetX && cuY >= candidate.targetY
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

  torch::NoGradGuard noGrad;
  auto input = torch::empty({ 1, 2, gridHeight, gridWidth }, torch::kFloat32);
  float* inputData = input.data_ptr<float>();
  const int roiArea = gridHeight * gridWidth;
  for (int c = 0; c < 2; c++)
  {
    for (int y = 0; y < gridHeight; y++)
    {
      for (int x = 0; x < gridWidth; x++)
      {
        inputData[c * roiArea + y * gridWidth + x] =
          gridmap->values[size_t(c * 8 * 8 + (gridY + y) * 8 + gridX + x)];
      }
    }
  }

  const auto inferStart = std::chrono::steady_clock::now();
  auto output = m_impl->classifierModule->forward({ input, c10::IValue(), true }).toTensor().contiguous();
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.classifierCalls++;
  g_fastPartitionInferStats.classifierSeconds += elapsedSeconds(inferStart, inferEnd);
  if (output.numel() != int64_t(splitProbabilities.size()))
  {
    THROW("FastPartition chroma Classifier_I output size mismatch");
  }
  const float* outputData = output.data_ptr<float>();
  std::copy(outputData, outputData + splitProbabilities.size(), splitProbabilities.begin());
  return true;
}

#endif
