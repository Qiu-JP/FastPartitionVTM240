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
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
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

struct JsonValue
{
  enum Type { Null, Boolean, Number, String, Array, Object } type = Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;
};

class JsonParser
{
public:
  explicit JsonParser(const std::string& text) : m_text(text) {}

  JsonValue parse()
  {
    JsonValue value = parseValue();
    skipWhitespace();
    if (m_pos != m_text.size()) fail("trailing data");
    return value;
  }

private:
  const std::string& m_text;
  size_t m_pos = 0;

  [[noreturn]] void fail(const std::string& message) const
  {
    throw std::runtime_error("Invalid native classifier JSON at byte " + std::to_string(m_pos) + ": " + message);
  }

  void skipWhitespace()
  {
    while (m_pos < m_text.size() && std::isspace(static_cast<unsigned char>(m_text[m_pos]))) ++m_pos;
  }

  bool consume(char c)
  {
    skipWhitespace();
    if (m_pos < m_text.size() && m_text[m_pos] == c)
    {
      ++m_pos;
      return true;
    }
    return false;
  }

  JsonValue parseValue()
  {
    skipWhitespace();
    if (m_pos >= m_text.size()) fail("unexpected end of file");
    const char c = m_text[m_pos];
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"')
    {
      JsonValue value;
      value.type = JsonValue::String;
      value.string = parseString();
      return value;
    }
    if (c == 't') return parseLiteral("true", JsonValue::Boolean, true);
    if (c == 'f') return parseLiteral("false", JsonValue::Boolean, false);
    if (c == 'n') return parseLiteral("null", JsonValue::Null, false);
    if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
    fail("unexpected character");
  }

  JsonValue parseObject()
  {
    JsonValue value;
    value.type = JsonValue::Object;
    ++m_pos;
    if (consume('}')) return value;
    for (;;)
    {
      skipWhitespace();
      if (m_pos >= m_text.size() || m_text[m_pos] != '"') fail("expected object key");
      std::string key = parseString();
      if (!consume(':')) fail("expected ':'");
      if (!value.object.emplace(key, parseValue()).second) fail("duplicate object key '" + key + "'");
      if (consume('}')) return value;
      if (!consume(',')) fail("expected ',' or '}'");
    }
  }

  JsonValue parseArray()
  {
    JsonValue value;
    value.type = JsonValue::Array;
    ++m_pos;
    if (consume(']')) return value;
    for (;;)
    {
      value.array.push_back(parseValue());
      if (consume(']')) return value;
      if (!consume(',')) fail("expected ',' or ']'");
    }
  }

  std::string parseString()
  {
    ++m_pos;
    std::string result;
    while (m_pos < m_text.size())
    {
      char c = m_text[m_pos++];
      if (c == '"') return result;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\')
      {
        result.push_back(c);
        continue;
      }
      if (m_pos >= m_text.size()) fail("unfinished string escape");
      c = m_text[m_pos++];
      switch (c)
      {
      case '"': result.push_back('"'); break;
      case '\\': result.push_back('\\'); break;
      case '/': result.push_back('/'); break;
      case 'b': result.push_back('\b'); break;
      case 'f': result.push_back('\f'); break;
      case 'n': result.push_back('\n'); break;
      case 'r': result.push_back('\r'); break;
      case 't': result.push_back('\t'); break;
      default: fail("unsupported string escape");
      }
    }
    fail("unterminated string");
  }

  JsonValue parseLiteral(const char* literal, JsonValue::Type type, bool boolean)
  {
    const size_t length = std::char_traits<char>::length(literal);
    if (m_text.compare(m_pos, length, literal) != 0) fail("invalid literal");
    m_pos += length;
    JsonValue value;
    value.type = type;
    value.boolean = boolean;
    return value;
  }

  JsonValue parseNumber()
  {
    const char* begin = m_text.c_str() + m_pos;
    char* end = nullptr;
    const double number = std::strtod(begin, &end);
    if (end == begin) fail("invalid number");
    m_pos += size_t(end - begin);
    JsonValue value;
    value.type = JsonValue::Number;
    value.number = number;
    return value;
  }
};

struct NativeClassifierBranch
{
  int gridH = 0;
  int gridW = 0;
  int inputDim = 0;
  int hiddenDim = 0;
  std::vector<float> fc1Weight;
  std::vector<float> fc1Bias;
  std::vector<float> fc2Weight;
  std::vector<float> fc2Bias;
  std::array<bool, 6> staticMask{};
};

int branchKey(int gridH, int gridW)
{
  return gridH * 100 + gridW;
}

const JsonValue& jsonMember(const JsonValue& value, const std::string& name)
{
  if (value.type != JsonValue::Object) throw std::runtime_error("Native classifier JSON: expected object");
  auto it = value.object.find(name);
  if (it == value.object.end()) throw std::runtime_error("Native classifier JSON: missing field '" + name + "'");
  return it->second;
}

int jsonInt(const JsonValue& value, const std::string& name)
{
  if (value.type != JsonValue::Number || !std::isfinite(value.number) || std::floor(value.number) != value.number)
    throw std::runtime_error("Native classifier JSON: '" + name + "' must be an integer");
  return int(value.number);
}

float jsonFloat(const JsonValue& value, const std::string& name)
{
  if (value.type != JsonValue::Number || !std::isfinite(value.number))
    throw std::runtime_error("Native classifier JSON: '" + name + "' must be a finite number");
  return float(value.number);
}

std::vector<float> jsonVector(const JsonValue& value, size_t expected, const std::string& name)
{
  if (value.type != JsonValue::Array || value.array.size() != expected)
    throw std::runtime_error("Native classifier JSON: invalid length for '" + name + "'");
  std::vector<float> result;
  result.reserve(expected);
  for (const JsonValue& item : value.array) result.push_back(jsonFloat(item, name));
  return result;
}

std::vector<float> jsonMatrix(const JsonValue& value, size_t rows, size_t cols, const std::string& name)
{
  if (value.type != JsonValue::Array || value.array.size() != rows)
    throw std::runtime_error("Native classifier JSON: invalid row count for '" + name + "'");
  std::vector<float> result;
  result.reserve(rows * cols);
  for (const JsonValue& row : value.array)
  {
    std::vector<float> values = jsonVector(row, cols, name);
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

std::array<bool, 6> jsonMask(const JsonValue& value, const std::string& name)
{
  if (value.type != JsonValue::Array || value.array.size() != 6)
    throw std::runtime_error("Native classifier JSON: invalid length for '" + name + "'");
  std::array<bool, 6> result{};
  for (size_t i = 0; i < result.size(); ++i)
  {
    if (value.array[i].type != JsonValue::Number || (value.array[i].number != 0.0 && value.array[i].number != 1.0))
      throw std::runtime_error("Native classifier JSON: mask values must be 0 or 1");
    result[i] = value.array[i].number != 0.0;
  }
  return result;
}

std::map<int, NativeClassifierBranch> loadNativeClassifierJson(const std::string& path, float& maskValue)
{
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open native classifier JSON: " + path);
  const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const JsonValue root = JsonParser(text).parse();

  const JsonValue& format = jsonMember(root, "format");
  if (format.type != JsonValue::String || format.string != "ClassifierI.NativeJSON")
    throw std::runtime_error("Native classifier JSON: unsupported format");
  if (jsonInt(jsonMember(root, "version"), "version") != 1
      || jsonInt(jsonMember(root, "num_classes"), "num_classes") != 6
      || jsonInt(jsonMember(root, "input_channel"), "input_channel") != 2)
    throw std::runtime_error("Native classifier JSON: unsupported version or model shape");
  maskValue = jsonFloat(jsonMember(root, "mask_value"), "mask_value");

  const JsonValue& branches = jsonMember(root, "branches");
  if (branches.type != JsonValue::Array || branches.array.size() != 16)
    throw std::runtime_error("Native classifier JSON: expected 16 branches");
  std::map<int, NativeClassifierBranch> result;
  for (const JsonValue& item : branches.array)
  {
    NativeClassifierBranch branch;
    branch.gridH = jsonInt(jsonMember(item, "grid_h"), "grid_h");
    branch.gridW = jsonInt(jsonMember(item, "grid_w"), "grid_w");
    const JsonValue& name = jsonMember(item, "name");
    if (name.type != JsonValue::String
        || name.string != std::to_string(branch.gridH) + "x" + std::to_string(branch.gridW))
      throw std::runtime_error("Native classifier JSON: branch name does not match its dimensions");
    branch.inputDim = jsonInt(jsonMember(item, "input_dim"), "input_dim");
    branch.hiddenDim = jsonInt(jsonMember(item, "hidden_dim"), "hidden_dim");
    if (branch.inputDim != 2 * branch.gridH * branch.gridW || branch.hiddenDim <= 0)
      throw std::runtime_error("Native classifier JSON: inconsistent branch dimensions");
    branch.fc1Weight = jsonMatrix(jsonMember(item, "fc1_weight"), branch.hiddenDim, branch.inputDim, "fc1_weight");
    branch.fc1Bias = jsonVector(jsonMember(item, "fc1_bias"), branch.hiddenDim, "fc1_bias");
    branch.fc2Weight = jsonMatrix(jsonMember(item, "fc2_weight"), 6, branch.hiddenDim, "fc2_weight");
    branch.fc2Bias = jsonVector(jsonMember(item, "fc2_bias"), 6, "fc2_bias");
    branch.staticMask = jsonMask(jsonMember(item, "static_mask"), "static_mask");
    if (!result.emplace(branchKey(branch.gridH, branch.gridW), std::move(branch)).second)
      throw std::runtime_error("Native classifier JSON: duplicate branch");
  }

  static const int expected[][2] = {
    {16,16}, {8,8}, {8,4}, {4,8}, {8,2}, {2,8}, {8,1}, {1,8},
    {4,2}, {2,4}, {4,1}, {1,4}, {4,4}, {2,2}, {2,1}, {1,2}
  };
  for (const auto& shape : expected)
    if (result.find(branchKey(shape[0], shape[1])) == result.end())
      throw std::runtime_error("Native classifier JSON: missing required branch");

  const JsonValue& branch1x1 = jsonMember(root, "branch_1x1");
  if (jsonInt(jsonMember(branch1x1, "grid_h"), "grid_h") != 1
      || jsonInt(jsonMember(branch1x1, "grid_w"), "grid_w") != 1)
    throw std::runtime_error("Native classifier JSON: invalid 1x1 branch");
  const std::array<bool, 6> mask1x1 = jsonMask(jsonMember(branch1x1, "static_mask"), "static_mask");
  if (!mask1x1[0] || std::count(mask1x1.begin(), mask1x1.end(), true) != 1)
    throw std::runtime_error("Native classifier JSON: invalid 1x1 mask");
  return result;
}

void inferNativeClassifier(const NativeClassifierBranch& branch, const std::vector<float>& input,
                           float maskValue, std::array<float, 6>& probabilities)
{
  std::vector<float> hidden(size_t(branch.hiddenDim));
  for (int h = 0; h < branch.hiddenDim; ++h)
  {
    float value = branch.fc1Bias[size_t(h)];
    const float* weight = branch.fc1Weight.data() + size_t(h) * branch.inputDim;
    for (int i = 0; i < branch.inputDim; ++i) value += weight[i] * input[size_t(i)];
    hidden[size_t(h)] = std::max(value, 0.0f);
  }

  for (size_t c = 0; c < probabilities.size(); ++c)
  {
    float value = branch.fc2Bias[c];
    const float* weight = branch.fc2Weight.data() + c * size_t(branch.hiddenDim);
    for (int h = 0; h < branch.hiddenDim; ++h) value += weight[h] * hidden[size_t(h)];
    probabilities[c] = branch.staticMask[c] ? value : maskValue;
  }
  const float maxLogit = *std::max_element(probabilities.begin(), probabilities.end());
  float sum = 0.0f;
  for (float& value : probabilities)
  {
    value = std::exp(value - maxLogit);
    sum += value;
  }
  for (float& value : probabilities) value /= sum;
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
  std::map<int, NativeClassifierBranch> nativeBranches;
  float nativeMaskValue = -1.0e9f;
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
  return !m_impl->nativeBranches.empty();
}

void EncFastPartitionClassifierInfer::init(const std::string& modelPath)
{
  if (isInitialized())
  {
    return;
  }

  if (modelPath.empty())
  {
    THROW("FastPartitionClassifierModel must point to a Classifier_I native JSON model");
  }

  m_impl->nativeBranches = loadNativeClassifierJson(modelPath, m_impl->nativeMaskValue);
}

bool EncFastPartitionClassifierInfer::inferCu(const FastPartitionCtuCache& ctuCache, int cuX, int cuY, int cuWidth,
                                              int cuHeight, std::array<float, 6>& splitProbabilities)
{
  if (!isInitialized() || !ctuCache.valid)
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

  const int roiArea = gridHeight * gridWidth;
  std::vector<float> inputData(size_t(2 * roiArea));
  for (int c = 0; c < 2; c++)
  {
    for (int y = 0; y < gridHeight; y++)
    {
      for (int x = 0; x < gridWidth; x++)
      {
        inputData[size_t(c * roiArea + y * gridWidth + x)] =
          gridmap->values[size_t(c * 16 * 16 + (gridY + y) * 16 + gridX + x)];
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
    auto branch = m_impl->nativeBranches.find(branchKey(gridHeight, gridWidth));
    if (branch == m_impl->nativeBranches.end()) return false;
    inferNativeClassifier(branch->second, inputData, m_impl->nativeMaskValue, splitProbabilities);
  }
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.classifierCalls++;
  const double classifierElapsed = elapsedSeconds(inferStart, inferEnd);
  g_fastPartitionInferStats.classifierSeconds += classifierElapsed;
  return true;
}

bool EncFastPartitionClassifierInfer::inferCu(const FastPartitionChromaCtuCache& ctuCache, int cuX, int cuY,
                                              int cuWidth, int cuHeight,
                                              std::array<float, 6>& splitProbabilities)
{
  if (!isInitialized() || !ctuCache.valid)
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
    auto branch = m_impl->nativeBranches.find(branchKey(gridHeight, gridWidth));
    if (branch == m_impl->nativeBranches.end()) return false;
    inferNativeClassifier(branch->second, inputData, m_impl->nativeMaskValue, splitProbabilities);
  }
  const auto inferEnd = std::chrono::steady_clock::now();
  g_fastPartitionInferStats.classifierCalls++;
  g_fastPartitionInferStats.classifierSeconds += elapsedSeconds(inferStart, inferEnd);
  return true;
}

#endif
