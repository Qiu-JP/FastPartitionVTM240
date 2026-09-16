/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 */

#ifndef __ENCFASTPARTITION__
#define __ENCFASTPARTITION__

#include "CommonLib/CommonDef.h"

#include <array>
#include <cstdio>
#include <memory>
#include <string>

#if FastPartition

FILE* fastPartitionStatFile();


struct FastPartitionLuma32SwinInput
{
  int targetX = 0;
  int targetY = 0;
  std::array<float, 48 * 48> luma;
};

struct FastPartitionLuma32Gridmap
{
  bool valid = false;
  int  targetX = 0;
  int  targetY = 0;
  int  validWidthUnits = 0;
  int  validHeightUnits = 0;
  std::array<float, 2 * 8 * 8> values;
};

struct FastPartitionLuma32CtuCache
{
  bool valid = false;
  int  ctuX = 0;
  int  ctuY = 0;
  int  ctuWidth = 0;
  int  ctuHeight = 0;
  std::array<FastPartitionLuma32SwinInput, 16> swinInputs;
  std::array<FastPartitionLuma32Gridmap, 16> gridmaps;

  void reset()
  {
    valid = false;
    ctuX = ctuY = ctuWidth = ctuHeight = 0;
    for (auto& gridmap : gridmaps)
    {
      gridmap.valid = false;
      gridmap.validWidthUnits = 0;
      gridmap.validHeightUnits = 0;
    }
  }
};


class EncFastPartitionLuma32SwinInfer
{
public:
  EncFastPartitionLuma32SwinInfer();
  ~EncFastPartitionLuma32SwinInfer();

  EncFastPartitionLuma32SwinInfer(const EncFastPartitionLuma32SwinInfer&) = delete;
  EncFastPartitionLuma32SwinInfer& operator=(const EncFastPartitionLuma32SwinInfer&) = delete;

  void init(const std::string& modelPath);
  bool isInitialized() const;
  void inferCtu(const std::array<FastPartitionLuma32SwinInput, 16>& swinInputs, int qp,
                std::array<FastPartitionLuma32Gridmap, 16>& gridmaps);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};


class EncFastPartitionClassifierInfer
{
public:
  EncFastPartitionClassifierInfer();
  ~EncFastPartitionClassifierInfer();

  EncFastPartitionClassifierInfer(const EncFastPartitionClassifierInfer&) = delete;
  EncFastPartitionClassifierInfer& operator=(const EncFastPartitionClassifierInfer&) = delete;

  void init(const std::string& modelPath);
  bool isInitialized() const;
  bool inferCu(const FastPartitionLuma32CtuCache& ctuCache, int cuX, int cuY, int cuWidth, int cuHeight,
               std::array<float, 6>& splitProbabilities);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

#endif

#endif // __ENCFASTPARTITION__
