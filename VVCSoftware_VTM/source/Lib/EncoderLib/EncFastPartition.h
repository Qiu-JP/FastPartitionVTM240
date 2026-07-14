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

struct FastPartitionSwinInput
{
  int targetX = 0;
  int targetY = 0;
  std::array<float, 96 * 96> luma;
};

struct FastPartitionGridmap64
{
  bool valid = false;
  int  targetX = 0;
  int  targetY = 0;
  int  validWidthUnits = 0;
  int  validHeightUnits = 0;
  std::array<float, 2 * 16 * 16> values;
};

struct FastPartitionCtuCache
{
  bool valid = false;
  int  ctuX = 0;
  int  ctuY = 0;
  int  ctuWidth = 0;
  int  ctuHeight = 0;
  std::array<FastPartitionSwinInput, 4>  swinInputs;
  std::array<FastPartitionGridmap64, 4>  gridmaps;

  void reset()
  {
    valid = false;
    ctuX = ctuY = ctuWidth = ctuHeight = 0;
    for (auto& gridmap: gridmaps)
    {
      gridmap.valid = false;
      gridmap.validWidthUnits = 0;
      gridmap.validHeightUnits = 0;
    }
  }
};

struct FastPartitionChromaSwinInput
{
  int targetX = 0;
  int targetY = 0;
  std::array<float, 2 * 48 * 48> chroma;
};

struct FastPartitionChromaGridmap32
{
  bool valid = false;
  int  targetX = 0;
  int  targetY = 0;
  int  validWidthUnits = 0;
  int  validHeightUnits = 0;
  std::array<float, 2 * 8 * 8> values;
};

struct FastPartitionChromaCtuCache
{
  bool valid = false;
  int  ctuX = 0;
  int  ctuY = 0;
  int  ctuWidth = 0;
  int  ctuHeight = 0;
  std::array<FastPartitionChromaSwinInput, 4> swinInputs;
  std::array<FastPartitionChromaGridmap32, 4> gridmaps;

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

class EncFastPartitionSwinInfer
{
public:
  EncFastPartitionSwinInfer();
  ~EncFastPartitionSwinInfer();

  EncFastPartitionSwinInfer(const EncFastPartitionSwinInfer&) = delete;
  EncFastPartitionSwinInfer& operator=(const EncFastPartitionSwinInfer&) = delete;

  void init(const std::string& modelPath);
  bool isInitialized() const;
  void inferCtu(const std::array<FastPartitionSwinInput, 4>& swinInputs, int qp,
                std::array<FastPartitionGridmap64, 4>& gridmaps);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

class EncFastPartitionChromaSwinInfer
{
public:
  EncFastPartitionChromaSwinInfer();
  ~EncFastPartitionChromaSwinInfer();

  EncFastPartitionChromaSwinInfer(const EncFastPartitionChromaSwinInfer&) = delete;
  EncFastPartitionChromaSwinInfer& operator=(const EncFastPartitionChromaSwinInfer&) = delete;

  void init(const std::string& modelPath);
  bool isInitialized() const;
  void inferCtu(const std::array<FastPartitionChromaSwinInput, 4>& swinInputs, int qp,
                std::array<FastPartitionChromaGridmap32, 4>& gridmaps);

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
  bool inferCu(const FastPartitionCtuCache& ctuCache, int cuX, int cuY, int cuWidth, int cuHeight,
               std::array<float, 6>& splitProbabilities);
  bool inferCu(const FastPartitionChromaCtuCache& ctuCache, int cuX, int cuY, int cuWidth, int cuHeight,
               std::array<float, 6>& splitProbabilities);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

#endif

#endif // __ENCFASTPARTITION__
