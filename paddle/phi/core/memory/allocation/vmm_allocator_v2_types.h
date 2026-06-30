// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

#if defined(PADDLE_WITH_CUDA)
using VMMDevicePtr = CUdeviceptr;
using VMMAllocHandle = CUmemGenericAllocationHandle;
#else
using VMMDevicePtr = uintptr_t;
using VMMAllocHandle = uint64_t;
#endif

// V2 keeps the bottom-layer shared types independent from the best-fit layer
// so that CUDAVirtualMemAllocatorV2 can be reviewed and compiled separately.
enum class PoolType : uint8_t {
  kSmall = 0,
  kLarge = 1,
};

// Fixed-size handle metadata returned by the bottom VMM provider. Upper layers
// may later reference these handles from block-level views.
struct VMMHandleMeta {
  VMMHandleMeta() = default;

  VMMHandleMeta(VMMDevicePtr base,
                size_t size,
                VMMAllocHandle handle,
                int device)
      : base_(base), size_(size), handle_(handle), device_(device) {}

  VMMDevicePtr Base() const { return base_; }
  size_t Size() const { return size_; }
  VMMAllocHandle AllocationHandle() const { return handle_; }
  int Device() const { return device_; }

 private:
  VMMDevicePtr base_{0};
  size_t size_{0};
  VMMAllocHandle handle_{0};
  int device_{0};
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is used to bootstrap upper-layer block state.
using HandleLayout = std::vector<std::shared_ptr<VMMHandleMeta>>;

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kUnmappedFree = 2,
};

struct BlockV2 {
  static BlockV2 MakeMappedBlock(BlockType type,
                                 void* ptr,
                                 size_t size,
                                 PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, type, pool_type);
    return block;
  }

  static BlockV2 MakeUnmappedFreeBlock(void* ptr,
                                       size_t size,
                                       PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, BlockType::kUnmappedFree, pool_type);
    return block;
  }

  bool IsActive() const { return type_ == BlockType::kActive; }
  bool IsFree() const { return type_ == BlockType::kFree; }
  bool IsMappedFree() const { return IsFree(); }
  bool IsUnmappedFree() const { return type_ == BlockType::kUnmappedFree; }
  void* Ptr() const { return ptr_; }
  size_t Size() const { return size_; }
  PoolType Pool() const { return pool_type_; }
  uint8_t* BeginPtr() const { return reinterpret_cast<uint8_t*>(ptr_); }
  uint8_t* EndPtr() const { return BeginPtr() + size_; }
  VMMDevicePtr BeginVA() const {
    return reinterpret_cast<VMMDevicePtr>(BeginPtr());
  }
  VMMDevicePtr EndVA() const { return BeginVA() + size_; }
  std::pair<VMMDevicePtr, size_t> VARange() const { return {BeginVA(), size_}; }
  bool ContainsVARange(VMMDevicePtr va, size_t size) const {
    return va >= BeginVA() && va <= EndVA() && size <= EndVA() - va;
  }
  bool IsAdjacentBefore(const BlockV2& next) const {
    return EndPtr() == next.BeginPtr();
  }
  bool CanAbsorbAdjacentFreeBlock(const BlockV2& next) const {
    return IsFree() && next.IsFree() && IsAdjacentBefore(next);
  }
  bool CanAbsorbAdjacentUnmappedFreeBlock(const BlockV2& next) const {
    return IsUnmappedFree() && next.IsUnmappedFree() && IsAdjacentBefore(next);
  }
  BlockV2 MakeMappedSubBlock(BlockType type, size_t offset, size_t len) const {
    return MakeMappedBlock(type, BeginPtr() + offset, len, pool_type_);
  }
  BlockV2 MakeMappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeMappedSubBlock(BlockType::kFree, offset, len);
  }
  BlockV2 MakeMappedActiveSubBlock(size_t offset, size_t len) const {
    return MakeMappedSubBlock(BlockType::kActive, offset, len);
  }
  BlockV2 MakeUnmappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeUnmappedFreeBlock(BeginPtr() + offset, len, pool_type_);
  }
  void MarkActive() { type_ = BlockType::kActive; }
  void MarkFree() { type_ = BlockType::kFree; }
  void MarkMappedFree() { MarkFree(); }
  void MarkUnmappedFree() { type_ = BlockType::kUnmappedFree; }
  void Reset(void* ptr, size_t size, BlockType type, PoolType pool_type) {
    ptr_ = ptr;
    size_ = size;
    type_ = type;
    pool_type_ = pool_type;
  }
  void TrimToPrefix(size_t keep) { size_ = keep; }
  void TrimToSuffix(size_t trim, size_t keep) {
    ptr_ = reinterpret_cast<uint8_t*>(ptr_) + trim;
    size_ = keep;
  }
  void AbsorbAdjacentBlock(const BlockV2& src) { size_ += src.size_; }
  void AbsorbAdjacentUnmappedFreeBlock(const BlockV2& src) {
    size_ += src.size_;
  }

  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kUnmappedFree};

  PoolType pool_type_{PoolType::kLarge};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
