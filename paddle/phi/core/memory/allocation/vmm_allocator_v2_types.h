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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
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

// RAII wrapper around gpuEvent_t so that multiple blocks can share
// ownership of the same event via shared_ptr.  The event is only
// destroyed when the last reference is dropped, preventing the
// double-destroy SIGSEGV that occurred when raw gpuEvent_t pointers
// were shallow-copied across split blocks and then independently
// destroyed during merge.
#if defined(PADDLE_WITH_CUDA)
struct CUDAEventGuard {
  gpuEvent_t event{nullptr};

  explicit CUDAEventGuard(gpuEvent_t e) : event(e) {}
  ~CUDAEventGuard() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }

  // Non-copyable: ownership is shared via shared_ptr.
  CUDAEventGuard(const CUDAEventGuard&) = delete;
  CUDAEventGuard& operator=(const CUDAEventGuard&) = delete;
};

class VMMRemapEventAllocation {
 public:
  virtual ~VMMRemapEventAllocation() = default;
  virtual bool SetVMMRemapEvent(gpuStream_t stream,
                                std::shared_ptr<CUDAEventGuard> event) = 0;
};

struct VMMBlockRemapState {
  gpuStream_t stream{nullptr};
  std::shared_ptr<CUDAEventGuard> event;
};
#endif

// V2 keeps the bottom-layer shared types independent from the best-fit layer
// so that CUDAVirtualMemAllocatorV2 can be reviewed and compiled separately.
enum class PoolType : uint8_t {
  kSmall = 0,
  kLarge = 1,
};

// Fixed-size handle metadata returned by the bottom VMM provider. Upper layers
// may later reference these handles from block-level views, remap metadata, or
// IPC export state.
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

  bool IsOwnedByRemapDestination() const { return owned_by_remap_destination_; }
  void MarkOwnedByRemapDestination() { owned_by_remap_destination_ = true; }
  void RestoreOriginalOwnership() { owned_by_remap_destination_ = false; }

 private:
  VMMDevicePtr base_{0};
  size_t size_{0};
  VMMAllocHandle handle_{0};
  int device_{0};
  // Set while a handle's lifetime has moved from the original free block to a
  // synthetic destination block created by remap compaction. FreeImpl must skip
  // the original owner because the destination allocation now releases it.
  bool owned_by_remap_destination_{false};
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is only used to bootstrap upper-layer block state or
// answer allocation-level IPC/export queries.
using HandleLayout = std::vector<std::shared_ptr<VMMHandleMeta>>;

struct IpcBlockPartDescriptor {
  VMMDevicePtr handle_base;
  size_t handle_size;
  VMMAllocHandle handle;
  int device;
  size_t handle_rel_off;
  size_t len;
};

// A logical slice of one fixed-size VMM handle. This is the block-level view
// owned by VMMAutoGrowthBestFitAllocatorV2 and is updated by split / merge /
// remap after the initial HandleLayout has been consumed. Future IPC export
// still exports whole handles at the driver layer; BlockPartV2 carries the
// slice metadata needed to rebuild the logical tensor view on import.
struct BlockPartV2 {
  BlockPartV2() = default;

  BlockPartV2(std::shared_ptr<VMMHandleMeta> handle,
              size_t handle_rel_off,
              size_t len)
      : handle_(std::move(handle)),
        handle_rel_off_(handle_rel_off),
        len_(len) {}

  bool HasHandle() const { return handle_ != nullptr; }
  const std::shared_ptr<VMMHandleMeta>& HandleMeta() const { return handle_; }
  VMMDevicePtr HandleBase() const { return handle_->Base(); }
  VMMDevicePtr SliceBase() const { return HandleBase() + handle_rel_off_; }
  size_t HandleSize() const { return handle_->Size(); }
  size_t HandleRelOffset() const { return handle_rel_off_; }
  size_t ByteSize() const { return len_; }
  VMMAllocHandle AllocationHandle() const {
    return handle_->AllocationHandle();
  }
  int Device() const { return handle_->Device(); }

  bool FullyCoversHandle() const {
    return handle_rel_off_ == 0 && handle_ != nullptr &&
           len_ == handle_->Size();
  }

  bool IsOwnedByRemapDestination() const {
    return handle_ != nullptr && handle_->IsOwnedByRemapDestination();
  }

  // Return a sub-slice of this part in handle-relative coordinates.
  BlockPartV2 Slice(size_t offset, size_t slice_len) const {
    return {handle_, handle_rel_off_ + offset, slice_len};
  }

  // Fold an adjacent slice from the same handle into this part.
  bool TryExtend(const BlockPartV2& next) {
    if (handle_.get() != next.handle_.get()) {
      return false;
    }
    if (handle_rel_off_ + len_ != next.handle_rel_off_) {
      return false;
    }
    len_ += next.len_;
    return true;
  }

 private:
  std::shared_ptr<VMMHandleMeta> handle_;
  size_t handle_rel_off_{0};
  size_t len_{0};
};

inline std::vector<BlockPartV2> BuildBlockPartsFromHandleLayout(
    const HandleLayout& layout) {
  std::vector<BlockPartV2> parts;
  parts.reserve(layout.size());
  for (const auto& handle : layout) {
    parts.push_back(BlockPartV2{handle, 0, handle->Size()});
  }
  return parts;
}

inline std::vector<BlockPartV2> SliceBlockPartsForRange(
    const std::vector<BlockPartV2>& parts,
    size_t range_offset,
    size_t range_len) {
  // parts describes one logical block as an ordered list of handle slices.
  // The target range is expressed in that same logical block address space.
  std::vector<BlockPartV2> sliced_parts;
  if (range_len == 0 || parts.empty()) {
    return sliced_parts;
  }

  PADDLE_ENFORCE_LE(
      range_offset,
      std::numeric_limits<size_t>::max() - range_len,
      common::errors::InvalidArgument(
          "Invalid VMM V2 block-part slice range: offset %zu plus length %zu "
          "overflows.",
          range_offset,
          range_len));

  if (parts.size() == 1) {
    const auto& part = parts.front();
    PADDLE_ENFORCE_LE(
        range_offset,
        part.ByteSize(),
        common::errors::InvalidArgument(
            "Invalid VMM V2 block-part slice offset %zu for part length %zu.",
            range_offset,
            part.ByteSize()));
    PADDLE_ENFORCE_LE(
        range_len,
        part.ByteSize() - range_offset,
        common::errors::InvalidArgument(
            "Invalid VMM V2 block-part slice length %zu at offset %zu for "
            "part length %zu.",
            range_len,
            range_offset,
            part.ByteSize()));
    return {part.Slice(range_offset, range_len)};
  }

  sliced_parts.reserve(parts.size());
  const size_t range_end = range_offset + range_len;
  size_t cursor = 0;
  size_t sliced_len = 0;

  for (const auto& part : parts) {
    const size_t part_block_begin = cursor;
    const size_t part_block_end = cursor + part.ByteSize();
    cursor = part_block_end;

    if (part_block_end <= range_offset) {
      continue;
    }
    if (part_block_begin >= range_end) {
      break;
    }

    const size_t slice_begin = std::max(part_block_begin, range_offset);
    const size_t slice_end = std::min(part_block_end, range_end);
    auto slice =
        part.Slice(slice_begin - part_block_begin, slice_end - slice_begin);
    const size_t slice_len = slice.ByteSize();

    if (sliced_parts.empty() || !sliced_parts.back().TryExtend(slice)) {
      sliced_parts.push_back(std::move(slice));
    }
    sliced_len += slice_len;
  }
  PADDLE_ENFORCE_EQ(
      sliced_len,
      range_len,
      common::errors::InvalidArgument(
          "Invalid VMM V2 block-part slice range: requested %zu bytes at "
          "offset %zu, but only sliced %zu bytes from %zu parts.",
          range_len,
          range_offset,
          sliced_len,
          parts.size()));
  return sliced_parts;
}

inline void AppendBlockPartsTail(std::vector<BlockPartV2>* dst,
                                 std::vector<BlockPartV2>* src) {
  if (src->empty()) {
    return;
  }
  dst->reserve(dst->size() + src->size());
  auto begin = src->begin();
  if (!dst->empty() && dst->back().TryExtend(src->front())) {
    ++begin;
  }
  dst->insert(dst->end(),
              std::make_move_iterator(begin),
              std::make_move_iterator(src->end()));
}

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kUnmappedFree = 2,
};

enum class BlockRestoreMappedFreeResult : uint8_t {
  kOutside = 0,
  kRangeExceedsBlock = 1,
  kBuilt = 2,
};

struct BlockV2 {
  static BlockV2 MakeMappedBlock(BlockType type,
                                 void* ptr,
                                 size_t size,
                                 const std::vector<BlockPartV2>& parts,
                                 size_t parts_offset,
                                 size_t parts_len,
                                 PoolType pool_type) {
    BlockV2 block;
    block.ResetAsMappedBlock(
        type, ptr, size, parts, parts_offset, parts_len, pool_type);
    return block;
  }

  static BlockV2 MakeMappedActiveBlock(void* ptr,
                                       size_t size,
                                       const std::vector<BlockPartV2>& parts,
                                       size_t parts_offset,
                                       size_t parts_len,
                                       PoolType pool_type) {
    return MakeMappedBlock(BlockType::kActive,
                           ptr,
                           size,
                           parts,
                           parts_offset,
                           parts_len,
                           pool_type);
  }

  static BlockV2 MakeMappedFreeBlock(void* ptr,
                                     size_t size,
                                     const std::vector<BlockPartV2>& parts,
                                     size_t parts_offset,
                                     size_t parts_len,
                                     PoolType pool_type) {
    return MakeMappedBlock(
        BlockType::kFree, ptr, size, parts, parts_offset, parts_len, pool_type);
  }

  static BlockV2 MakeMappedFreeBlock(void* ptr,
                                     size_t size,
                                     std::vector<BlockPartV2>&& parts,
                                     PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, BlockType::kFree, pool_type);
    block.SetParts(std::move(parts));
    return block;
  }

  static BlockV2 MakeMappedFreeBlockFromLayout(void* ptr,
                                               size_t size,
                                               const HandleLayout& layout,
                                               PoolType pool_type) {
    return MakeMappedFreeBlock(
        ptr, size, BuildBlockPartsFromHandleLayout(layout), 0, size, pool_type);
  }

  static BlockV2 MakeMappedBlockWithoutParts(BlockType type,
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

  static BlockV2 MakeSinglePartMappedFreeBlock(
      void* ptr,
      size_t size,
      std::shared_ptr<VMMHandleMeta> meta,
      PoolType pool_type) {
    BlockV2 block;
    block.ResetAsSinglePartMappedFree(ptr, size, std::move(meta), pool_type);
    return block;
  }

  static BlockV2 MakeFreeSegment(BlockType type,
                                 void* ptr,
                                 size_t size,
                                 const BlockPartV2* part,
                                 PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, type, pool_type);
    if (part != nullptr) {
      block.AddPart(*part);
    }
    return block;
  }

  static void AppendFreeSegment(std::vector<BlockV2>* segments,
                                BlockType type,
                                void* ptr,
                                size_t size,
                                const BlockPartV2* part,
                                PoolType pool_type) {
    if (size == 0) {
      return;
    }

    if (!segments->empty() && segments->back().type_ == type) {
      segments->back().ExtendSegment(size, part);
      return;
    }

    segments->push_back(MakeFreeSegment(type, ptr, size, part, pool_type));
  }

  bool HasParts() const { return !parts_.empty(); }
  bool IsActive() const { return type_ == BlockType::kActive; }
  bool IsFree() const { return type_ == BlockType::kFree; }
  bool IsMappedFree() const { return IsFree(); }
  bool CanBeRemapSource() const { return IsMappedFree(); }
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
  BlockV2 MakeMappedFreeSubBlock(size_t offset, size_t len) const {
    auto block = MakeMappedFreeBlock(
        BeginPtr() + offset, len, parts_, offset, len, pool_type_);
    block.ipc_exported_ = ipc_exported_;
#if defined(PADDLE_WITH_CUDA)
    block.CopyRemapSafetyFrom(*this);
#endif
    return block;
  }
  BlockV2 MakeMappedActiveSubBlock(size_t offset, size_t len) const {
    auto block = MakeMappedActiveBlock(
        BeginPtr() + offset, len, parts_, offset, len, pool_type_);
    block.ipc_exported_ = ipc_exported_;
    return block;
  }
  BlockV2 MakeMappedSubBlockWithoutParts(BlockType type,
                                         size_t offset,
                                         size_t len) const {
    auto block =
        MakeMappedBlockWithoutParts(type, BeginPtr() + offset, len, pool_type_);
    block.ipc_exported_ = ipc_exported_;
#if defined(PADDLE_WITH_CUDA)
    block.CopyRemapSafetyFrom(*this);
#endif
    return block;
  }
  BlockV2 MakeMappedFreeSubBlockWithoutParts(size_t offset, size_t len) const {
    return MakeMappedSubBlockWithoutParts(BlockType::kFree, offset, len);
  }
  BlockV2 MakeMappedActiveSubBlockWithoutParts(size_t offset,
                                               size_t len) const {
    auto block =
        MakeMappedSubBlockWithoutParts(BlockType::kActive, offset, len);
#if defined(PADDLE_WITH_CUDA)
    block.ClearRemapSafety();
#endif
    return block;
  }
  BlockV2 MakeUnmappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeUnmappedFreeBlock(BeginPtr() + offset, len, pool_type_);
  }
  BlockRestoreMappedFreeResult BuildRestoreMappedFreeSegments(
      VMMDevicePtr va,
      size_t size,
      const std::shared_ptr<VMMHandleMeta>& meta,
      std::vector<BlockV2>* segments) const {
    if (!IsUnmappedFree() || va < BeginVA() || va >= EndVA()) {
      return BlockRestoreMappedFreeResult::kOutside;
    }
    if (size > EndVA() - va) {
      return BlockRestoreMappedFreeResult::kRangeExceedsBlock;
    }

    segments->clear();
    const size_t prefix = va - BeginVA();
    const size_t suffix = EndVA() - (va + size);
    if (prefix > 0) {
      segments->push_back(MakeUnmappedFreeSubBlock(0, prefix));
    }
    segments->push_back(MakeSinglePartMappedFreeBlock(
        reinterpret_cast<void*>(va), size, meta, pool_type_));
    if (suffix > 0) {
      segments->push_back(MakeUnmappedFreeSubBlock(prefix + size, suffix));
    }
    return BlockRestoreMappedFreeResult::kBuilt;
  }
  void MarkActive() {
    type_ = BlockType::kActive;
#if defined(PADDLE_WITH_CUDA)
    ClearRemapSafety();
#endif
  }
  void MarkFree() { type_ = BlockType::kFree; }
  void MarkMappedFree() { MarkFree(); }
  void MarkUnmappedFree() { type_ = BlockType::kUnmappedFree; }
  void Reset(void* ptr, size_t size, BlockType type, PoolType pool_type) {
    ptr_ = ptr;
    size_ = size;
    type_ = type;
    pool_type_ = pool_type;
    ipc_exported_ = false;
    parts_.clear();
#if defined(PADDLE_WITH_CUDA)
    ClearRemapSafety();
#endif
  }
  void ResetAsMappedBlock(BlockType type,
                          void* ptr,
                          size_t size,
                          const std::vector<BlockPartV2>& parts,
                          size_t parts_offset,
                          size_t parts_len,
                          PoolType pool_type) {
    Reset(ptr, size, type, pool_type);
    SetPartsFromRange(parts, parts_offset, parts_len);
  }
  // Logical allocation-view slices. Ownership/reuse/release/remap safety must
  // be decided by the backing-state APIs, not by inspecting part layout.
  size_t AllocationPartCount() const { return parts_.size(); }
  size_t AllocationPartsByteSize() const {
    size_t total = 0;
    for (const auto& part : parts_) {
      total += part.ByteSize();
    }
    return total;
  }
  bool HasCompleteAllocationParts() const {
    return AllocationPartsByteSize() == size_;
  }
  bool HasSingleAllocationPart(size_t handle_rel_off, size_t len) const {
    return parts_.size() == 1 &&
           parts_.front().HandleRelOffset() == handle_rel_off &&
           parts_.front().ByteSize() == len;
  }
  const std::shared_ptr<VMMHandleMeta>& FirstAllocationPartHandleMeta() const {
    return parts_.front().HandleMeta();
  }
  size_t AllocationPartHandleRelOffset(size_t index) const {
    return parts_.at(index).HandleRelOffset();
  }
  size_t AllocationPartByteSize(size_t index) const {
    return parts_.at(index).ByteSize();
  }
  void TrimToPrefix(size_t keep) {
    if (HasParts()) {
      TrimPartsToRange(0, keep);
    }
    size_ = keep;
  }
  BlockV2 SplitMappedFreeSuffixFromPrefix(size_t keep) {
    PADDLE_ENFORCE_GT(
        keep,
        0,
        common::errors::InvalidArgument(
            "VMM V2 split prefix size must be greater than zero."));
    PADDLE_ENFORCE_LT(
        keep,
        size_,
        common::errors::InvalidArgument(
            "VMM V2 split prefix size %zu must be smaller than block size %zu.",
            keep,
            size_));
    BlockV2 suffix;
    suffix.Reset(BeginPtr() + keep, size_ - keep, BlockType::kFree, pool_type_);
    suffix.ipc_exported_ = ipc_exported_;
#if defined(PADDLE_WITH_CUDA)
    suffix.CopyRemapSafetyFrom(*this);
#endif
    if (HasParts()) {
      std::vector<BlockPartV2> prefix_parts;
      std::vector<BlockPartV2> suffix_parts;
      SplitPartsAt(keep, &prefix_parts, &suffix_parts);
      parts_ = std::move(prefix_parts);
      suffix.parts_ = std::move(suffix_parts);
    }
    size_ = keep;
    return suffix;
  }
  void TrimToSuffix(size_t trim, size_t keep) {
    if (HasParts()) {
      TrimPartsToRange(trim, keep);
    }
    ptr_ = reinterpret_cast<uint8_t*>(ptr_) + trim;
    size_ = keep;
  }
  template <typename Fn>
  void ForEachPartWithPtr(Fn&& fn) const {
    auto* base = reinterpret_cast<uint8_t*>(ptr_);
    size_t offset = 0;
    for (const auto& part : parts_) {
      fn(part, base + offset);
      offset += part.ByteSize();
    }
  }
  void AbsorbAdjacentBlock(BlockV2* src) {
    size_ += src->size_;
    ipc_exported_ = ipc_exported_ || src->ipc_exported_;
    AppendPartsFrom(src);
#if defined(PADDLE_WITH_CUDA)
    AppendRemapSafetyFrom(*src);
#endif
  }
  void AbsorbAdjacentBlockWithoutParts(BlockV2* src) {
    size_ += src->size_;
    ipc_exported_ = ipc_exported_ || src->ipc_exported_;
#if defined(PADDLE_WITH_CUDA)
    AppendRemapSafetyFrom(*src);
#endif
  }
  void AbsorbAdjacentUnmappedFreeBlock(const BlockV2& src) {
    size_ += src.size_;
  }
  void ExtendSegment(size_t size, const BlockPartV2* part) {
    size_ += size;
    if (part != nullptr) {
      TryAppendPart(*part);
    }
  }
  void ResetAsSinglePartMappedFree(void* ptr,
                                   size_t size,
                                   std::shared_ptr<VMMHandleMeta> meta,
                                   PoolType pool_type) {
    Reset(ptr, size, BlockType::kFree, pool_type);
    SetSinglePart(std::move(meta), size);
  }

  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kUnmappedFree};
  bool ipc_exported_{false};

 private:
  void SetParts(const std::vector<BlockPartV2>& parts) { parts_ = parts; }
  void SetParts(std::vector<BlockPartV2>&& parts) { parts_ = std::move(parts); }
  void SetPartsFromRange(const std::vector<BlockPartV2>& parts,
                         size_t offset,
                         size_t len) {
    parts_ = SliceBlockPartsForRange(parts, offset, len);
  }
  void TrimPartsToRange(size_t offset, size_t len) {
    parts_ = SliceBlockPartsForRange(parts_, offset, len);
  }
  void SplitPartsAt(size_t split_offset,
                    std::vector<BlockPartV2>* prefix_parts,
                    std::vector<BlockPartV2>* suffix_parts) const {
    prefix_parts->clear();
    suffix_parts->clear();
    prefix_parts->reserve(parts_.size());
    suffix_parts->reserve(parts_.size());

    auto append_part = [](std::vector<BlockPartV2>* parts, BlockPartV2 part) {
      if (part.ByteSize() == 0) {
        return;
      }
      if (parts->empty() || !parts->back().TryExtend(part)) {
        parts->push_back(std::move(part));
      }
    };

    size_t cursor = 0;
    size_t prefix_len = 0;
    size_t suffix_len = 0;
    for (const auto& part : parts_) {
      const size_t part_begin = cursor;
      const size_t part_end = part_begin + part.ByteSize();
      cursor = part_end;

      if (part_end <= split_offset) {
        append_part(prefix_parts, part);
        prefix_len += part.ByteSize();
        continue;
      }
      if (part_begin >= split_offset) {
        append_part(suffix_parts, part);
        suffix_len += part.ByteSize();
        continue;
      }

      const size_t prefix_part_len = split_offset - part_begin;
      const size_t suffix_part_len = part_end - split_offset;
      append_part(prefix_parts, part.Slice(0, prefix_part_len));
      append_part(suffix_parts, part.Slice(prefix_part_len, suffix_part_len));
      prefix_len += prefix_part_len;
      suffix_len += suffix_part_len;
    }

    PADDLE_ENFORCE_EQ(
        prefix_len,
        split_offset,
        common::errors::InvalidArgument(
            "Invalid VMM V2 split prefix: expected %zu bytes, got %zu.",
            split_offset,
            prefix_len));
    PADDLE_ENFORCE_EQ(
        suffix_len,
        size_ - split_offset,
        common::errors::InvalidArgument(
            "Invalid VMM V2 split suffix: expected %zu bytes, got %zu.",
            size_ - split_offset,
            suffix_len));
  }
  void SetSinglePart(std::shared_ptr<VMMHandleMeta> meta, size_t len) {
    parts_.clear();
    parts_.push_back(BlockPartV2{std::move(meta), 0, len});
  }
  bool TryAppendPart(const BlockPartV2& part) {
    if (parts_.empty() || !parts_.back().TryExtend(part)) {
      parts_.push_back(part);
      return false;
    }
    return true;
  }
  void AppendPartsFrom(BlockV2* src) {
    AppendBlockPartsTail(&parts_, &src->parts_);
  }
  void AddPart(BlockPartV2 part) { parts_.push_back(std::move(part)); }

  std::vector<BlockPartV2> parts_;

 public:
  PoolType pool_type_{PoolType::kLarge};
#if defined(PADDLE_WITH_CUDA)
  void ClearRemapSafety() {
    owning_stream_ = nullptr;
    remap_safe_event_.reset();
    remap_pending_states_.clear();
    remap_safety_unknown_ = false;
  }
  void SetRemapSafety(gpuStream_t stream,
                      std::shared_ptr<CUDAEventGuard> event) {
    ClearRemapSafety();
    if (stream == nullptr && event == nullptr) {
      remap_safety_unknown_ = true;
      return;
    }
    owning_stream_ = stream;
    remap_safe_event_ = std::move(event);
  }
  void CopyRemapSafetyFrom(const BlockV2& src) {
    owning_stream_ = src.owning_stream_;
    remap_safe_event_ = src.remap_safe_event_;
    remap_pending_states_ = src.remap_pending_states_;
    remap_safety_unknown_ = src.remap_safety_unknown_;
  }
  void AppendRemapSafety(gpuStream_t stream,
                         std::shared_ptr<CUDAEventGuard> event) {
    if (stream == nullptr && event == nullptr) {
      return;
    }
    if (owning_stream_ == stream && remap_safe_event_.get() == event.get()) {
      return;
    }
    for (const auto& state : remap_pending_states_) {
      if (state.stream == stream && state.event.get() == event.get()) {
        return;
      }
    }
    if (owning_stream_ == nullptr && remap_safe_event_ == nullptr) {
      owning_stream_ = stream;
      remap_safe_event_ = std::move(event);
      return;
    }
    remap_pending_states_.push_back({stream, std::move(event)});
  }
  void AppendRemapSafetyFrom(const BlockV2& src) {
    remap_safety_unknown_ = remap_safety_unknown_ || src.remap_safety_unknown_;
    AppendRemapSafety(src.owning_stream_, src.remap_safe_event_);
    for (const auto& state : src.remap_pending_states_) {
      AppendRemapSafety(state.stream, state.event);
    }
  }
  bool HasUnknownRemapSafety() const { return remap_safety_unknown_; }

  gpuStream_t owning_stream_{nullptr};
  std::shared_ptr<CUDAEventGuard> remap_safe_event_;
  std::vector<VMMBlockRemapState> remap_pending_states_;
  bool remap_safety_unknown_{false};
#endif
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
