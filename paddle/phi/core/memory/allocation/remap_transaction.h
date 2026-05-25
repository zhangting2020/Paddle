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

#include <cstddef>
#include <functional>
#include <list>
#include <vector>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

class RemapTransaction {
 public:
  using VaRanges = std::vector<std::pair<VmmDevicePtr, size_t>>;
  using BlockList = std::list<BlockV2>;
  using BlockIterator = BlockList::iterator;

  struct PendingDestinationRange {
    VmmDevicePtr dst{0};
    size_t handle_count{0};
  };
  struct CandidateValidation {
    bool source_ok{true};
    bool target_ok{true};
  };
  struct MaterializedRange {
    HandleLayout layout;
    BlockV2 free_block;
    size_t bytes{0};
  };

  RemapTransaction(CUDAVirtualMemAllocatorV2* vmm_allocator, size_t handle_size)
      : vmm_allocator_(vmm_allocator), handle_size_(handle_size) {}

  void PrepareCandidates(const VaRanges& source_ranges,
                         const VaRanges& target_ranges,
                         size_t target_bytes);
  const VmmBackingMap::CompactCandidates& candidates() const {
    return candidates_;
  }

  bool ValidateSourcePages(const char* context) const;
  bool ValidateTargetPages(const char* context) const;
  CandidateValidation ValidateCandidates(const char* source_context,
                                         const char* target_context) const;

  void AddRollbackAction(std::function<void()> action);
  void AddSourceRestoreAction(
      std::list<BlockV2>* blocks,
      const std::vector<VmmAllocHandle>* handles,
      const std::vector<std::shared_ptr<VmmHandleMeta>>* metas);
  void SetSyntheticAllocationSink(
      std::list<DecoratedAllocationPtr>* underlying_allocations);
  void MapHandlesToDestination(
      VmmDevicePtr dst,
      const std::vector<VmmAllocHandle>& handles,
      const std::vector<std::shared_ptr<VmmHandleMeta>>* metas = nullptr);
  void MapHandleRangeToDestination(
      VmmDevicePtr dst,
      const std::vector<VmmAllocHandle>& handles,
      size_t start,
      size_t count,
      const std::vector<std::shared_ptr<VmmHandleMeta>>* metas = nullptr);
  HandleLayout BuildDestinationLayout(
      VmmDevicePtr dst,
      const std::vector<VmmAllocHandle>& handles,
      size_t start,
      size_t count) const;
  MaterializedRange MaterializeMappedRange(
      VmmDevicePtr dst,
      const std::vector<VmmAllocHandle>& handles,
      size_t start,
      size_t count,
      PoolType pool_type);
  void InstallTailFreeBlock(BlockList* blocks, BlockV2 free_block) const;
  BlockIterator InstallMappedGapRange(BlockList* blocks,
                                      BlockIterator gap_it,
                                      BlockV2 free_block,
                                      PoolType pool_type) const;
  void NormalizeBlocks(BlockList* blocks) const;
  void Commit();
  void Rollback();

 private:
  // Record destination intent before map so later bookkeeping failures can
  // still unmap every destination touched by this transaction.
  void RecordDestinationRange(VmmDevicePtr dst, size_t handle_count);
  void UnmapPartialDestination(VmmDevicePtr dst_base, size_t handle_count);
  void RollbackPendingDestinations();
  void StageSyntheticAllocation(DecoratedAllocationPtr allocation);
  void ClearPendingDestinations() { pending_destination_ranges_.clear(); }

  CUDAVirtualMemAllocatorV2* vmm_allocator_;
  size_t handle_size_;
  VmmBackingMap::CompactCandidates candidates_;
  std::vector<PendingDestinationRange> pending_destination_ranges_;
  std::list<DecoratedAllocationPtr>* underlying_allocations_{nullptr};
  std::vector<DecoratedAllocationPtr> pending_synthetic_allocations_;
  std::vector<std::function<void()>> rollback_actions_;
  bool completed_{false};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
