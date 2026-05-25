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

#include "paddle/phi/core/memory/allocation/remap_transaction.h"

#include <list>
#include <utility>

#include "glog/logging.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

bool TryAppendRollbackPart(std::vector<BlockPartV2>* dst,
                           const BlockPartV2& part) {
  if (dst->empty() || !dst->back().TryExtend(part)) {
    dst->push_back(part);
    return false;
  }
  return true;
}

void MergeAdjacentFreeBlocksForRollback(std::list<BlockV2>* blocks) {
  for (auto it = blocks->begin(); it != blocks->end();) {
    if (it->type_ != BlockType::kFree) {
      ++it;
      continue;
    }
    auto next = std::next(it);
    if (next != blocks->end() && next->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(it->ptr_) + it->size_ ==
            reinterpret_cast<uint8_t*>(next->ptr_)) {
      it->size_ += next->size_;
      for (const auto& part : next->parts_) {
        TryAppendRollbackPart(&it->parts_, part);
      }
      blocks->erase(next);
      continue;
    }
    ++it;
  }
}

bool RestoreGapToFree(std::list<BlockV2>* blocks,
                      VmmDevicePtr va,
                      size_t size,
                      const std::shared_ptr<VmmHandleMeta>& meta) {
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    if (it->type_ != BlockType::kGap) continue;
    auto blk_start = reinterpret_cast<VmmDevicePtr>(it->ptr_);
    auto blk_end = blk_start + it->size_;
    if (va < blk_start || va >= blk_end) continue;
    if (size > blk_end - va) {
      VLOG(0) << "RestoreGapToFree: range exceeds GAP, va="
              << reinterpret_cast<void*>(va) << " size=" << size
              << " gap_start=" << reinterpret_cast<void*>(blk_start)
              << " gap_size=" << it->size_;
      return false;
    }

    size_t prefix = va - blk_start;
    size_t suffix = blk_end - (va + size);

    if (prefix > 0) {
      BlockV2 prefix_gap;
      prefix_gap.ptr_ = it->ptr_;
      prefix_gap.size_ = prefix;
      prefix_gap.type_ = BlockType::kGap;
      prefix_gap.pool_type_ = it->pool_type_;
      blocks->insert(it, std::move(prefix_gap));
    }

    it->ptr_ = reinterpret_cast<void*>(va);
    it->size_ = size;
    it->type_ = BlockType::kFree;
    it->parts_.clear();
    it->parts_.push_back(BlockPartV2{meta, 0, size});

    if (suffix > 0) {
      BlockV2 suffix_gap;
      suffix_gap.ptr_ = reinterpret_cast<void*>(va + size);
      suffix_gap.size_ = suffix;
      suffix_gap.type_ = BlockType::kGap;
      suffix_gap.pool_type_ = it->pool_type_;
      blocks->insert(std::next(it), std::move(suffix_gap));
    }
    return true;
  }
  VLOG(0) << "RestoreGapToFree: GAP not found for VA "
          << reinterpret_cast<void*>(va) << " — force-release will follow";
  return false;
}

void RestoreSourceMappings(
    std::list<BlockV2>* blocks,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    CUDAVirtualMemAllocatorV2* vmm_allocator,
    size_t handle_size) {
  platform::CUDADeviceGuard guard(vmm_allocator->place().device);
  size_t restored = 0, force_released = 0;
  for (size_t i = 0; i < handles.size(); ++i) {
    if (!metas[i]->remapped) continue;
    VmmDevicePtr original_va = metas[i]->base;

    auto map_status =
        phi::dynload::cuMemMap(original_va, handle_size, 0, handles[i], 0);
    if (map_status != CUDA_SUCCESS) {
      VLOG(0) << "RestoreSourceMappings: cuMemMap(" << std::hex << original_va
              << std::dec << ") failed status=" << map_status
              << ", force-releasing handle";
      auto release_status = platform::RecordedGpuMemRelease(
          handles[i], handle_size, vmm_allocator->place().device);
      if (release_status == CUDA_SUCCESS) {
        vmm_allocator->MarkBackingReleased(
            original_va, handles[i], handle_size);
      }
      if (release_status != CUDA_SUCCESS) {
        VLOG(0) << "RestoreSourceMappings: force-release after cuMemMap "
                << "failure returned status=" << release_status;
      }
      force_released++;
      continue;
    }
    CUmemAccessDesc access_desc;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = vmm_allocator->place().device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    auto access_status =
        phi::dynload::cuMemSetAccess(original_va, handle_size, &access_desc, 1);
    if (access_status != CUDA_SUCCESS) {
      VLOG(0) << "RestoreSourceMappings: cuMemSetAccess failed for VA "
              << std::hex << original_va << std::dec
              << " status=" << access_status;
      phi::dynload::cuMemUnmap(original_va, handle_size);
      auto release_status = platform::RecordedGpuMemRelease(
          handles[i], handle_size, vmm_allocator->place().device);
      if (release_status == CUDA_SUCCESS) {
        vmm_allocator->MarkBackingReleased(
            original_va, handles[i], handle_size);
      }
      if (release_status != CUDA_SUCCESS) {
        VLOG(0) << "RestoreSourceMappings: force-release after cuMemSetAccess "
                << "failure returned status=" << release_status;
      }
      force_released++;
      continue;
    }
    vmm_allocator->MarkBackingMapped(original_va, handles[i], handle_size);
    if (RestoreGapToFree(blocks, original_va, handle_size, metas[i])) {
      metas[i]->remapped = false;
      restored++;
    } else {
      phi::dynload::cuMemUnmap(original_va, handle_size);
      vmm_allocator->MarkBackingUnmapped(original_va, handle_size);
      auto release_status = platform::RecordedGpuMemRelease(
          handles[i], handle_size, vmm_allocator->place().device);
      if (release_status == CUDA_SUCCESS) {
        vmm_allocator->MarkBackingReleased(
            original_va, handles[i], handle_size);
      }
      if (release_status != CUDA_SUCCESS) {
        VLOG(0) << "RestoreSourceMappings: force-release after block restore "
                << "failure returned status=" << release_status;
      }
      force_released++;
    }
  }
  MergeAdjacentFreeBlocksForRollback(blocks);
  VLOG(3) << "RestoreSourceMappings: restored=" << restored
          << " force_released=" << force_released;
}

}  // namespace

void RemapTransaction::PrepareCandidates(const VaRanges& source_ranges,
                                         const VaRanges& target_ranges,
                                         size_t target_bytes) {
  candidates_ = vmm_allocator_->CollectBackingCompactCandidates(
      source_ranges, target_ranges, target_bytes);
}

bool RemapTransaction::ValidateSourcePages(const char* context) const {
  return vmm_allocator_->ValidateMappedBackingPages(candidates_.source_pages,
                                                    context);
}

bool RemapTransaction::ValidateTargetPages(const char* context) const {
  return vmm_allocator_->ValidateUnmappedBackingPages(candidates_.target_pages,
                                                      context);
}

RemapTransaction::CandidateValidation RemapTransaction::ValidateCandidates(
    const char* source_context, const char* target_context) const {
  CandidateValidation validation;
  validation.source_ok = ValidateSourcePages(source_context);
  validation.target_ok = ValidateTargetPages(target_context);
  return validation;
}

void RemapTransaction::AddRollbackAction(std::function<void()> action) {
  rollback_actions_.push_back(std::move(action));
}

void RemapTransaction::AddSourceRestoreAction(
    std::list<BlockV2>* blocks,
    const std::vector<VmmAllocHandle>* handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  AddRollbackAction([=] {
    if (handles->empty()) {
      return;
    }
    RestoreSourceMappings(
        blocks, *handles, *metas, vmm_allocator_, handle_size_);
  });
}

void RemapTransaction::SetSyntheticAllocationSink(
    std::list<DecoratedAllocationPtr>* underlying_allocations) {
  underlying_allocations_ = underlying_allocations;
}

void RemapTransaction::MapHandlesToDestination(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  RecordDestinationRange(dst, handles.size());
  vmm_allocator_->MapHandlesToVA(dst, handles, metas);
}

void RemapTransaction::MapHandleRangeToDestination(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    size_t start,
    size_t count,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  std::vector<VmmAllocHandle> handle_slice(handles.begin() + start,
                                           handles.begin() + start + count);
  if (metas == nullptr) {
    MapHandlesToDestination(dst, handle_slice, nullptr);
    return;
  }

  std::vector<std::shared_ptr<VmmHandleMeta>> meta_slice(metas->begin() + start,
                                                         metas->begin() + start +
                                                             count);
  MapHandlesToDestination(dst, handle_slice, &meta_slice);
}

HandleLayout RemapTransaction::BuildDestinationLayout(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    size_t start,
    size_t count) const {
  HandleLayout layout;
  layout.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    layout.push_back(std::make_shared<VmmHandleMeta>(
        VmmHandleMeta{dst + i * handle_size_,
                      handle_size_,
                      handles[start + i],
                      vmm_allocator_->place().device}));
  }
  return layout;
}

RemapTransaction::MaterializedRange RemapTransaction::MaterializeMappedRange(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    size_t start,
    size_t count,
    PoolType pool_type) {
  MaterializedRange range;
  range.layout = BuildDestinationLayout(dst, handles, start, count);
  range.bytes = count * handle_size_;
  StageSyntheticAllocation(
      vmm_allocator_->CreateSyntheticAllocation(dst, range.bytes, range.layout));
  range.free_block.ptr_ = reinterpret_cast<void*>(dst);
  range.free_block.size_ = range.bytes;
  range.free_block.type_ = BlockType::kFree;
  range.free_block.pool_type_ = pool_type;
  range.free_block.parts_.reserve(range.layout.size());
  for (const auto& meta : range.layout) {
    range.free_block.parts_.push_back(BlockPartV2{meta, 0, handle_size_});
  }
  return range;
}

void RemapTransaction::RecordDestinationRange(VmmDevicePtr dst,
                                              size_t handle_count) {
  pending_destination_ranges_.push_back({dst, handle_count});
}

void RemapTransaction::Commit() {
  if (underlying_allocations_ != nullptr) {
    for (auto& allocation : pending_synthetic_allocations_) {
      underlying_allocations_->emplace_back(std::move(allocation));
    }
  }
  pending_synthetic_allocations_.clear();
  ClearPendingDestinations();
  rollback_actions_.clear();
  completed_ = true;
}

void RemapTransaction::UnmapPartialDestination(VmmDevicePtr dst_base,
                                               size_t handle_count) {
  for (size_t i = 0; i < handle_count; ++i) {
    vmm_allocator_->TryUnmapHandle(dst_base + i * handle_size_, handle_size_);
  }
}

void RemapTransaction::Rollback() {
  if (completed_) {
    return;
  }
  pending_synthetic_allocations_.clear();
  RollbackPendingDestinations();
  for (auto it = rollback_actions_.rbegin(); it != rollback_actions_.rend();
       ++it) {
    if (*it) {
      (*it)();
    }
  }
  rollback_actions_.clear();
  completed_ = true;
}

void RemapTransaction::RollbackPendingDestinations() {
  for (auto it = pending_destination_ranges_.rbegin();
       it != pending_destination_ranges_.rend();
       ++it) {
    VLOG(0) << "VMM V2 remap transaction: unmapping pending destination "
            << reinterpret_cast<void*>(it->dst)
            << " handles=" << it->handle_count;
    UnmapPartialDestination(it->dst, it->handle_count);
  }
  pending_destination_ranges_.clear();
}

void RemapTransaction::StageSyntheticAllocation(
    DecoratedAllocationPtr allocation) {
  pending_synthetic_allocations_.emplace_back(std::move(allocation));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
