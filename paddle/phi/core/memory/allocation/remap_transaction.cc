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

bool TryAppendPart(std::vector<BlockPartV2>* dst, const BlockPartV2& part) {
  if (dst->empty() || !dst->back().TryExtend(part)) {
    dst->push_back(part);
    return false;
  }
  return true;
}

void MergeAdjacentFreeBlocks(std::list<BlockV2>* blocks) {
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
        TryAppendPart(&it->parts_, part);
      }
      blocks->erase(next);
      continue;
    }
    ++it;
  }
}

void MergeAdjacentGaps(std::list<BlockV2>* blocks) {
  for (auto it = blocks->begin(); it != blocks->end();) {
    auto next = std::next(it);
    if (next != blocks->end() && it->type_ == BlockType::kGap &&
        next->type_ == BlockType::kGap &&
        reinterpret_cast<uint8_t*>(it->ptr_) + it->size_ ==
            reinterpret_cast<uint8_t*>(next->ptr_)) {
      it->size_ += next->size_;
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
  MergeAdjacentFreeBlocks(blocks);
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

bool RemapTransaction::TailIsUsable(VmmDevicePtr tail_va,
                                    size_t total_bytes,
                                    VmmDevicePtr va_limit) const {
  if (tail_va + total_bytes > va_limit) {
    return false;
  }
  for (size_t off = 0; off < total_bytes; off += handle_size_) {
    CUmemGenericAllocationHandle probe_handle;
    CUresult probe = phi::dynload::cuMemRetainAllocationHandle(
        &probe_handle, reinterpret_cast<void*>(tail_va + off));
    if (probe == CUDA_SUCCESS) {
      phi::dynload::cuMemRelease(probe_handle);
      VLOG(3) << "VMM V2 remap transaction: tail VA slot "
              << reinterpret_cast<void*>(tail_va + off) << " (offset " << off
              << "/" << total_bytes
              << ") is unexpectedly mapped, skipping tail path";
      return false;
    }
  }
  return true;
}

bool RemapTransaction::FindSingleGap(BlockList* blocks,
                                     size_t required_bytes,
                                     BlockIterator* gap_it) const {
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    if (it->type_ == BlockType::kGap && it->size_ >= required_bytes) {
      *gap_it = it;
      return true;
    }
  }
  return false;
}

size_t RemapTransaction::CollectGapCapacity(const BlockList& blocks) const {
  size_t total_gap_capacity = 0;
  for (const auto& blk : blocks) {
    if (blk.type_ == BlockType::kGap) {
      total_gap_capacity += (blk.size_ / handle_size_) * handle_size_;
    }
  }
  return total_gap_capacity;
}

bool RemapTransaction::PlanGapScatter(
    BlockList* blocks,
    size_t handle_count,
    std::vector<GapPlacement>* placements) const {
  placements->clear();
  size_t handle_idx = 0;
  for (auto it = blocks->begin(); it != blocks->end() && handle_idx < handle_count;
       ++it) {
    if (it->type_ != BlockType::kGap) continue;
    size_t gap_cap = it->size_ / handle_size_;
    size_t to_fill = std::min(gap_cap, handle_count - handle_idx);
    if (to_fill == 0) continue;
    placements->push_back(
        {it, reinterpret_cast<VmmDevicePtr>(it->ptr_), handle_idx, to_fill});
    handle_idx += to_fill;
  }
  return handle_idx == handle_count;
}

bool RemapTransaction::TryCommitTailPlacement(
    BlockList* blocks,
    VmmDevicePtr tail_va,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    PoolType pool_type) {
  try {
    MapHandlesToDestination(tail_va, handles, &metas);
  } catch (...) {
    VLOG(0) << "VMM V2 remap transaction: tail MapHandlesToVA failed";
    Rollback();
    return false;
  }

  auto mapped = MaterializeMappedRange(
      tail_va, handles, 0, metas.size(), pool_type);
  InstallTailFreeBlock(blocks, std::move(mapped.free_block));
  Commit();
  return true;
}

bool RemapTransaction::TryCommitSingleGapPlacement(
    BlockList* blocks,
    BlockIterator gap_it,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    PoolType pool_type) {
  VmmDevicePtr gap_va = reinterpret_cast<VmmDevicePtr>(gap_it->ptr_);
  try {
    MapHandlesToDestination(gap_va, handles, &metas);
  } catch (...) {
    VLOG(0) << "VMM V2 remap transaction: gap MapHandlesToVA failed";
    Rollback();
    return false;
  }

  auto mapped = MaterializeMappedRange(gap_va, handles, 0, metas.size(), pool_type);
  InstallMappedGapRange(blocks, gap_it, std::move(mapped.free_block), pool_type);
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

bool RemapTransaction::TryCommitGapScatter(
    BlockList* blocks,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    const std::vector<GapPlacement>& placements,
    PoolType pool_type) {
  for (const auto& p : placements) {
    try {
      MapHandleRangeToDestination(
          p.dst, handles, p.handle_start_idx, p.count, &metas);
    } catch (...) {
      VLOG(0) << "VMM V2 remap transaction: gap-scatter MapHandlesToVA failed at "
              << "handle_idx=" << p.handle_start_idx << "/" << handles.size();
      Rollback();
      return false;
    }
  }

  for (const auto& p : placements) {
    auto mapped = MaterializeMappedRange(
        p.dst, handles, p.handle_start_idx, p.count, pool_type);
    InstallMappedGapRange(
        blocks, p.gap_it, std::move(mapped.free_block), pool_type);
  }
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

RemapTransaction::PlacementResult RemapTransaction::ExecutePlacementStrategy(
    BlockList* blocks,
    VmmDevicePtr tail_va,
    VmmDevicePtr va_limit,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    PoolType pool_type) {
  PlacementResult result;
  const size_t total_remapped = handles.size() * handle_size_;

  if (TailIsUsable(tail_va, total_remapped, va_limit)) {
    VLOG(10) << "VMM remap compact using tail path, dst_va="
             << reinterpret_cast<void*>(tail_va)
             << " bytes=" << total_remapped;
    result.success =
        TryCommitTailPlacement(blocks, tail_va, handles, metas, pool_type);
    result.used_tail = result.success;
    return result;
  }

  BlockIterator gap_it = blocks->end();
  if (FindSingleGap(blocks, total_remapped, &gap_it)) {
    const VmmDevicePtr gap_va = reinterpret_cast<VmmDevicePtr>(gap_it->ptr_);
    VLOG(10) << "VMM remap compact using gap path, dst_va="
             << reinterpret_cast<void*>(gap_va)
             << " gap_size=" << gap_it->size_ << " bytes=" << total_remapped;
    result.success =
        TryCommitSingleGapPlacement(blocks, gap_it, handles, metas, pool_type);
    return result;
  }

  VLOG(3) << "VMM V2 remap transaction: tail unavailable and no single gap >= "
          << total_remapped << " bytes, falling back to gap-scatter remap";

  size_t total_gap_capacity = CollectGapCapacity(*blocks);
  if (total_gap_capacity < total_remapped) {
    VLOG(0) << "VMM V2 remap transaction: gap capacity " << total_gap_capacity
            << " < total_remapped " << total_remapped
            << ", rolling back to original VA";
    Rollback();
    return result;
  }

  std::vector<GapPlacement> placements;
  bool planned = PlanGapScatter(blocks, handles.size(), &placements);
  if (!planned) {
    size_t planned_handles = 0;
    for (const auto& p : placements) {
      planned_handles += p.count;
    }
    VLOG(0) << "VMM V2 remap transaction gap-scatter: placed "
            << planned_handles << " of " << handles.size()
            << " handles despite precheck; rolling back";
    Rollback();
    return result;
  }

  result.success = TryCommitGapScatter(blocks, handles, metas, placements, pool_type);
  return result;
}

void RemapTransaction::InstallTailFreeBlock(BlockList* blocks,
                                            BlockV2 free_block) const {
  if (!blocks->empty()) {
    auto last = std::prev(blocks->end());
    if (last->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(last->ptr_) + last->size_ ==
            reinterpret_cast<uint8_t*>(free_block.ptr_)) {
      last->size_ += free_block.size_;
      for (const auto& part : free_block.parts_) {
        TryAppendPart(&last->parts_, part);
      }
      return;
    }
  }
  blocks->push_back(std::move(free_block));
}

RemapTransaction::BlockIterator RemapTransaction::InstallMappedGapRange(
    BlockList* blocks,
    BlockIterator gap_it,
    BlockV2 free_block,
    PoolType pool_type) const {
  VmmDevicePtr gap_va = reinterpret_cast<VmmDevicePtr>(gap_it->ptr_);
  size_t gap_size = gap_it->size_;
  size_t filled_bytes = free_block.size_;

  if (gap_size == filled_bytes) {
    *gap_it = std::move(free_block);
  } else {
    BlockV2 remaining_gap;
    remaining_gap.ptr_ = reinterpret_cast<void*>(gap_va + filled_bytes);
    remaining_gap.size_ = gap_size - filled_bytes;
    remaining_gap.type_ = BlockType::kGap;
    remaining_gap.pool_type_ = pool_type;
    *gap_it = std::move(free_block);
    blocks->insert(std::next(gap_it), std::move(remaining_gap));
  }

  auto result = gap_it;
  if (result != blocks->begin()) {
    auto prev = std::prev(result);
    if (prev->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(prev->ptr_) + prev->size_ ==
            reinterpret_cast<uint8_t*>(result->ptr_)) {
      prev->size_ += result->size_;
      for (const auto& part : result->parts_) {
        TryAppendPart(&prev->parts_, part);
      }
      blocks->erase(result);
      result = prev;
    }
  }

  auto next = std::next(result);
  if (next != blocks->end() && next->type_ == BlockType::kFree &&
      reinterpret_cast<uint8_t*>(result->ptr_) + result->size_ ==
          reinterpret_cast<uint8_t*>(next->ptr_)) {
    result->size_ += next->size_;
    for (const auto& part : next->parts_) {
      TryAppendPart(&result->parts_, part);
    }
    blocks->erase(next);
  }
  return result;
}

void RemapTransaction::NormalizeBlocks(BlockList* blocks) const {
  MergeAdjacentFreeBlocks(blocks);
  MergeAdjacentGaps(blocks);
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
