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
#include <map>
#include <utility>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
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

std::vector<std::pair<VmmDevicePtr, size_t>> CollectFreeRanges(
    const std::list<BlockV2>& blocks) {
  std::vector<std::pair<VmmDevicePtr, size_t>> ranges;
  for (const auto& block : blocks) {
    if (block.type_ != BlockType::kFree || block.ipc_exported_) {
      continue;
    }
    ranges.emplace_back(reinterpret_cast<VmmDevicePtr>(block.ptr_),
                        block.size_);
  }
  return ranges;
}

std::vector<std::pair<VmmDevicePtr, size_t>> CollectGapRanges(
    const std::list<BlockV2>& blocks) {
  std::vector<std::pair<VmmDevicePtr, size_t>> ranges;
  for (const auto& block : blocks) {
    if (block.type_ != BlockType::kGap) {
      continue;
    }
    ranges.emplace_back(reinterpret_cast<VmmDevicePtr>(block.ptr_),
                        block.size_);
  }
  return ranges;
}

void AppendGapOrFreeSegment(std::vector<BlockV2>* segments,
                            BlockType type,
                            size_t size,
                            const BlockPartV2* part,
                            void* ptr,
                            PoolType pool_type) {
  if (size == 0) {
    return;
  }

  if (!segments->empty() && segments->back().type_ == type) {
    segments->back().size_ += size;
    if (part != nullptr) {
      TryAppendPart(&segments->back().parts_, *part);
    }
    return;
  }

  BlockV2 segment;
  segment.ptr_ = ptr;
  segment.size_ = size;
  segment.type_ = type;
  segment.pool_type_ = pool_type;
  if (part != nullptr) {
    segment.parts_.push_back(*part);
  }
  segments->push_back(std::move(segment));
}

using EventReadyCache =
    std::map<std::shared_ptr<CudaEventGuard>,
             bool,
             std::owner_less<std::shared_ptr<CudaEventGuard>>>;

bool IsFullyCoveredHandle(const BlockPartV2& part, EventReadyCache* cache) {
  if (part.handle->remapped) {
    return false;
  }
  if (part.handle_rel_off != 0 || part.len != part.handle->size) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (part.handle->remap_safe_event) {
    auto event = part.handle->remap_safe_event;
    auto it = cache->find(event);
    bool ready = false;
    if (it != cache->end()) {
      ready = it->second;
    } else {
#ifdef PADDLE_WITH_CUDA
      gpuError_t err = cudaEventQuery(event->event);
      if (err != gpuSuccess && err != cudaErrorNotReady) {
        PADDLE_ENFORCE_GPU_SUCCESS(err);
      }
#else
      gpuError_t err = hipEventQuery(event->event);
      if (err != gpuSuccess && err != hipErrorNotReady) {
        PADDLE_ENFORCE_GPU_SUCCESS(err);
      }
#endif
      ready = (err == gpuSuccess);
      cache->emplace(event, ready);
    }
    if (!ready) {
      return false;
    }
    part.handle->remap_safe_event.reset();
  }
#endif
  return true;
}

bool IsRemapSafe(const BlockV2& block) {
  return block.type_ == BlockType::kFree && !block.ipc_exported_;
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

RemapTransaction::PreScanResult RemapTransaction::PreparePhase1Diagnostics(
    BlockList* blocks,
    size_t requested_size,
    const char* source_context,
    const char* target_context) {
  PreScanResult result;
  auto free_ranges = CollectFreeRanges(*blocks);
  auto gap_ranges = CollectGapRanges(*blocks);
  result.free_range_count = free_ranges.size();
  result.gap_range_count = gap_ranges.size();
  PrepareCandidates(free_ranges, gap_ranges, requested_size);
  result.mapped_page_count = candidates_.source_pages.size();
  result.target_page_count = candidates_.target_pages.size();
  auto validation = ValidateCandidates(source_context, target_context);
  result.source_ok = validation.source_ok;
  result.target_ok = validation.target_ok;
  return result;
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

RemapTransaction::SourceCollectionStats RemapTransaction::CollectRemapSources(
    BlockList* blocks,
    size_t requested_size,
    PoolType pool_type,
    std::vector<VmmAllocHandle>* handles,
    std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  SourceCollectionStats stats;
  EventReadyCache event_ready_cache;
  bool logged_first_candidate = false;
  for (auto it = blocks->begin(); it != blocks->end();) {
    auto current = it++;
    if (current->type_ == BlockType::kFree) stats.free_block_count++;
    if (!IsRemapSafe(*current)) {
      continue;
    }
    stats.safe_block_count++;

    std::vector<BlockV2> replacement_segments;
    size_t block_offset = 0;
    size_t remapped_count_before = handles->size();
    for (const auto& part : current->parts_) {
      void* part_ptr =
          reinterpret_cast<uint8_t*>(current->ptr_) + block_offset;
      block_offset += part.len;
      if (IsFullyCoveredHandle(part, &event_ready_cache)) {
        stats.fully_covered_count++;
        stats.fully_covered_bytes += part.len;
        if (!logged_first_candidate) {
          VLOG(0) << "First remap candidate pool="
                  << static_cast<int>(pool_type)
                  << " block_ptr=" << current->ptr_
                  << " block_size=" << current->size_ << " handle_base="
                  << reinterpret_cast<void*>(part.handle->base)
                  << " handle_size=" << part.handle->size << " handle="
                  << reinterpret_cast<void*>(part.handle->handle);
          logged_first_candidate = true;
        }
        if (!vmm_allocator_->TryUnmapHandle(part.handle->base, part.len)) {
          VLOG(0) << "VMM V2 remap transaction: TryUnmapHandle failed, "
                     "skipping handle "
                  << reinterpret_cast<void*>(part.handle->base);
          AppendGapOrFreeSegment(&replacement_segments,
                                 BlockType::kFree,
                                 part.len,
                                 &part,
                                 part_ptr,
                                 pool_type);
          continue;
        }
        part.handle->remapped = true;
        handles->push_back(part.handle->handle);
        metas->push_back(part.handle);
        AppendGapOrFreeSegment(&replacement_segments,
                               BlockType::kGap,
                               part.len,
                               nullptr,
                               part_ptr,
                               pool_type);
        continue;
      }
      if (part.handle->remapped) {
        stats.remapped_blocked_count++;
        stats.remapped_blocked_bytes += part.len;
      }
      if (part.handle->remap_safe_event) {
        stats.event_blocked_count++;
        stats.event_blocked_bytes += part.len;
      } else {
        stats.partial_count++;
        stats.partial_bytes += part.len;
      }
      AppendGapOrFreeSegment(&replacement_segments,
                             BlockType::kFree,
                             part.len,
                             &part,
                             part_ptr,
                             pool_type);
    }

    if (handles->size() == remapped_count_before) {
      continue;
    }

    auto insert_pos = current;
    for (auto& segment : replacement_segments) {
      blocks->insert(insert_pos, std::move(segment));
    }
    blocks->erase(current);

    if (requested_size > 0 && handles->size() * handle_size_ >= requested_size) {
      VLOG(3) << "VMM V2 remap transaction: bounded exit, collected "
              << handles->size() << " handles (" << handles->size() * handle_size_
              << " bytes) >= requested=" << requested_size;
      break;
    }
  }
  return stats;
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

RemapTransaction::CompactResult RemapTransaction::CompactFreeBlocks(
    BlockList* blocks, size_t requested_size, PoolType pool_type) {
  CompactResult result;
  std::vector<VmmAllocHandle> remapped_handles;
  std::vector<std::shared_ptr<VmmHandleMeta>> remapped_metas;

  AddSourceRestoreAction(blocks, &remapped_handles, &remapped_metas);
  result.source_stats = CollectRemapSources(
      blocks, requested_size, pool_type, &remapped_handles, &remapped_metas);
  result.remapped_handle_count = remapped_handles.size();
  result.remapped_bytes = remapped_handles.size() * handle_size_;
  if (remapped_handles.empty()) {
    return result;
  }

  NormalizeBlocks(blocks);

  VmmDevicePtr tail_va = vmm_allocator_->virtual_mem_base();
  if (!blocks->empty()) {
    const auto& last = blocks->back();
    tail_va = reinterpret_cast<VmmDevicePtr>(
        reinterpret_cast<uint8_t*>(last.ptr_) + last.size_);
  }
  const VmmDevicePtr va_limit =
      vmm_allocator_->virtual_mem_base() + vmm_allocator_->virtual_mem_size();

  auto placement = ExecutePlacementStrategy(
      blocks, tail_va, va_limit, remapped_handles, remapped_metas, pool_type);
  result.success = placement.success;
  result.used_tail = placement.used_tail;
  if (placement.success && placement.used_tail) {
    vmm_allocator_->AdvanceTailOffset(result.remapped_bytes);
  }
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
