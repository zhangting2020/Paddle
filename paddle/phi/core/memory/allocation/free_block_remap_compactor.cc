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

#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
  // Skip handles that were already remapped by a previous compact — their
  // physical memory is owned by a synthetic allocation at a different VA.
  // Attempting to remap them again would use a stale or released handle.
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

BlockV2 CreateTailFreeBlock(
    VmmDevicePtr dst_va,
    size_t total_remapped,
    PoolType pool_type,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& remapped_metas,
    size_t handle_size) {
  BlockV2 free_block;
  free_block.ptr_ = reinterpret_cast<void*>(dst_va);
  free_block.size_ = total_remapped;
  free_block.type_ = BlockType::kFree;
  free_block.pool_type_ = pool_type;
  free_block.parts_.reserve(remapped_metas.size());
  for (size_t i = 0; i < remapped_metas.size(); ++i) {
    remapped_metas[i]->base = dst_va + i * handle_size;
    free_block.parts_.push_back(BlockPartV2{remapped_metas[i], 0, handle_size});
  }
  return free_block;
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

// Maps each handle back to its original VA (meta->base) and restores
// the corresponding GAP block to FREE in the block list.
// Invariant: meta->base was unmapped in Phase 1 and is currently a GAP.
void RollbackToOriginalVA(
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
      VLOG(0) << "RollbackToOriginalVA: cuMemMap(" << std::hex << original_va
              << std::dec << ") failed status=" << map_status
              << ", force-releasing handle";
      auto release_status = platform::RecordedGpuMemRelease(
          handles[i], handle_size, vmm_allocator->place().device);
      if (release_status == CUDA_SUCCESS) {
        vmm_allocator->MarkBackingReleased(
            original_va, handles[i], handle_size);
      }
      if (release_status != CUDA_SUCCESS) {
        VLOG(0) << "RollbackToOriginalVA: force-release after cuMemMap "
                << "failure returned status=" << release_status;
      }
      // Keep remapped=true so FreeImpl skips this already-released handle.
      force_released++;
      continue;
    }
    // Set access permissions.
    CUmemAccessDesc access_desc;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = vmm_allocator->place().device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    auto access_status =
        phi::dynload::cuMemSetAccess(original_va, handle_size, &access_desc, 1);
    if (access_status != CUDA_SUCCESS) {
      VLOG(0) << "RollbackToOriginalVA: cuMemSetAccess failed for VA "
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
        VLOG(0) << "RollbackToOriginalVA: force-release after cuMemSetAccess "
                << "failure returned status=" << release_status;
      }
      // Keep remapped=true so FreeImpl skips this already-released handle.
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
        VLOG(0) << "RollbackToOriginalVA: force-release after block restore "
                << "failure returned status=" << release_status;
      }
      // Keep remapped=true so FreeImpl skips this handle.
      force_released++;
    }
  }
  MergeAdjacentFreeBlocks(blocks);
  VLOG(3) << "RollbackToOriginalVA: restored=" << restored
          << " force_released=" << force_released;
}

}  // namespace

size_t FreeBlockRemapCompactor::Compact(std::list<BlockV2>* blocks,
                                        size_t requested_size) {
  std::vector<VmmAllocHandle> remapped_handles;
  std::vector<std::shared_ptr<VmmHandleMeta>> remapped_metas;
  bool logged_first_candidate = false;
  const size_t handle_size = vmm_allocator_->handle_size();
  RemapTransaction transaction(vmm_allocator_.get(), handle_size);
  transaction.SetSourceRollbackAction([&] {
    if (!remapped_handles.empty()) {
      RollbackToOriginalVA(blocks,
                           remapped_handles,
                           remapped_metas,
                           vmm_allocator_.get(),
                           handle_size);
    }
  });

  LOG(INFO) << "VMM V2 compactor: entering Compact, blocks=" << blocks->size()
            << " handle_size=" << handle_size;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Clear any sticky CUDA error before we start.
  cudaGetLastError();
  // No cudaDeviceSynchronize here — IsRemapSafe uses per-event
  // cudaEventQuery to check individual blocks, avoiding a full
  // pipeline stall.
#endif

  // The entire compact is wrapped in try-catch.  If ANY CUDA API call
  // fails (e.g. cuMemUnmap during Phase 1, cuMemMap during Phase 2),
  // we rollback all unmapped handles via gap-scatter so that the block
  // list stays consistent for subsequent FreeImpl/TryMerge calls.
  try {
    // ---- Phase 1: Unmap fully-covered handles from FREE blocks ----
    LOG(INFO) << "VMM V2 compactor: Phase 1 - scanning FREE blocks for "
              << "fully-covered handles";
    if (VLOG_IS_ON(4)) {
      const auto free_ranges = CollectFreeRanges(*blocks);
      const auto gap_ranges = CollectGapRanges(*blocks);
      transaction.PrepareCandidates(free_ranges, gap_ranges, requested_size);
      const auto& candidates = transaction.candidates();
      const auto validation = transaction.ValidateCandidates(
          "FreeBlockRemapCompactor::pre_phase1",
          "FreeBlockRemapCompactor::pre_phase1_target");
      VLOG(4) << "VMM V2 compactor BackingMap pre-scan pool="
              << static_cast<int>(pool_type_)
              << " free_ranges=" << free_ranges.size()
              << " gap_ranges=" << gap_ranges.size()
              << " mapped_pages=" << candidates.source_pages.size()
              << " mapped_bytes="
              << candidates.source_pages.size() * handle_size
              << " target_unmapped_pages=" << candidates.target_pages.size()
              << " target_unmapped_bytes="
              << candidates.target_pages.size() * handle_size
              << " requested=" << requested_size
              << " snapshot_ok=" << validation.source_ok
              << " target_snapshot_ok=" << validation.target_ok;
    }
    size_t free_block_count = 0, safe_block_count = 0;
    size_t fully_covered_count = 0, partial_count = 0;
    size_t event_blocked_count = 0;
    size_t fully_covered_bytes = 0, partial_bytes = 0;
    size_t event_blocked_bytes = 0;
    size_t remapped_blocked_count = 0, remapped_blocked_bytes = 0;
    EventReadyCache event_ready_cache;
    for (auto it = blocks->begin(); it != blocks->end();) {
      auto current = it++;
      if (current->type_ == BlockType::kFree) free_block_count++;
      if (!IsRemapSafe(*current)) {
        continue;
      }
      safe_block_count++;

      std::vector<BlockV2> replacement_segments;
      size_t block_offset = 0;
      size_t remapped_count_before = remapped_handles.size();
      for (const auto& part : current->parts_) {
        void* part_ptr =
            reinterpret_cast<uint8_t*>(current->ptr_) + block_offset;
        block_offset += part.len;
        if (IsFullyCoveredHandle(part, &event_ready_cache)) {
          fully_covered_count++;
          fully_covered_bytes += part.len;
          if (!logged_first_candidate) {
            VLOG(0) << "First remap candidate pool="
                    << static_cast<int>(pool_type_)
                    << " block_ptr=" << current->ptr_
                    << " block_size=" << current->size_ << " handle_base="
                    << reinterpret_cast<void*>(part.handle->base)
                    << " handle_size=" << part.handle->size << " handle="
                    << reinterpret_cast<void*>(part.handle->handle);
            logged_first_candidate = true;
          }
          // Use TryUnmapHandle (non-throwing) so that a cuMemUnmap failure
          // does not corrupt the block list mid-iteration.  If unmap fails,
          // skip this handle and keep the block as-is (FREE, not GAP).
          if (!vmm_allocator_->TryUnmapHandle(part.handle->base, part.len)) {
            VLOG(0)
                << "VMM V2 compactor: TryUnmapHandle failed, skipping handle "
                << reinterpret_cast<void*>(part.handle->base);
            AppendGapOrFreeSegment(&replacement_segments,
                                   BlockType::kFree,
                                   part.len,
                                   &part,
                                   part_ptr,
                                   pool_type_);
            continue;
          }
          // Mark the handle as remapped so that FreeImpl (called when
          // FreeIdleChunks releases the original underlying allocation)
          // skips cuMemUnmap+cuMemRelease for this handle — its physical
          // memory is now owned by the destination block.
          part.handle->remapped = true;
          remapped_handles.push_back(part.handle->handle);
          remapped_metas.push_back(part.handle);
          AppendGapOrFreeSegment(&replacement_segments,
                                 BlockType::kGap,
                                 part.len,
                                 nullptr,
                                 part_ptr,
                                 pool_type_);
          continue;
        }
        if (part.handle->remapped) {
          remapped_blocked_count++;
          remapped_blocked_bytes += part.len;
        }
        if (part.handle->remap_safe_event) {
          event_blocked_count++;
          event_blocked_bytes += part.len;
        } else {
          partial_count++;
          partial_bytes += part.len;
        }

        AppendGapOrFreeSegment(&replacement_segments,
                               BlockType::kFree,
                               part.len,
                               &part,
                               part_ptr,
                               pool_type_);
      }

      if (remapped_handles.size() == remapped_count_before) {
        continue;
      }

      auto insert_pos = current;
      for (auto& segment : replacement_segments) {
        blocks->insert(insert_pos, std::move(segment));
      }
      blocks->erase(current);

      // Bounded compact: stop collecting handles once we have enough to
      // satisfy the requested allocation.  This avoids unnecessary remap
      // work beyond what is needed for this OOM retry.
      if (requested_size > 0 &&
          remapped_handles.size() * handle_size >= requested_size) {
        VLOG(3) << "VMM V2 compactor: bounded exit, collected "
                << remapped_handles.size() << " handles ("
                << remapped_handles.size() * handle_size
                << " bytes) >= requested=" << requested_size;
        break;
      }
    }

    // Log per-handle coverage to diagnose why handles are partial.
    // A handle is "fully covered" when a single part spans the entire
    // handle (off=0, len=handle->size).  Partial handles are split
    // between ACTIVE and FREE blocks — the ACTIVE portion prevents remap.
    LOG(INFO) << "VMM V2 compactor pool=" << static_cast<int>(pool_type_)
              << " Phase 1 stats: free_blocks=" << free_block_count
              << " safe_blocks=" << safe_block_count
              << " event_blocked=" << event_blocked_count
              << " event_blocked_bytes=" << event_blocked_bytes
              << " fully_covered_parts=" << fully_covered_count
              << " fully_covered_bytes=" << fully_covered_bytes
              << " partial_parts=" << partial_count
              << " partial_bytes=" << partial_bytes
              << " remapped_blocked=" << remapped_blocked_count
              << " remapped_blocked_bytes=" << remapped_blocked_bytes;
    // Log details of first few partial parts for debugging.
    if (partial_count > 0 && fully_covered_count == 0) {
      size_t logged = 0;
      for (auto& block : *blocks) {
        if (block.type_ != BlockType::kFree || logged >= 5) break;
        for (const auto& part : block.parts_) {
          if (!IsFullyCoveredHandle(part, &event_ready_cache) && logged < 5) {
            const bool is_partial =
                !(part.handle_rel_off == 0 && part.len == part.handle->size);
            const char* reason =
                part.handle->remapped
                    ? "remapped"
                    : (part.handle->remap_safe_event
                           ? "event_blocked"
                           : (is_partial ? "partial" : "other"));
            LOG(INFO) << "  partial part: block_ptr=" << block.ptr_
                      << " block_size=" << block.size_ << " handle_base="
                      << reinterpret_cast<void*>(part.handle->base)
                      << " handle_size=" << part.handle->size
                      << " part_off=" << part.handle_rel_off
                      << " part_len=" << part.len
                      << " coverage=" << (part.len * 100 / part.handle->size)
                      << "% reason=" << reason << " has_event="
                      << (part.handle->remap_safe_event != nullptr)
                      << " remapped=" << part.handle->remapped;
            logged++;
          }
        }
      }
    }
    if (remapped_handles.empty()) {
      LOG(INFO) << "VMM V2 compactor: Phase 1 done, no handles to remap"
                << " (safe_blocks=" << safe_block_count
                << " fully_covered=" << fully_covered_count << ")";
      return 0;
    }

    MergeAdjacentGaps(blocks);

    const size_t total_remapped = remapped_handles.size() * handle_size;
    LOG(INFO) << "VMM V2 compactor: Phase 1 done, " << remapped_handles.size()
              << " handles (" << total_remapped << " bytes) unmapped";

    // -------------------------------------------------------------------
    // Compute the real tail VA from the block list.
    // -------------------------------------------------------------------
    VmmDevicePtr tail_va = vmm_allocator_->virtual_mem_base();
    if (!blocks->empty()) {
      const auto& last = blocks->back();
      tail_va = reinterpret_cast<VmmDevicePtr>(
          reinterpret_cast<uint8_t*>(last.ptr_) + last.size_);
    }
    const VmmDevicePtr va_limit =
        vmm_allocator_->virtual_mem_base() + vmm_allocator_->virtual_mem_size();
    VLOG(10) << "VMM remap compact pool=" << static_cast<int>(pool_type_)
             << " remapped_handles=" << remapped_handles.size()
             << " total_remapped=" << total_remapped
             << " handle_size=" << handle_size
             << " tail_va=" << reinterpret_cast<void*>(tail_va)
             << " va_limit=" << reinterpret_cast<void*>(va_limit);

    // -------------------------------------------------------------------
    // Phase 2: Remap handles to destination VA.
    // If remap fails, map each handle back to its original VA (meta->base).
    // -------------------------------------------------------------------
    auto rollback = [&] {
      VLOG(0) << "VMM V2 compactor Phase 2 failed, rolling back "
              << remapped_handles.size() << " handles to original VA";
      transaction.Rollback();
    };

    bool tail_usable = (tail_va + total_remapped <= va_limit);

    // Safety probe: verify tail range is unmapped.
    if (tail_usable) {
      for (size_t off = 0; off < total_remapped; off += handle_size) {
        CUmemGenericAllocationHandle probe_handle;
        CUresult probe = phi::dynload::cuMemRetainAllocationHandle(
            &probe_handle, reinterpret_cast<void*>(tail_va + off));
        if (probe == CUDA_SUCCESS) {
          phi::dynload::cuMemRelease(probe_handle);
          tail_usable = false;
          VLOG(3) << "VMM V2 compactor: tail VA slot "
                  << reinterpret_cast<void*>(tail_va + off) << " (offset "
                  << off << "/" << total_remapped
                  << ") is unexpectedly mapped, skipping tail path";
          break;
        }
      }
    }

    // ---- Path 1: tail path ----
    if (tail_usable) {
      VLOG(10) << "VMM remap compact using tail path, dst_va="
               << reinterpret_cast<void*>(tail_va)
               << " bytes=" << total_remapped;
      try {
        transaction.MapHandlesToDestination(
            tail_va, remapped_handles, &remapped_metas);
      } catch (...) {
        VLOG(0) << "VMM V2 compactor: tail MapHandlesToVA failed";
        rollback();
        return 0;
      }

      // Register a synthetic allocation so FreeIdleChunks can release
      // these handles when the tail block becomes entirely free.
      HandleLayout tail_layout;
      for (size_t i = 0; i < remapped_metas.size(); ++i) {
        tail_layout.push_back(std::make_shared<VmmHandleMeta>(
            VmmHandleMeta{tail_va + i * handle_size,
                          handle_size,
                          remapped_handles[i],
                          vmm_allocator_->place().device}));
      }
      auto synth = vmm_allocator_->CreateSyntheticAllocation(
          tail_va, total_remapped, tail_layout);
      underlying_allocations_->emplace_back(std::move(synth));

      BlockV2 tail_free = CreateTailFreeBlock(
          tail_va, total_remapped, pool_type_, tail_layout, handle_size);

      if (!blocks->empty()) {
        auto last = std::prev(blocks->end());
        if (last->type_ == BlockType::kFree &&
            reinterpret_cast<uint8_t*>(last->ptr_) + last->size_ ==
                reinterpret_cast<uint8_t*>(tail_free.ptr_)) {
          last->size_ += tail_free.size_;
          for (const auto& part : tail_free.parts_) {
            TryAppendPart(&last->parts_, part);
          }
          vmm_allocator_->AdvanceTailOffset(total_remapped);
          transaction.Commit();
          return total_remapped;
        }
      }
      blocks->push_back(std::move(tail_free));
      vmm_allocator_->AdvanceTailOffset(total_remapped);
      transaction.Commit();
      return total_remapped;
    }

    // ---- Path 2: single-gap path ----
    auto gap_it = blocks->end();
    for (auto it = blocks->begin(); it != blocks->end(); ++it) {
      if (it->type_ == BlockType::kGap && it->size_ >= total_remapped) {
        gap_it = it;
        break;
      }
    }

    if (gap_it != blocks->end()) {
      const VmmDevicePtr gap_va = reinterpret_cast<VmmDevicePtr>(gap_it->ptr_);
      VLOG(10) << "VMM remap compact using gap path, dst_va="
               << reinterpret_cast<void*>(gap_va)
               << " gap_size=" << gap_it->size_ << " bytes=" << total_remapped;
      try {
        transaction.MapHandlesToDestination(
            gap_va, remapped_handles, &remapped_metas);
      } catch (...) {
        VLOG(0) << "VMM V2 compactor: gap MapHandlesToVA failed";
        rollback();
        return 0;
      }

      // Register synthetic allocation for gap-remapped handles.
      HandleLayout gap_layout;
      for (size_t i = 0; i < remapped_metas.size(); ++i) {
        gap_layout.push_back(std::make_shared<VmmHandleMeta>(
            VmmHandleMeta{gap_va + i * handle_size,
                          handle_size,
                          remapped_handles[i],
                          vmm_allocator_->place().device}));
      }
      auto synth = vmm_allocator_->CreateSyntheticAllocation(
          gap_va, total_remapped, gap_layout);
      underlying_allocations_->emplace_back(std::move(synth));

      BlockV2 free_block = CreateTailFreeBlock(
          gap_va, total_remapped, pool_type_, gap_layout, handle_size);
      if (gap_it->size_ == total_remapped) {
        *gap_it = std::move(free_block);
      } else {
        BlockV2 remaining_gap;
        remaining_gap.ptr_ = reinterpret_cast<void*>(gap_va + total_remapped);
        remaining_gap.size_ = gap_it->size_ - total_remapped;
        remaining_gap.type_ = BlockType::kGap;
        remaining_gap.pool_type_ = pool_type_;
        gap_it->size_ = total_remapped;
        *gap_it = std::move(free_block);
        blocks->insert(std::next(gap_it), std::move(remaining_gap));
      }

      if (gap_it != blocks->begin()) {
        auto prev = std::prev(gap_it);
        if (prev->type_ == BlockType::kFree &&
            reinterpret_cast<uint8_t*>(prev->ptr_) + prev->size_ ==
                reinterpret_cast<uint8_t*>(gap_it->ptr_)) {
          prev->size_ += gap_it->size_;
          for (const auto& part : gap_it->parts_) {
            TryAppendPart(&prev->parts_, part);
          }
          blocks->erase(gap_it);
          gap_it = prev;
        }
      }
      if (gap_it != blocks->end()) {
        auto next = std::next(gap_it);
        if (next != blocks->end() && next->type_ == BlockType::kFree &&
            reinterpret_cast<uint8_t*>(gap_it->ptr_) + gap_it->size_ ==
                reinterpret_cast<uint8_t*>(next->ptr_)) {
          gap_it->size_ += next->size_;
          for (const auto& part : next->parts_) {
            TryAppendPart(&gap_it->parts_, part);
          }
          blocks->erase(next);
        }
      }
      MergeAdjacentGaps(blocks);
      transaction.Commit();
      return total_remapped;
    }

    // ---- Path 3: gap-scatter (two-phase commit) ----
    VLOG(3) << "VMM V2 compactor: tail unavailable and no single gap >= "
            << total_remapped << " bytes, falling back to gap-scatter remap";

    // Capacity precheck: verify total GAP can hold all handles.
    size_t total_gap_capacity = 0;
    for (const auto& blk : *blocks) {
      if (blk.type_ == BlockType::kGap) {
        total_gap_capacity += (blk.size_ / handle_size) * handle_size;
      }
    }
    if (total_gap_capacity < total_remapped) {
      VLOG(0) << "VMM V2 compactor: gap capacity " << total_gap_capacity
              << " < total_remapped " << total_remapped
              << ", rolling back to original VA";
      transaction.Rollback();
      return 0;
    }

    // Phase 3a: tentative placement (cuMemMap only, no bookkeeping).
    struct GapPlacement {
      std::list<BlockV2>::iterator gap_it;
      VmmDevicePtr dst;
      size_t handle_start_idx;
      size_t count;
    };
    std::vector<GapPlacement> placements;
    size_t handle_idx = 0;

    for (auto it = blocks->begin();
         it != blocks->end() && handle_idx < remapped_handles.size();
         ++it) {
      if (it->type_ != BlockType::kGap) continue;
      size_t gap_cap = it->size_ / handle_size;
      size_t to_fill = std::min(gap_cap, remapped_handles.size() - handle_idx);
      if (to_fill == 0) continue;

      VmmDevicePtr dst = reinterpret_cast<VmmDevicePtr>(it->ptr_);
      std::vector<VmmAllocHandle> chunk(
          remapped_handles.begin() + handle_idx,
          remapped_handles.begin() + handle_idx + to_fill);
      std::vector<std::shared_ptr<VmmHandleMeta>> chunk_metas(
          remapped_metas.begin() + handle_idx,
          remapped_metas.begin() + handle_idx + to_fill);

      try {
        transaction.MapHandlesToDestination(dst, chunk, &chunk_metas);
      } catch (...) {
        VLOG(0) << "VMM V2 compactor: gap-scatter MapHandlesToVA failed at "
                << "handle_idx=" << handle_idx << "/"
                << remapped_handles.size();
        transaction.Rollback();
        return 0;
      }

      placements.push_back({it, dst, handle_idx, to_fill});
      handle_idx += to_fill;
    }

    // Defensive: capacity precheck should prevent this.
    if (handle_idx != remapped_handles.size()) {
      VLOG(0) << "VMM V2 compactor gap-scatter: placed " << handle_idx << " of "
              << remapped_handles.size()
              << " handles despite precheck; rolling back";
      transaction.Rollback();
      return 0;
    }

    // Phase 3b: commit — all handles mapped successfully.
    for (auto& p : placements) {
      auto it = p.gap_it;
      size_t filled_bytes = p.count * handle_size;

      HandleLayout chunk_layout;
      for (size_t i = 0; i < p.count; ++i) {
        chunk_layout.push_back(std::make_shared<VmmHandleMeta>(
            VmmHandleMeta{p.dst + i * handle_size,
                          handle_size,
                          remapped_handles[p.handle_start_idx + i],
                          vmm_allocator_->place().device}));
      }
      auto synth = vmm_allocator_->CreateSyntheticAllocation(
          p.dst, filled_bytes, chunk_layout);
      underlying_allocations_->emplace_back(std::move(synth));

      it->type_ = BlockType::kFree;
      it->parts_.clear();
      for (size_t i = 0; i < p.count; ++i) {
        it->parts_.push_back(BlockPartV2{chunk_layout[i], 0, handle_size});
      }

      if (filled_bytes < it->size_) {
        BlockV2 leftover_gap;
        leftover_gap.ptr_ = reinterpret_cast<void*>(p.dst + filled_bytes);
        leftover_gap.size_ = it->size_ - filled_bytes;
        leftover_gap.type_ = BlockType::kGap;
        leftover_gap.pool_type_ = pool_type_;
        it->size_ = filled_bytes;
        blocks->insert(std::next(it), std::move(leftover_gap));
      }
    }

    MergeAdjacentFreeBlocks(blocks);
    MergeAdjacentGaps(blocks);
    transaction.Commit();
    return total_remapped;
  } catch (...) {
    VLOG(0)
        << "VMM V2 compactor: exception caught during Compact, rolling back "
        << remapped_handles.size() << " unmapped handles";
    transaction.Rollback();
    return 0;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
