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
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

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
      if (segments->back().parts_.empty() ||
          !segments->back().parts_.back().TryExtend(*part)) {
        segments->back().parts_.push_back(*part);
      }
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

}  // namespace

size_t FreeBlockRemapCompactor::Compact(std::list<BlockV2>* blocks,
                                        size_t requested_size) {
  std::vector<VmmAllocHandle> remapped_handles;
  std::vector<std::shared_ptr<VmmHandleMeta>> remapped_metas;
  bool logged_first_candidate = false;
  const size_t handle_size = vmm_allocator_->handle_size();
  RemapTransaction transaction(vmm_allocator_.get(), handle_size);
  transaction.AddSourceRestoreAction(
      blocks, &remapped_handles, &remapped_metas);
  transaction.SetSyntheticAllocationSink(underlying_allocations_);

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

    transaction.NormalizeBlocks(blocks);

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
    bool tail_usable = transaction.TailIsUsable(tail_va, total_remapped, va_limit);

    // ---- Path 1: tail path ----
    if (tail_usable) {
      VLOG(10) << "VMM remap compact using tail path, dst_va="
               << reinterpret_cast<void*>(tail_va)
               << " bytes=" << total_remapped;
      if (!transaction.TryCommitTailPlacement(
              blocks, tail_va, remapped_handles, remapped_metas, pool_type_)) {
        return 0;
      }
      vmm_allocator_->AdvanceTailOffset(total_remapped);
      return total_remapped;
    }

    // ---- Path 2: single-gap path ----
    auto gap_it = blocks->end();
    bool has_single_gap =
        transaction.FindSingleGap(blocks, total_remapped, &gap_it);

    if (has_single_gap) {
      const VmmDevicePtr gap_va = reinterpret_cast<VmmDevicePtr>(gap_it->ptr_);
      VLOG(10) << "VMM remap compact using gap path, dst_va="
               << reinterpret_cast<void*>(gap_va)
               << " gap_size=" << gap_it->size_ << " bytes=" << total_remapped;
      if (!transaction.TryCommitSingleGapPlacement(
              blocks, gap_it, remapped_handles, remapped_metas, pool_type_)) {
        return 0;
      }
      return total_remapped;
    }

    // ---- Path 3: gap-scatter (two-phase commit) ----
    VLOG(3) << "VMM V2 compactor: tail unavailable and no single gap >= "
            << total_remapped << " bytes, falling back to gap-scatter remap";

    // Capacity precheck: verify total GAP can hold all handles.
    size_t total_gap_capacity = transaction.CollectGapCapacity(*blocks);
    if (total_gap_capacity < total_remapped) {
      VLOG(0) << "VMM V2 compactor: gap capacity " << total_gap_capacity
              << " < total_remapped " << total_remapped
              << ", rolling back to original VA";
      transaction.Rollback();
      return 0;
    }

    // Phase 3a: tentative placement (cuMemMap only, no bookkeeping).
    std::vector<RemapTransaction::GapPlacement> placements;
    bool planned =
        transaction.PlanGapScatter(blocks, remapped_handles.size(), &placements);

    // Defensive: capacity precheck should prevent this.
    if (!planned) {
      size_t planned_handles = 0;
      for (const auto& p : placements) {
        planned_handles += p.count;
      }
      VLOG(0) << "VMM V2 compactor gap-scatter: placed " << planned_handles
              << " of " << remapped_handles.size()
              << " handles despite precheck; rolling back";
      transaction.Rollback();
      return 0;
    }

    if (!transaction.TryCommitGapScatter(
            blocks, remapped_handles, remapped_metas, placements, pool_type_)) {
      return 0;
    }
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
