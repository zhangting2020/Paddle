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

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

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
                            PoolType pool_type,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                            gpuStream_t last_use_stream,
                            std::shared_ptr<CudaEventGuard> remap_safe_event
#endif
) {
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  segment.last_use_stream_ = last_use_stream;
  segment.remap_safe_event_ = remap_safe_event;
#endif
  segments->push_back(std::move(segment));
}

bool IsFullyCoveredHandle(const BlockPartV2& part) {
  // Skip handles that were already remapped by a previous compact — their
  // physical memory is owned by a synthetic allocation at a different VA.
  // Attempting to remap them again would use a stale or released handle.
  if (part.handle->remapped) {
    return false;
  }
  return part.handle_rel_off == 0 && part.len == part.handle->size;
}

bool IsRemapSafe(BlockV2* block) {
  if (block->type_ != BlockType::kFree || block->ipc_exported_) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Per-event query: no cudaDeviceSynchronize needed.
  // If the block has a remap_safe_event_, query whether the GPU work
  // that last used this memory has completed.  Only remap blocks whose
  // events are done; skip those still in flight.
  if (block->remap_safe_event_) {
#ifdef PADDLE_WITH_CUDA
    gpuError_t err = cudaEventQuery(block->remap_safe_event_->event);
#else
    gpuError_t err = hipEventQuery(block->remap_safe_event_->event);
#endif
    if (err == gpuSuccess) {
      // Event completed — release the shared_ptr (may destroy the event
      // if this was the last holder).
      block->remap_safe_event_.reset();
    } else {
      // GPU work still pending on this block's memory — not safe to remap.
      return false;
    }
  }
#endif
  return true;
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

// Rollback helper: remap handles back into their original GAP positions
// and convert those GAPs back to FREE blocks.  Called when Phase 2 fails
// so that the block list stays consistent for subsequent allocator ops.
void RollbackUnmappedHandles(
    std::list<BlockV2>* blocks,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>& metas,
    CUDAVirtualMemAllocatorV2* vmm_allocator,
    size_t handle_size) {
  size_t handle_idx = 0;
  for (auto it = blocks->begin();
       it != blocks->end() && handle_idx < handles.size();
       ++it) {
    if (it->type_ != BlockType::kGap) continue;

    const size_t gap_capacity = it->size_ / handle_size;
    const size_t remaining = handles.size() - handle_idx;
    const size_t to_fill = std::min(gap_capacity, remaining);
    if (to_fill == 0) continue;

    const VmmDevicePtr dst = reinterpret_cast<VmmDevicePtr>(it->ptr_);
    std::vector<VmmAllocHandle> chunk(handles.begin() + handle_idx,
                                      handles.begin() + handle_idx + to_fill);
    std::vector<std::shared_ptr<VmmHandleMeta>> chunk_metas(
        metas.begin() + handle_idx, metas.begin() + handle_idx + to_fill);

    // Best-effort remap back. If this also fails, we log and continue.
    try {
      vmm_allocator->MapHandlesToVA(dst, chunk, &chunk_metas);
    } catch (...) {
      VLOG(0) << "VMM V2 compactor rollback: MapHandlesToVA failed for "
              << to_fill << " handles at gap VA "
              << reinterpret_cast<void*>(dst) << ", some memory may be leaked";
      handle_idx += to_fill;
      continue;
    }

    // Convert the filled gap portion back to FREE.
    const size_t filled_bytes = to_fill * handle_size;
    it->type_ = BlockType::kFree;
    it->parts_.clear();
    for (size_t i = 0; i < to_fill; ++i) {
      chunk_metas[i]->base = dst + i * handle_size;
      chunk_metas[i]->remapped = false;  // Undo the remapped flag
      it->parts_.push_back(BlockPartV2{chunk_metas[i], 0, handle_size});
    }

    if (filled_bytes < it->size_) {
      BlockV2 leftover_gap;
      leftover_gap.ptr_ = reinterpret_cast<void*>(dst + filled_bytes);
      leftover_gap.size_ = it->size_ - filled_bytes;
      leftover_gap.type_ = BlockType::kGap;
      leftover_gap.pool_type_ = it->pool_type_;
      it->size_ = filled_bytes;
      blocks->insert(std::next(it), std::move(leftover_gap));
    }
    handle_idx += to_fill;
  }

  // Merge adjacent FREE blocks created by rollback
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

}  // namespace

size_t FreeBlockRemapCompactor::Compact(std::list<BlockV2>* blocks) {
  std::vector<VmmAllocHandle> remapped_handles;
  std::vector<std::shared_ptr<VmmHandleMeta>> remapped_metas;
  bool logged_first_candidate = false;
  const size_t handle_size = vmm_allocator_->handle_size();

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
    size_t free_block_count = 0, safe_block_count = 0;
    size_t fully_covered_count = 0, partial_count = 0;
    for (auto it = blocks->begin(); it != blocks->end();) {
      auto current = it++;
      if (current->type_ == BlockType::kFree) free_block_count++;
      if (!IsRemapSafe(&(*current))) {
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
        if (IsFullyCoveredHandle(part)) {
          fully_covered_count++;
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
                                   pool_type_
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                                   ,
                                   current->last_use_stream_,
                                   current->remap_safe_event_
#endif
            );
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
                                 pool_type_
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                                 ,
                                 nullptr,
                                 nullptr
#endif
          );
          continue;
        } else {
          partial_count++;
        }

        AppendGapOrFreeSegment(&replacement_segments,
                               BlockType::kFree,
                               part.len,
                               &part,
                               part_ptr,
                               pool_type_
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                               ,
                               current->last_use_stream_,
                               current->remap_safe_event_
#endif
        );
      }

      if (remapped_handles.size() == remapped_count_before) {
        continue;
      }

      auto insert_pos = current;
      for (auto& segment : replacement_segments) {
        blocks->insert(insert_pos, std::move(segment));
      }
      blocks->erase(current);
    }

    // Log per-handle coverage to diagnose why handles are partial.
    // A handle is "fully covered" when a single part spans the entire
    // handle (off=0, len=handle->size).  Partial handles are split
    // between ACTIVE and FREE blocks — the ACTIVE portion prevents remap.
    size_t event_blocked_count = 0;
    for (auto& block : *blocks) {
      if (block.type_ == BlockType::kFree && !block.ipc_exported_ &&
          block.remap_safe_event_) {
        event_blocked_count++;
      }
    }
    LOG(INFO) << "VMM V2 compactor pool=" << static_cast<int>(pool_type_)
              << " Phase 1 stats: free_blocks=" << free_block_count
              << " safe_blocks=" << safe_block_count
              << " event_blocked=" << event_blocked_count
              << " fully_covered_parts=" << fully_covered_count
              << " partial_parts=" << partial_count;
    // Log details of first few partial parts for debugging.
    if (partial_count > 0 && fully_covered_count == 0) {
      size_t logged = 0;
      for (auto& block : *blocks) {
        if (block.type_ != BlockType::kFree || logged >= 5) break;
        for (const auto& part : block.parts_) {
          if (!IsFullyCoveredHandle(part) && logged < 5) {
            LOG(INFO) << "  partial part: block_ptr=" << block.ptr_
                      << " block_size=" << block.size_ << " handle_base="
                      << reinterpret_cast<void*>(part.handle->base)
                      << " handle_size=" << part.handle->size
                      << " part_off=" << part.handle_rel_off
                      << " part_len=" << part.len
                      << " coverage=" << (part.len * 100 / part.handle->size)
                      << "%";
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
    // Wrapped in try-catch: if remap fails, rollback by gap-scattering
    // handles back into their original GAP positions.
    // -------------------------------------------------------------------
    auto rollback = [&]() {
      VLOG(0) << "VMM V2 compactor Phase 2 failed, rolling back "
              << remapped_handles.size() << " handles via gap-scatter";
      RollbackUnmappedHandles(blocks,
                              remapped_handles,
                              remapped_metas,
                              vmm_allocator_.get(),
                              handle_size);
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
        vmm_allocator_->MapHandlesToVA(
            tail_va, remapped_handles, &remapped_metas);
      } catch (...) {
        VLOG(0) << "VMM V2 compactor: tail MapHandlesToVA failed";
        rollback();
        return 0;
      }
      vmm_allocator_->AdvanceTailOffset(total_remapped);

      // Register a synthetic allocation so FreeIdleChunks can release
      // these handles when the tail block becomes entirely free.
      // Creates NEW VmmHandleMeta objects with the new VA as base and
      // remapped=false, so that FreeImpl correctly unmaps+releases them.
      // The original allocation's layout retains the old metas with
      // remapped=true (set in Phase 1), so FreeImpl skips them.
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
          tail_va, total_remapped, pool_type_, remapped_metas, handle_size);

      if (!blocks->empty()) {
        auto last = std::prev(blocks->end());
        if (last->type_ == BlockType::kFree &&
            reinterpret_cast<uint8_t*>(last->ptr_) + last->size_ ==
                reinterpret_cast<uint8_t*>(tail_free.ptr_)) {
          last->size_ += tail_free.size_;
          for (const auto& part : tail_free.parts_) {
            TryAppendPart(&last->parts_, part);
          }
          return total_remapped;
        }
      }
      blocks->push_back(std::move(tail_free));
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
        vmm_allocator_->MapHandlesToVA(
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
          gap_va, total_remapped, pool_type_, remapped_metas, handle_size);
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
      return total_remapped;
    }

    // ---- Path 3: gap-scatter fallback ----
    VLOG(3) << "VMM V2 compactor: tail unavailable and no single gap >= "
            << total_remapped << " bytes, falling back to gap-scatter remap";
    size_t handle_idx = 0;
    for (auto it = blocks->begin();
         it != blocks->end() && handle_idx < remapped_handles.size();
         ++it) {
      if (it->type_ != BlockType::kGap) continue;

      const size_t gap_capacity = it->size_ / handle_size;
      const size_t remaining = remapped_handles.size() - handle_idx;
      const size_t to_fill = std::min(gap_capacity, remaining);
      if (to_fill == 0) continue;

      const VmmDevicePtr dst = reinterpret_cast<VmmDevicePtr>(it->ptr_);
      std::vector<VmmAllocHandle> chunk(
          remapped_handles.begin() + handle_idx,
          remapped_handles.begin() + handle_idx + to_fill);
      std::vector<std::shared_ptr<VmmHandleMeta>> chunk_metas(
          remapped_metas.begin() + handle_idx,
          remapped_metas.begin() + handle_idx + to_fill);
      try {
        vmm_allocator_->MapHandlesToVA(dst, chunk, &chunk_metas);
      } catch (...) {
        // Gap-scatter IS the rollback path. If even this fails, we have
        // orphaned handles. Log and return 0 to indicate no useful work.
        VLOG(0) << "VMM V2 compactor: gap-scatter MapHandlesToVA failed at "
                << "handle_idx=" << handle_idx << "/" << remapped_handles.size()
                << ", some handles are orphaned (memory leak)";
        return 0;
      }

      // Register synthetic allocation for this gap chunk.
      HandleLayout chunk_layout;
      for (size_t i = 0; i < to_fill; ++i) {
        chunk_layout.push_back(std::make_shared<VmmHandleMeta>(
            VmmHandleMeta{dst + i * handle_size,
                          handle_size,
                          chunk[i],
                          vmm_allocator_->place().device}));
      }
      const size_t filled_bytes = to_fill * handle_size;
      auto synth = vmm_allocator_->CreateSyntheticAllocation(
          dst, filled_bytes, chunk_layout);
      underlying_allocations_->emplace_back(std::move(synth));
      it->type_ = BlockType::kFree;
      it->parts_.clear();
      for (size_t i = 0; i < to_fill; ++i) {
        chunk_metas[i]->base = dst + i * handle_size;
        it->parts_.push_back(BlockPartV2{chunk_metas[i], 0, handle_size});
      }

      if (filled_bytes < it->size_) {
        BlockV2 leftover_gap;
        leftover_gap.ptr_ = reinterpret_cast<void*>(dst + filled_bytes);
        leftover_gap.size_ = it->size_ - filled_bytes;
        leftover_gap.type_ = BlockType::kGap;
        leftover_gap.pool_type_ = pool_type_;
        it->size_ = filled_bytes;
        blocks->insert(std::next(it), std::move(leftover_gap));
      }

      handle_idx += to_fill;
    }

    if (handle_idx != remapped_handles.size()) {
      VLOG(0) << "VMM V2 compactor gap-scatter: placed " << handle_idx << " of "
              << remapped_handles.size()
              << " handles; not enough gap space in pool "
              << static_cast<int>(pool_type_);
      // Don't crash — return 0 to indicate partial/failed compaction.
      return 0;
    }

    // Merge adjacent FREE blocks
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

    MergeAdjacentGaps(blocks);
    return total_remapped;
  } catch (...) {
    VLOG(0)
        << "VMM V2 compactor: exception caught during Compact, rolling back "
        << remapped_handles.size() << " unmapped handles";
    if (!remapped_handles.empty()) {
      RollbackUnmappedHandles(blocks,
                              remapped_handles,
                              remapped_metas,
                              vmm_allocator_.get(),
                              handle_size);
    }
    return 0;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
