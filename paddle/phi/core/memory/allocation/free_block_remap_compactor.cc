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
                            gpuEvent_t remap_safe_event
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
  return part.handle_rel_off == 0 && part.len == part.handle->size;
}

bool IsRemapSafe(BlockV2* block) {
  if (block->type_ != BlockType::kFree || block->ipc_exported_) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (block->remap_safe_event_ == nullptr) {
    return true;
  }

  gpuError_t err = cudaEventQuery(block->remap_safe_event_);
  if (err == cudaSuccess) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(block->remap_safe_event_));
    block->remap_safe_event_ = nullptr;
    return true;
  }
#ifdef PADDLE_WITH_HIP
  if (err == hipErrorNotReady) {
    return false;
  }
#else
  if (err == cudaErrorNotReady) {
    return false;
  }
#endif
  PADDLE_ENFORCE_GPU_SUCCESS(err);
  return false;
#else
  return true;
#endif
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

}  // namespace

size_t FreeBlockRemapCompactor::Compact(std::list<BlockV2>* blocks) {
  std::vector<VmmAllocHandle> remapped_handles;
  std::vector<std::shared_ptr<VmmHandleMeta>> remapped_metas;
  bool logged_first_candidate = false;

  for (auto it = blocks->begin(); it != blocks->end();) {
    auto current = it++;
    if (!IsRemapSafe(&(*current))) {
      continue;
    }

    std::vector<BlockV2> replacement_segments;
    size_t block_offset = 0;
    size_t remapped_count_before = remapped_handles.size();
    for (const auto& part : current->parts_) {
      void* part_ptr = reinterpret_cast<uint8_t*>(current->ptr_) + block_offset;
      block_offset += part.len;
      if (IsFullyCoveredHandle(part)) {
        if (!logged_first_candidate) {
          VLOG(0) << "First remap candidate pool="
                  << static_cast<int>(pool_type_)
                  << " block_ptr=" << current->ptr_
                  << " block_size=" << current->size_ << " handle_base="
                  << reinterpret_cast<void*>(part.handle->base)
                  << " handle_size=" << part.handle->size
                  << " handle=" << reinterpret_cast<void*>(part.handle->handle);
          logged_first_candidate = true;
        }
        vmm_allocator_->UnmapHandle(part.handle->base, part.len);
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

  if (remapped_handles.empty()) {
    return 0;
  }

  MergeAdjacentGaps(blocks);

  const size_t handle_size = vmm_allocator_->handle_size();
  const size_t total_remapped = remapped_handles.size() * handle_size;

  // -------------------------------------------------------------------
  // Compute the real tail VA from the block list.
  //
  // virtual_mem_alloced_offset_ is a high-water mark that never retreats.
  // After FreeIdleChunks releases tail-end underlying allocations (or
  // other bookkeeping drift), the recorded tail may point to VA that is
  // still mapped by another entity.  Deriving the real tail from the
  // actual block list is always correct: everything beyond the last
  // block is guaranteed to be unmapped reserved VA.
  // -------------------------------------------------------------------
  VmmDevicePtr real_tail_va = vmm_allocator_->virtual_mem_base();
  if (!blocks->empty()) {
    const auto& last = blocks->back();
    real_tail_va = reinterpret_cast<VmmDevicePtr>(
        reinterpret_cast<uint8_t*>(last.ptr_) + last.size_);
  }
  const size_t real_tail_offset =
      real_tail_va - vmm_allocator_->virtual_mem_base();
  if (real_tail_offset < vmm_allocator_->tail_offset()) {
    VLOG(3) << "VMM V2 compactor: retreating tail_offset from "
            << vmm_allocator_->tail_offset() << " to " << real_tail_offset;
    vmm_allocator_->SetTailOffset(real_tail_offset);
  }

  const VmmDevicePtr tail_va = real_tail_va;
  const VmmDevicePtr va_limit =
      vmm_allocator_->virtual_mem_base() + vmm_allocator_->virtual_mem_size();
  VLOG(10) << "VMM remap compact pool=" << static_cast<int>(pool_type_)
           << " remapped_handles=" << remapped_handles.size()
           << " total_remapped=" << total_remapped
           << " handle_size=" << handle_size
           << " tail_va=" << reinterpret_cast<void*>(tail_va)
           << " va_limit=" << reinterpret_cast<void*>(va_limit);

  // -------------------------------------------------------------------
  // Decide remap destination: tail path → single-gap → gap-scatter.
  // -------------------------------------------------------------------
  bool tail_usable = (tail_va + total_remapped <= va_limit);

  // Safety probe: even if VA arithmetic says tail is within limits, verify
  // that the first handle-sized slot is actually unmapped.  A stale
  // tail_offset or external mapping would otherwise cause cuMemMap to
  // fail with CUDA_ERROR_INVALID_VALUE.
  if (tail_usable) {
    CUmemGenericAllocationHandle probe_handle;
    CUresult probe = phi::dynload::cuMemRetainAllocationHandle(
        &probe_handle, reinterpret_cast<void*>(tail_va));
    if (probe == CUDA_SUCCESS) {
      // tail VA is already mapped – must not use tail path.
      phi::dynload::cuMemRelease(probe_handle);
      tail_usable = false;
      VLOG(3) << "VMM V2 compactor: tail VA "
              << reinterpret_cast<void*>(tail_va)
              << " is unexpectedly mapped, skipping tail path";
    }
    // Any other CUresult (e.g. CUDA_ERROR_INVALID_VALUE) means the VA
    // is unmapped – exactly what we want.
  }

  // ---- Path 1: tail path (consolidate all handles at the tail) --------
  if (tail_usable) {
    VLOG(10) << "VMM remap compact using tail path, dst_va="
             << reinterpret_cast<void*>(tail_va) << " bytes=" << total_remapped;
    vmm_allocator_->MapHandlesToVA(tail_va, remapped_handles, &remapped_metas);
    vmm_allocator_->AdvanceTailOffset(total_remapped);

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

  // ---- Path 2: single-gap path (one gap large enough) -----------------
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
             << reinterpret_cast<void*>(gap_va) << " gap_size=" << gap_it->size_
             << " bytes=" << total_remapped;
    vmm_allocator_->MapHandlesToVA(gap_va, remapped_handles, &remapped_metas);

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

  // ---- Path 3: gap-scatter fallback -----------------------------------
  // Neither tail nor a single large gap is available.  Remap handles back
  // into the individual gaps they came from so that at least we restore a
  // valid mapping and do not leave dangling unmapped VA.  This does not
  // consolidate fragmentation, but it is safe and avoids a hard error.
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
    vmm_allocator_->MapHandlesToVA(dst, chunk, &chunk_metas);

    // Turn the filled portion of this gap into a FREE block.
    const size_t filled_bytes = to_fill * handle_size;
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

  PADDLE_ENFORCE_EQ(
      handle_idx,
      remapped_handles.size(),
      common::errors::ResourceExhausted(
          "VMM V2 compactor gap-scatter: placed %zu of %zu handles; "
          "not enough gap space in pool %d.",
          handle_idx,
          remapped_handles.size(),
          static_cast<int>(pool_type_)));

  MergeAdjacentGaps(blocks);
  return total_remapped;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
