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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"

COMMON_DECLARE_bool(vmm_v2_compact_all);

namespace paddle {
namespace memory {
namespace allocation {

namespace {

template <typename Map, typename Key, typename Value>
void EmplaceOrEnforce(Map* map,
                      Key&& key,
                      Value&& value,
                      const char* map_name) {
  const bool inserted =
      map->try_emplace(std::forward<Key>(key), std::forward<Value>(value))
          .second;
  PADDLE_ENFORCE_EQ(
      inserted,
      true,
      common::errors::AlreadyExists(
          "Duplicate key inserted into %s, allocator state is inconsistent.",
          map_name));
}

std::vector<BlockPartV2> SlicePartsForRange(
    const std::vector<BlockPartV2>& parts,
    size_t range_offset,
    size_t range_len) {
  // parts describes one logical block as an ordered list of handle slices.
  // The target range is also expressed in that logical block address space.
  //
  // Example:
  //   parts:        [part0 len=2][part1 len=3][part2 len=4]
  //   logical idx:   0          2            5            9
  //   range:              [------ range ------)
  //                       1                    7
  //
  //   result:
  //     - part0 contributes a right-side slice [1,2)
  //     - part1 is fully covered and copied as-is
  //     - part2 contributes a left-side slice [5,7)
  //
  // A block-level parts_ list always describes one contiguous logical range,
  // but each element may only cover a slice of its underlying handle. Scan the
  // logical range once, intersect each part with [range_offset, range_end),
  // and rebuild the sliced view in order.
  std::vector<BlockPartV2> sliced_parts;
  if (range_len == 0 || parts.empty()) {
    return sliced_parts;
  }

  sliced_parts.reserve(parts.size());
  const size_t range_end = range_offset + range_len;
  size_t cursor = 0;

  for (const auto& part : parts) {
    const size_t part_block_begin = cursor;
    const size_t part_block_end = cursor + part.len;
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

    if (sliced_parts.empty() || !sliced_parts.back().TryExtend(slice)) {
      sliced_parts.push_back(std::move(slice));
    }
  }
  return sliced_parts;
}

std::vector<BlockPartV2> BuildBlockPartsFromHandleLayout(
    const HandleLayout& layout) {
  std::vector<BlockPartV2> parts;
  parts.reserve(layout.size());
  // The bottom allocator only returns the fixed-handle list for one raw
  // allocation. Best-fit turns that list into block-level slices that will
  // later be split/merged/remapped as BlockV2::parts_ evolves.
  for (const auto& handle : layout) {
    parts.push_back(BlockPartV2{handle, 0, handle->size});
  }
  return parts;
}

void AppendPartsTail(std::vector<BlockPartV2>* dst,
                     std::vector<BlockPartV2>* src) {
  // dst and src each describe one logical block. When merge joins two adjacent
  // FREE blocks, concatenate their parts_ while collapsing the boundary if it
  // happens to land in the middle of one handle. The source block is erased
  // right after merge, so its parts_ can be moved instead of copied.
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

}  // namespace

VMMAutoGrowthBestFitAllocatorV2::VMMAutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
    size_t alignment,
    const GPUPlace& place,
    PoolType pool_type)
    : underlying_allocator_(underlying_allocator),
      alignment_(alignment),
      place_(place),
      pool_type_(pool_type) {}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocateImpl(size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  const size_t requested_size = AlignedSize(size, alignment_);
  if (auto* allocation = AllocFromFreeBlocks(requested_size)) {
    return allocation;
  }

  // Tail reuse: if the last block in the address space is FREE, detach it
  // and only request the difference from the underlying allocator. The
  // underlying VMM provider maps new handles at a monotonically increasing
  // VA cursor, so the new allocation is guaranteed to be contiguous with
  // the tail FREE block.
  size_t tail_reuse_size = 0;
  std::vector<BlockPartV2> tail_parts;
  if (!all_blocks_.empty()) {
    auto tail_it = std::prev(all_blocks_.end());
    if (tail_it->type_ == BlockType::kFree) {
      tail_reuse_size = tail_it->size_;
      tail_parts = std::move(tail_it->parts_);
      EraseFreeBlock(tail_it);
      all_blocks_.erase(tail_it);
    }
  }

  const size_t grow_size = (requested_size > tail_reuse_size)
                               ? (requested_size - tail_reuse_size)
                               : 0;

  // Grow: obtain a new raw allocation from the bottom VMM provider.
  // If cuMemCreate fails due to physical memory exhaustion (CU error 2),
  // the driver-level allocator throws EnforceNotMet.  Convert it to BadAlloc
  // so that RetryAllocator can catch it and trigger try_remap / offload.
  AllocationPtr raw_alloc;
  if (grow_size > 0) {
    try {
      raw_alloc = underlying_allocator_->Allocate(grow_size);
    } catch (...) {
      // Grow failed — restore the tail FREE block before propagating.
      if (tail_reuse_size > 0) {
        BlockV2 restored;
        restored.ptr_ = reinterpret_cast<uint8_t*>(
            all_blocks_.empty()
                ? nullptr
                : reinterpret_cast<uint8_t*>(all_blocks_.back().ptr_) +
                      all_blocks_.back().size_);
        restored.size_ = tail_reuse_size;
        restored.type_ = BlockType::kFree;
        restored.parts_ = std::move(tail_parts);
        restored.pool_type_ = pool_type_;
        auto restored_it =
            all_blocks_.insert(all_blocks_.end(), std::move(restored));
        InsertFreeBlock(restored_it);
      }
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes.",
          static_cast<int>(pool_type_),
          grow_size));
    }
  }

  // Build combined parts: tail_parts + new_parts
  std::vector<BlockPartV2> combined_parts = std::move(tail_parts);
  size_t total_new_size = tail_reuse_size;

  if (raw_alloc) {
    auto allocation = static_unique_ptr_cast<Allocation>(std::move(raw_alloc));
    HandleLayout layout;
    PADDLE_ENFORCE_EQ(
        underlying_allocator_->CollectAllocationHandleLayout(allocation->ptr(),
                                                             &layout),
        true,
        common::errors::NotFound(
            "Can not collect VMM handle layout for allocation %p.",
            allocation->ptr()));
    auto new_parts = BuildBlockPartsFromHandleLayout(layout);
    total_new_size += allocation->size();
    underlying_allocations_.emplace_back(std::move(allocation));
    AppendPartsTail(&combined_parts, &new_parts);
  }

  // The active block starts at the beginning of the combined region.
  uint8_t* combined_ptr =
      combined_parts.empty()
          ? nullptr
          : reinterpret_cast<uint8_t*>(combined_parts.front().handle->base) +
                combined_parts.front().handle_rel_off;
  auto active_parts = SlicePartsForRange(combined_parts, 0, requested_size);
  const size_t remaining_size = total_new_size - requested_size;

  BlockV2 block;
  block.ptr_ = combined_ptr;
  block.size_ = requested_size;
  block.type_ = BlockType::kActive;
  block.parts_ = std::move(active_parts);
  block.pool_type_ = pool_type_;
  auto it = all_blocks_.insert(all_blocks_.end(), std::move(block));
  EmplaceOrEnforce(&allocated_blocks_, it->ptr_, it, "allocated_blocks_");

  if (remaining_size > 0) {
    BlockV2 remaining_block;
    remaining_block.ptr_ = combined_ptr + requested_size;
    remaining_block.size_ = remaining_size;
    remaining_block.type_ = BlockType::kFree;
    remaining_block.parts_ =
        SlicePartsForRange(combined_parts, requested_size, remaining_size);
    remaining_block.pool_type_ = pool_type_;
    auto remain_it =
        all_blocks_.insert(std::next(it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  return new Allocation(it->ptr_, it->ptr_, it->size_, place_);
}

size_t VMMAutoGrowthBestFitAllocatorV2::CompactImpl(const Place& place,
                                                    size_t requested_size) {
  // Defensive place validation: the call chain
  // (RetryAllocator → StreamSafe → MultiPool → SinglePool) guarantees
  // place consistency.  Log a warning on mismatch but do not throw,
  // since CompactImpl is called inside a try-catch that would silently
  // swallow the exception and skip compaction.
  if (UNLIKELY(place != Place(place_))) {
    LOG(WARNING) << "CompactImpl place mismatch: got " << place.DebugString()
                 << " but allocator serves " << Place(place_).DebugString();
  }
  std::lock_guard<SpinLock> guard(spinlock_);

  size_t total_free = 0;
  size_t max_free = 0;
  for (const auto& blk : all_blocks_) {
    if (blk.type_ == BlockType::kFree) {
      total_free += blk.size_;
      max_free = std::max(max_free, blk.size_);
    }
  }

  if (requested_size > 0) {
    if (total_free < requested_size) {
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact skip: total_free=" << total_free
              << " < requested=" << requested_size;
      return 0;
    }
    if (max_free >= requested_size) {
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact skip: max_free=" << max_free
              << " >= requested=" << requested_size;
      return 0;
    }
  }

  std::unordered_set<VmmHandleMeta*> active_handles;
  for (const auto& blk : all_blocks_) {
    if (blk.type_ == BlockType::kActive) {
      for (const auto& part : blk.parts_) {
        active_handles.insert(part.handle.get());
      }
    }
  }

  // Count potentially releasable handles.  Use only static (non-timing)
  // conditions here: handle not already remapped, fully covered, not active.
  // Runtime checks (ipc_exported_, per-handle remap_safe_event) are left to
  // the compactor — they depend on timing and would cause false negatives in
  // the pre-check (e.g. event completes between pre-check and compactor
  // execution).
  size_t releasable_handles = 0;
  size_t remapped_count = 0, partial_count = 0, active_count = 0;
  size_t releasable_bytes = 0;
  size_t remapped_bytes = 0, partial_bytes = 0, active_bytes = 0;
  size_t total_parts_in_free = 0;
  for (const auto& blk : all_blocks_) {
    if (blk.type_ != BlockType::kFree) continue;
    for (const auto& part : blk.parts_) {
      ++total_parts_in_free;
      if (part.handle->remapped) {
        ++remapped_count;
        remapped_bytes += part.len;
        continue;
      }
      if (!(part.handle_rel_off == 0 && part.len == part.handle->size)) {
        ++partial_count;
        partial_bytes += part.len;
        continue;
      }
      if (active_handles.find(part.handle.get()) != active_handles.end()) {
        ++active_count;
        active_bytes += part.len;
        continue;
      }
      ++releasable_handles;
      releasable_bytes += part.len;
    }
  }

  if (releasable_handles == 0) {
    VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " compact skip: no releasable handles"
            << " (total_free=" << total_free << " max_free=" << max_free
            << " requested=" << requested_size
            << " parts_in_free=" << total_parts_in_free
            << " releasable_handles=" << releasable_handles
            << " releasable_bytes=" << releasable_bytes
            << " remapped=" << remapped_count
            << " remapped_bytes=" << remapped_bytes
            << " partial=" << partial_count
            << " partial_bytes=" << partial_bytes << " active=" << active_count
            << " active_bytes=" << active_bytes << ")";
    // Dump the largest free block's parts for debugging.
    size_t largest_free_size = 0;
    const BlockV2* largest_free = nullptr;
    for (const auto& blk : all_blocks_) {
      if (blk.type_ == BlockType::kFree && blk.size_ > largest_free_size) {
        largest_free_size = blk.size_;
        largest_free = &blk;
      }
    }
    if (largest_free) {
      size_t largest_full_count = 0, largest_partial_count = 0;
      size_t largest_remapped_count = 0, largest_active_count = 0;
      size_t largest_full_bytes = 0, largest_partial_bytes = 0;
      size_t largest_remapped_bytes = 0, largest_active_bytes = 0;
      for (const auto& part : largest_free->parts_) {
        if (part.handle->remapped) {
          ++largest_remapped_count;
          largest_remapped_bytes += part.len;
          continue;
        }
        const bool fully_covered =
            part.handle_rel_off == 0 && part.len == part.handle->size;
        if (!fully_covered) {
          ++largest_partial_count;
          largest_partial_bytes += part.len;
          continue;
        }
        if (active_handles.find(part.handle.get()) != active_handles.end()) {
          ++largest_active_count;
          largest_active_bytes += part.len;
          continue;
        }
        ++largest_full_count;
        largest_full_bytes += part.len;
      }
      VLOG(4) << "  Largest free block: ptr=" << largest_free->ptr_
              << " size=" << largest_free->size_
              << " num_parts=" << largest_free->parts_.size()
              << " fully_covered_handles=" << largest_full_count
              << " fully_covered_bytes=" << largest_full_bytes
              << " partial_handles=" << largest_partial_count
              << " partial_bytes=" << largest_partial_bytes
              << " active_overlap_handles=" << largest_active_count
              << " active_overlap_bytes=" << largest_active_bytes
              << " remapped_handles=" << largest_remapped_count
              << " remapped_bytes=" << largest_remapped_bytes;
      size_t logged = 0;
      for (const auto& part : largest_free->parts_) {
        if (logged >= 8) {
          VLOG(4) << "  ... (" << (largest_free->parts_.size() - logged)
                  << " more parts)";
          break;
        }
        VLOG(4) << "  part[" << logged << "]: handle_base="
                << reinterpret_cast<void*>(part.handle->base)
                << " handle_size=" << part.handle->size
                << " rel_off=" << part.handle_rel_off << " len=" << part.len
                << " remapped=" << part.handle->remapped;
        ++logged;
      }
    }
    return 0;
  }

  VLOG(3) << "VMM V2 pool " << static_cast<int>(pool_type_)
          << " compact: total_free=" << total_free << " max_free=" << max_free
          << " requested=" << requested_size
          << " releasable_handles=" << releasable_handles
          << " releasable_bytes=" << releasable_bytes
          << " partial_handles=" << partial_count
          << " partial_bytes=" << partial_bytes
          << " active_overlap_handles=" << active_count
          << " active_overlap_bytes=" << active_bytes
          << " remapped_handles=" << remapped_count
          << " remapped_bytes=" << remapped_bytes
          << ", proceeding with compaction";

  FreeBlockRemapCompactor compactor(
      underlying_allocator_, pool_type_, &underlying_allocations_);
  const size_t compact_target = FLAGS_vmm_v2_compact_all ? 0 : requested_size;
  const size_t remapped = compactor.Compact(&all_blocks_, compact_target);
  // Always rebuild: Phase 1 may have replaced FREE blocks with GAP/FREE
  // segments before Phase 2 fails.  Without rebuild, free_blocks_ holds
  // stale iterators to erased list nodes → use-after-free on next alloc.
  RebuildFreeBlockIndex();
  return remapped;
}

void VMMAutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto ptr = allocation->ptr();
  auto found = allocated_blocks_.find(ptr);
  PADDLE_ENFORCE_NE(
      found,
      allocated_blocks_.end(),
      common::errors::NotFound("Can not find active block for allocation %p in "
                               "VMMAutoGrowthBestFitAllocatorV2.",
                               ptr));
  auto it = found->second;
  allocated_blocks_.erase(it->ptr_);
  it->type_ = BlockType::kFree;
  TryMerge(it);
  delete allocation;
}

void VMMAutoGrowthBestFitAllocatorV2::GetFreeBlockStats(size_t* total_free,
                                                        size_t* max_free) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t total = 0;
  for (const auto& entry : free_blocks_) {
    total += entry.first.first;
  }
  size_t max_sz = 0;
  if (!free_blocks_.empty()) {
    max_sz = free_blocks_.rbegin()->first.first;
  }
  *total_free = total;
  *max_free = max_sz;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(
    void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    gpuStream_t stream,
    std::shared_ptr<CudaEventGuard> event
#else
    void* stream,
    void* event
#endif
) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::unordered_set<VmmHandleMeta*> seen;
  for (auto& part : it->second->parts_) {
    auto* handle = part.handle.get();
    if (!seen.insert(handle).second) {
      continue;
    }
    handle->last_use_stream = stream;
    handle->remap_safe_event = event;
  }
#else
  (void)stream;
  (void)event;
#endif
  return true;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size) {
  auto it = free_blocks_.lower_bound({size, nullptr});
  if (it == free_blocks_.end()) {
    return nullptr;
  }

  auto block_it = it->second;
  EraseFreeBlock(block_it);

  if (block_it->size_ > size) {
    const size_t remaining_size = block_it->size_ - size;
    BlockV2 remaining_block;
    remaining_block.ptr_ = reinterpret_cast<uint8_t*>(block_it->ptr_) + size;
    remaining_block.size_ = remaining_size;
    remaining_block.type_ = BlockType::kFree;
    remaining_block.parts_ =
        SlicePartsForRange(block_it->parts_, size, remaining_size);
    remaining_block.pool_type_ = block_it->pool_type_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    // owning_stream_ is cleared: nobody "owns" a free fragment. Remap safety
    // lives on each handle meta, so the remaining fragment observes the same
    // event state naturally through its sliced parts.
    remaining_block.owning_stream_ = nullptr;
#endif

    block_it->size_ = size;
    block_it->parts_ = SlicePartsForRange(block_it->parts_, 0, size);
    auto remain_it =
        all_blocks_.insert(std::next(block_it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  block_it->type_ = BlockType::kActive;
  EmplaceOrEnforce(
      &allocated_blocks_, block_it->ptr_, block_it, "allocated_blocks_");
  return new Allocation(
      block_it->ptr_, block_it->ptr_, block_it->size_, place_);
}

void VMMAutoGrowthBestFitAllocatorV2::InsertFreeBlock(BlockListIt it) {
  EmplaceOrEnforce(
      &free_blocks_, std::make_pair(it->size_, it->ptr_), it, "free_blocks_");
}

void VMMAutoGrowthBestFitAllocatorV2::EraseFreeBlock(BlockListIt it) {
  free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::RebuildFreeBlockIndex() {
  free_blocks_.clear();
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (it->type_ == BlockType::kFree) {
      InsertFreeBlock(it);
    }
  }
}

void VMMAutoGrowthBestFitAllocatorV2::TryMerge(BlockListIt it) {
  // Only adjacent FREE blocks are merged here. ACTIVE blocks are never touched,
  // and GAP blocks remain as explicit holes for later remap/GAP handling.
  // all_blocks_ is the full VA-ordered block list, so adjacency is checked
  // against neighboring entries in that list.
  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(prev->ptr_) + prev->size_ ==
            reinterpret_cast<uint8_t*>(it->ptr_)) {
      EraseFreeBlock(prev);
      AppendPartsTail(&prev->parts_, &it->parts_);
      prev->size_ += it->size_;
      all_blocks_.erase(it);
      it = prev;
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() && next->type_ == BlockType::kFree &&
      reinterpret_cast<uint8_t*>(it->ptr_) + it->size_ ==
          reinterpret_cast<uint8_t*>(next->ptr_)) {
    EraseFreeBlock(next);
    AppendPartsTail(&it->parts_, &next->parts_);
    it->size_ += next->size_;
    all_blocks_.erase(next);
  }

  InsertFreeBlock(it);
}

// ---------------------------------------------------------------------------
// ReleaseImpl / FreeIdleChunks – release underlying allocations whose entire
// VA range is covered by FREE blocks back to the CUDA VMM driver.
//
// Because TryMerge may have merged FREE blocks across allocation boundaries,
// we must split the spanning block at the allocation edges before removing
// the inner portion and freeing the allocation.
// ---------------------------------------------------------------------------

uint64_t VMMAutoGrowthBestFitAllocatorV2::ReleaseImpl(
    const Place& place UNUSED) {
  std::lock_guard<SpinLock> guard(spinlock_);
  return FreeIdleChunks();
}

uint64_t VMMAutoGrowthBestFitAllocatorV2::FreeIdleChunks() {
  uint64_t released = 0;

  for (auto alloc_it = underlying_allocations_.begin();
       alloc_it != underlying_allocations_.end();) {
    auto* base = reinterpret_cast<uint8_t*>((*alloc_it)->ptr());
    const size_t alloc_size = (*alloc_it)->size();

    if (!IsRangeEntirelyFree(base, alloc_size)) {
      ++alloc_it;
      continue;
    }

    SplitAndRemoveRange(base, alloc_size);
    released += alloc_size;
    VLOG(5) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " released idle chunk: " << alloc_size << " bytes";
    // Erasing the DecoratedAllocationPtr triggers its deleter, which calls
    // CUDAVirtualMemAllocatorV2::FreeImpl → cuMemUnmap + cuMemRelease.
    // FreeImpl already skips handles with remapped==true (their physical
    // memory is owned by the compactor's destination block), so it is
    // safe to erase even when the allocation contains remapped handles.
    alloc_it = underlying_allocations_.erase(alloc_it);
  }

  return released;
}

bool VMMAutoGrowthBestFitAllocatorV2::IsRangeEntirelyFree(uint8_t* base,
                                                          size_t size) const {
  auto* end = base + size;
  size_t covered = 0;
  for (const auto& block : all_blocks_) {
    auto* bptr = reinterpret_cast<uint8_t*>(block.ptr_);
    auto* bend = bptr + block.size_;
    if (bend <= base) continue;
    if (bptr >= end) break;
    // Accept both FREE and GAP: GAP blocks represent VA ranges whose
    // physical memory was remapped elsewhere by the compactor.  The
    // original allocation can still be released because FreeImpl skips
    // handles marked remapped==true.
    if (block.type_ != BlockType::kFree && block.type_ != BlockType::kGap) {
      return false;
    }
    covered += static_cast<size_t>(std::min(bend, end) - std::max(bptr, base));
  }
  return covered == size;
}

void VMMAutoGrowthBestFitAllocatorV2::SplitAndRemoveRange(uint8_t* base,
                                                          size_t size) {
  auto* end = base + size;

  for (auto it = all_blocks_.begin(); it != all_blocks_.end();) {
    auto* bptr = reinterpret_cast<uint8_t*>(it->ptr_);
    auto* bend = bptr + it->size_;

    if (bend <= base) {
      ++it;
      continue;
    }
    if (bptr >= end) break;

    const bool is_gap = (it->type_ == BlockType::kGap);

    // Case 1: block entirely within [base, end) → remove it.
    if (bptr >= base && bend <= end) {
      if (!is_gap) EraseFreeBlock(it);
      it = all_blocks_.erase(it);
      continue;
    }

    // Case 2: block straddles left boundary only → keep left remnant.
    if (bptr < base && bend <= end) {
      const size_t keep = static_cast<size_t>(base - bptr);
      if (!is_gap) {
        EraseFreeBlock(it);
        it->parts_ = SlicePartsForRange(it->parts_, 0, keep);
        it->size_ = keep;
        InsertFreeBlock(it);
      } else {
        it->size_ = keep;
      }
      ++it;
      continue;
    }

    // Case 3: block straddles right boundary only → keep right remnant.
    if (bptr >= base && bend > end) {
      const size_t trim = static_cast<size_t>(end - bptr);
      const size_t keep = it->size_ - trim;
      if (!is_gap) {
        EraseFreeBlock(it);
        it->parts_ = SlicePartsForRange(it->parts_, trim, keep);
        it->ptr_ = end;
        it->size_ = keep;
        InsertFreeBlock(it);
      } else {
        it->ptr_ = end;
        it->size_ = keep;
      }
      break;  // nothing more in range
    }

    // Case 4: block fully encompasses [base, end) → split into two.
    if (bptr < base && bend > end) {
      const size_t left_size = static_cast<size_t>(base - bptr);
      const size_t right_offset = static_cast<size_t>(end - bptr);
      const size_t right_size = it->size_ - right_offset;

      if (!is_gap) {
        const auto orig_parts = it->parts_;
        EraseFreeBlock(it);
        it->parts_ = SlicePartsForRange(orig_parts, 0, left_size);
        it->size_ = left_size;
        InsertFreeBlock(it);

        BlockV2 right;
        right.ptr_ = end;
        right.size_ = right_size;
        right.type_ = BlockType::kFree;
        right.parts_ = SlicePartsForRange(orig_parts, right_offset, right_size);
        right.pool_type_ = it->pool_type_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        right.owning_stream_ = nullptr;
#endif
        auto right_it = all_blocks_.insert(std::next(it), std::move(right));
        InsertFreeBlock(right_it);
      } else {
        // GAP: just shrink left and insert right GAP.
        it->size_ = left_size;
        BlockV2 right;
        right.ptr_ = end;
        right.size_ = right_size;
        right.type_ = BlockType::kGap;
        right.pool_type_ = it->pool_type_;
        all_blocks_.insert(std::next(it), std::move(right));
      }
      break;  // done
    }

    ++it;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
