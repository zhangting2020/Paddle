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

#if defined(PADDLE_WITH_CUDA)

#include <chrono>
#include <exception>
#include <limits>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"
#include "paddle/phi/core/memory/allocation/vmm_v2_step_stats.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

COMMON_DECLARE_bool(vmm_v2_compact_all);
COMMON_DECLARE_bool(vmm_v2_remap_on_oom);
PHI_DECLARE_bool(vmm_v2_round_alloc_to_handle_size);
PHI_DECLARE_bool(vmm_v2_round_large_pool_alloc_to_handle_size);
PHI_DECLARE_bool(vmm_v2_skip_remap_safety_hot_path);
PHI_DECLARE_bool(vmm_v2_consume_whole_free_block);
PHI_DECLARE_uint64(vmm_v2_consume_whole_free_block_max_waste_mb);
PHI_DECLARE_bool(vmm_v2_exact_free_block_cache);

namespace paddle {
namespace memory {
namespace allocation {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kSlowAllocatorOpLogUs = 1000;

uint64_t ElapsedMicros(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
}

void ClearGpuLastError() { (void)platform::GpuGetLastError(); }

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

}  // namespace

void VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Add(
    DecoratedAllocationPtr allocation) {
  allocations_.emplace_back(std::move(allocation));
  auto it = std::prev(allocations_.end());
  auto* begin = Begin(*it);
  PADDLE_ENFORCE_EQ(
      allocations_by_ptr_.emplace(begin, it).second,
      true,
      common::errors::AlreadyExists(
          "Duplicate underlying allocation base %p in VMM V2 registry.",
          begin));
}

namespace {

bool RangesOverlap(void* lhs_ptr,
                   size_t lhs_size,
                   void* rhs_ptr,
                   size_t rhs_size) {
  const auto* lhs_begin = reinterpret_cast<const uint8_t*>(lhs_ptr);
  const auto* lhs_end = lhs_begin + lhs_size;
  const auto* rhs_begin = reinterpret_cast<const uint8_t*>(rhs_ptr);
  const auto* rhs_end = rhs_begin + rhs_size;
  return lhs_end > rhs_begin && rhs_end > lhs_begin;
}

bool ShouldConsumeWholeFreeBlock(size_t remainder_size) {
  if (!FLAGS_vmm_v2_consume_whole_free_block || remainder_size == 0) {
    return false;
  }
  if (FLAGS_vmm_v2_consume_whole_free_block_max_waste_mb == 0) {
    return true;
  }
  const size_t max_waste =
      static_cast<size_t>(FLAGS_vmm_v2_consume_whole_free_block_max_waste_mb)
      << 20;
  return remainder_size <= max_waste;
}

}  // namespace

uint8_t* VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Begin(
    const DecoratedAllocationPtr& allocation) {
  return reinterpret_cast<uint8_t*>(allocation->ptr());
}

uint8_t* VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::End(
    const DecoratedAllocationPtr& allocation) {
  return Begin(allocation) + allocation->size();
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::HasOverlap(
    void* ptr, size_t size) const {
  auto* begin = reinterpret_cast<uint8_t*>(ptr);
  auto* end = begin + size;
  auto it = allocations_by_ptr_.lower_bound(begin);
  if (it != allocations_by_ptr_.begin()) {
    auto prev = std::prev(it);
    if (End(*prev->second) > begin) {
      return true;
    }
  }
  return it != allocations_by_ptr_.end() && it->first < end;
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Overlaps(
    void* ptr, size_t size) const {
  return HasOverlap(ptr, size);
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    AllOverlapsSatisfy(void* ptr,
                       size_t size,
                       const OverlapPredicate& predicate) const {
  for (const auto& allocation : allocations_) {
    if (!RangesOverlap(ptr, size, allocation->ptr(), allocation->size())) {
      continue;
    }
    if (!predicate(allocation)) {
      return false;
    }
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    EraseOverlapsIf(void* ptr, size_t size, const OverlapPredicate& predicate) {
  bool ok = true;
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    if (!RangesOverlap(ptr, size, (*it)->ptr(), (*it)->size())) {
      ++it;
      continue;
    }
    if (!predicate(*it)) {
      ok = false;
      ++it;
      continue;
    }
    allocations_by_ptr_.erase(Begin(*it));
    it = allocations_.erase(it);
  }
  return ok;
}

VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::iterator
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Erase(
    iterator it) {
  allocations_by_ptr_.erase(Begin(*it));
  return allocations_.erase(it);
}

VMMAutoGrowthBestFitAllocatorV2::VMMAutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
    size_t alignment,
    const GPUPlace& place,
    PoolType pool_type)
    : underlying_allocator_(underlying_allocator),
      alignment_(alignment),
      place_(place),
      pool_type_(pool_type) {}

bool VMMAutoGrowthBestFitBlockAllocationV2::SetVMMRemapEvent(
    gpuStream_t stream, std::shared_ptr<CUDAEventGuard> event) {
  if (owner_ == nullptr) {
    return false;
  }
  remap_stream_ = stream;
  remap_event_ = std::move(event);
  has_remap_state_ = true;
  return true;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocateImpl(size_t size) {
  const bool trace_perf = VLOG_IS_ON(4);
  const bool trace_step = VMMV2StepStatsEnabled();
  const auto lock_wait_start =
      (trace_perf || trace_step) ? Clock::now() : Clock::time_point{};
  std::lock_guard<SpinLock> guard(spinlock_);
  const uint64_t lock_wait_us =
      (trace_perf || trace_step) ? ElapsedMicros(lock_wait_start, Clock::now())
                                 : 0;
  const auto op_start =
      (trace_perf || trace_step) ? Clock::now() : Clock::time_point{};
  size_t requested_size = AlignedSize(size, alignment_);
  if (FLAGS_vmm_v2_round_alloc_to_handle_size ||
      (FLAGS_vmm_v2_round_large_pool_alloc_to_handle_size &&
       pool_type_ == PoolType::kLarge)) {
    requested_size =
        AlignedSize(requested_size, underlying_allocator_->HandleSize());
  }
  VMMV2MappedFreeDetailStats mapped_free_detail_stats;
  VMMV2MappedFreeDetailStats* mapped_free_detail_stats_ptr =
      VMMV2DetailStatsEnabled() ? &mapped_free_detail_stats : nullptr;
  auto record_alloc = [&](const char* path, phi::Allocation* allocation) {
    const uint64_t elapsed_us =
        (trace_perf || trace_step) ? ElapsedMicros(op_start, Clock::now()) : 0;
    if (trace_step) {
      RecordVMMV2Alloc(place_.device,
                       pool_type_,
                       path,
                       size,
                       elapsed_us,
                       lock_wait_us,
                       all_blocks_.size(),
                       free_blocks_.size(),
                       unmapped_free_blocks_.size(),
                       underlying_allocator_->TailOffset());
    }
    if (!trace_perf) {
      return;
    }
    if (elapsed_us < kSlowAllocatorOpLogUs) {
      return;
    }
    VLOG(4) << "VMM V2 best-fit slow alloc"
            << " pool=" << static_cast<int>(pool_type_) << " path=" << path
            << " request=" << size << " aligned=" << requested_size
            << " elapsed_us=" << elapsed_us
            << " ptr=" << (allocation == nullptr ? nullptr : allocation->ptr())
            << " block_count=" << all_blocks_.size()
            << " free_blocks=" << free_blocks_.size()
            << " unmapped_free_blocks=" << unmapped_free_blocks_.size()
            << " tail_offset=" << underlying_allocator_->TailOffset();
  };
  if (auto* allocation =
          AllocFromFreeBlocks(requested_size, mapped_free_detail_stats_ptr)) {
    if (mapped_free_detail_stats_ptr != nullptr) {
      RecordVMMV2MappedFreeDetail(place_.device, mapped_free_detail_stats);
    }
    record_alloc("mapped_free", allocation);
    return allocation;
  }
  if (auto* allocation = AllocFromUnmappedFreeBlocks(requested_size)) {
    record_alloc("unmapped_free", allocation);
    return allocation;
  }

  // Tail reuse: if the last block in the address space is FREE, detach it
  // and only request the difference from the underlying allocator. The
  // underlying VMM provider maps new handles at a monotonically increasing
  // VA cursor, so the new allocation is guaranteed to be contiguous with
  // the tail FREE block.
  bool has_tail_reuse = false;
  size_t tail_reuse_size = 0;
  BlockV2 combined_free_block;
  if (!all_blocks_.empty()) {
    auto tail_it = std::prev(all_blocks_.end());
    if (CanIndexFreeBlock(*tail_it)) {
      has_tail_reuse = true;
      tail_reuse_size = tail_it->size_;
      EraseFreeBlock(tail_it);
      combined_free_block = std::move(*tail_it);
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
  CUDAVirtualMemAllocatorV2::AllocationWithBlock grow_alloc;
  if (grow_size > 0) {
    try {
      grow_alloc = underlying_allocator_->AppendWithBlock(grow_size);
    } catch (const BadAlloc& bad_alloc) {
      // Grow failed: restore the tail FREE block before propagating.
      if (has_tail_reuse) {
        auto restored_it = all_blocks_.insert(all_blocks_.end(),
                                              std::move(combined_free_block));
        InsertFreeBlock(restored_it);
      }
      ClearGpuLastError();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes.\n"
          "Underlying VMM allocation failure:\n%s",
          static_cast<int>(pool_type_),
          grow_size,
          bad_alloc.what()));
    } catch (const std::exception& e) {
      // Grow failed: restore the tail FREE block before propagating.
      if (has_tail_reuse) {
        auto restored_it = all_blocks_.insert(all_blocks_.end(),
                                              std::move(combined_free_block));
        InsertFreeBlock(restored_it);
      }
      ClearGpuLastError();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes.\n"
          "Underlying VMM allocation exception:\n%s",
          static_cast<int>(pool_type_),
          grow_size,
          e.what()));
    } catch (...) {
      // Grow failed: restore the tail FREE block before propagating.
      if (has_tail_reuse) {
        auto restored_it = all_blocks_.insert(all_blocks_.end(),
                                              std::move(combined_free_block));
        InsertFreeBlock(restored_it);
      }
      ClearGpuLastError();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes "
          "with an unknown underlying VMM allocation exception.",
          static_cast<int>(pool_type_),
          grow_size));
    }
  }

  size_t total_new_size = tail_reuse_size;

  if (grow_alloc.HasAllocation()) {
    BlockV2 grow_block = AdoptBackingBlock(&grow_alloc);
    total_new_size += grow_block.size_;
    if (has_tail_reuse) {
      combined_free_block.AbsorbAdjacentBlockWithoutParts(&grow_block);
    } else {
      combined_free_block = std::move(grow_block);
    }
  }

  const size_t remaining_size = total_new_size - requested_size;

  BlockV2 block = combined_free_block.MakeMappedActiveSubBlockWithoutParts(
      0, requested_size);
  auto it = all_blocks_.insert(all_blocks_.end(), std::move(block));

  if (remaining_size > 0) {
    BlockV2 remaining_block =
        combined_free_block.MakeMappedFreeSubBlockWithoutParts(requested_size,
                                                               remaining_size);
    auto remain_it =
        all_blocks_.insert(std::next(it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  auto* allocation =
      new VMMAutoGrowthBestFitBlockAllocationV2(it, place_, this);
  record_alloc(grow_size > 0 ? "grow" : "tail_reuse", allocation);
  return allocation;
}

size_t VMMAutoGrowthBestFitAllocatorV2::CompactImpl(const Place& place,
                                                    size_t requested_size) {
  // Defensive place validation: the call chain
  // (RetryAllocator -> StreamSafe -> MultiPool -> SinglePool) guarantees
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
  size_t tail_free = 0;
  std::vector<std::pair<VMMDevicePtr, size_t>> compact_source_ranges;
  for (const auto& blk : all_blocks_) {
    if (blk.IsMappedFree()) {
      total_free += blk.size_;
      compact_source_ranges.emplace_back(blk.VARange());
    }
    if (CanIndexFreeBlock(blk)) {
      max_free = std::max(max_free, blk.size_);
    }
  }
  if (!all_blocks_.empty() && CanIndexFreeBlock(all_blocks_.back())) {
    tail_free = all_blocks_.back().size_;
  }

  size_t compact_target = requested_size;
  if (requested_size > 0) {
    if (max_free >= requested_size) {
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact skip: max_free=" << max_free
              << " >= requested=" << requested_size;
      return 0;
    }

    if (total_free < requested_size) {
      if (total_free <= tail_free) {
        VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
                << " compact skip: total_free=" << total_free
                << " < requested=" << requested_size
                << " and no non-tail free bytes are available"
                << " (tail_free=" << tail_free << ")";
        return 0;
      }
      // Partial compact: under tight training pressure, mapped-free bytes may
      // be insufficient to cover the whole request but still reduce the next
      // grow attempt. Move the non-tail free backing to tail/gaps and let the
      // following allocation retry grow only the remaining deficit.
      compact_target = total_free;
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact partial: total_free=" << total_free
              << " < requested=" << requested_size << " tail_free=" << tail_free
              << " compact_target=" << compact_target;
    }
  }

  // Count potentially movable source pages through the BackingMap mirror.
  // Runtime event readiness remains in RemapTransaction; this precheck only
  // verifies that free VA ranges fully cover mapped, reusable backing pages.
  // The source ranges include all mapped-free blocks. BackingMap page state
  // decides which individual handles are movable.
  const size_t required_releasable_bytes =
      compact_target > tail_free ? compact_target - tail_free : 0;
  const size_t releasable_target_bytes =
      requested_size > 0 && !FLAGS_vmm_v2_compact_all
          ? required_releasable_bytes
          : compact_target;

  auto source_pages = underlying_allocator_->CollectRemapSourcePages(
      compact_source_ranges, releasable_target_bytes);
  size_t releasable_handles = 0;
  for (const auto& page : source_pages) {
    if (page.remap_source_state == VMMBackingMap::RemapSourceState::kReady) {
      ++releasable_handles;
    }
  }
  const size_t releasable_bytes =
      releasable_handles * underlying_allocator_->HandleSize();

  if (requested_size > 0 && !FLAGS_vmm_v2_compact_all &&
      releasable_bytes < required_releasable_bytes) {
    VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " compact skip: releasable_bytes=" << releasable_bytes
            << " < required=" << required_releasable_bytes
            << " requested=" << requested_size << " total_free=" << total_free
            << " max_free=" << max_free << " tail_free=" << tail_free
            << " compact_target=" << compact_target
            << " source_ranges=" << compact_source_ranges.size();
    return 0;
  }

  if (releasable_handles == 0) {
    VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " compact skip: no releasable handles"
            << " (total_free=" << total_free << " max_free=" << max_free
            << " tail_free=" << tail_free << " requested=" << requested_size
            << " compact_target=" << compact_target
            << " releasable_handles=" << releasable_handles
            << " releasable_bytes=" << releasable_bytes
            << " source_ranges=" << compact_source_ranges.size() << ")";
    return 0;
  }

  VLOG(3) << "VMM V2 pool " << static_cast<int>(pool_type_)
          << " compact: total_free=" << total_free << " max_free=" << max_free
          << " tail_free=" << tail_free << " requested=" << requested_size
          << " compact_target=" << compact_target
          << " partial=" << (compact_target < requested_size)
          << " required_releasable_bytes=" << required_releasable_bytes
          << " releasable_handles=" << releasable_handles
          << " releasable_bytes=" << releasable_bytes
          << " source_ranges=" << compact_source_ranges.size()
          << ", proceeding with compaction";

  auto commit_synthetic_allocation = [this](DecoratedAllocationPtr allocation) {
    TrackUnderlyingAllocation(std::move(allocation));
  };
  auto can_prepare_synthetic_allocation = [this](void* ptr, size_t size) {
    return CanReleaseRemapDestinationUnderlyingAllocations(ptr, size);
  };
  auto prepare_synthetic_allocation = [this](void* ptr, size_t size) {
    return ReleaseRemapDestinationUnderlyingAllocations(ptr, size);
  };
  FreeBlockRemapCompactor compactor(underlying_allocator_,
                                    pool_type_,
                                    commit_synthetic_allocation,
                                    can_prepare_synthetic_allocation,
                                    prepare_synthetic_allocation);
  const bool compact_all = FLAGS_vmm_v2_compact_all;
  const size_t bounded_compact_target = compact_all ? 0 : compact_target;
  const size_t remapped =
      compactor.Compact(&all_blocks_, bounded_compact_target);
  // Always rebuild: Phase 1 may have replaced FREE blocks with
  // UNMAPPED-FREE/FREE
  // segments before Phase 2 fails.  Without rebuild, free_blocks_ holds
  // stale iterators to erased list nodes, causing use-after-free on next alloc.
  RebuildFreeBlockIndex();
  return remapped;
}

void VMMAutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  const bool trace_perf = VLOG_IS_ON(4);
  const bool trace_step = VMMV2StepStatsEnabled();
  const auto lock_wait_start =
      (trace_perf || trace_step) ? Clock::now() : Clock::time_point{};
  std::lock_guard<SpinLock> guard(spinlock_);
  const uint64_t lock_wait_us =
      (trace_perf || trace_step) ? ElapsedMicros(lock_wait_start, Clock::now())
                                 : 0;
  const auto op_start =
      (trace_perf || trace_step) ? Clock::now() : Clock::time_point{};
  auto* wrapped_allocation =
      static_cast<VMMAutoGrowthBestFitBlockAllocationV2*>(allocation);
  void* ptr = allocation->ptr();
  size_t allocation_size = allocation->size();
  auto it = wrapped_allocation->block_it();
  PADDLE_ENFORCE_NE(
      it,
      all_blocks_.end(),
      common::errors::NotFound("Can not find active block for allocation %p in "
                               "VMMAutoGrowthBestFitAllocatorV2.",
                               allocation->ptr()));
  VMMV2FreeDetailStats free_detail_stats;
  VMMV2FreeDetailStats* free_detail_stats_ptr =
      VMMV2DetailStatsEnabled() ? &free_detail_stats : nullptr;
  auto detail_tick = [&]() {
    return free_detail_stats_ptr != nullptr ? Clock::now()
                                            : Clock::time_point{};
  };
  auto detail_elapsed = [&](Clock::time_point start) -> uint64_t {
    return free_detail_stats_ptr != nullptr ? ElapsedMicros(start, Clock::now())
                                            : 0;
  };
  bool remap_safety_touched = false;
  auto remap_event = wrapped_allocation->TakeRemapEvent();
  const bool has_remap_state = wrapped_allocation->has_remap_state();
  const auto remap_stream = wrapped_allocation->remap_stream();
  if (has_remap_state) {
    remap_safety_touched = true;
  } else if (FLAGS_vmm_v2_remap_on_oom &&
             !FLAGS_vmm_v2_skip_remap_safety_hot_path) {
    remap_safety_touched = true;
    auto remap_safety_start = detail_tick();
    if (remap_stream == nullptr) {
      it->ClearRemapSafety();
    } else {
      it->SetRemapSafety(remap_stream, nullptr);
    }
    if (free_detail_stats_ptr != nullptr) {
      free_detail_stats.remap_safety_us += detail_elapsed(remap_safety_start);
    }
  }
  if (free_detail_stats_ptr != nullptr && remap_safety_touched) {
    free_detail_stats.remap_safety_count += 1;
  }
  auto mark_start = detail_tick();
  it->MarkFree();
  if (free_detail_stats_ptr != nullptr) {
    free_detail_stats.mark_free_us += detail_elapsed(mark_start);
  }
  auto merge_start = detail_tick();
  it = TryMerge(it, free_detail_stats_ptr);
  if (free_detail_stats_ptr != nullptr) {
    free_detail_stats.try_merge_us += detail_elapsed(merge_start);
  }
  if (has_remap_state) {
    auto remap_safety_start = detail_tick();
    it->AppendRemapSafety(remap_stream, remap_event);
    if (free_detail_stats_ptr != nullptr) {
      free_detail_stats.remap_safety_us += detail_elapsed(remap_safety_start);
    }
  }
  if (free_detail_stats_ptr != nullptr) {
    RecordVMMV2FreeDetail(place_.device, free_detail_stats);
  }
  if (trace_perf || trace_step) {
    const uint64_t elapsed_us = ElapsedMicros(op_start, Clock::now());
    if (trace_step) {
      RecordVMMV2Free(place_.device,
                      pool_type_,
                      allocation_size,
                      elapsed_us,
                      lock_wait_us,
                      all_blocks_.size(),
                      free_blocks_.size(),
                      unmapped_free_blocks_.size());
    }
    if (trace_perf && elapsed_us >= kSlowAllocatorOpLogUs) {
      VLOG(4) << "VMM V2 best-fit slow free"
              << " pool=" << static_cast<int>(pool_type_) << " ptr=" << ptr
              << " size=" << allocation_size << " elapsed_us=" << elapsed_us
              << " block_count=" << all_blocks_.size()
              << " free_blocks=" << free_blocks_.size()
              << " unmapped_free_blocks=" << unmapped_free_blocks_.size();
    }
  }
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

bool VMMAutoGrowthBestFitAllocatorV2::CollectTensorParts(
    void* ptr,
    size_t size,
    std::vector<BlockPart>* parts,
    bool mark_ipc_exported) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto target_va = reinterpret_cast<VMMDevicePtr>(ptr);
  PADDLE_ENFORCE_LE(
      size,
      std::numeric_limits<VMMDevicePtr>::max() - target_va,
      common::errors::InvalidArgument(
          "Invalid VMM V2 tensor range: ptr %p plus size %zu overflows.",
          ptr,
          size));
  BlockListIt block_it = all_blocks_.end();
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (!it->IsActive()) {
      continue;
    }
    if (it->ContainsVARange(target_va, size)) {
      block_it = it;
      break;
    }
  }
  if (block_it == all_blocks_.end()) {
    VLOG(4) << "[VMM-IPC/export] VMM v2 best-fit no active block for "
            << "target_ptr=" << ptr << " target_size=" << size
            << " pool=" << static_cast<int>(pool_type_)
            << " block_count=" << all_blocks_.size();
    return false;
  }

  std::vector<BlockPart> collected;
  if (!underlying_allocator_->CollectIpcParts(
          target_va, size, parts != nullptr ? &collected : nullptr)) {
    VLOG(4) << "[VMM-IPC/export] VMM v2 best-fit failed to collect backing "
            << "parts for active block ptr=" << block_it->ptr_
            << " block_size=" << block_it->size_ << " target_ptr=" << ptr
            << " target_size=" << size
            << " pool=" << static_cast<int>(pool_type_);
    return false;
  }
  if (mark_ipc_exported) {
    if (!underlying_allocator_->MarkIpcExported(target_va, size)) {
      VLOG(4) << "[VMM-IPC/export] VMM v2 best-fit failed to mark IPC exported "
              << "for active block ptr=" << block_it->ptr_
              << " block_size=" << block_it->size_ << " target_ptr=" << ptr
              << " target_size=" << size
              << " pool=" << static_cast<int>(pool_type_);
      return false;
    }
    block_it->ipc_exported_ = true;
  }
  if (parts != nullptr) {
    *parts = std::move(collected);
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(
    void* ptr, gpuStream_t stream, std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (!it->IsActive() || it->ptr_ != ptr) {
      continue;
    }
    it->SetRemapSafety(stream, std::move(event));
    return true;
  }
  return false;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(
    BlockListIt block_it,
    gpuStream_t stream,
    std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (block_it == all_blocks_.end() || !block_it->IsActive()) {
    return false;
  }
  block_it->SetRemapSafety(stream, std::move(event));
  return true;
}

BlockList VMMAutoGrowthBestFitAllocatorV2::SnapshotAllBlocks() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return all_blocks_;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size, VMMV2MappedFreeDetailStats* detail_stats) {
  auto detail_tick = [&]() {
    return detail_stats != nullptr ? Clock::now() : Clock::time_point{};
  };
  auto detail_elapsed = [&](Clock::time_point start) -> uint64_t {
    return detail_stats != nullptr ? ElapsedMicros(start, Clock::now()) : 0;
  };
  auto block_it = TryPopExactFreeBlock(size);
  if (block_it == all_blocks_.end()) {
    auto lower_bound_start = detail_tick();
    auto it = free_blocks_.lower_bound({size, nullptr});
    if (detail_stats != nullptr) {
      detail_stats->lower_bound_us += detail_elapsed(lower_bound_start);
    }
    while (it != free_blocks_.end() && !CanIndexFreeBlock(*it->second)) {
      auto stale_erase_start = detail_tick();
      it = free_blocks_.erase(it);
      if (detail_stats != nullptr) {
        ++detail_stats->stale_erase_count;
        detail_stats->stale_erase_us += detail_elapsed(stale_erase_start);
      }
    }
    if (it == free_blocks_.end()) {
      return nullptr;
    }
    block_it = it->second;
  }

  const size_t block_size = block_it->size_;
  auto erase_free_start = detail_tick();
  EraseFreeBlock(block_it);
  if (detail_stats != nullptr) {
    detail_stats->erase_free_us += detail_elapsed(erase_free_start);
  }

  const bool has_remainder = block_size > size;
  const size_t remaining_size = has_remainder ? block_size - size : 0;
  const bool consume_whole_block = ShouldConsumeWholeFreeBlock(remaining_size);
  if (has_remainder && !consume_whole_block) {
    auto split_start = detail_tick();
    BlockV2 remaining_block =
        block_it->MakeMappedFreeSubBlockWithoutParts(size, remaining_size);
    // The free remainder keeps the source block's remap-safety stream. The
    // reused prefix is cleared by MarkActive().

    *block_it = block_it->MakeMappedActiveSubBlockWithoutParts(0, size);
    if (detail_stats != nullptr) {
      ++detail_stats->split_count;
      detail_stats->split_us += detail_elapsed(split_start);
    }
    auto insert_block_start = detail_tick();
    auto remain_it =
        all_blocks_.insert(std::next(block_it), std::move(remaining_block));
    if (detail_stats != nullptr) {
      detail_stats->insert_block_us += detail_elapsed(insert_block_start);
    }
    auto insert_free_start = detail_tick();
    InsertFreeBlock(remain_it);
    if (detail_stats != nullptr) {
      detail_stats->insert_free_us += detail_elapsed(insert_free_start);
    }
  }

  auto mark_active_start = detail_tick();
  if (!has_remainder || consume_whole_block) {
    *block_it = block_it->MakeMappedActiveSubBlockWithoutParts(
        0, consume_whole_block ? block_size : size);
  }
  if (detail_stats != nullptr) {
    detail_stats->mark_active_us += detail_elapsed(mark_active_start);
  }
  auto wrapper_new_start = detail_tick();
  auto* allocation =
      new VMMAutoGrowthBestFitBlockAllocationV2(block_it, place_, this);
  if (detail_stats != nullptr) {
    detail_stats->wrapper_new_us += detail_elapsed(wrapper_new_start);
  }
  return allocation;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromUnmappedFreeBlocks(
    size_t size) {
  const size_t backing_size =
      AlignedSize(size, underlying_allocator_->HandleSize());
  BlockListIt best = all_blocks_.end();
  for (auto iter = unmapped_free_blocks_.lower_bound({backing_size, nullptr});
       iter != unmapped_free_blocks_.end();) {
    auto it = iter->second;
    if (!it->IsUnmappedFree()) {
      iter = unmapped_free_blocks_.erase(iter);
      continue;
    }
    if (RangeOverlapsUnderlyingAllocation(it->ptr_, backing_size)) {
      VLOG(6) << "VMM V2 AllocFromUnmappedFreeBlocks skip ownership-overlapped "
                 "unmapped-free ptr="
              << it->ptr_ << " backing_size=" << backing_size
              << " block_size=" << it->size_;
      ++iter;
      continue;
    }
    best = it;
    break;
  }
  if (best == all_blocks_.end()) {
    return nullptr;
  }

  const auto unmapped_free_ptr = best->BeginVA();
  VLOG(6) << "VMM V2 AllocFromUnmappedFreeBlocks ptr="
          << reinterpret_cast<void*>(unmapped_free_ptr) << " requested=" << size
          << " backing_size=" << backing_size
          << " original_unmapped_free_size=" << best->size_
          << " tail_offset=" << underlying_allocator_->TailOffset();
  CUDAVirtualMemAllocatorV2::AllocationWithBlock unmapped_free_alloc;
  try {
    unmapped_free_alloc = underlying_allocator_->PlaceAtVAWithBlock(
        unmapped_free_ptr, backing_size);
  } catch (...) {
    // Do not mutate the allocation view if backing cannot be created in this
    // unmapped-free range. The normal grow path will surface the allocation
    // failure if needed.
    return nullptr;
  }

  BlockV2 mapped_block = AdoptBackingBlock(&unmapped_free_alloc);
  PADDLE_ENFORCE_EQ(
      mapped_block.size_,
      backing_size,
      common::errors::InvalidArgument(
          "Unexpected unmapped-free backing size: got %zu, expected %zu.",
          mapped_block.size_,
          backing_size));

  const size_t original_unmapped_free_size = best->size_;
  const PoolType original_pool_type = best->pool_type_;

  EraseUnmappedFreeBlock(best);
  *best = mapped_block.MakeMappedActiveSubBlockWithoutParts(0, size);

  auto insert_pos = std::next(best);
  if (backing_size > size) {
    BlockV2 mapped_remain = mapped_block.MakeMappedFreeSubBlockWithoutParts(
        size, backing_size - size);
    mapped_remain.owning_stream_ = nullptr;
    mapped_remain.remap_safe_event_.reset();
    mapped_remain.remap_pending_states_.clear();
    auto free_it = all_blocks_.insert(insert_pos, std::move(mapped_remain));
    InsertFreeBlock(free_it);
    insert_pos = std::next(free_it);
  }

  if (original_unmapped_free_size > backing_size) {
    BlockV2 tail_unmapped_free = BlockV2::MakeUnmappedFreeBlock(
        reinterpret_cast<uint8_t*>(best->ptr_) + backing_size,
        original_unmapped_free_size - backing_size,
        original_pool_type);
    auto tail_it =
        all_blocks_.insert(insert_pos, std::move(tail_unmapped_free));
    InsertUnmappedFreeBlock(tail_it);
  }

  return new VMMAutoGrowthBestFitBlockAllocationV2(best, place_, this);
}

void VMMAutoGrowthBestFitAllocatorV2::TrackUnderlyingAllocation(
    DecoratedAllocationPtr allocation) {
  underlying_allocations_.Add(std::move(allocation));
}

bool VMMAutoGrowthBestFitAllocatorV2::AllocationOwnedByRemapDestination(
    const DecoratedAllocationPtr& allocation,
    void* target_ptr,
    size_t target_size) const {
  if (!underlying_allocator_->IsAllocationOwnedByRemapDestination(
          allocation->ptr())) {
    VLOG(0) << "VMM V2 synthetic allocation preparation: target range "
            << target_ptr << " size=" << target_size
            << " overlaps non-remap-destination underlying allocation "
            << allocation->ptr() << " size=" << allocation->size();
    return false;
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::
    CanReleaseRemapDestinationUnderlyingAllocations(void* ptr,
                                                    size_t size) const {
  return underlying_allocations_.AllOverlapsSatisfy(
      ptr, size, [this, ptr, size](const DecoratedAllocationPtr& allocation) {
        return AllocationOwnedByRemapDestination(allocation, ptr, size);
      });
}

bool VMMAutoGrowthBestFitAllocatorV2::
    ReleaseRemapDestinationUnderlyingAllocations(void* ptr, size_t size) {
  return underlying_allocations_.EraseOverlapsIf(
      ptr, size, [this, ptr, size](const DecoratedAllocationPtr& allocation) {
        if (!AllocationOwnedByRemapDestination(allocation, ptr, size)) {
          return false;
        }
        VLOG(3) << "VMM V2 synthetic allocation preparation: releasing stale "
                   "remap-destination allocation "
                << allocation->ptr() << " size=" << allocation->size();
        return true;
      });
}

BlockV2 VMMAutoGrowthBestFitAllocatorV2::AdoptBackingBlock(
    CUDAVirtualMemAllocatorV2::AllocationWithBlock* allocation_with_block) {
  PADDLE_ENFORCE_NOT_NULL(
      allocation_with_block,
      common::errors::InvalidArgument(
          "AllocationWithBlock must not be null when adopting block."));
  BlockV2 block = allocation_with_block->TakeBlock();
  auto allocation = static_unique_ptr_cast<Allocation>(
      allocation_with_block->TakeAllocation());
  TrackUnderlyingAllocation(std::move(allocation));
  return block;
}

bool VMMAutoGrowthBestFitAllocatorV2::RangeOverlapsUnderlyingAllocation(
    void* ptr, size_t size) const {
  return underlying_allocations_.Overlaps(ptr, size);
}

bool VMMAutoGrowthBestFitAllocatorV2::CanReleaseIdleUnderlyingAllocation(
    uint8_t* base, size_t size) const {
  if (!IsRangeEntirelyFree(base, size)) {
    return false;
  }
  return underlying_allocator_->IsRangeReleasable(
      reinterpret_cast<VMMDevicePtr>(base), size);
}

bool VMMAutoGrowthBestFitAllocatorV2::TryReleaseIdleUnderlyingAllocation(
    UnderlyingAllocationRegistry::iterator* alloc_it, uint64_t* released) {
  auto& allocation = **alloc_it;
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  const size_t alloc_size = allocation->size();
  if (!CanReleaseIdleUnderlyingAllocation(base, alloc_size)) {
    return false;
  }

  SplitAndReplaceRangeWithUnmappedFree(base, alloc_size);
  *released += alloc_size;
  VLOG(5) << "VMM V2 pool " << static_cast<int>(pool_type_)
          << " released idle chunk: " << alloc_size << " bytes";
  *alloc_it = underlying_allocations_.Erase(*alloc_it);
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanIndexFreeBlock(
    const BlockV2& block) const {
  return block.IsMappedFree() && !block.ipc_exported_;
}

void VMMAutoGrowthBestFitAllocatorV2::InsertFreeBlock(BlockListIt it) {
  if (!CanIndexFreeBlock(*it)) {
    return;
  }
  EmplaceOrEnforce(
      &free_blocks_, std::make_pair(it->size_, it->ptr_), it, "free_blocks_");
  if (FLAGS_vmm_v2_exact_free_block_cache) {
    exact_free_block_cache_ = it;
    has_exact_free_block_cache_ = true;
  }
}

void VMMAutoGrowthBestFitAllocatorV2::EraseFreeBlock(BlockListIt it) {
  ClearExactFreeBlockCacheIf(it);
  free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::InsertUnmappedFreeBlock(BlockListIt it) {
  if (!it->IsUnmappedFree()) {
    return;
  }
  EmplaceOrEnforce(&unmapped_free_blocks_,
                   std::make_pair(it->size_, it->ptr_),
                   it,
                   "unmapped_free_blocks_");
}

void VMMAutoGrowthBestFitAllocatorV2::EraseUnmappedFreeBlock(BlockListIt it) {
  unmapped_free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::RebuildFreeBlockIndex() {
  free_blocks_.clear();
  unmapped_free_blocks_.clear();
  has_exact_free_block_cache_ = false;
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (CanIndexFreeBlock(*it)) {
      InsertFreeBlock(it);
    }
    if (it->IsUnmappedFree()) {
      InsertUnmappedFreeBlock(it);
    }
  }
}

void VMMAutoGrowthBestFitAllocatorV2::ClearExactFreeBlockCacheIf(
    BlockListIt it) {
  if (has_exact_free_block_cache_ && exact_free_block_cache_ == it) {
    has_exact_free_block_cache_ = false;
  }
}

BlockListIt VMMAutoGrowthBestFitAllocatorV2::TryPopExactFreeBlock(size_t size) {
  if (!FLAGS_vmm_v2_exact_free_block_cache || !has_exact_free_block_cache_) {
    return all_blocks_.end();
  }
  auto cached = exact_free_block_cache_;
  if (cached == all_blocks_.end() || !CanIndexFreeBlock(*cached) ||
      cached->size_ != size) {
    has_exact_free_block_cache_ = false;
    return all_blocks_.end();
  }
  return cached;
}

BlockListIt VMMAutoGrowthBestFitAllocatorV2::TryMerge(
    BlockListIt it, VMMV2FreeDetailStats* detail) {
  auto detail_tick = [&]() {
    return detail != nullptr ? Clock::now() : Clock::time_point{};
  };
  auto detail_elapsed = [&](Clock::time_point start) -> uint64_t {
    return detail != nullptr ? ElapsedMicros(start, Clock::now()) : 0;
  };
  // Only adjacent FREE blocks are merged here. ACTIVE blocks are never touched,
  // and unmapped-free blocks remain as explicit holes for later remap/reuse.
  // all_blocks_ is the full VA-ordered block list, so adjacency is checked
  // against neighboring entries in that list.
  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->CanAbsorbAdjacentFreeBlock(*it)) {
      if (detail != nullptr) {
        ++detail->merge_prev_count;
      }
      auto erase_free_start = detail_tick();
      EraseFreeBlock(prev);
      if (detail != nullptr) {
        detail->erase_free_us += detail_elapsed(erase_free_start);
      }
      auto absorb_start = detail_tick();
      prev->AbsorbAdjacentBlockWithoutParts(&*it);
      if (detail != nullptr) {
        detail->absorb_us += detail_elapsed(absorb_start);
      }
      auto erase_block_start = detail_tick();
      all_blocks_.erase(it);
      if (detail != nullptr) {
        detail->erase_block_us += detail_elapsed(erase_block_start);
      }
      it = prev;
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() && it->CanAbsorbAdjacentFreeBlock(*next)) {
    if (detail != nullptr) {
      ++detail->merge_next_count;
    }
    auto erase_free_start = detail_tick();
    EraseFreeBlock(next);
    if (detail != nullptr) {
      detail->erase_free_us += detail_elapsed(erase_free_start);
    }
    auto absorb_start = detail_tick();
    it->AbsorbAdjacentBlockWithoutParts(&*next);
    if (detail != nullptr) {
      detail->absorb_us += detail_elapsed(absorb_start);
    }
    auto erase_block_start = detail_tick();
    all_blocks_.erase(next);
    if (detail != nullptr) {
      detail->erase_block_us += detail_elapsed(erase_block_start);
    }
  }

  if (CanIndexFreeBlock(*it)) {
    auto insert_free_start = detail_tick();
    InsertFreeBlock(it);
    if (detail != nullptr) {
      detail->insert_free_us += detail_elapsed(insert_free_start);
    }
  }
  return it;
}

void VMMAutoGrowthBestFitAllocatorV2::TryMergeUnmappedFree(BlockListIt it) {
  if (it == all_blocks_.end() || !it->IsUnmappedFree()) {
    return;
  }

  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->CanAbsorbAdjacentUnmappedFreeBlock(*it)) {
      EraseUnmappedFreeBlock(prev);
      EraseUnmappedFreeBlock(it);
      prev->AbsorbAdjacentUnmappedFreeBlock(*it);
      all_blocks_.erase(it);
      it = prev;
      InsertUnmappedFreeBlock(it);
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() &&
      it->CanAbsorbAdjacentUnmappedFreeBlock(*next)) {
    EraseUnmappedFreeBlock(it);
    EraseUnmappedFreeBlock(next);
    it->AbsorbAdjacentUnmappedFreeBlock(*next);
    all_blocks_.erase(next);
    InsertUnmappedFreeBlock(it);
  }
}

// ---------------------------------------------------------------------------
// ReleaseImpl / FreeIdleChunks: release underlying allocations whose entire
// VA range is covered by FREE blocks back to the CUDA VMM driver.
//
// Because TryMerge may have merged FREE blocks across allocation boundaries,
// we must split the spanning block at the allocation edges, release the
// backing, and keep the released VA range as explicit unmapped-free space for
// later reuse.
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
    if (!TryReleaseIdleUnderlyingAllocation(&alloc_it, &released)) {
      ++alloc_it;
    }
  }

  underlying_allocator_->SetTailOffset(ComputeTailOffset());
  return released;
}

size_t VMMAutoGrowthBestFitAllocatorV2::ComputeTailOffset() const {
  if (all_blocks_.empty()) {
    return 0;
  }
  return static_cast<size_t>(all_blocks_.back().EndVA() -
                             underlying_allocator_->VirtualMemBase());
}

bool VMMAutoGrowthBestFitAllocatorV2::IsRangeEntirelyFree(uint8_t* base,
                                                          size_t size) const {
  auto* end = base + size;
  for (const auto& block : all_blocks_) {
    auto* bptr = block.BeginPtr();
    auto* bend = block.EndPtr();
    if (bend <= base) continue;
    if (bptr >= end) break;
    if (block.IsActive()) {
      return false;
    }
  }
  // Returns true when the range contains only FREE/unmapped-free blocks or
  // when
  // blocks have already been removed by a prior FreeIdleChunks pass
  // (unmapped-free scatter / single-unmapped-free path case: the original
  // allocation's cleanup removes
  // blocks in the overlapping VA range before the synthetic allocation
  // is processed).  FreeImpl handles this safely: original allocation
  // skips remapped handles; synthetic allocation unmaps+releases its own.
  return true;
}

void VMMAutoGrowthBestFitAllocatorV2::SplitAndReplaceRangeWithUnmappedFree(
    uint8_t* base, size_t size) {
  auto* end = base + size;

  for (auto it = all_blocks_.begin(); it != all_blocks_.end();) {
    auto* bptr = it->BeginPtr();
    auto* bend = it->EndPtr();

    if (bend <= base) {
      ++it;
      continue;
    }
    if (bptr >= end) break;

    const bool is_unmapped_free = it->IsUnmappedFree();

    // Case 1: block entirely within [base, end): remove it.
    if (bptr >= base && bend <= end) {
      if (!is_unmapped_free) EraseFreeBlock(it);
      if (is_unmapped_free) EraseUnmappedFreeBlock(it);
      it = all_blocks_.erase(it);
      continue;
    }

    // Case 2: block straddles left boundary only: keep left remnant.
    if (bptr < base && bend <= end) {
      const size_t keep = static_cast<size_t>(base - bptr);
      if (!is_unmapped_free) {
        EraseFreeBlock(it);
        *it = it->MakeMappedFreeSubBlockWithoutParts(0, keep);
        InsertFreeBlock(it);
      } else {
        EraseUnmappedFreeBlock(it);
        it->TrimToPrefix(keep);
        InsertUnmappedFreeBlock(it);
      }
      ++it;
      continue;
    }

    // Case 3: block straddles right boundary only: keep right remnant.
    if (bptr >= base && bend > end) {
      const size_t trim = static_cast<size_t>(end - bptr);
      const size_t keep = it->size_ - trim;
      if (!is_unmapped_free) {
        EraseFreeBlock(it);
        *it = it->MakeMappedFreeSubBlockWithoutParts(trim, keep);
        InsertFreeBlock(it);
      } else {
        EraseUnmappedFreeBlock(it);
        it->TrimToSuffix(trim, keep);
        InsertUnmappedFreeBlock(it);
      }
      break;  // nothing more in range
    }

    // Case 4: block fully encompasses [base, end): split into two.
    if (bptr < base && bend > end) {
      const size_t left_size = static_cast<size_t>(base - bptr);
      const size_t right_offset = static_cast<size_t>(end - bptr);
      const size_t right_size = it->size_ - right_offset;

      if (!is_unmapped_free) {
        BlockV2 right =
            it->MakeMappedFreeSubBlockWithoutParts(right_offset, right_size);
        EraseFreeBlock(it);
        *it = it->MakeMappedFreeSubBlockWithoutParts(0, left_size);
        InsertFreeBlock(it);
        right.CopyRemapSafetyFrom(*it);
        auto right_it = all_blocks_.insert(std::next(it), std::move(right));
        InsertFreeBlock(right_it);
      } else {
        // Unmapped-free: just shrink left and insert right unmapped-free
        // block.
        BlockV2 right = it->MakeUnmappedFreeSubBlock(right_offset, right_size);
        EraseUnmappedFreeBlock(it);
        it->TrimToPrefix(left_size);
        InsertUnmappedFreeBlock(it);
        auto right_it = all_blocks_.insert(std::next(it), std::move(right));
        InsertUnmappedFreeBlock(right_it);
      }
      break;  // done
    }

    ++it;
  }

  auto insert_pos = all_blocks_.begin();
  while (insert_pos != all_blocks_.end() && insert_pos->BeginPtr() < base) {
    ++insert_pos;
  }
  auto unmapped_it = all_blocks_.insert(
      insert_pos, BlockV2::MakeUnmappedFreeBlock(base, size, pool_type_));
  InsertUnmappedFreeBlock(unmapped_it);
  TryMergeUnmappedFree(unmapped_it);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
