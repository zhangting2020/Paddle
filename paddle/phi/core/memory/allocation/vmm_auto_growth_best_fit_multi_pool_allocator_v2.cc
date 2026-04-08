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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/alloc_hint.h"

COMMON_DECLARE_int32(vmm_v2_pool_mode);

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

}  // namespace

VMMAutoGrowthBestFitMultiPoolAllocatorV2::
    VMMAutoGrowthBestFitMultiPoolAllocatorV2(
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            stable_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            longlived_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            transient_small_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            transient_large_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            oversized_allocator,
        size_t transient_small_threshold,
        size_t oversized_threshold,
        const GPUPlace& place)
    : stable_allocator_(stable_allocator),
      longlived_allocator_(longlived_allocator),
      transient_small_allocator_(transient_small_allocator),
      transient_large_allocator_(transient_large_allocator),
      oversized_allocator_(oversized_allocator),
      transient_small_threshold_(transient_small_threshold),
      oversized_threshold_(oversized_threshold),
      place_(place) {}

phi::Allocation* VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocateImpl(
    size_t size) {
  const auto route = RouteAllocation(size);
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  auto allocation = route.allocator->Allocate(size);
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    EmplaceOrEnforce(&active_allocations_,
                     allocation->ptr(),
                     AllocationRoute{route.pool_type, route.allocator},
                     "active_allocations_");
  }
  return allocation.release();
}

size_t VMMAutoGrowthBestFitMultiPoolAllocatorV2::CompactImpl(
    const Place& place) {
  PADDLE_ENFORCE_EQ(
      place,
      place_,
      common::errors::InvalidArgument("VMM multipool V2 compact only supports "
                                      "its own place %s, but got %s.",
                                      place_,
                                      place));
  return stable_allocator_->Compact(place_) +
         longlived_allocator_->Compact(place_) +
         transient_small_allocator_->Compact(place_) +
         transient_large_allocator_->Compact(place_) +
         oversized_allocator_->Compact(place_);
}

uint64_t VMMAutoGrowthBestFitMultiPoolAllocatorV2::ReleaseImpl(
    const Place& place) {
  PADDLE_ENFORCE_EQ(
      place,
      place_,
      common::errors::InvalidArgument(
          "VMM multipool V2 release only supports its own place %s, "
          "but got %s.",
          place_,
          place));
  return stable_allocator_->Release(place_) +
         longlived_allocator_->Release(place_) +
         transient_small_allocator_->Release(place_) +
         transient_large_allocator_->Release(place_) +
         oversized_allocator_->Release(place_);
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::FreeImpl(
    phi::Allocation* allocation) {
  AllocationRoute route{PoolType::kTransientSmall, nullptr};
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(allocation->ptr());
    PADDLE_ENFORCE_NE(
        it,
        active_allocations_.end(),
        common::errors::NotFound(
            "No VMM pool routing metadata found for allocation %p.",
            allocation->ptr()));
    route = it->second;
    active_allocations_.erase(it);
  }
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  route.allocator->Free(allocation);
}

bool VMMAutoGrowthBestFitMultiPoolAllocatorV2::SetBlockRemapEvent(
    void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    gpuStream_t stream,
    gpuEvent_t event
#else
    void* stream,
    void* event
#endif
) {
  AllocationRoute route{PoolType::kTransientSmall, nullptr};
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      return false;
    }
    route = it->second;
  }
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  return route.allocator->SetBlockRemapEvent(ptr, stream, event);
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ExportForIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set "
      "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=0 or wait for W5."));
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ImportFromIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set "
      "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=0 or wait for W5."));
}

VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocationRoute
VMMAutoGrowthBestFitMultiPoolAllocatorV2::RouteAllocation(size_t size) const {
  const int pool_mode = FLAGS_vmm_v2_pool_mode;

  // Mode 2: single pool — everything to transient_small, all hints ignored.
  if (pool_mode == 2) {
    return {PoolType::kTransientSmall, transient_small_allocator_.get()};
  }

  // Mode 3: size-based 2-pool — small + large by threshold, all hints
  // ignored. Mirrors V1's small_pool / large_pool split for A/B comparison.
  if (pool_mode == 3) {
    if (size < transient_small_threshold_) {
      return {PoolType::kTransientSmall, transient_small_allocator_.get()};
    }
    return {PoolType::kTransientLarge, transient_large_allocator_.get()};
  }

  // Mode 0 (default 5-pool) and Mode 1 (Stable + Transient) both
  // respect kStable hint.
  const auto hint = GetCurrentPoolHint();
  if (hint == PoolHint::kStable) {
    return {PoolType::kStable, stable_allocator_.get()};
  }

  // Mode 1: Stable + Transient small/large.
  // kLongLived hint is ignored — falls through to size-based routing.
  // Oversized is disabled (routed to transient_large).
  if (pool_mode == 1) {
    if (size < transient_small_threshold_) {
      return {PoolType::kTransientSmall, transient_small_allocator_.get()};
    }
    return {PoolType::kTransientLarge, transient_large_allocator_.get()};
  }

  // Mode 0: default 5-pool routing.
  if (hint == PoolHint::kLongLived) {
    return {PoolType::kLongLived, longlived_allocator_.get()};
  }
  if (size >= oversized_threshold_) {
    return {PoolType::kOversized, oversized_allocator_.get()};
  }
  if (size < transient_small_threshold_) {
    return {PoolType::kTransientSmall, transient_small_allocator_.get()};
  }
  return {PoolType::kTransientLarge, transient_large_allocator_.get()};
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
