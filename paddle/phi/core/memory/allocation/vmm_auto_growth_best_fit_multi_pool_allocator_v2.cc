// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace memory {
namespace allocation {

VMMAutoGrowthBestFitMultiPoolAllocatorV2::
    VMMAutoGrowthBestFitMultiPoolAllocatorV2(
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            stable_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            longlived_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            transient_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            oversized_allocator,
        size_t oversized_threshold,
        const GPUPlace& place)
    : stable_allocator_(stable_allocator),
      longlived_allocator_(longlived_allocator),
      transient_allocator_(transient_allocator),
      oversized_allocator_(oversized_allocator),
      oversized_threshold_(oversized_threshold),
      place_(place) {}

phi::Allocation* VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocateImpl(
    size_t size) {
  auto pool = RouteAllocation(size);
  auto* allocator = GetPoolAllocator(pool);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(pool)));
  per_pool_stats_[static_cast<size_t>(pool)].alloc_count.fetch_add(1);
  per_pool_stats_[static_cast<size_t>(pool)].alloc_bytes.fetch_add(size);
  auto allocation = allocator->Allocate(size);
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    active_allocations_[allocation->ptr()] = pool;
  }
  return allocation.release();
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::FreeImpl(
    phi::Allocation* allocation) {
  PoolType pool = PoolType::kTransient;
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(allocation->ptr());
    PADDLE_ENFORCE_NE(
        it,
        active_allocations_.end(),
        common::errors::NotFound(
            "No VMM pool routing metadata found for allocation %p.",
            allocation->ptr()));
    pool = it->second;
    active_allocations_.erase(it);
  }
  auto* allocator = GetPoolAllocator(pool);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(pool)));
  per_pool_stats_[static_cast<size_t>(pool)].free_count.fetch_add(1);
  allocator->Free(allocation);
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
  PoolType pool = PoolType::kTransient;
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      return false;
    }
    pool = it->second;
  }
  auto* allocator = GetPoolAllocator(pool);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(pool)));
  return allocator->SetBlockRemapEvent(ptr, stream, event);
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ExportForIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set FLAGS_use_vmm_v2=0 or wait for W5"));
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ImportFromIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set FLAGS_use_vmm_v2=0 or wait for W5"));
}

PoolType VMMAutoGrowthBestFitMultiPoolAllocatorV2::RouteAllocation(
    size_t size) const {
  // TODO(zhangting35): W3 should route by tls_pool_hint first, then fallback
  // to size-based oversized routing.
  if (size >= oversized_threshold_) {
    return PoolType::kOversized;
  }
  return PoolType::kTransient;
}

VMMAutoGrowthBestFitAllocatorV2*
VMMAutoGrowthBestFitMultiPoolAllocatorV2::GetPoolAllocator(
    PoolType pool) const {
  switch (pool) {
    case PoolType::kStable:
      return stable_allocator_.get();
    case PoolType::kLongLived:
      return longlived_allocator_.get();
    case PoolType::kTransient:
      return transient_allocator_.get();
    case PoolType::kOversized:
      return oversized_allocator_.get();
  }
  return nullptr;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
