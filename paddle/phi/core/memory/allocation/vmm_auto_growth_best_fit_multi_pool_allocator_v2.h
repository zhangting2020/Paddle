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

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2_types.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

class VMMAutoGrowthBestFitMultiPoolAllocatorV2 : public Allocator {
 public:
  VMMAutoGrowthBestFitMultiPoolAllocatorV2(
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& stable_allocator,
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
          longlived_allocator,
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
          transient_allocator,
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
          oversized_allocator,
      size_t oversized_threshold,
      const GPUPlace& place);

  bool IsAllocThreadSafe() const override { return true; }

  bool SetBlockRemapEvent(void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                          gpuStream_t stream,
                          gpuEvent_t event
#else
                          void* stream,
                          void* event
#endif
  );

  [[noreturn]] void ExportForIpc();
  [[noreturn]] void ImportFromIpc();

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  PoolType RouteAllocation(size_t size) const;
  VMMAutoGrowthBestFitAllocatorV2* GetPoolAllocator(PoolType pool) const;

  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> stable_allocator_;
  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> longlived_allocator_;
  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> transient_allocator_;
  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> oversized_allocator_;
  size_t oversized_threshold_;
  GPUPlace place_;
  std::array<PerPoolStats, 4> per_pool_stats_;
  std::unordered_map<void*, PoolType> active_allocations_;
  mutable SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
