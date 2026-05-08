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

#pragma once

#include <list>
#include <memory>

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

class FreeBlockRemapCompactor {
 public:
  FreeBlockRemapCompactor(
      const std::shared_ptr<CUDAVirtualMemAllocatorV2>& vmm_allocator,
      PoolType pool_type,
      std::list<DecoratedAllocationPtr>* underlying_allocations)
      : vmm_allocator_(vmm_allocator),
        pool_type_(pool_type),
        underlying_allocations_(underlying_allocations) {}

  // Remap fully-covered handles from FREE blocks to consolidate fragmented VA.
  // If requested_size > 0, performs bounded compaction: stops collecting
  // handles once enough are gathered to satisfy the requested allocation size.
  // If requested_size == 0, compacts all eligible handles (unbounded).
  size_t Compact(std::list<BlockV2>* blocks, size_t requested_size = 0);

 private:
  std::shared_ptr<CUDAVirtualMemAllocatorV2> vmm_allocator_;
  PoolType pool_type_;
  std::list<DecoratedAllocationPtr>* underlying_allocations_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
