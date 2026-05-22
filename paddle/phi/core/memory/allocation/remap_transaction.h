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

#include <cstddef>
#include <vector>

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

class RemapTransaction {
 public:
  struct PendingMappedRange {
    VmmDevicePtr dst{0};
    size_t handle_count{0};
  };

  RemapTransaction(CUDAVirtualMemAllocatorV2* vmm_allocator, size_t handle_size)
      : vmm_allocator_(vmm_allocator), handle_size_(handle_size) {}

  void SetCandidates(const VmmBackingMap::CompactCandidates& candidates);
  const VmmBackingMap::CompactCandidates& candidates() const {
    return candidates_;
  }

  bool ValidateSourcePages(const char* context) const;
  bool ValidateTargetPages(const char* context) const;

  void RecordMappedRange(VmmDevicePtr dst, size_t handle_count);
  void RollbackPendingMappings();
  void ClearPendingMappings() { pending_mapped_ranges_.clear(); }

 private:
  void UnmapPartialDestination(VmmDevicePtr dst_base, size_t handle_count);

  CUDAVirtualMemAllocatorV2* vmm_allocator_;
  size_t handle_size_;
  VmmBackingMap::CompactCandidates candidates_;
  std::vector<PendingMappedRange> pending_mapped_ranges_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
