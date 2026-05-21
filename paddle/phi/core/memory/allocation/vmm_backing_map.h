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

#if defined(PADDLE_WITH_CUDA)

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

// Phase-1 backing view mirror for VMM V2. It does not own physical handles yet;
// it only mirrors successful driver map/unmap/release operations so we can
// validate the future AllocationView/BackingView split without changing
// allocation behavior.
class VmmBackingMap {
 public:
  struct MappedPage {
    VmmDevicePtr va{0};
    VmmAllocHandle handle{0};
    uint64_t epoch{0};
  };

  void Configure(VmmDevicePtr base, size_t size, size_t page_size, int device);

  bool configured() const { return configured_; }

  void MarkMapped(VmmDevicePtr va, VmmAllocHandle handle, size_t size);
  void MarkUnmapped(VmmDevicePtr va, size_t size);
  void MarkReleased(VmmDevicePtr va, VmmAllocHandle handle, size_t size);

  bool ValidateLayout(const HandleLayout& layout, const char* context) const;

  bool IsRangeMapped(VmmDevicePtr va, size_t size) const;
  bool IsRangeUnmapped(VmmDevicePtr va, size_t size) const;
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectMappedRanges(
      VmmDevicePtr va, size_t size) const;
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectUnmappedRanges(
      VmmDevicePtr va, size_t size) const;
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectMappedRanges(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const;
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectUnmappedRanges(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const;
  std::vector<MappedPage> CollectMappedPages(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const;
  size_t TotalMappedBytes() const;

 private:
  struct Page {
    VmmAllocHandle handle{0};
    bool mapped{false};
    uint64_t epoch{0};
  };

  bool CheckRangeLocked(VmmDevicePtr va,
                        size_t size,
                        const char* context,
                        size_t* start,
                        size_t* count) const;
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectRangesLocked(
      VmmDevicePtr va, size_t size, bool mapped, const char* context) const;
  void AppendRangesLocked(VmmDevicePtr va,
                          size_t size,
                          bool mapped,
                          const char* context,
                          std::vector<std::pair<VmmDevicePtr, size_t>>*
                              ranges) const;
  void AppendMappedPagesLocked(VmmDevicePtr va,
                               size_t size,
                               const char* context,
                               std::vector<MappedPage>* pages) const;

  VmmDevicePtr base_{0};
  size_t size_{0};
  size_t page_size_{0};
  int device_{-1};
  bool configured_{false};
  std::vector<Page> pages_;
  mutable SpinLock mu_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
