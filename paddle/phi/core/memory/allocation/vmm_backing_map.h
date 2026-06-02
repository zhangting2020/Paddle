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
#include <functional>
#include <utility>
#include <vector>

#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

// Page-granular backing state for VMM V2. Allocation blocks keep only logical
// VA layout; ownership, IPC pinning, event readiness, release safety and remap
// source eligibility are decided from this backing map.
class VMMBackingMap {
 public:
  enum class RemapSourceState : uint8_t {
    kReady = 0,
    kRemapDestinationOwned = 1,
    kPendingEvent = 2,
    kPartialOrInvalid = 3,
  };

  struct MappedPage {
    VMMDevicePtr va{0};
    VMMAllocHandle handle{0};
    std::shared_ptr<VMMHandleMeta> meta;
    uint64_t epoch{0};
    RemapSourceState remap_source_state{RemapSourceState::kReady};
  };
  struct UnmappedPage {
    VMMDevicePtr va{0};
    uint64_t epoch{0};
  };
  struct CompactCandidates {
    std::vector<MappedPage> source_pages;
    std::vector<UnmappedPage> target_pages;
  };

  void Configure(VMMDevicePtr base, size_t size, size_t page_size, int device);

  bool IsConfigured() const { return configured_; }

  void MarkMapped(VMMDevicePtr va, VMMAllocHandle handle, size_t size);
  void MarkMapped(VMMDevicePtr va,
                  const std::shared_ptr<VMMHandleMeta>& meta,
                  size_t size);
  void MarkUnmapped(VMMDevicePtr va, size_t size);
  void MarkReleased(VMMDevicePtr va, VMMAllocHandle handle, size_t size);
  void MarkIpcExported(VMMDevicePtr va, size_t size);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void MarkPendingEvent(VMMDevicePtr va,
                        size_t size,
                        gpuStream_t stream,
                        std::shared_ptr<CUDAEventGuard> event);
#endif

  bool ValidateLayout(const HandleLayout& layout, const char* context) const;
  bool CollectIpcPartDescriptors(
      VMMDevicePtr va,
      size_t size,
      std::vector<IpcBlockPartDescriptor>* descriptors) const;
  bool ForEachUniqueMappedHandle(
      VMMDevicePtr va,
      size_t size,
      const std::function<bool(const std::shared_ptr<VMMHandleMeta>&)>& fn)
      const;

  bool IsRangeMapped(VMMDevicePtr va, size_t size) const;
  bool IsRangeUnmapped(VMMDevicePtr va, size_t size) const;
  bool IsRangeReleasable(VMMDevicePtr va, size_t size) const;
  bool IsRangeReusableForAllocation(VMMDevicePtr va, size_t size) const;
  bool HasIpcExportedPages(VMMDevicePtr va, size_t size) const;
  std::vector<std::pair<VMMDevicePtr, size_t>> CollectMappedRanges(
      VMMDevicePtr va, size_t size) const;
  std::vector<std::pair<VMMDevicePtr, size_t>> CollectUnmappedRanges(
      VMMDevicePtr va, size_t size) const;
  std::vector<std::pair<VMMDevicePtr, size_t>> CollectMappedRanges(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  std::vector<std::pair<VMMDevicePtr, size_t>> CollectUnmappedRanges(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  std::vector<MappedPage> CollectMappedPages(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  std::vector<MappedPage> CollectMappedPages(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<MappedPage> CollectMappedPagesFullyCoveredBy(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  std::vector<MappedPage> CollectMappedPagesFullyCoveredBy(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<MappedPage> CollectRemapSourcePagesFullyCoveredBy(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<UnmappedPage> CollectUnmappedPagesFullyCoveredBy(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  std::vector<UnmappedPage> CollectUnmappedPagesFullyCoveredBy(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  CompactCandidates CollectCompactCandidates(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& source_ranges,
      const std::vector<std::pair<VMMDevicePtr, size_t>>& target_ranges,
      size_t target_bytes) const;
  bool ValidateMappedPages(const std::vector<MappedPage>& pages,
                           const char* context) const;
  bool ValidateUnmappedPages(const std::vector<UnmappedPage>& pages,
                             const char* context) const;
  size_t TotalMappedBytes() const;

 private:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  struct PendingEvent {
    gpuStream_t stream{nullptr};
    std::shared_ptr<CUDAEventGuard> event;
  };
#endif

  struct Page {
    VMMAllocHandle handle{0};
    std::shared_ptr<VMMHandleMeta> meta;
    bool mapped{false};
    bool ipc_exported{false};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    std::vector<PendingEvent> pending_events;
#endif
    uint64_t epoch{0};
  };

  bool CheckRangeLocked(VMMDevicePtr va,
                        size_t size,
                        const char* context,
                        size_t* start,
                        size_t* count) const;
  std::vector<std::pair<VMMDevicePtr, size_t>> CollectRangesLocked(
      VMMDevicePtr va, size_t size, bool mapped, const char* context) const;
  void AppendRangesLocked(VMMDevicePtr va,
                          size_t size,
                          bool mapped,
                          const char* context,
                          std::vector<std::pair<VMMDevicePtr, size_t>>*
                              ranges) const;
  void AppendMappedPagesLocked(VMMDevicePtr va,
                               size_t size,
                               const char* context,
                               size_t max_pages,
                               std::vector<MappedPage>* pages) const;
  bool CollectIpcPartDescriptorsLocked(
      VMMDevicePtr va,
      size_t size,
      std::vector<IpcBlockPartDescriptor>* descriptors) const;
  void AppendMappedPagesFullyCoveredByLocked(
      VMMDevicePtr va,
      size_t size,
      const char* context,
      size_t max_pages,
      bool require_events_ready,
      bool annotate_remap_source_state,
      std::vector<MappedPage>* pages) const;
  void AppendUnmappedPagesFullyCoveredByLocked(
      VMMDevicePtr va,
      size_t size,
      const char* context,
      size_t max_pages,
      std::vector<UnmappedPage>* pages) const;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool PageEventsReadyLocked(Page* page, const char* context) const;
#endif
  bool PageCanUseBackingLocked(Page* page, const char* context) const;
  RemapSourceState GetRemapSourceStateLocked(Page* page,
                                             const char* context) const;

  VMMDevicePtr base_{0};
  size_t size_{0};
  size_t page_size_{0};
  int device_{-1};
  bool configured_{false};
  mutable std::vector<Page> pages_;
  size_t mapped_page_count_{0};
  mutable SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
