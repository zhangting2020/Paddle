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

#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

#if defined(PADDLE_WITH_CUDA)

#include <algorithm>
#include <mutex>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

bool ComputeOverlappedPages(VMMDevicePtr base,
                            size_t backing_size,
                            size_t page_size,
                            VMMDevicePtr va,
                            size_t size,
                            const char* context,
                            size_t* start,
                            size_t* count) {
  if (size == 0 || page_size == 0 || va < base || va + size < va ||
      va + size > base + backing_size) {
    VLOG(0) << "VMM V2 BackingMap invalid overlap range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base)
            << " backing_size=" << backing_size
            << " page_size=" << page_size;
    return false;
  }
  const size_t begin_offset = va - base;
  const size_t end_offset = va + size - base;
  *start = begin_offset / page_size;
  const size_t end_page = (end_offset + page_size - 1) / page_size;
  *count = end_page - *start;
  return true;
}

}  // namespace

void VMMBackingMap::Configure(VMMDevicePtr base,
                              size_t size,
                              size_t page_size,
                              int device) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (configured_) {
    if (base_ != base || size_ != size || page_size_ != page_size ||
        device_ != device) {
      VLOG(0) << "VMM V2 BackingMap reconfigure mismatch: old_base="
              << reinterpret_cast<void*>(base_) << " new_base="
              << reinterpret_cast<void*>(base) << " old_size=" << size_
              << " new_size=" << size << " old_page_size=" << page_size_
              << " new_page_size=" << page_size << " old_device=" << device_
              << " new_device=" << device;
    }
    return;
  }

  base_ = base;
  size_ = size;
  page_size_ = page_size;
  device_ = device;
  configured_ = true;
  pages_.resize(size_ / page_size_);
  mapped_page_count_ = 0;
}

bool VMMBackingMap::CheckRangeLocked(VMMDevicePtr va,
                                     size_t size,
                                     const char* context,
                                     size_t* start,
                                     size_t* count) const {
  if (!configured_) {
    VLOG(0) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return false;
  }
  if (size == 0 || page_size_ == 0 || size % page_size_ != 0 ||
      va < base_ || va + size < va || va + size > base_ + size_ ||
      (va - base_) % page_size_ != 0) {
    VLOG(0) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return false;
  }
  *start = (va - base_) / page_size_;
  *count = size / page_size_;
  return true;
}

void VMMBackingMap::MarkMapped(VMMDevicePtr va,
                               VMMAllocHandle handle,
                               size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkMapped(handle)", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    PADDLE_ENFORCE_EQ(
        page.mapped && handle != 0 && page.handle != handle,
        false,
        common::errors::PreconditionNotMet(
            "VMM V2 BackingMap cannot overwrite mapped page at %p from "
            "handle %p to %p.",
            reinterpret_cast<void*>(va + i * page_size_),
            reinterpret_cast<void*>(page.handle),
            reinterpret_cast<void*>(handle)));
    if (!page.mapped) {
      mapped_page_count_++;
    }
    page.handle = handle;
    page.meta.reset();
    page.mapped = true;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    page.pending_events.clear();
#endif
    page.epoch++;
  }
}

void VMMBackingMap::MarkMapped(
    VMMDevicePtr va,
    const std::shared_ptr<VMMHandleMeta>& meta,
    size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkMapped", &start, &count)) {
    return;
  }
  const VMMAllocHandle handle =
      meta == nullptr ? static_cast<VMMAllocHandle>(0)
                      : meta->AllocationHandle();
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    PADDLE_ENFORCE_EQ(
        page.mapped && handle != 0 && page.handle != handle,
        false,
        common::errors::PreconditionNotMet(
            "VMM V2 BackingMap cannot overwrite mapped page at %p from "
            "handle %p to %p.",
            reinterpret_cast<void*>(va + i * page_size_),
            reinterpret_cast<void*>(page.handle),
            reinterpret_cast<void*>(handle)));
    if (!page.mapped) {
      mapped_page_count_++;
    }
    page.handle = handle;
    page.meta = meta;
    page.mapped = true;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    page.pending_events.clear();
#endif
    page.epoch++;
  }
}

void VMMBackingMap::MarkUnmapped(VMMDevicePtr va, size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkUnmapped", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(5) << "VMM V2 BackingMap unmapping already-unmapped page at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    if (page.mapped && mapped_page_count_ > 0) {
      mapped_page_count_--;
    }
    page.handle = 0;
    page.meta.reset();
    page.mapped = false;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    page.pending_events.clear();
#endif
    page.epoch++;
  }
}

void VMMBackingMap::MarkReleased(VMMDevicePtr va,
                                 VMMAllocHandle handle,
                                 size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkReleased", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (handle != 0 && page.handle != 0 && page.handle != handle) {
      VLOG(0) << "VMM V2 BackingMap release handle mismatch at "
              << reinterpret_cast<void*>(va + i * page_size_)
              << " tracked=" << reinterpret_cast<void*>(page.handle)
              << " released=" << reinterpret_cast<void*>(handle);
    }
    if (page.mapped && mapped_page_count_ > 0) {
      mapped_page_count_--;
    }
    page.handle = 0;
    page.meta.reset();
    page.mapped = false;
    page.ipc_exported = false;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    page.pending_events.clear();
#endif
    page.epoch++;
  }
}

void VMMBackingMap::MarkIpcExported(VMMDevicePtr va, size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(
          base_, size_, page_size_, va, size, "MarkIpcExported", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(4) << "VMM V2 BackingMap marks unmapped page as IPC-exported at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    page.ipc_exported = true;
    page.epoch++;
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void VMMBackingMap::MarkPendingEvent(VMMDevicePtr va,
                                     size_t size,
                                     gpuStream_t stream,
                                     std::shared_ptr<CUDAEventGuard> event) {
  if (event == nullptr) {
    return;
  }
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkPendingEvent", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(6) << "VMM V2 BackingMap marks unmapped page event-pending at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    auto same_stream = std::find_if(
        page.pending_events.begin(),
        page.pending_events.end(),
        [stream](const PendingEvent& pending) {
          return pending.stream == stream;
        });
    if (same_stream != page.pending_events.end()) {
      same_stream->event = event;
    } else {
      page.pending_events.push_back(PendingEvent{stream, event});
    }
    page.epoch++;
  }
}
#endif

bool VMMBackingMap::ValidateLayout(const HandleLayout& layout,
                                   const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& meta : layout) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(meta->Base(), meta->Size(), context, &start, &count)) {
      ok = false;
      continue;
    }
    for (size_t i = 0; i < count; ++i) {
      const auto& page = pages_[start + i];
      const bool expected_mapped = !meta->IsOwnedByRemapDestination();
      if (page.mapped != expected_mapped) {
        VLOG(0) << "VMM V2 BackingMap mapped-state mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->Base() + i * page_size_)
                << " tracked_mapped=" << page.mapped
                << " meta_owned_by_remap_destination="
                << meta->IsOwnedByRemapDestination();
        ok = false;
      }
      if (expected_mapped && page.handle != meta->AllocationHandle()) {
        VLOG(0) << "VMM V2 BackingMap handle mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->Base() + i * page_size_)
                << " tracked=" << reinterpret_cast<void*>(page.handle)
                << " meta="
                << reinterpret_cast<void*>(meta->AllocationHandle());
        ok = false;
      }
    }
  }
  return ok;
}

bool VMMBackingMap::CollectIpcPartDescriptors(
    VMMDevicePtr va,
    size_t size,
    std::vector<IpcBlockPartDescriptor>* descriptors) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return CollectIpcPartDescriptorsLocked(va, size, descriptors);
}

bool VMMBackingMap::ForEachUniqueMappedHandle(
    VMMDevicePtr va,
    size_t size,
    const std::function<bool(const std::shared_ptr<VMMHandleMeta>&)>& fn)
    const {
  std::vector<std::shared_ptr<VMMHandleMeta>> handles;
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    size_t start = 0;
    size_t count = 0;
    if (!ComputeOverlappedPages(base_,
                                size_,
                                page_size_,
                                va,
                                size,
                                "ForEachUniqueMappedHandle",
                                &start,
                                &count)) {
      return false;
    }
    std::unordered_set<VMMHandleMeta*> seen;
    handles.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const auto& page = pages_[start + i];
      if (!page.mapped || page.meta == nullptr) {
        return false;
      }
      if (!seen.insert(page.meta.get()).second) {
        continue;
      }
      handles.push_back(page.meta);
    }
  }
  for (const auto& handle : handles) {
    if (!fn(handle)) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeMapped(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "IsRangeMapped", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!pages_[start + i].mapped) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeUnmapped(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "IsRangeUnmapped", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].mapped) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeReleasable(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "IsRangeReleasable",
                              &start,
                              &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].ipc_exported ||
        !PageCanUseBackingLocked(&pages_[start + i], "IsRangeReleasable")) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeReusableForAllocation(VMMDevicePtr va,
                                                 size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "IsRangeReusableForAllocation",
                              &start,
                              &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    auto* page = &pages_[start + i];
    if (!page->mapped || page->ipc_exported ||
        !PageCanUseBackingLocked(page, "IsRangeReusableForAllocation")) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::HasIpcExportedPages(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "HasIpcExportedPages",
                              &start,
                              &count)) {
    return true;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].ipc_exported) {
      return true;
    }
  }
  return false;
}

std::vector<std::pair<VMMDevicePtr, size_t>>
VMMBackingMap::CollectMappedRanges(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return CollectRangesLocked(va, size, true, "CollectMappedRanges");
}

std::vector<std::pair<VMMDevicePtr, size_t>>
VMMBackingMap::CollectUnmappedRanges(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return CollectRangesLocked(va, size, false, "CollectUnmappedRanges");
}

std::vector<std::pair<VMMDevicePtr, size_t>>
VMMBackingMap::CollectMappedRanges(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<std::pair<VMMDevicePtr, size_t>> mapped_ranges;
  for (const auto& range : ranges) {
    AppendRangesLocked(
        range.first, range.second, true, "CollectMappedRanges", &mapped_ranges);
  }
  return mapped_ranges;
}

std::vector<std::pair<VMMDevicePtr, size_t>>
VMMBackingMap::CollectUnmappedRanges(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<std::pair<VMMDevicePtr, size_t>> unmapped_ranges;
  for (const auto& range : ranges) {
    AppendRangesLocked(range.first,
                       range.second,
                       false,
                       "CollectUnmappedRanges",
                       &unmapped_ranges);
  }
  return unmapped_ranges;
}

std::vector<VMMBackingMap::MappedPage> VMMBackingMap::CollectMappedPages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  for (const auto& range : ranges) {
    AppendMappedPagesLocked(
        range.first, range.second, "CollectMappedPages", 0, &mapped_pages);
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage> VMMBackingMap::CollectMappedPages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  if (target_bytes == 0 || page_size_ == 0) {
    return mapped_pages;
  }

  const size_t target_pages = (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesLocked(range.first,
                            range.second,
                            "CollectMappedPages",
                            target_pages,
                            &mapped_pages);
    if (mapped_pages.size() >= target_pages) {
      break;
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage>
VMMBackingMap::CollectMappedPagesFullyCoveredBy(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyCoveredByLocked(range.first,
                                          range.second,
                                          "CollectMappedPagesFullyCoveredBy",
                                          0,
                                          true,
                                          false,
                                          &mapped_pages);
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage>
VMMBackingMap::CollectMappedPagesFullyCoveredBy(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  if (target_bytes == 0 || page_size_ == 0) {
    return mapped_pages;
  }

  const size_t target_pages = (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyCoveredByLocked(range.first,
                                          range.second,
                                          "CollectMappedPagesFullyCoveredBy",
                                          target_pages,
                                          true,
                                          false,
                                          &mapped_pages);
    if (mapped_pages.size() >= target_pages) {
      break;
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage>
VMMBackingMap::CollectRemapSourcePagesFullyCoveredBy(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  const size_t target_pages =
      (target_bytes == 0 || page_size_ == 0)
          ? 0
          : (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyCoveredByLocked(range.first,
                                          range.second,
                                          "CollectRemapSourcePages"
                                          "FullyCoveredBy",
                                          0,
                                          false,
                                          true,
                                          &mapped_pages);
    if (target_pages != 0) {
      size_t ready_pages = 0;
      for (const auto& page : mapped_pages) {
        if (page.remap_source_state == RemapSourceState::kReady) {
          ++ready_pages;
        }
      }
      if (ready_pages >= target_pages) {
        break;
      }
    }
  }
  if (target_pages != 0) {
    size_t ready_pages = 0;
    size_t keep_pages = mapped_pages.size();
    for (size_t i = 0; i < mapped_pages.size(); ++i) {
      if (mapped_pages[i].remap_source_state == RemapSourceState::kReady) {
        ++ready_pages;
        if (ready_pages >= target_pages) {
          keep_pages = i + 1;
          break;
        }
      }
    }
    if (keep_pages < mapped_pages.size()) {
      mapped_pages.resize(keep_pages);
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::UnmappedPage>
VMMBackingMap::CollectUnmappedPagesFullyCoveredBy(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<UnmappedPage> unmapped_pages;
  for (const auto& range : ranges) {
    AppendUnmappedPagesFullyCoveredByLocked(
        range.first,
        range.second,
        "CollectUnmappedPagesFullyCoveredBy",
        0,
        &unmapped_pages);
  }
  return unmapped_pages;
}

std::vector<VMMBackingMap::UnmappedPage>
VMMBackingMap::CollectUnmappedPagesFullyCoveredBy(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<UnmappedPage> unmapped_pages;
  if (target_bytes == 0 || page_size_ == 0) {
    return unmapped_pages;
  }

  const size_t target_pages = (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendUnmappedPagesFullyCoveredByLocked(
        range.first,
        range.second,
        "CollectUnmappedPagesFullyCoveredBy",
        target_pages,
        &unmapped_pages);
    if (unmapped_pages.size() >= target_pages) {
      break;
    }
  }
  return unmapped_pages;
}

VMMBackingMap::CompactCandidates VMMBackingMap::CollectCompactCandidates(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& source_ranges,
    const std::vector<std::pair<VMMDevicePtr, size_t>>& target_ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  CompactCandidates candidates;
  const size_t max_pages =
      (target_bytes == 0 || page_size_ == 0)
          ? 0
          : (target_bytes + page_size_ - 1) / page_size_;

  for (const auto& range : source_ranges) {
    AppendMappedPagesFullyCoveredByLocked(range.first,
                                          range.second,
                                          "CollectCompactCandidates.source",
                                          max_pages,
                                          true,
                                          false,
                                          &candidates.source_pages);
    if (max_pages != 0 && candidates.source_pages.size() >= max_pages) {
      break;
    }
  }
  for (const auto& range : target_ranges) {
    AppendUnmappedPagesFullyCoveredByLocked(range.first,
                                            range.second,
                                            "CollectCompactCandidates.target",
                                            max_pages,
                                            &candidates.target_pages);
    if (max_pages != 0 && candidates.target_pages.size() >= max_pages) {
      break;
    }
  }
  return candidates;
}

bool VMMBackingMap::ValidateMappedPages(
    const std::vector<MappedPage>& mapped_pages, const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& mapped_page : mapped_pages) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(
            mapped_page.va, page_size_, context, &start, &count)) {
      ok = false;
      continue;
    }
    if (count != 1) {
      VLOG(0) << "VMM V2 BackingMap invalid mapped page count in " << context
              << ": va=" << reinterpret_cast<void*>(mapped_page.va)
              << " count=" << count;
      ok = false;
      continue;
    }

    const auto& page = pages_[start];
    if (!page.mapped) {
      VLOG(0) << "VMM V2 BackingMap mapped page became unmapped in "
              << context << ": va="
              << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_epoch=" << mapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
      continue;
    }
    if (page.handle != mapped_page.handle) {
      VLOG(0) << "VMM V2 BackingMap mapped page handle changed in "
              << context << ": va="
              << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_handle="
              << reinterpret_cast<void*>(mapped_page.handle)
              << " current_handle=" << reinterpret_cast<void*>(page.handle);
      ok = false;
    }
    if (page.epoch != mapped_page.epoch) {
      VLOG(0) << "VMM V2 BackingMap mapped page epoch changed in " << context
              << ": va=" << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_epoch=" << mapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
    }
  }
  return ok;
}

bool VMMBackingMap::ValidateUnmappedPages(
    const std::vector<UnmappedPage>& unmapped_pages, const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& unmapped_page : unmapped_pages) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(
            unmapped_page.va, page_size_, context, &start, &count)) {
      ok = false;
      continue;
    }
    if (count != 1) {
      VLOG(0) << "VMM V2 BackingMap invalid unmapped page count in "
              << context << ": va="
              << reinterpret_cast<void*>(unmapped_page.va)
              << " count=" << count;
      ok = false;
      continue;
    }

    const auto& page = pages_[start];
    if (page.mapped) {
      VLOG(0) << "VMM V2 BackingMap unmapped page became mapped in "
              << context << ": va="
              << reinterpret_cast<void*>(unmapped_page.va)
              << " snapshot_epoch=" << unmapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
      continue;
    }
    if (page.handle != 0) {
      VLOG(0) << "VMM V2 BackingMap unmapped page retains handle in "
              << context << ": va="
              << reinterpret_cast<void*>(unmapped_page.va)
              << " handle=" << reinterpret_cast<void*>(page.handle);
      ok = false;
    }
    if (page.epoch != unmapped_page.epoch) {
      VLOG(0) << "VMM V2 BackingMap unmapped page epoch changed in "
              << context << ": va="
              << reinterpret_cast<void*>(unmapped_page.va)
              << " snapshot_epoch=" << unmapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
    }
  }
  return ok;
}

std::vector<std::pair<VMMDevicePtr, size_t>>
VMMBackingMap::CollectRangesLocked(VMMDevicePtr va,
                                   size_t size,
                                   bool mapped,
                                   const char* context) const {
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges;
  AppendRangesLocked(va, size, mapped, context, &ranges);
  return ranges;
}

void VMMBackingMap::AppendRangesLocked(
    VMMDevicePtr va,
    size_t size,
    bool mapped,
    const char* context,
    std::vector<std::pair<VMMDevicePtr, size_t>>* ranges) const {
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, context, &start, &count)) {
    return;
  }

  bool in_range = false;
  VMMDevicePtr range_begin = 0;
  size_t range_size = 0;
  for (size_t i = 0; i < count; ++i) {
    const bool selected = pages_[start + i].mapped == mapped;
    const VMMDevicePtr page_va = va + i * page_size_;
    if (selected) {
      if (!in_range) {
        in_range = true;
        range_begin = page_va;
        range_size = 0;
      }
      range_size += page_size_;
      continue;
    }

    if (in_range) {
      if (!ranges->empty() &&
          ranges->back().first + ranges->back().second == range_begin) {
        ranges->back().second += range_size;
      } else {
        ranges->emplace_back(range_begin, range_size);
      }
      in_range = false;
    }
  }
  if (in_range) {
    if (!ranges->empty() &&
        ranges->back().first + ranges->back().second == range_begin) {
      ranges->back().second += range_size;
    } else {
      ranges->emplace_back(range_begin, range_size);
    }
  }
}

void VMMBackingMap::AppendMappedPagesLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    std::vector<MappedPage>* mapped_pages) const {
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, context, &start, &count)) {
    return;
  }

  for (size_t i = 0; i < count; ++i) {
    if (max_pages != 0 && mapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[start + i];
    if (!page.mapped || page.ipc_exported ||
        !PageCanUseBackingLocked(&pages_[start + i], context)) {
      continue;
    }
    mapped_pages->push_back(
        MappedPage{va + i * page_size_, page.handle, page.meta, page.epoch});
  }
}

void VMMBackingMap::AppendMappedPagesFullyCoveredByLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    bool require_events_ready,
    bool annotate_remap_source_state,
    std::vector<MappedPage>* mapped_pages) const {
  if (!configured_) {
    VLOG(0) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return;
  }
  if (size == 0 || page_size_ == 0 || va < base_ || va + size < va ||
      va + size > base_ + size_) {
    VLOG(0) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return;
  }

  const VMMDevicePtr range_end = va + size;
  const size_t start_offset = va - base_;
  const size_t end_offset = range_end - base_;
  const size_t first_page =
      (start_offset + page_size_ - 1) / page_size_;
  const size_t end_page = end_offset / page_size_;
  if (first_page >= end_page) {
    return;
  }

  for (size_t page_idx = first_page; page_idx < end_page; ++page_idx) {
    if (max_pages != 0 && mapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[page_idx];
    if (!page.mapped || page.ipc_exported) {
      continue;
    }
    auto remap_source_state = RemapSourceState::kReady;
    if (annotate_remap_source_state) {
      remap_source_state =
          GetRemapSourceStateLocked(&pages_[page_idx], context);
    } else if (require_events_ready &&
               !PageCanUseBackingLocked(&pages_[page_idx], context)) {
      continue;
    }
    mapped_pages->push_back(MappedPage{
        base_ + page_idx * page_size_,
        page.handle,
        page.meta,
        page.epoch,
        remap_source_state});
  }
}

bool VMMBackingMap::CollectIpcPartDescriptorsLocked(
    VMMDevicePtr va,
    size_t size,
    std::vector<IpcBlockPartDescriptor>* descriptors) const {
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "CollectIpcPartDescriptors",
                              &start,
                              &count)) {
    return false;
  }
  if (descriptors != nullptr) {
    descriptors->clear();
    descriptors->reserve(count);
  }
  for (size_t i = 0; i < count; ++i) {
    const auto& page = pages_[start + i];
    if (!page.mapped || page.meta == nullptr ||
        page.meta->IsOwnedByRemapDestination()) {
      return false;
    }
    if (descriptors != nullptr) {
      const VMMDevicePtr page_va = base_ + (start + i) * page_size_;
      const VMMDevicePtr slice_begin = std::max(va, page_va);
      const VMMDevicePtr slice_end = std::min(va + size, page_va + page_size_);
      descriptors->push_back(IpcBlockPartDescriptor{
          page.meta->Base(),
          page.meta->Size(),
          page.meta->AllocationHandle(),
          page.meta->Device(),
          static_cast<size_t>(slice_begin - page_va),
          static_cast<size_t>(slice_end - slice_begin),
      });
    }
  }
  return true;
}

void VMMBackingMap::AppendUnmappedPagesFullyCoveredByLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    std::vector<UnmappedPage>* unmapped_pages) const {
  if (!configured_) {
    VLOG(0) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return;
  }
  if (size == 0 || page_size_ == 0 || va < base_ || va + size < va ||
      va + size > base_ + size_) {
    VLOG(0) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return;
  }

  const VMMDevicePtr range_end = va + size;
  const size_t start_offset = va - base_;
  const size_t end_offset = range_end - base_;
  const size_t first_page = (start_offset + page_size_ - 1) / page_size_;
  const size_t end_page = end_offset / page_size_;
  if (first_page >= end_page) {
    return;
  }

  for (size_t page_idx = first_page; page_idx < end_page; ++page_idx) {
    if (max_pages != 0 && unmapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[page_idx];
    if (page.mapped) {
      continue;
    }
    unmapped_pages->push_back(
        UnmappedPage{base_ + page_idx * page_size_, page.epoch});
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool VMMBackingMap::PageEventsReadyLocked(Page* page,
                                          const char* context) const {
  for (auto it = page->pending_events.begin();
       it != page->pending_events.end();) {
    if (it->event == nullptr || it->event->event == nullptr) {
      it = page->pending_events.erase(it);
      continue;
    }
#ifdef PADDLE_WITH_CUDA
    gpuError_t err = cudaEventQuery(it->event->event);
    if (err != cudaSuccess && err != cudaErrorNotReady) {
      PADDLE_ENFORCE_GPU_SUCCESS(err);
    }
#else
    gpuError_t err = hipEventQuery(it->event->event);
    if (err != hipSuccess && err != hipErrorNotReady) {
      PADDLE_ENFORCE_GPU_SUCCESS(err);
    }
#endif
    if (
#ifdef PADDLE_WITH_CUDA
        err == cudaSuccess
#else
        err == hipSuccess
#endif
    ) {
      it = page->pending_events.erase(it);
      continue;
    }
    VLOG(6) << "VMM V2 BackingMap page blocked by pending event in "
            << context;
    return false;
  }
  return true;
}
#endif

bool VMMBackingMap::PageCanUseBackingLocked(Page* page,
                                            const char* context) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return PageEventsReadyLocked(page, context);
#else
  (void)page;
  (void)context;
  return true;
#endif
}

VMMBackingMap::RemapSourceState VMMBackingMap::GetRemapSourceStateLocked(
    Page* page,
    const char* context) const {
  if (page == nullptr || page->meta == nullptr) {
    return RemapSourceState::kPartialOrInvalid;
  }
  if (page->meta->IsOwnedByRemapDestination()) {
    return RemapSourceState::kRemapDestinationOwned;
  }
  return PageCanUseBackingLocked(page, context)
             ? RemapSourceState::kReady
             : RemapSourceState::kPendingEvent;
}

size_t VMMBackingMap::TotalMappedBytes() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return mapped_page_count_ * page_size_;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
