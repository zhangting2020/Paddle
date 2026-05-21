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

#include <mutex>

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace allocation {

void VmmBackingMap::Configure(VmmDevicePtr base,
                              size_t size,
                              size_t page_size,
                              int device) {
  std::lock_guard<SpinLock> guard(mu_);
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
}

bool VmmBackingMap::CheckRangeLocked(VmmDevicePtr va,
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

void VmmBackingMap::MarkMapped(VmmDevicePtr va,
                               VmmAllocHandle handle,
                               size_t size) {
  std::lock_guard<SpinLock> guard(mu_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkMapped", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (page.mapped && page.handle != handle) {
      VLOG(0) << "VMM V2 BackingMap remapping mapped page at "
              << reinterpret_cast<void*>(va + i * page_size_)
              << " old_handle=" << reinterpret_cast<void*>(page.handle)
              << " new_handle=" << reinterpret_cast<void*>(handle);
    }
    page.handle = handle;
    page.mapped = true;
    page.epoch++;
  }
}

void VmmBackingMap::MarkUnmapped(VmmDevicePtr va, size_t size) {
  std::lock_guard<SpinLock> guard(mu_);
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
    page.mapped = false;
    page.epoch++;
  }
}

void VmmBackingMap::MarkReleased(VmmDevicePtr va,
                                 VmmAllocHandle handle,
                                 size_t size) {
  std::lock_guard<SpinLock> guard(mu_);
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
    page.handle = 0;
    page.mapped = false;
    page.epoch++;
  }
}

bool VmmBackingMap::ValidateLayout(const HandleLayout& layout,
                                   const char* context) const {
  std::lock_guard<SpinLock> guard(mu_);
  bool ok = true;
  for (const auto& meta : layout) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(meta->base, meta->size, context, &start, &count)) {
      ok = false;
      continue;
    }
    for (size_t i = 0; i < count; ++i) {
      const auto& page = pages_[start + i];
      const bool expected_mapped = !meta->remapped;
      if (page.mapped != expected_mapped) {
        VLOG(0) << "VMM V2 BackingMap mapped-state mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->base + i * page_size_)
                << " tracked_mapped=" << page.mapped
                << " meta_remapped=" << meta->remapped;
        ok = false;
      }
      if (expected_mapped && page.handle != meta->handle) {
        VLOG(0) << "VMM V2 BackingMap handle mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->base + i * page_size_)
                << " tracked=" << reinterpret_cast<void*>(page.handle)
                << " meta=" << reinterpret_cast<void*>(meta->handle);
        ok = false;
      }
    }
  }
  return ok;
}

bool VmmBackingMap::IsRangeMapped(VmmDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(mu_);
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

bool VmmBackingMap::IsRangeUnmapped(VmmDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(mu_);
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

std::vector<std::pair<VmmDevicePtr, size_t>>
VmmBackingMap::CollectMappedRanges(VmmDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(mu_);
  return CollectRangesLocked(va, size, true, "CollectMappedRanges");
}

std::vector<std::pair<VmmDevicePtr, size_t>>
VmmBackingMap::CollectUnmappedRanges(VmmDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(mu_);
  return CollectRangesLocked(va, size, false, "CollectUnmappedRanges");
}

std::vector<std::pair<VmmDevicePtr, size_t>>
VmmBackingMap::CollectMappedRanges(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(mu_);
  std::vector<std::pair<VmmDevicePtr, size_t>> mapped_ranges;
  for (const auto& range : ranges) {
    AppendRangesLocked(
        range.first, range.second, true, "CollectMappedRanges", &mapped_ranges);
  }
  return mapped_ranges;
}

std::vector<std::pair<VmmDevicePtr, size_t>>
VmmBackingMap::CollectUnmappedRanges(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(mu_);
  std::vector<std::pair<VmmDevicePtr, size_t>> unmapped_ranges;
  for (const auto& range : ranges) {
    AppendRangesLocked(range.first,
                       range.second,
                       false,
                       "CollectUnmappedRanges",
                       &unmapped_ranges);
  }
  return unmapped_ranges;
}

std::vector<VmmBackingMap::MappedPage> VmmBackingMap::CollectMappedPages(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(mu_);
  std::vector<MappedPage> mapped_pages;
  for (const auto& range : ranges) {
    AppendMappedPagesLocked(
        range.first, range.second, "CollectMappedPages", 0, &mapped_pages);
  }
  return mapped_pages;
}

std::vector<VmmBackingMap::MappedPage> VmmBackingMap::CollectMappedPages(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(mu_);
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

std::vector<VmmBackingMap::MappedPage>
VmmBackingMap::CollectMappedPagesFullyCoveredBy(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(mu_);
  std::vector<MappedPage> mapped_pages;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyCoveredByLocked(range.first,
                                          range.second,
                                          "CollectMappedPagesFullyCoveredBy",
                                          0,
                                          &mapped_pages);
  }
  return mapped_pages;
}

std::vector<VmmBackingMap::MappedPage>
VmmBackingMap::CollectMappedPagesFullyCoveredBy(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(mu_);
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
                                          &mapped_pages);
    if (mapped_pages.size() >= target_pages) {
      break;
    }
  }
  return mapped_pages;
}

bool VmmBackingMap::ValidateMappedPages(
    const std::vector<MappedPage>& mapped_pages, const char* context) const {
  std::lock_guard<SpinLock> guard(mu_);
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

std::vector<std::pair<VmmDevicePtr, size_t>>
VmmBackingMap::CollectRangesLocked(VmmDevicePtr va,
                                   size_t size,
                                   bool mapped,
                                   const char* context) const {
  std::vector<std::pair<VmmDevicePtr, size_t>> ranges;
  AppendRangesLocked(va, size, mapped, context, &ranges);
  return ranges;
}

void VmmBackingMap::AppendRangesLocked(
    VmmDevicePtr va,
    size_t size,
    bool mapped,
    const char* context,
    std::vector<std::pair<VmmDevicePtr, size_t>>* ranges) const {
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, context, &start, &count)) {
    return;
  }

  bool in_range = false;
  VmmDevicePtr range_begin = 0;
  size_t range_size = 0;
  for (size_t i = 0; i < count; ++i) {
    const bool selected = pages_[start + i].mapped == mapped;
    const VmmDevicePtr page_va = va + i * page_size_;
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

void VmmBackingMap::AppendMappedPagesLocked(
    VmmDevicePtr va,
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
    if (!page.mapped) {
      continue;
    }
    mapped_pages->push_back(
        MappedPage{va + i * page_size_, page.handle, page.epoch});
  }
}

void VmmBackingMap::AppendMappedPagesFullyCoveredByLocked(
    VmmDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
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

  const VmmDevicePtr range_end = va + size;
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
    if (!page.mapped) {
      continue;
    }
    mapped_pages->push_back(
        MappedPage{base_ + page_idx * page_size_, page.handle, page.epoch});
  }
}

size_t VmmBackingMap::TotalMappedBytes() const {
  std::lock_guard<SpinLock> guard(mu_);
  size_t mapped_pages = 0;
  for (const auto& page : pages_) {
    if (page.mapped) {
      mapped_pages++;
    }
  }
  return mapped_pages * page_size_;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
