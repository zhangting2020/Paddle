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
