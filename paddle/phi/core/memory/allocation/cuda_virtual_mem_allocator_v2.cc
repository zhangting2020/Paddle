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

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

#if defined(PADDLE_WITH_CUDA)

#include <algorithm>
#include <limits>

#include "glog/logging.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

size_t GetPoolVAMultiplier(PoolType pool_type) {
  switch (pool_type) {
    case PoolType::kSmall:
      return 1;
    case PoolType::kLarge:
      return 4;
  }
  return 1;
}

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

CUDAVirtualMemAllocatorV2::CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                                                     size_t handle_size,
                                                     PoolType pool)
    : place_(place), handle_size_(handle_size), pool_type_(pool) {}

bool CUDAVirtualMemAllocatorV2::IsAllocThreadSafe() const { return false; }

void CUDAVirtualMemAllocatorV2::InitOnce() {
  std::call_once(init_flag_, [this] {
    platform::CUDADeviceGuard guard(place_.device);
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = place_.device;
#if defined(_WIN32)
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
#else
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemGetAllocationGranularity(
        &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // V2 uses a per-pool fixed handle size. Unlike V1, the allocator rounds
    // user input up to the device granularity so upper layers can treat every
    // handle in one HandleLayout as a stable fixed-size building block.
    handle_size_ =
        AlignedSize(std::max(handle_size_, granularity_), granularity_);
    size_t actual_avail = 0;
    size_t actual_total = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));
    const size_t va_multiplier = GetPoolVAMultiplier(pool_type_);
    PADDLE_ENFORCE_LE(va_multiplier,
                      std::numeric_limits<size_t>::max() / actual_total,
                      common::errors::InvalidArgument(
                          "VA multiplier %d for pool %d overflows size_t.",
                          va_multiplier,
                          static_cast<int>(pool_type_)));
    // Reserves VA by pool to leave room for later split/remap growth.
    virtual_mem_size_ = AlignedSize(actual_total * va_multiplier, granularity_);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressReserve(
        &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));
    backing_map_.Configure(
        virtual_mem_base_, virtual_mem_size_, handle_size_, place_.device);
    CUmemAccessDesc self = {};
    self.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    self.location.id = place_.device;
    self.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc_.push_back(self);
  });
}

phi::Allocation* CUDAVirtualMemAllocatorV2::AllocateImpl(size_t size) {
  InitOnce();
  size_t aligned = AlignedSize(size, handle_size_);
  size_t num_handles = aligned / handle_size_;
  VmmDevicePtr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;
  PADDLE_ENFORCE_LE(
      ptr + aligned,
      virtual_mem_base_ + virtual_mem_size_,
      common::errors::ResourceExhausted("VMMAllocatorV2 virtual address space "
                                        "is exhausted for place %s.",
                                        place_));

  platform::CUDADeviceGuard guard(place_.device);
  HandleLayout layout;
  layout.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    VmmAllocHandle handle;
    auto ce = platform::RecordedGpuMemCreate(
        &handle, handle_size_, &prop_, 0, place_.device);
    if (ce != CUDA_SUCCESS) {
      for (const auto& m : layout) {
        phi::dynload::cuMemUnmap(m->base, m->size);
        platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
      }
      if (ce == CUDA_ERROR_OUT_OF_MEMORY) {
        PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
            "cuMemCreate failed: out of GPU memory at handle %zu/%zu "
            "(handle_size=%zu).",
            i,
            num_handles,
            handle_size_));
      }
      PADDLE_ENFORCE_GPU_SUCCESS(ce);
    }
    auto me = phi::dynload::cuMemMap(
        ptr + i * handle_size_, handle_size_, 0, handle, 0);
    if (me != CUDA_SUCCESS) {
      platform::RecordedGpuMemRelease(handle, handle_size_, place_.device);
      for (const auto& m : layout) {
        phi::dynload::cuMemUnmap(m->base, m->size);
        platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
      }
      PADDLE_THROW(common::errors::External(
          "cuMemMap failed at handle %zu/%zu.", i, num_handles));
    }
    layout.push_back(std::make_shared<VmmHandleMeta>(VmmHandleMeta{
        ptr + i * handle_size_, handle_size_, handle, place_.device}));
  }
  auto access_status = phi::dynload::cuMemSetAccess(
      ptr, aligned, access_desc_.data(), access_desc_.size());
  if (access_status != CUDA_SUCCESS) {
    for (const auto& m : layout) {
      phi::dynload::cuMemUnmap(m->base, m->size);
      platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(access_status);
  }

  for (const auto& m : layout) {
    backing_map_.MarkMapped(m->base, m->handle, m->size);
  }
  RegisterHandleLayout(reinterpret_cast<void*>(ptr), layout);
  AdvanceTailOffset(aligned);
  return new Allocation(reinterpret_cast<void*>(ptr), aligned, place_);
}

DecoratedAllocationPtr CUDAVirtualMemAllocatorV2::AllocateAtVA(
    VmmDevicePtr ptr, size_t size) {
  InitOnce();
  const size_t aligned = AlignedSize(size, handle_size_);
  const size_t num_handles = aligned / handle_size_;
  PADDLE_ENFORCE_GE(
      ptr,
      virtual_mem_base_,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 AllocateAtVA ptr is before reserved VA range."));
  PADDLE_ENFORCE_LE(
      ptr,
      virtual_mem_base_ + virtual_mem_size_,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 AllocateAtVA ptr is outside reserved VA range."));
  PADDLE_ENFORCE_EQ(
      (ptr - virtual_mem_base_) % handle_size_,
      0UL,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 AllocateAtVA requires handle-aligned VA, ptr=%p.",
          reinterpret_cast<void*>(ptr)));
  PADDLE_ENFORCE_LE(
      aligned,
      virtual_mem_base_ + virtual_mem_size_ - ptr,
      common::errors::ResourceExhausted(
          "VMMAllocatorV2 AllocateAtVA range exceeds reserved VA space."));

  platform::CUDADeviceGuard guard(place_.device);
  VLOG(6) << "VMM V2 AllocateAtVA ptr=" << reinterpret_cast<void*>(ptr)
          << " requested=" << size << " aligned=" << aligned
          << " handle_count=" << num_handles
          << " tail_offset=" << virtual_mem_alloced_offset_;
  HandleLayout layout;
  layout.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    VmmAllocHandle handle;
    auto ce = platform::RecordedGpuMemCreate(
        &handle, handle_size_, &prop_, 0, place_.device);
    if (ce != CUDA_SUCCESS) {
      for (const auto& m : layout) {
        phi::dynload::cuMemUnmap(m->base, m->size);
        platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
      }
      if (ce == CUDA_ERROR_OUT_OF_MEMORY) {
        PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
            "cuMemCreate failed in AllocateAtVA: out of GPU memory at "
            "handle %zu/%zu (handle_size=%zu).",
            i,
            num_handles,
            handle_size_));
      }
      PADDLE_ENFORCE_GPU_SUCCESS(ce);
    }

    const VmmDevicePtr dst = ptr + i * handle_size_;
    auto me = phi::dynload::cuMemMap(dst, handle_size_, 0, handle, 0);
    if (me != CUDA_SUCCESS) {
      platform::RecordedGpuMemRelease(handle, handle_size_, place_.device);
      for (const auto& m : layout) {
        phi::dynload::cuMemUnmap(m->base, m->size);
        platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
      }
      PADDLE_THROW(common::errors::External(
          "cuMemMap failed in AllocateAtVA at handle %zu/%zu.",
          i,
          num_handles));
    }
    layout.push_back(std::make_shared<VmmHandleMeta>(
        VmmHandleMeta{dst, handle_size_, handle, place_.device}));
  }

  auto access_status = phi::dynload::cuMemSetAccess(
      ptr, aligned, access_desc_.data(), access_desc_.size());
  if (access_status != CUDA_SUCCESS) {
    for (const auto& m : layout) {
      phi::dynload::cuMemUnmap(m->base, m->size);
      platform::RecordedGpuMemRelease(m->handle, m->size, place_.device);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(access_status);
  }

  for (const auto& m : layout) {
    backing_map_.MarkMapped(m->base, m->handle, m->size);
  }
  RegisterHandleLayout(reinterpret_cast<void*>(ptr), layout);
  auto* alloc = new Allocation(reinterpret_cast<void*>(ptr), aligned, place_);
  CUDAVirtualMemAllocatorV2* self = this;
  return DecoratedAllocationPtr(alloc, [self](phi::Allocation* a) {
    self->FreeImpl(static_cast<Allocation*>(a));
  });
}

void CUDAVirtualMemAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  auto* ptr = allocation->ptr();
  HandleLayout layout;
  {
    std::lock_guard<SpinLock> guard(allocation_layout_mu_);
    auto it = allocation_layout_map_.find(ptr);
    PADDLE_ENFORCE_NE(
        it == allocation_layout_map_.end(),
        true,
        common::errors::NotFound(
            "No VMMAllocatorV2 handle layout found for allocation %p.", ptr));
    layout = it->second;
  }

  platform::CUDADeviceGuard guard(place_.device);
  for (const auto& handle : layout) {
    if (handle->remapped) {
      VLOG(5) << "FreeImpl: skipping remapped handle base="
              << reinterpret_cast<void*>(handle->base)
              << " size=" << handle->size;
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemUnmap(handle->base, handle->size));
    backing_map_.MarkUnmapped(handle->base, handle->size);
    // Use non-throwing release: if the handle was already released by a
    // subsequent compactor remap (which created a new synthetic allocation
    // for the same physical handle), cuMemRelease returns
    // CUDA_ERROR_INVALID_VALUE.  This is expected and safe to ignore —
    // the handle's physical memory is now owned by the newer synthetic
    // allocation.
    auto release_status = platform::RecordedGpuMemRelease(
        handle->handle, handle->size, place_.device);
    if (release_status != CUDA_SUCCESS) {
      VLOG(3) << "FreeImpl: cuMemRelease returned " << release_status
              << " for handle " << handle->handle
              << " (likely already released by re-remap), skipping";
    } else {
      backing_map_.MarkReleased(handle->base, handle->handle, handle->size);
    }
  }

  UnregisterHandleLayout(ptr);
  delete allocation;
}

void CUDAVirtualMemAllocatorV2::UnmapHandle(VmmDevicePtr ptr, size_t size) {
  platform::CUDADeviceGuard guard(place_.device);
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemUnmap(ptr, size));
  backing_map_.MarkUnmapped(ptr, size);
}

bool CUDAVirtualMemAllocatorV2::TryUnmapHandle(VmmDevicePtr ptr, size_t size) {
  platform::CUDADeviceGuard guard(place_.device);
  auto status = phi::dynload::cuMemUnmap(ptr, size);
  if (status != CUDA_SUCCESS) {
    VLOG(0) << "VMM V2 TryUnmapHandle: cuMemUnmap failed at "
            << reinterpret_cast<void*>(ptr) << " size=" << size
            << " status=" << status;
    return false;
  }
  backing_map_.MarkUnmapped(ptr, size);
  return true;
}

void CUDAVirtualMemAllocatorV2::MapHandlesToVA(
    VmmDevicePtr ptr,
    const std::vector<VmmAllocHandle>& hs,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  (void)metas;
  platform::CUDADeviceGuard guard(place_.device);
  // V2 currently assumes one uniform handle size per pool, so remap can
  // re-materialize a contiguous VA range by replaying fixed-size mappings.
  VLOG(10) << "MapHandlesToVA dst=" << reinterpret_cast<void*>(ptr)
           << " handle_count=" << hs.size() << " handle_size=" << handle_size_
           << " total_bytes=" << hs.size() * handle_size_
           << " tail_offset=" << virtual_mem_alloced_offset_
           << " virtual_mem_size=" << virtual_mem_size_;
  for (size_t i = 0; i < hs.size(); ++i) {
    auto dst = ptr + i * handle_size_;
    auto status = phi::dynload::cuMemMap(dst, handle_size_, 0, hs[i], 0);
    if (status != CUDA_SUCCESS) {
      VLOG(0) << "cuMemMap failed at index=" << i
              << " dst=" << reinterpret_cast<void*>(dst)
              << " handle_size=" << handle_size_
              << " handle=" << reinterpret_cast<void*>(hs[i])
              << " total_handles=" << hs.size();
      CUmemGenericAllocationHandle retained = 0;
      auto retain_status = phi::dynload::cuMemRetainAllocationHandle(
          &retained, reinterpret_cast<void*>(dst));
      VLOG(0) << "Probe dst retain status=" << retain_status
              << " retained_handle=" << reinterpret_cast<void*>(retained);
      if (retain_status == CUDA_SUCCESS) {
        auto release_status = phi::dynload::cuMemRelease(retained);
        VLOG(0) << "Probe dst release retained_handle status="
                << release_status;
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(status);
    backing_map_.MarkMapped(dst, hs[i], handle_size_);
  }
  auto status = phi::dynload::cuMemSetAccess(
      ptr, hs.size() * handle_size_, access_desc_.data(), access_desc_.size());
  if (status != CUDA_SUCCESS) {
    VLOG(0) << "cuMemSetAccess failed dst=" << reinterpret_cast<void*>(ptr)
            << " total_bytes=" << hs.size() * handle_size_
            << " access_desc_count=" << access_desc_.size();
  }
  PADDLE_ENFORCE_GPU_SUCCESS(status);
}

bool CUDAVirtualMemAllocatorV2::CollectAllocationHandleLayout(
    void* ptr, HandleLayout* layout) const {
  std::lock_guard<SpinLock> guard(allocation_layout_mu_);
  auto it = allocation_layout_map_.find(ptr);
  if (it == allocation_layout_map_.end()) {
    return false;
  }
  if (layout) {
    *layout = it->second;
  }
  return true;
}

void CUDAVirtualMemAllocatorV2::RegisterHandleLayout(
    void* ptr, const HandleLayout& layout) {
  std::lock_guard<SpinLock> guard(allocation_layout_mu_);
  EmplaceOrEnforce(
      &allocation_layout_map_, ptr, layout, "allocation_layout_map_");
  if (!backing_map_.ValidateLayout(layout, "RegisterHandleLayout")) {
    VLOG(0) << "VMM V2 BackingMap validation failed while registering layout "
            << ptr;
  }
}

void CUDAVirtualMemAllocatorV2::UnregisterHandleLayout(void* ptr) {
  std::lock_guard<SpinLock> guard(allocation_layout_mu_);
  allocation_layout_map_.erase(ptr);
}

DecoratedAllocationPtr CUDAVirtualMemAllocatorV2::CreateSyntheticAllocation(
    VmmDevicePtr ptr, size_t size, const HandleLayout& layout) {
  RegisterHandleLayout(reinterpret_cast<void*>(ptr), layout);
  auto* alloc = new Allocation(reinterpret_cast<void*>(ptr), size, place_);
  // Use a custom deleter that calls FreeImpl directly, since the
  // synthetic allocation bypasses the normal Allocate() path and
  // cannot use RegisterDecoratedAllocator (which is private).
  CUDAVirtualMemAllocatorV2* self = this;
  return DecoratedAllocationPtr(alloc, [self](phi::Allocation* a) {
    self->FreeImpl(static_cast<Allocation*>(a));
  });
}

void CUDAVirtualMemAllocatorV2::MarkBackingMapped(VmmDevicePtr ptr,
                                                  VmmAllocHandle handle,
                                                  size_t size) {
  backing_map_.MarkMapped(ptr, handle, size);
}

void CUDAVirtualMemAllocatorV2::MarkBackingUnmapped(VmmDevicePtr ptr,
                                                    size_t size) {
  backing_map_.MarkUnmapped(ptr, size);
}

void CUDAVirtualMemAllocatorV2::MarkBackingReleased(VmmDevicePtr ptr,
                                                    VmmAllocHandle handle,
                                                    size_t size) {
  backing_map_.MarkReleased(ptr, handle, size);
}

bool CUDAVirtualMemAllocatorV2::ValidateBackingLayout(
    const HandleLayout& layout, const char* context) const {
  return backing_map_.ValidateLayout(layout, context);
}

std::vector<VmmBackingMap::MappedPage>
CUDAVirtualMemAllocatorV2::CollectMappedBackingPagesFullyCoveredBy(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  if (target_bytes == 0) {
    return backing_map_.CollectMappedPagesFullyCoveredBy(ranges);
  }
  return backing_map_.CollectMappedPagesFullyCoveredBy(ranges, target_bytes);
}

bool CUDAVirtualMemAllocatorV2::ValidateMappedBackingPages(
    const std::vector<VmmBackingMap::MappedPage>& pages,
    const char* context) const {
  return backing_map_.ValidateMappedPages(pages, context);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
