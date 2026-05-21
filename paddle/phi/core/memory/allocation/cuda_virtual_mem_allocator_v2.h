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

#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

// Compared with CUDAVirtualMemAllocator, V2 does not expose a single
// VA<->handle mapping per allocation. Instead it returns a lightweight
// HandleLayout (a handle list) for one allocation. Upper layers later
// transform that list into block-level BlockPartV2 state.
class CUDAVirtualMemAllocatorV2 : public Allocator {
 public:
  // Standalone use defaults to the large pool. Upper layers may also choose
  // explicit small/large pool types.
  CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                            size_t handle_size,
                            PoolType pool = PoolType::kLarge);

  bool IsAllocThreadSafe() const override;

  size_t handle_size() const { return handle_size_; }
  PoolType pool_type() const { return pool_type_; }
  VmmDevicePtr virtual_mem_base() const { return virtual_mem_base_; }
  size_t virtual_mem_size() const { return virtual_mem_size_; }
  size_t tail_offset() const { return virtual_mem_alloced_offset_; }
  // Best-fit/remap layers may consume VA from the reserved range incrementally.
  // V2 keeps this as an explicit cursor instead of reusing V1's
  // virtual_2_physical_map_ bookkeeping.
  void AdvanceTailOffset(size_t bytes) { virtual_mem_alloced_offset_ += bytes; }
  // Retreat the tail cursor when the compactor discovers that blocks no
  // longer span up to the previous high-water mark (e.g. after
  // FreeIdleChunks released tail-end underlying allocations).
  void SetTailOffset(size_t offset) { virtual_mem_alloced_offset_ = offset; }

  void UnmapHandle(VmmDevicePtr ptr, size_t size);
  // Non-throwing variant: returns true if cuMemUnmap succeeds.
  bool TryUnmapHandle(VmmDevicePtr ptr, size_t size);

  const GPUPlace& place() const { return place_; }
  void MapHandlesToVA(
      VmmDevicePtr ptr,
      const std::vector<VmmAllocHandle>& hs,
      const std::vector<std::shared_ptr<VmmHandleMeta>>* metas = nullptr);
  // Create fresh physical backing and map it at an existing reserved VA range.
  // This is used by upper layers to reuse FREE+UNMAPPED gap space in place.
  DecoratedAllocationPtr AllocateAtVA(VmmDevicePtr ptr, size_t size);
  // Exposes the allocation-level handle list for IPC/export queries. The key
  // is the raw allocation ptr returned by this allocator.
  bool CollectAllocationHandleLayout(void* ptr, HandleLayout* layout) const;

  // Register a handle layout for handles that were remapped by the compactor
  // to a new VA.  This allows FreeImpl to release them when the synthetic
  // allocation is eventually freed.
  void RegisterHandleLayout(void* ptr, const HandleLayout& layout);

  // Create a synthetic Allocation object for remapped handles.  The handles
  // already exist (cuMemCreate was done earlier), this just registers
  // ownership so that FreeImpl can properly release them later.
  DecoratedAllocationPtr CreateSyntheticAllocation(VmmDevicePtr ptr,
                                                   size_t size,
                                                   const HandleLayout& layout);

  // Phase-1 BackingMap mirror hooks for driver operations that still happen
  // outside the bottom allocator (e.g. compactor rollback).
  void MarkBackingMapped(VmmDevicePtr ptr, VmmAllocHandle handle, size_t size);
  void MarkBackingUnmapped(VmmDevicePtr ptr, size_t size);
  void MarkBackingReleased(VmmDevicePtr ptr,
                           VmmAllocHandle handle,
                           size_t size);
  bool ValidateBackingLayout(const HandleLayout& layout,
                             const char* context) const;
  std::vector<VmmBackingMap::MappedPage> CollectMappedBackingPagesFullyCoveredBy(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<VmmBackingMap::UnmappedPage>
  CollectUnmappedBackingPagesFullyCoveredBy(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  bool ValidateMappedBackingPages(
      const std::vector<VmmBackingMap::MappedPage>& pages,
      const char* context) const;
  bool ValidateUnmappedBackingPages(
      const std::vector<VmmBackingMap::UnmappedPage>& pages,
      const char* context) const;

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  void InitOnce();
  void UnregisterHandleLayout(void* ptr);

  GPUPlace place_;
  size_t handle_size_;
  PoolType pool_type_;
  std::once_flag init_flag_;

  VmmDevicePtr virtual_mem_base_{0};
  size_t virtual_mem_size_{0};
  size_t virtual_mem_alloced_offset_{0};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  std::vector<CUmemAccessDesc> access_desc_;

  mutable std::unordered_map<void*, HandleLayout> allocation_layout_map_;
  mutable SpinLock allocation_layout_mu_;
  VmmBackingMap backing_map_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
