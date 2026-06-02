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

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

__global__ void VMMBackingMapBusyWaitKernel(uint64_t cycles) {
  const auto start = clock64();
  while (clock64() - start < cycles) {
  }
}

}  // namespace

TEST(VMMBackingMap, TracksMappedAndUnmappedRanges) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x10000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_EQ(map.TotalMappedBytes(), 0UL);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x101);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x102);
  map.MarkMapped(base, first_handle, page_size);
  map.MarkMapped(base + page_size, second_handle, page_size);
  map.MarkMapped(base, first_handle, page_size);
  EXPECT_EQ(map.TotalMappedBytes(), page_size * 2);
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeReleasable(base - page_size, page_size));

  auto mapped_ranges = map.CollectMappedRanges(base, page_size * 4);
  ASSERT_EQ(mapped_ranges.size(), 1UL);
  EXPECT_EQ(mapped_ranges[0].first, base);
  EXPECT_EQ(mapped_ranges[0].second, page_size * 2);
  auto unmapped_ranges = map.CollectUnmappedRanges(base, page_size * 4);
  ASSERT_EQ(unmapped_ranges.size(), 1UL);
  EXPECT_EQ(unmapped_ranges[0].first, base + page_size * 2);
  EXPECT_EQ(unmapped_ranges[0].second, page_size * 2);

  std::vector<std::pair<VMMDevicePtr, size_t>> free_ranges = {
      {base, page_size}, {base + page_size, page_size * 3}};
  mapped_ranges = map.CollectMappedRanges(free_ranges);
  ASSERT_EQ(mapped_ranges.size(), 1UL);
  EXPECT_EQ(mapped_ranges[0].first, base);
  EXPECT_EQ(mapped_ranges[0].second, page_size * 2);
  unmapped_ranges = map.CollectUnmappedRanges(free_ranges);
  ASSERT_EQ(unmapped_ranges.size(), 1UL);
  EXPECT_EQ(unmapped_ranges[0].first, base + page_size * 2);
  EXPECT_EQ(unmapped_ranges[0].second, page_size * 2);

  auto mapped_pages = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(mapped_pages.size(), 2UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  EXPECT_EQ(mapped_pages[0].handle, first_handle);
  EXPECT_EQ(mapped_pages[1].va, base + page_size);
  EXPECT_EQ(mapped_pages[1].handle, second_handle);
  mapped_pages = map.CollectMappedPages(free_ranges, page_size);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  const auto two_page_snapshot = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(two_page_snapshot.size(), 2UL);
  EXPECT_TRUE(map.ValidateMappedPages(two_page_snapshot, "unit_test"));

  EXPECT_TRUE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 3));
  EXPECT_FALSE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeUnmapped(base + page_size * 2, page_size * 2));

  map.MarkUnmapped(base, page_size);
  map.MarkUnmapped(base, page_size);
  EXPECT_FALSE(map.ValidateMappedPages(two_page_snapshot, "unit_test_stale"));
  std::vector<std::pair<VMMDevicePtr, size_t>> unmapped_base_range = {
      {base, page_size}};
  auto unmapped_base_pages =
      map.CollectUnmappedPagesFullyCoveredBy(unmapped_base_range);
  ASSERT_EQ(unmapped_base_pages.size(), 1UL);
  EXPECT_TRUE(map.ValidateUnmappedPages(unmapped_base_pages,
                                        "unit_test_unmapped_clears_handle"));
  mapped_ranges = map.CollectMappedRanges(base, page_size * 4);
  ASSERT_EQ(mapped_ranges.size(), 1UL);
  EXPECT_EQ(mapped_ranges[0].first, base + page_size);
  EXPECT_EQ(mapped_ranges[0].second, page_size);
  unmapped_ranges = map.CollectUnmappedRanges(base, page_size * 4);
  ASSERT_EQ(unmapped_ranges.size(), 2UL);
  EXPECT_EQ(unmapped_ranges[0].first, base);
  EXPECT_EQ(unmapped_ranges[0].second, page_size);
  EXPECT_EQ(unmapped_ranges[1].first, base + page_size * 2);
  EXPECT_EQ(unmapped_ranges[1].second, page_size * 2);

  std::vector<std::pair<VMMDevicePtr, size_t>> unaligned_free_ranges = {
      {base + page_size / 2, page_size * 3}};
  mapped_pages =
      map.CollectMappedPagesFullyCoveredBy(unaligned_free_ranges, page_size);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  auto unmapped_pages =
      map.CollectUnmappedPagesFullyCoveredBy(unaligned_free_ranges, page_size);
  ASSERT_EQ(unmapped_pages.size(), 1UL);
  EXPECT_EQ(unmapped_pages[0].va, base + page_size * 2);
  auto candidates = map.CollectCompactCandidates(
      unaligned_free_ranges, unaligned_free_ranges, page_size);
  ASSERT_EQ(candidates.source_pages.size(), 1UL);
  ASSERT_EQ(candidates.target_pages.size(), 1UL);
  EXPECT_EQ(candidates.source_pages[0].va, base + page_size);
  EXPECT_EQ(candidates.target_pages[0].va, base + page_size * 2);

  map.MarkIpcExported(base + page_size, page_size);
  EXPECT_FALSE(map.HasIpcExportedPages(base, page_size));
  EXPECT_TRUE(map.HasIpcExportedPages(base + page_size, page_size));
  EXPECT_TRUE(map.HasIpcExportedPages(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size * 2));
  mapped_pages = map.CollectMappedPagesFullyCoveredBy(unaligned_free_ranges);
  EXPECT_TRUE(mapped_pages.empty());
  candidates = map.CollectCompactCandidates(
      unaligned_free_ranges, unaligned_free_ranges, page_size);
  EXPECT_TRUE(candidates.source_pages.empty());
  ASSERT_EQ(candidates.target_pages.size(), 1UL);
  EXPECT_EQ(candidates.target_pages[0].va, base + page_size * 2);

  map.MarkReleased(base + page_size, second_handle, page_size);
  map.MarkReleased(base + page_size, second_handle, page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_EQ(map.TotalMappedBytes(), 0UL);
  std::vector<std::pair<VMMDevicePtr, size_t>> all_ranges = {
      {base, page_size * 4}};
  auto all_unmapped_pages =
      map.CollectUnmappedPagesFullyCoveredBy(all_ranges, page_size * 2);
  ASSERT_EQ(all_unmapped_pages.size(), 2UL);
  EXPECT_TRUE(
      map.ValidateUnmappedPages(all_unmapped_pages, "unit_test_all_unmapped"));
  const VMMAllocHandle third_handle = static_cast<VMMAllocHandle>(0x103);
  map.MarkMapped(base, third_handle, page_size);
  EXPECT_FALSE(map.ValidateUnmappedPages(all_unmapped_pages,
                                         "unit_test_unmapped_stale"));
}

TEST(VMMBackingMap, RejectsMappedPageHandleOverwrite) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x18000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size, page_size, 0);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x181);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x182);
  map.MarkMapped(base, first_handle, page_size);
  EXPECT_THROW(map.MarkMapped(base, second_handle, page_size),
               common::enforce::EnforceNotMet);

  map.MarkUnmapped(base, page_size);
  auto meta = std::make_shared<VMMHandleMeta>(base, page_size, first_handle, 0);
  map.MarkMapped(base, meta, page_size);
  auto other_meta =
      std::make_shared<VMMHandleMeta>(base, page_size, second_handle, 0);
  EXPECT_THROW(map.MarkMapped(base, other_meta, page_size),
               common::enforce::EnforceNotMet);
}

TEST(VMMBackingMap, ReplacesPendingEventForSameStream) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x20000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size, page_size, 0);
  auto meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x201), 0);
  map.MarkMapped(base, meta, page_size);

  gpuStream_t key_stream;
  gpuStream_t busy_stream;
  ASSERT_EQ(cudaStreamCreate(&key_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&busy_stream), cudaSuccess);

  VMMBackingMapBusyWaitKernel<<<1, 1, 0, busy_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  gpuEvent_t pending_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&pending_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(pending_event, busy_stream), cudaSuccess);
  map.MarkPendingEvent(base,
                       page_size,
                       key_stream,
                       std::make_shared<CUDAEventGuard>(pending_event));

  gpuEvent_t ready_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(ready_event, key_stream), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(ready_event), cudaSuccess);
  map.MarkPendingEvent(base,
                       page_size,
                       key_stream,
                       std::make_shared<CUDAEventGuard>(ready_event));

  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{base, page_size}};
  auto mapped_snapshot = map.CollectMappedPages(ranges);
  ASSERT_EQ(mapped_snapshot.size(), 1UL);
  auto pages = map.CollectRemapSourcePagesFullyCoveredBy(ranges, page_size);
  ASSERT_EQ(pages.size(), 1UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_TRUE(map.ValidateMappedPages(mapped_snapshot,
                                      "unit_test_ready_event_gc_keeps_epoch"));

  ASSERT_EQ(cudaStreamSynchronize(busy_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(key_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(busy_stream), cudaSuccess);
}

TEST(CUDAVirtualMemAllocatorV2, HandleSizeAligned) {
  CUDAVirtualMemAllocatorV2 allocator(phi::GPUPlace(), 1, PoolType::kLarge);

  auto allocation = allocator.Allocate(1);
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(allocation->ptr(), nullptr);
  ASSERT_GT(allocator.HandleSize(), 0UL);
  ASSERT_EQ(allocation->size() % allocator.HandleSize(), 0UL);
}

TEST(CUDAVirtualMemAllocatorV2, CollectAllocationHandleLayout) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.HandleSize() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  auto base = reinterpret_cast<VMMDevicePtr>(allocation->ptr());
  for (size_t i = 0; i < layout.size(); ++i) {
    ASSERT_TRUE(layout[i]);
    EXPECT_EQ(layout[i]->base, base + i * allocator.HandleSize());
    EXPECT_EQ(layout[i]->size, allocator.HandleSize());
  }
}

TEST(CUDAVirtualMemAllocatorV2, TailOffsetAdvancesWithAllocationSize) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto first = allocator.Allocate(1);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(allocator.TailOffset(), first->size());

  auto second = allocator.Allocate(allocator.HandleSize() + 1);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(allocator.TailOffset(), first->size() + second->size());
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.HandleSize());
  ASSERT_NE(allocation, nullptr);
  void* ptr = allocation->ptr();

  HandleLayout layout;
  ASSERT_TRUE(allocator.CollectAllocationHandleLayout(ptr, &layout));
  ASSERT_EQ(layout.size(), 1UL);

  allocation.reset();

  EXPECT_FALSE(allocator.CollectAllocationHandleLayout(ptr, &layout));
}

TEST(CUDAVirtualMemAllocatorV2, UnmapAndMapHandleBackToSameVA) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.HandleSize() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  const auto remap_ptr = layout[0]->base;
  const auto remap_handle = layout[0]->handle;
  allocator.UnmapHandle(remap_ptr, allocator.HandleSize());
  allocator.MapHandlesToVA(remap_ptr, {remap_handle});

  HandleLayout layout_after_remap;
  EXPECT_TRUE(allocator.CollectAllocationHandleLayout(allocation->ptr(),
                                                      &layout_after_remap));
  ASSERT_EQ(layout_after_remap.size(), layout.size());
  EXPECT_EQ(layout_after_remap[0]->base, layout[0]->base);
  EXPECT_EQ(layout_after_remap[0]->handle, layout[0]->handle);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
