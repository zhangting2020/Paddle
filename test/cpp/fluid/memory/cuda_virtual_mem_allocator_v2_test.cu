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

#include "gtest/gtest.h"

#include <utility>
#include <vector>

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(VmmBackingMap, TracksMappedAndUnmappedRanges) {
  VmmBackingMap map;
  const VmmDevicePtr base = 0x10000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_EQ(map.TotalMappedBytes(), 0UL);

  const VmmAllocHandle first_handle = static_cast<VmmAllocHandle>(0x101);
  const VmmAllocHandle second_handle = static_cast<VmmAllocHandle>(0x102);
  map.MarkMapped(base, first_handle, page_size);
  map.MarkMapped(base + page_size, second_handle, page_size);

  auto mapped_ranges = map.CollectMappedRanges(base, page_size * 4);
  ASSERT_EQ(mapped_ranges.size(), 1UL);
  EXPECT_EQ(mapped_ranges[0].first, base);
  EXPECT_EQ(mapped_ranges[0].second, page_size * 2);
  auto unmapped_ranges = map.CollectUnmappedRanges(base, page_size * 4);
  ASSERT_EQ(unmapped_ranges.size(), 1UL);
  EXPECT_EQ(unmapped_ranges[0].first, base + page_size * 2);
  EXPECT_EQ(unmapped_ranges[0].second, page_size * 2);
  std::vector<std::pair<VmmDevicePtr, size_t>> free_ranges = {
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
  mapped_pages = map.CollectMappedPages(free_ranges, page_size + 1);
  ASSERT_EQ(mapped_pages.size(), 2UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  EXPECT_EQ(mapped_pages[1].va, base + page_size);
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
  EXPECT_EQ(map.TotalMappedBytes(), page_size * 2);

  map.MarkUnmapped(base, page_size);
  EXPECT_FALSE(map.ValidateMappedPages(two_page_snapshot, "unit_test_stale"));
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
  unmapped_ranges = map.CollectUnmappedRanges(free_ranges);
  ASSERT_EQ(unmapped_ranges.size(), 2UL);
  EXPECT_EQ(unmapped_ranges[0].first, base);
  EXPECT_EQ(unmapped_ranges[0].second, page_size);
  EXPECT_EQ(unmapped_ranges[1].first, base + page_size * 2);
  EXPECT_EQ(unmapped_ranges[1].second, page_size * 2);
  mapped_pages = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  EXPECT_EQ(mapped_pages[0].handle, second_handle);
  mapped_pages = map.CollectMappedPages(free_ranges, page_size * 4);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);

  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeMapped(base + page_size, page_size));
  EXPECT_EQ(map.TotalMappedBytes(), page_size);

  map.MarkReleased(base + page_size, second_handle, page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_EQ(map.TotalMappedBytes(), 0UL);
}

TEST(CUDAVirtualMemAllocatorV2, HandleSizeAligned) {
  CUDAVirtualMemAllocatorV2 allocator(phi::GPUPlace(), 1, PoolType::kLarge);

  auto allocation = allocator.Allocate(1);
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(allocation->ptr(), nullptr);
  ASSERT_GT(allocator.handle_size(), 0UL);
  ASSERT_EQ(allocation->size() % allocator.handle_size(), 0UL);
}

TEST(CUDAVirtualMemAllocatorV2, CollectAllocationHandleLayout) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  auto base = reinterpret_cast<VmmDevicePtr>(allocation->ptr());
  for (size_t i = 0; i < layout.size(); ++i) {
    ASSERT_TRUE(layout[i]);
    EXPECT_EQ(layout[i]->base, base + i * allocator.handle_size());
    EXPECT_EQ(layout[i]->size, allocator.handle_size());
  }
}

TEST(CUDAVirtualMemAllocatorV2, TailOffsetAdvancesWithAllocationSize) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto first = allocator.Allocate(1);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(allocator.tail_offset(), first->size());

  auto second = allocator.Allocate(allocator.handle_size() + 1);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(allocator.tail_offset(), first->size() + second->size());
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.handle_size());
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

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  const auto remap_ptr = layout[0]->base;
  const auto remap_handle = layout[0]->handle;
  allocator.UnmapHandle(remap_ptr, allocator.handle_size());
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
