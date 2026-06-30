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
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(VMMBackingMap, TracksMappedAndUnmappedRanges) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x10000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x101);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x102);
  map.MarkMapped(base, first_handle, page_size);
  map.MarkMapped(base + page_size, second_handle, page_size);
  map.MarkMapped(base, first_handle, page_size);
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);
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
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);

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
  std::vector<std::pair<VMMDevicePtr, size_t>> unaligned_free_ranges = {
      {base + page_size / 2, page_size * 3}};
  mapped_pages = map.CollectMappedPagesFullyCoveredBy(unaligned_free_ranges);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  EXPECT_EQ(mapped_pages[0].handle, second_handle);
  mapped_pages =
      map.CollectMappedPagesFullyCoveredBy(unaligned_free_ranges, page_size);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  auto unmapped_pages =
      map.CollectUnmappedPagesFullyCoveredBy(unaligned_free_ranges);
  ASSERT_EQ(unmapped_pages.size(), 1UL);
  EXPECT_EQ(unmapped_pages[0].va, base + page_size * 2);
  EXPECT_TRUE(map.ValidateUnmappedPages(unmapped_pages, "unit_test_unmapped"));
  unmapped_pages =
      map.CollectUnmappedPagesFullyCoveredBy(unaligned_free_ranges, page_size);
  ASSERT_EQ(unmapped_pages.size(), 1UL);
  EXPECT_EQ(unmapped_pages[0].va, base + page_size * 2);

  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeMapped(base + page_size, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), page_size);

  map.MarkReleased(base + page_size, second_handle, page_size);
  map.MarkReleased(base + page_size, second_handle, page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);
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

TEST(CUDAVirtualMemAllocatorV2, AppendWithBlockReturnsMappedFreeBlock) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto& block = allocation_with_block.block;
  ASSERT_EQ(block.size_, allocation_with_block.allocation->size());
  EXPECT_EQ(block.ptr_, allocation_with_block.allocation->ptr());
  EXPECT_TRUE(block.IsMappedFree());
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  void* ptr = allocation_with_block.allocation->ptr();

  allocation_with_block.allocation.reset();

  auto reused = allocator.PlaceAtVAWithBlock(
      reinterpret_cast<VMMDevicePtr>(ptr), allocator.handle_size());
  ASSERT_NE(reused.allocation, nullptr);
  EXPECT_EQ(reused.allocation->ptr(), ptr);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
