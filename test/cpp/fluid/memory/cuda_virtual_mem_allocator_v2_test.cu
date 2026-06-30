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

  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeMapped(base + page_size, page_size));
  EXPECT_EQ(map.TotalMappedBytes(), page_size);

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

TEST(VMMBackingMap, MarksPendingEventForUnalignedRangeOnce) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x24000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);
  auto first_meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x241), 0);
  auto second_meta = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x242), 0);
  map.MarkMapped(base, first_meta, page_size);
  map.MarkMapped(base + page_size, second_meta, page_size);

  gpuStream_t busy_stream;
  ASSERT_EQ(cudaStreamCreate(&busy_stream), cudaSuccess);
  VMMBackingMapBusyWaitKernel<<<1, 1, 0, busy_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  gpuEvent_t pending_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&pending_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(pending_event, busy_stream), cudaSuccess);
  EXPECT_TRUE(map.MarkPendingEventForRange(
      base + 128UL,
      page_size,
      busy_stream,
      std::make_shared<CUDAEventGuard>(pending_event)));

  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{base, page_size * 2}};
  auto pages = map.CollectRemapSourcePagesFullyCoveredBy(ranges, page_size * 2);
  ASSERT_EQ(pages.size(), 2UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);
  EXPECT_EQ(pages[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);

  ASSERT_EQ(cudaStreamSynchronize(busy_stream), cudaSuccess);
  pages = map.CollectRemapSourcePagesFullyCoveredBy(ranges, page_size * 2);
  ASSERT_EQ(pages.size(), 2UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_EQ(pages[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  ASSERT_EQ(cudaStreamDestroy(busy_stream), cudaSuccess);
}

TEST(CUDAVirtualMemAllocatorV2, DetectsDriverVARangeMapping) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.HandleSize());
  ASSERT_NE(allocation, nullptr);
  auto va = reinterpret_cast<VMMDevicePtr>(allocation->ptr());
  EXPECT_FALSE(allocator.IsDriverVARangeUnmapped(va, allocator.HandleSize()));

  allocation.reset();
  EXPECT_TRUE(allocator.IsDriverVARangeUnmapped(va, allocator.HandleSize()));
}

TEST(CUDAVirtualMemAllocatorV2, AppendWithBlockReturnsMappedFreeBlock) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto& block = allocation_with_block.block;
  ASSERT_EQ(block.size_, allocation_with_block.allocation->size());
  EXPECT_TRUE(block.IsMappedFree());

  auto base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {base, allocation_with_block.allocation->size()}};
  auto pages = allocator.CollectMappedPages(
      ranges, allocation_with_block.allocation->size());
  ASSERT_EQ(pages.size(), 2UL);
  for (size_t i = 0; i < pages.size(); ++i) {
    EXPECT_EQ(pages[i].va, base + i * allocator.HandleSize());
    ASSERT_NE(pages[i].meta, nullptr);
    EXPECT_EQ(pages[i].meta->Base(), pages[i].va);
    EXPECT_EQ(pages[i].meta->Size(), allocator.HandleSize());
  }
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  void* ptr = allocation_with_block.allocation->ptr();

  allocation_with_block.allocation.reset();

  auto reused = allocator.PlaceAtVAWithBlock(
      reinterpret_cast<VMMDevicePtr>(ptr), allocator.HandleSize());
  ASSERT_NE(reused.allocation, nullptr);
  EXPECT_EQ(reused.allocation->ptr(), ptr);
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {reinterpret_cast<VMMDevicePtr>(ptr), allocator.HandleSize()}};
  EXPECT_EQ(allocator.CollectMappedPages(ranges, allocator.HandleSize()).size(),
            1UL);
}

TEST(CUDAVirtualMemAllocatorV2, MoveBackingPageRoundTripsHandle) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const auto target_va = allocator.VirtualMemBase() + allocator.TailOffset();
  ASSERT_NE(source_va, target_va);

  std::vector<std::pair<VMMDevicePtr, size_t>> source_ranges = {
      {source_va, allocator.HandleSize()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_ranges = {
      {target_va, allocator.HandleSize()}};
  auto source_pages =
      allocator.CollectMappedPages(source_ranges, allocator.HandleSize());
  auto target_pages =
      allocator.CollectUnmappedPages(target_ranges, allocator.HandleSize());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  ASSERT_TRUE(allocator.MoveBackingPage(source_pages[0], target_pages[0]));
  EXPECT_FALSE(allocator.ValidateUnmappedPages({target_pages[0]},
                                               "unit_test_move_stale_target"));
  EXPECT_FALSE(allocator.ValidateMappedPages({source_pages[0]},
                                             "unit_test_move_stale_source"));
  auto moved_pages =
      allocator.CollectMappedPages(target_ranges, allocator.HandleSize());
  auto source_unmapped_pages =
      allocator.CollectUnmappedPages(source_ranges, allocator.HandleSize());
  ASSERT_EQ(moved_pages.size(), 1UL);
  ASSERT_EQ(source_unmapped_pages.size(), 1UL);
  EXPECT_EQ(moved_pages[0].handle, source_pages[0].handle);

  ASSERT_TRUE(
      allocator.MoveBackingPage(moved_pages[0], source_unmapped_pages[0]));
  auto restored_pages =
      allocator.CollectMappedPages(source_ranges, allocator.HandleSize());
  ASSERT_EQ(restored_pages.size(), 1UL);
  EXPECT_EQ(restored_pages[0].handle, source_pages[0].handle);
}

TEST(CUDAVirtualMemAllocatorV2, DetectsRemapDestinationOwnedLayouts) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  auto* ptr = allocation_with_block.allocation->ptr();
  EXPECT_FALSE(allocator.IsAllocationOwnedByRemapDestination(ptr));
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {reinterpret_cast<VMMDevicePtr>(ptr), allocator.HandleSize()}};
  auto pages = allocator.CollectMappedPages(ranges, allocator.HandleSize());
  ASSERT_EQ(pages.size(), 1UL);
  auto meta = pages[0].meta;
  ASSERT_NE(meta, nullptr);
  meta->MarkOwnedByRemapDestination();
  EXPECT_TRUE(allocator.IsAllocationOwnedByRemapDestination(ptr));
  meta->RestoreOriginalOwnership();
  EXPECT_FALSE(allocator.IsAllocationOwnedByRemapDestination(ptr));
  EXPECT_FALSE(allocator.IsAllocationOwnedByRemapDestination(nullptr));
}

TEST(CUDAVirtualMemAllocatorV2, StagedRemapDestinationBlocksSource) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.HandleSize();
  std::vector<std::pair<VMMDevicePtr, size_t>> source_ranges = {
      {source_va, allocator.HandleSize()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_ranges = {
      {target_va, allocator.HandleSize()}};
  auto source_pages =
      allocator.CollectMappedPages(source_ranges, allocator.HandleSize());
  auto target_pages =
      allocator.CollectUnmappedPages(target_ranges, allocator.HandleSize());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  auto meta = source_pages[0].meta;
  ASSERT_NE(meta, nullptr);
  ASSERT_TRUE(allocator.MoveBackingPageForRemap(
      source_pages[0], target_pages[0], meta));
  EXPECT_TRUE(meta->IsOwnedByRemapDestination());

  auto staged = allocator.CreateStagedRemapDestinationAllocationWithBlock(
      target_va,
      std::vector<VMMAllocHandle>{source_pages[0].handle},
      0,
      1,
      PoolType::kLarge);
  ASSERT_NE(staged.allocation, nullptr);
  EXPECT_FALSE(
      allocator.IsAllocationOwnedByRemapDestination(staged.allocation->ptr()));

  auto remap_sources =
      allocator.CollectRemapSourcePages(target_ranges, allocator.HandleSize());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);

  std::vector<BlockPart> ipc_parts;
  EXPECT_TRUE(allocator.CollectIpcParts(
      staged.block.BeginVA(), staged.block.Size(), &ipc_parts));
  EXPECT_EQ(ipc_parts.size(), 1UL);
  ASSERT_NE(ipc_parts[0].chunk, nullptr);
  EXPECT_EQ(ipc_parts[0].chunk->base, target_va);
  EXPECT_EQ(ipc_parts[0].chunk_rel_off, 0UL);
  EXPECT_EQ(ipc_parts[0].len, allocator.HandleSize());

  auto committed =
      allocator.AdoptCommittedSyntheticAllocation(staged.allocation);
  staged.allocation = nullptr;
}

TEST(CUDAVirtualMemAllocatorV2, DetectsReusableBlockBacking) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  BlockV2 block = allocation_with_block.block;
  EXPECT_TRUE(allocator.IsBlockReusableForAllocation(block));

  allocator.MarkBackingIpcExported(block.BeginVA(), allocator.HandleSize());
  EXPECT_FALSE(allocator.IsBlockReusableForAllocation(block));

  BlockV2 invalid_block = BlockV2::MakeMappedBlock(
      BlockType::kFree,
      reinterpret_cast<void*>(allocator.VirtualMemBase() -
                              allocator.HandleSize()),
      allocator.HandleSize(),
      PoolType::kLarge);
  EXPECT_FALSE(allocator.IsBlockReusableForAllocation(invalid_block));
}

TEST(CUDAVirtualMemAllocatorV2, CollectsAndPinsIpcBlockBacking) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  BlockV2 block = allocation_with_block.block.MakeMappedActiveSubBlock(
      128, allocator.HandleSize() - 128 + 2048);
  const auto block_base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {block_base, allocation_with_block.allocation->size()}};
  auto pages = allocator.CollectMappedPages(
      ranges, allocation_with_block.allocation->size());
  ASSERT_EQ(pages.size(), 2UL);
  std::vector<BlockPart> ipc_parts;
  ASSERT_TRUE(
      allocator.CollectIpcParts(block.BeginVA(), block.Size(), &ipc_parts));
  ASSERT_EQ(ipc_parts.size(), 2UL);
  EXPECT_EQ(ipc_parts[0].chunk->base, pages[0].va);
  EXPECT_EQ(ipc_parts[0].chunk->size, allocator.HandleSize());
  EXPECT_EQ(ipc_parts[0].chunk->handle, pages[0].handle);
  EXPECT_EQ(ipc_parts[0].chunk_rel_off, 128UL);
  EXPECT_EQ(ipc_parts[0].len, allocator.HandleSize() - 128);
  EXPECT_EQ(ipc_parts[1].chunk->base, pages[1].va);
  EXPECT_EQ(ipc_parts[1].chunk_rel_off, 0UL);
  EXPECT_EQ(ipc_parts[1].len, 2048UL);

  EXPECT_TRUE(allocator.IsBlockReusableForAllocation(block));
  ASSERT_TRUE(allocator.MarkIpcExported(block.BeginVA(), block.Size()));
  EXPECT_TRUE(allocator.HasIpcExportedRange(block.BeginVA(), block.Size()));
  EXPECT_FALSE(allocator.IsBlockReusableForAllocation(block));

  ASSERT_NE(pages[0].meta, nullptr);
  pages[0].meta->MarkOwnedByRemapDestination();
  EXPECT_FALSE(
      allocator.CollectIpcParts(block.BeginVA(), block.Size(), &ipc_parts));
  pages[0].meta->RestoreOriginalOwnership();

  BlockV2 invalid_block = BlockV2::MakeMappedBlock(
      BlockType::kActive,
      reinterpret_cast<void*>(allocator.VirtualMemBase() -
                              allocator.HandleSize()),
      allocator.HandleSize(),
      PoolType::kLarge);
  EXPECT_FALSE(allocator.CollectIpcParts(
      invalid_block.BeginVA(), invalid_block.Size(), &ipc_parts));
  EXPECT_FALSE(
      allocator.MarkIpcExported(invalid_block.BeginVA(), invalid_block.Size()));
  EXPECT_FALSE(allocator.HasIpcExportedRange(invalid_block.BeginVA(),
                                             invalid_block.Size()));
}

TEST(CUDAVirtualMemAllocatorV2, SetsBlockBackingRemapEvent) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  BlockV2 block = allocation_with_block.block.MakeMappedActiveSubBlock(0, 2048);
  cudaEvent_t raw_event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&raw_event, cudaEventDisableTiming),
            cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(raw_event);
  ASSERT_TRUE(allocator.SetBlockRemapEvent(block, nullptr, guard));

  BlockV2 invalid_block = BlockV2::MakeMappedBlock(
      BlockType::kActive,
      reinterpret_cast<void*>(allocator.VirtualMemBase() -
                              allocator.HandleSize()),
      allocator.HandleSize(),
      PoolType::kLarge);
  EXPECT_FALSE(allocator.SetBlockRemapEvent(invalid_block, nullptr, guard));
}

TEST(CUDAVirtualMemAllocatorV2, LazyPendingStreamBlocksRemapAndRelease) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.HandleSize());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  BlockV2 block = allocation_with_block.block;
  const auto block_base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  VMMBackingMapBusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_TRUE(allocator.SetBlockRemapEvent(block, stream, nullptr));
  EXPECT_TRUE(allocator.IsBlockReusableForAllocation(block));
  EXPECT_FALSE(allocator.IsRangeReleasable(block_base, allocator.HandleSize()));
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {block_base, allocator.HandleSize()}};
  EXPECT_EQ(allocator.CollectMappedPages(ranges, allocator.HandleSize()).size(),
            1UL);
  auto remap_sources =
      allocator.CollectRemapSourcePages(ranges, allocator.HandleSize());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].va, block_base);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_TRUE(allocator.IsBlockReusableForAllocation(block));
  EXPECT_TRUE(allocator.IsRangeReleasable(block_base, allocator.HandleSize()));
  remap_sources =
      allocator.CollectRemapSourcePages(ranges, allocator.HandleSize());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
