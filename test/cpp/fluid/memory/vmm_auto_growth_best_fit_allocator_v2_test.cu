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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

COMMON_DECLARE_bool(vmm_v2_compact_all);

namespace paddle {
namespace memory {
namespace allocation {

namespace {

std::shared_ptr<CUDAVirtualMemAllocatorV2> CreateUnderlyingAllocator() {
  return std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
}

__global__ void BusyWaitKernel(uint64_t cycles) {
  uint64_t start = clock64();
  while (clock64() - start < cycles) {
  }
}

size_t CountBlocksOfType(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                         BlockType type) {
  size_t count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == type) {
      ++count;
    }
  }
  return count;
}

const BlockV2* FindBlockByPtr(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                              void* ptr) {
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == ptr) {
      return &block;
    }
  }
  return nullptr;
}

void ExpectIndexedFreeStats(VMMAutoGrowthBestFitAllocatorV2* allocator,
                            size_t total_free,
                            size_t max_free) {
  size_t actual_total_free = 0;
  size_t actual_max_free = 0;
  allocator->GetFreeBlockStats(&actual_total_free, &actual_max_free);
  EXPECT_EQ(actual_total_free, total_free);
  EXPECT_EQ(actual_max_free, max_free);
}

size_t AllocationPartBytes(const BlockV2& block) {
  size_t bytes = 0;
  for (size_t i = 0; i < block.AllocationPartCount(); ++i) {
    bytes += block.AllocationPartByteSize(i);
  }
  return bytes;
}

}  // namespace

TEST(VMMAutoGrowthBestFitAllocatorV2, ReuseSmallestSufficientFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // Layout after allocation:
  //   [ACTIVE 4MB] [ACTIVE 2MB separator] [ACTIVE 2MB small]
  // The separator prevents TryMerge from coalescing large and small on free.
  auto large = allocator.Allocate(underlying->HandleSize() * 2);
  auto separator = allocator.Allocate(underlying->HandleSize());
  auto small = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(large, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(small, nullptr);

  auto* small_ptr = small->ptr();
  large.reset();
  small.reset();
  // Layout: [FREE 4MB] [ACTIVE 2MB separator] [FREE 2MB]
  // free_blocks_: {(2MB, ptr_small), (4MB, ptr_large)}

  auto reused = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(reused, nullptr);

  // lower_bound({2MB, nullptr}) picks the exact-fit 2MB free block over the
  // larger 4MB one.
  EXPECT_EQ(reused->ptr(), small_ptr);
  // Layout: [FREE 4MB] [ACTIVE 2MB separator] [ACTIVE 2MB reused]
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  size_t free_block_count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == BlockType::kFree) {
      ++free_block_count;
      EXPECT_EQ(block.size_, underlying->HandleSize() * 2);
    }
  }
  EXPECT_EQ(free_block_count, 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockOnFirstAllocation) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // The bottom allocator rounds this grow to one full handle, but best-fit
  // should immediately split it into [ACTIVE requested_size] + [FREE remain].
  auto allocation = allocator.Allocate(256);
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->size(), 256UL);
  auto* alloc = static_cast<Allocation*>(allocation.get());
  EXPECT_EQ(alloc->ptr(), alloc->base_ptr());

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, 256UL);
  EXPECT_TRUE(it->HasSingleAllocationPart(0UL, 256UL));

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks().end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->HandleSize() - 256UL);
  EXPECT_EQ(it->owning_stream_, nullptr);
  EXPECT_TRUE(
      it->HasSingleAllocationPart(256UL, underlying->HandleSize() - 256UL));
  ExpectIndexedFreeStats(&allocator,
                         underlying->HandleSize() - 256UL,
                         underlying->HandleSize() - 256UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockAcrossTwoHandles) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t requested_size = underlying->HandleSize() + 256UL;
  auto allocation = allocator.Allocate(requested_size);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, requested_size);
  ASSERT_EQ(it->AllocationPartCount(), 2UL);
  EXPECT_EQ(it->AllocationPartHandleRelOffset(0), 0UL);
  EXPECT_EQ(it->AllocationPartByteSize(0), underlying->HandleSize());
  EXPECT_EQ(it->AllocationPartHandleRelOffset(1), 0UL);
  EXPECT_EQ(it->AllocationPartByteSize(1), 256UL);

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks().end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->HandleSize() - 256UL);
  EXPECT_TRUE(
      it->HasSingleAllocationPart(256UL, underlying->HandleSize() - 256UL));
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     MergeSplitFreeSlicesIntoSingleHandlePart) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  allocation.reset();

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->HandleSize());
  EXPECT_TRUE(merged.HasSingleAllocationPart(0UL, underlying->HandleSize()));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockAfterRemapEvent) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(allocation, nullptr);

  // Simulate StreamSafeCUDAAllocator injecting remap-safety metadata on free.
  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(event);
  auto* ptr = allocation->ptr();
  gpuStream_t fake_stream = reinterpret_cast<gpuStream_t>(0x1);
  ASSERT_TRUE(allocator.SetBlockRemapEvent(ptr, fake_stream, guard));

  allocation.reset();

  // Reuse with a smaller size triggers split. Pending-event state is now
  // tracked by BackingMap rather than handle metadata.
  auto reused = allocator.Allocate(256UL);
  ASSERT_NE(reused, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ != BlockType::kFree) {
      continue;
    }
    ++free_count;
    // owning_stream_ is cleared; nobody "owns" a free fragment.
    EXPECT_EQ(block.owning_stream_, nullptr);
    ASSERT_GT(block.AllocationPartCount(), 0UL);
  }
  EXPECT_EQ(free_count, 1UL);

  reused.reset();
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeFreeBlocksWithDifferentStreams) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto second = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  gpuStream_t first_stream;
  gpuStream_t second_stream;
  ASSERT_EQ(cudaStreamCreate(&first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&second_stream), cudaSuccess);

  auto* first_remap = dynamic_cast<VMMRemapEventAllocation*>(first.get());
  auto* second_remap = dynamic_cast<VMMRemapEventAllocation*>(second.get());
  ASSERT_NE(first_remap, nullptr);
  ASSERT_NE(second_remap, nullptr);
  ASSERT_TRUE(first_remap->SetVMMRemapEvent(first_stream, nullptr));
  ASSERT_TRUE(second_remap->SetVMMRemapEvent(second_stream, nullptr));

  first.reset();
  second.reset();

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, 2UL * underlying->HandleSize());
  EXPECT_EQ(merged.remap_pending_states_.size(), 1UL);

  ASSERT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, FreeBlockTooSmallFallsBackToGrow) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // Create a small free block (handle_size).
  auto small = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(small, nullptr);
  small.reset();
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize(), underlying->HandleSize());

  // Request 2*handle_size: free block is too small, must grow.
  auto large = allocator.Allocate(underlying->HandleSize() * 2);
  ASSERT_NE(large, nullptr);

  // The old tail free block is used as the prefix of the new allocation, and
  // only the missing suffix is grown from the bottom allocator.
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);
  EXPECT_EQ(CountBlocksOfType(allocator, BlockType::kActive), 1UL);

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_EQ(allocator.all_blocks().front().type_, BlockType::kActive);
  EXPECT_EQ(allocator.all_blocks().front().size_, underlying->HandleSize() * 2);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ThreeWayMerge) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // Allocate 3 consecutive handle-sized blocks.
  auto a = allocator.Allocate(underlying->HandleSize());
  auto b = allocator.Allocate(underlying->HandleSize());
  auto c = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);

  // Free first and last: creates 2 non-adjacent FREE blocks.
  a.reset();
  c.reset();
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize() * 2, underlying->HandleSize());

  // Free middle: TryMerge merges prev+it (left), then merged+next (right)
  // into a single block spanning all 3 handles.
  b.reset();
  EXPECT_EQ(allocator.all_blocks().size(), 1UL);
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize() * 3, underlying->HandleSize() * 3);

  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->HandleSize() * 3);
  EXPECT_EQ(merged.AllocationPartCount(), 3UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactRemapsWholeFreeHandleToTail) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto middle = allocator.Allocate(underlying->HandleSize());
  auto last = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* first_ptr = first->ptr();
  auto* middle_ptr = middle->ptr();
  auto* last_ptr = last->ptr();
  std::vector<std::pair<VMMDevicePtr, size_t>> middle_range = {
      {reinterpret_cast<VMMDevicePtr>(middle_ptr), underlying->HandleSize()}};
  const auto middle_pages =
      underlying->CollectMappedPages(middle_range, underlying->HandleSize());
  ASSERT_EQ(middle_pages.size(), 1UL);

  middle.reset();
  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, underlying->HandleSize());

  ASSERT_EQ(allocator.all_blocks().size(), 4UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->ptr_, first_ptr);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kUnmappedFree);
  EXPECT_EQ(it->ptr_, middle_ptr);
  EXPECT_EQ(it->size_, underlying->HandleSize());
  EXPECT_EQ(it->AllocationPartCount(), 0UL);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->ptr_, last_ptr);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->HandleSize());
  EXPECT_TRUE(it->HasSingleAllocationPart(0UL, underlying->HandleSize()));
  std::vector<std::pair<VMMDevicePtr, size_t>> tail_range = {
      {reinterpret_cast<VMMDevicePtr>(it->ptr_), underlying->HandleSize()}};
  const auto tail_pages =
      underlying->CollectMappedPages(tail_range, underlying->HandleSize());
  ASSERT_EQ(tail_pages.size(), 1UL);
  EXPECT_EQ(tail_pages[0].handle, middle_pages[0].handle);
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize(), underlying->HandleSize());
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     AllocateSkipsOwnershipOverlappedUnmappedFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto middle = allocator.Allocate(underlying->HandleSize());
  auto last = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  middle.reset();

  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), underlying->HandleSize());
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize(), underlying->HandleSize());

  auto tail_reuse = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(tail_reuse, nullptr);
  EXPECT_NE(tail_reuse->ptr(), middle_ptr);
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);

  const size_t tail_before_unmapped_reuse = underlying->TailOffset();
  auto unmapped_reuse = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(unmapped_reuse, nullptr);
  EXPECT_NE(unmapped_reuse->ptr(), middle_ptr);
  EXPECT_GT(underlying->TailOffset(), tail_before_unmapped_reuse);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReleaseIdleMiddleChunkLeavesReusableUnmappedFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto middle = allocator.Allocate(underlying->HandleSize());
  auto last = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  const size_t tail_after_allocs = underlying->TailOffset();
  middle.reset();

  const uint64_t released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, underlying->HandleSize());
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  auto it = allocator.all_blocks().begin();
  EXPECT_TRUE(it->IsActive());
  ++it;
  ASSERT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, middle_ptr);
  EXPECT_EQ(it->size_, underlying->HandleSize());
  ++it;
  EXPECT_TRUE(it->IsActive());

  auto reused = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), middle_ptr);
  EXPECT_EQ(underlying->TailOffset(), tail_after_allocs);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     CollectTensorPartsMarksIpcExportedAndPinsFreedBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->HandleSize() * 2);
  ASSERT_NE(allocation, nullptr);
  auto* ptr = allocation->ptr();
  const size_t tail_after_first_alloc = underlying->TailOffset();
  auto* tensor_ptr = reinterpret_cast<uint8_t*>(ptr) + 128UL;
  const size_t tensor_size = underlying->HandleSize() - 128UL + 2048UL;

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(tensor_ptr, tensor_size, &parts));
  ASSERT_EQ(parts.size(), 2UL);
  EXPECT_EQ(parts[0].chunk_rel_off, 128UL);
  EXPECT_EQ(parts[0].len, underlying->HandleSize() - 128UL);
  EXPECT_EQ(parts[1].chunk_rel_off, 0UL);
  EXPECT_EQ(parts[1].len, 2048UL);

  EXPECT_EQ(parts[0].chunk->base, reinterpret_cast<VMMDevicePtr>(ptr));
  EXPECT_EQ(parts[0].chunk->size, underlying->HandleSize());
  EXPECT_EQ(parts[1].chunk->base,
            reinterpret_cast<VMMDevicePtr>(ptr) + underlying->HandleSize());
  std::vector<std::pair<VMMDevicePtr, size_t>> exported_ranges = {
      {reinterpret_cast<VMMDevicePtr>(ptr), underlying->HandleSize() * 2}};
  EXPECT_TRUE(
      underlying
          ->CollectMappedPages(exported_ranges, underlying->HandleSize() * 2)
          .empty());

  allocation.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(ptr), underlying->HandleSize() * 2));
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);

  auto released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, 0UL);
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(ptr), underlying->HandleSize() * 2));

  auto compacted =
      allocator.Compact(phi::GPUPlace(), underlying->HandleSize() * 2);
  EXPECT_EQ(compacted, 0UL);

  auto next = allocator.Allocate(underlying->HandleSize() * 2);
  ASSERT_NE(next, nullptr);
  EXPECT_NE(next->ptr(), ptr);
  EXPECT_GT(underlying->TailOffset(), tail_after_first_alloc);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BackingIpcPinAllowsRegularMergedNeighborRelease) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto exported = allocator.Allocate(underlying->HandleSize());
  auto regular = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(regular, nullptr);
  auto* exported_ptr = exported->ptr();
  auto* regular_ptr = regular->ptr();
  const size_t tail_after_two_allocs = underlying->TailOffset();

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(
      exported_ptr, underlying->HandleSize(), &parts));

  regular.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  ExpectIndexedFreeStats(
      &allocator, underlying->HandleSize(), underlying->HandleSize());

  exported.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->HandleSize()));
  EXPECT_EQ(allocator.all_blocks().front().size_, underlying->HandleSize() * 2);
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);

  auto released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, underlying->HandleSize());
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto block_it = allocator.all_blocks().begin();
  ASSERT_TRUE(block_it->IsFree());
  EXPECT_EQ(block_it->ptr_, exported_ptr);
  EXPECT_EQ(block_it->size_, underlying->HandleSize());
  ++block_it;
  ASSERT_TRUE(block_it->IsUnmappedFree());
  EXPECT_EQ(block_it->ptr_, regular_ptr);
  EXPECT_EQ(block_it->size_, underlying->HandleSize());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->HandleSize()));

  auto next = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(next, nullptr);
  EXPECT_NE(next->ptr(), exported_ptr);
  EXPECT_EQ(next->ptr(), regular_ptr);
  EXPECT_EQ(underlying->TailOffset(), tail_after_two_allocs);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BackingIpcPinAllowsRegularMergedNeighborCompact) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto exported = allocator.Allocate(underlying->HandleSize());
  auto regular = allocator.Allocate(underlying->HandleSize());
  auto anchor = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(regular, nullptr);
  ASSERT_NE(anchor, nullptr);
  auto* exported_ptr = exported->ptr();
  auto* regular_ptr = regular->ptr();

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(
      exported_ptr, underlying->HandleSize(), &parts));

  regular.reset();
  exported.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->HandleSize()));
  EXPECT_EQ(allocator.all_blocks().front().size_, underlying->HandleSize() * 2);
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);

  auto compacted = allocator.Compact(phi::GPUPlace(), underlying->HandleSize());
  EXPECT_EQ(compacted, underlying->HandleSize());
  ASSERT_GE(allocator.all_blocks().size(), 4UL);

  auto first = allocator.all_blocks().begin();
  EXPECT_TRUE(first->IsFree());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->HandleSize()));
  EXPECT_EQ(first->ptr_, exported_ptr);
  EXPECT_EQ(first->size_, underlying->HandleSize());

  auto second = std::next(first);
  EXPECT_TRUE(second->IsUnmappedFree());
  EXPECT_EQ(second->ptr_, regular_ptr);
  EXPECT_EQ(second->size_, underlying->HandleSize());
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     TailMoveCompactionNormalizesAdjacentUnmappedSources) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto second = allocator.Allocate(underlying->HandleSize());
  auto anchor = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(anchor, nullptr);
  auto* first_ptr = first->ptr();
  auto* second_ptr = second->ptr();

  first.reset();
  auto first_compacted = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(first_compacted, underlying->HandleSize());

  auto first_block = allocator.all_blocks().begin();
  ASSERT_NE(first_block, allocator.all_blocks().end());
  ASSERT_TRUE(first_block->IsUnmappedFree());
  EXPECT_EQ(first_block->ptr_, first_ptr);
  EXPECT_EQ(first_block->size_, underlying->HandleSize());

  second.reset();
  auto second_compacted = allocator.Compact(phi::GPUPlace());
  // The previous compacted tail mapped-free block is a valid source again
  // after its BackingMap meta is refreshed to the destination layout. This
  // call may therefore remap more than only the newly freed second block.
  EXPECT_GE(second_compacted, underlying->HandleSize());
  EXPECT_EQ(second_compacted % underlying->HandleSize(), 0UL);

  first_block = allocator.all_blocks().begin();
  ASSERT_NE(first_block, allocator.all_blocks().end());
  ASSERT_TRUE(first_block->IsUnmappedFree());
  EXPECT_EQ(first_block->ptr_, first_ptr);
  EXPECT_EQ(first_block->size_, underlying->HandleSize() * 2);

  auto next = std::next(first_block);
  ASSERT_NE(next, allocator.all_blocks().end());
  EXPECT_FALSE(next->IsUnmappedFree());
  EXPECT_EQ(reinterpret_cast<uint8_t*>(first_ptr) + underlying->HandleSize(),
            second_ptr);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, VMMTensorPartsVisitorFindsV2Blocks) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(allocation, nullptr);

  paddle::memory::VMMTensorPartsVisitor visitor(allocation->ptr(),
                                                allocation->size());
  allocator.Accept(&visitor);

  ASSERT_TRUE(visitor.Found());
  ASSERT_EQ(visitor.Parts().size(), 1UL);
  EXPECT_EQ(visitor.Parts()[0].chunk_rel_off, 0UL);
  EXPECT_EQ(visitor.Parts()[0].len, underlying->HandleSize());
  EXPECT_TRUE(underlying->HasIpcExportedRange(
      reinterpret_cast<VMMDevicePtr>(allocation->ptr()),
      underlying->HandleSize()));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactSkipsPartialFreeHandle) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, 0UL);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->ptr_,
            reinterpret_cast<uint8_t*>(allocation->ptr()) + allocation->size());
  EXPECT_EQ(it->size_, underlying->HandleSize() - 256UL);
  EXPECT_TRUE(
      it->HasSingleAllocationPart(256UL, underlying->HandleSize() - 256UL));
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BoundedCompactSkipsWhenReleasableBytesInsufficient) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto second = allocator.Allocate(underlying->HandleSize());
  auto separator = allocator.Allocate(underlying->HandleSize());
  auto third = allocator.Allocate(underlying->HandleSize());
  auto fourth = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(third, nullptr);
  ASSERT_NE(fourth, nullptr);

  second.reset();
  third.reset();
  auto partial = allocator.Allocate(256UL);
  ASSERT_NE(partial, nullptr);
  const size_t block_count_before = allocator.all_blocks().size();
  const size_t requested_size = underlying->HandleSize() + 1UL;

  const size_t remapped = allocator.Compact(phi::GPUPlace(), requested_size);
  EXPECT_EQ(remapped, 0UL);
  EXPECT_EQ(allocator.all_blocks().size(), block_count_before);
  for (const auto& block : allocator.all_blocks()) {
    EXPECT_FALSE(block.IsUnmappedFree());
  }
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BoundedCompactAllowsPartialRemapWhenFreeBytesAreInsufficient) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->HandleSize();
  auto first = allocator.Allocate(handle_size);
  auto movable = allocator.Allocate(handle_size);
  auto tail_guard = allocator.Allocate(handle_size);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(movable, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  auto* movable_ptr = movable->ptr();
  movable.reset();

  const size_t requested_size = 3UL * handle_size;
  const size_t remapped = allocator.Compact(phi::GPUPlace(), requested_size);
  EXPECT_EQ(remapped, handle_size);

  bool found_old_source = false;
  bool found_tail_free = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == movable_ptr) {
      found_old_source = true;
      EXPECT_TRUE(block.IsUnmappedFree());
      EXPECT_EQ(block.size_, handle_size);
    }
    if (block.IsMappedFree() && block.ptr_ != movable_ptr) {
      found_tail_free = true;
      EXPECT_EQ(block.size_, handle_size);
    }
  }
  EXPECT_TRUE(found_old_source);
  EXPECT_TRUE(found_tail_free);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BoundedCompactUsesTailFreeDeficitForReleasablePrecheck) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->HandleSize();
  auto unmapped_first = allocator.Allocate(handle_size);
  auto unmapped_second = allocator.Allocate(handle_size);
  auto separator = allocator.Allocate(handle_size);
  auto movable = allocator.Allocate(handle_size);
  auto tail_guard = allocator.Allocate(handle_size);
  ASSERT_NE(unmapped_first, nullptr);
  ASSERT_NE(unmapped_second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(movable, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  auto* unmapped_ptr = unmapped_first->ptr();
  unmapped_first.reset();
  unmapped_second.reset();
  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), 2UL * handle_size);

  movable.reset();
  const size_t requested_size = 2UL * handle_size + 1UL;
  const size_t remapped = allocator.Compact(phi::GPUPlace(), requested_size);
  EXPECT_GE(remapped, handle_size);
  EXPECT_EQ(remapped % handle_size, 0UL);

  bool found_unmapped_range = false;
  bool found_movable_source = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == unmapped_ptr) {
      found_unmapped_range = true;
      EXPECT_TRUE(block.IsUnmappedFree());
      EXPECT_EQ(block.size_, 2UL * handle_size);
    }
    if (block.IsUnmappedFree() && block.size_ == handle_size &&
        block.ptr_ != unmapped_ptr) {
      found_movable_source = true;
    }
  }
  EXPECT_TRUE(found_unmapped_range);
  EXPECT_TRUE(found_movable_source);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     CompactAllIgnoresReleasableBytesRequestedPrecheck) {
  const bool old_compact_all = FLAGS_vmm_v2_compact_all;
  FLAGS_vmm_v2_compact_all = true;

  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto second = allocator.Allocate(underlying->HandleSize());
  auto separator = allocator.Allocate(underlying->HandleSize());
  auto third = allocator.Allocate(underlying->HandleSize());
  auto fourth = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(third, nullptr);
  ASSERT_NE(fourth, nullptr);

  auto* third_ptr = third->ptr();
  second.reset();
  third.reset();
  auto partial = allocator.Allocate(256UL);
  ASSERT_NE(partial, nullptr);
  const size_t requested_size = underlying->HandleSize() + 1UL;

  const size_t remapped = allocator.Compact(phi::GPUPlace(), requested_size);
  FLAGS_vmm_v2_compact_all = old_compact_all;

  EXPECT_EQ(remapped, underlying->HandleSize());
  bool found_third_unmapped = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == third_ptr) {
      found_third_unmapped = block.IsUnmappedFree();
      EXPECT_EQ(block.size_, underlying->HandleSize());
    }
  }
  EXPECT_TRUE(found_third_unmapped);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactWaitsForBackingMapPendingEvent) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(allocation, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  gpuEvent_t event;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(event);
  ASSERT_TRUE(allocator.SetBlockRemapEvent(allocation->ptr(), stream, guard));

  allocation.reset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), underlying->HandleSize());

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kUnmappedFree);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->HandleSize());
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactWaitsForBlockOwningStream) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(allocation, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto* remap_allocation =
      dynamic_cast<VMMRemapEventAllocation*>(allocation.get());
  ASSERT_NE(remap_allocation, nullptr);
  ASSERT_TRUE(remap_allocation->SetVMMRemapEvent(stream, nullptr));

  allocation.reset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), underlying->HandleSize());

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactUsesBlockListTailPlacement) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->HandleSize());
  auto middle = allocator.Allocate(underlying->HandleSize());
  auto last = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  middle.reset();
  const size_t remaining_tail =
      underlying->VirtualMemSize() - underlying->TailOffset();
  underlying->AdvanceTailOffset(remaining_tail);

  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, underlying->HandleSize());

  void* remapped_free_ptr = nullptr;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == BlockType::kFree &&
        block.size_ == underlying->HandleSize()) {
      EXPECT_TRUE(block.HasSingleAllocationPart(0UL, underlying->HandleSize()));
      remapped_free_ptr = block.ptr_;
      break;
    }
  }
  ASSERT_NE(remapped_free_ptr, nullptr)
      << "expected one remapped tail free block";

  auto remap_sources = underlying->CollectRemapSourcePages(
      {{reinterpret_cast<VMMDevicePtr>(remapped_free_ptr),
        underlying->HandleSize()}},
      underlying->HandleSize());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);

  auto remapped_active = allocator.Allocate(underlying->HandleSize());
  ASSERT_NE(remapped_active, nullptr);
  std::vector<BlockPart> ipc_parts;
  EXPECT_TRUE(allocator.CollectTensorParts(
      remapped_active->ptr(), underlying->HandleSize(), &ipc_parts));
  ASSERT_EQ(ipc_parts.size(), 1UL);
  EXPECT_EQ(ipc_parts[0].chunk->base,
            reinterpret_cast<VMMDevicePtr>(remapped_active->ptr()));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactKeepsPartialSourceParts) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->HandleSize();
  auto large = allocator.Allocate(3UL * handle_size);
  ASSERT_NE(large, nullptr);
  large.reset();

  auto prefix = allocator.Allocate(256);
  ASSERT_NE(prefix, nullptr);

  const size_t remapped = allocator.Compact(phi::GPUPlace());
  ASSERT_EQ(remapped, 2UL * handle_size);

  for (const auto& block : allocator.all_blocks()) {
    if (block.IsMappedFree()) {
      EXPECT_EQ(AllocationPartBytes(block), block.Size());
    }
  }

  prefix.reset();
  for (const auto& block : allocator.all_blocks()) {
    if (block.IsMappedFree()) {
      EXPECT_EQ(AllocationPartBytes(block), block.Size());
    }
  }

  EXPECT_GT(allocator.Release(phi::GPUPlace()), 0UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     CompactScattersAcrossUnmappedFreeBlocksWhenTailMapped) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->HandleSize();
  auto target_a = allocator.Allocate(handle_size);
  auto source_a = allocator.Allocate(handle_size);
  auto target_b = allocator.Allocate(handle_size);
  auto source_b = allocator.Allocate(handle_size);
  ASSERT_NE(target_a, nullptr);
  ASSERT_NE(source_a, nullptr);
  ASSERT_NE(target_b, nullptr);
  ASSERT_NE(source_b, nullptr);
  auto* target_a_ptr = target_a->ptr();
  auto* source_a_ptr = source_a->ptr();
  auto* target_b_ptr = target_b->ptr();
  auto* source_b_ptr = source_b->ptr();

  target_a.reset();
  target_b.reset();
  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), 2UL * handle_size);
  auto tail_active = allocator.Allocate(2UL * handle_size);
  ASSERT_NE(tail_active, nullptr);

  auto hidden_tail_mapping = underlying->AllocateWithBlock(handle_size);
  ASSERT_TRUE(hidden_tail_mapping.HasAllocation());

  source_a.reset();
  source_b.reset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace(), 2UL * handle_size),
            2UL * handle_size);

  const auto* target_a_block = FindBlockByPtr(allocator, target_a_ptr);
  ASSERT_NE(target_a_block, nullptr);
  EXPECT_TRUE(target_a_block->IsFree());
  EXPECT_TRUE(target_a_block->HasSingleAllocationPart(0UL, handle_size));

  const auto* target_b_block = FindBlockByPtr(allocator, target_b_ptr);
  ASSERT_NE(target_b_block, nullptr);
  EXPECT_TRUE(target_b_block->IsFree());
  EXPECT_TRUE(target_b_block->HasSingleAllocationPart(0UL, handle_size));

  const auto* source_a_block = FindBlockByPtr(allocator, source_a_ptr);
  ASSERT_NE(source_a_block, nullptr);
  EXPECT_TRUE(source_a_block->IsUnmappedFree());
  EXPECT_EQ(source_a_block->Size(), handle_size);

  const auto* source_b_block = FindBlockByPtr(allocator, source_b_ptr);
  ASSERT_NE(source_b_block, nullptr);
  EXPECT_TRUE(source_b_block->IsUnmappedFree());
  EXPECT_EQ(source_b_block->Size(), handle_size);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
