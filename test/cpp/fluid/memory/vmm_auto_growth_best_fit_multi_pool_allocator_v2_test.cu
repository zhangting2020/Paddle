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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> CreatePoolAllocator(
    size_t handle_size, PoolType pool_type) {
  auto underlying = std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), handle_size, pool_type);
  return std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      underlying, 256, phi::GPUPlace(), pool_type);
}

std::unique_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2> CreateAllocator() {
  return std::make_unique<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
      CreatePoolAllocator(2UL << 20, PoolType::kSmall),
      CreatePoolAllocator(2UL << 20, PoolType::kLarge),
      2UL << 20,
      phi::GPUPlace());
}

bool HasBlockWithPtr(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                     void* ptr,
                     BlockType type) {
  for (const auto& block : allocator.all_blocks()) {
    if (block.Ptr() == ptr && block.type_ == type) {
      return true;
    }
  }
  return false;
}

}  // namespace

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, RouteSmallAndLarge) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->small_allocator(), small->ptr(), BlockType::kActive));
  EXPECT_FALSE(HasBlockWithPtr(
      *allocator->large_allocator(), small->ptr(), BlockType::kActive));
  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->large_allocator(), large->ptr(), BlockType::kActive));
  EXPECT_FALSE(HasBlockWithPtr(
      *allocator->small_allocator(), large->ptr(), BlockType::kActive));
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, SetBlockRemapEventRoutesByPtr) {
  auto allocator = CreateAllocator();

  auto allocation = allocator->Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto guard = std::make_shared<CudaEventGuard>(event);
  auto* ptr = allocation->ptr();
  ASSERT_TRUE(allocator->SetBlockRemapEvent(ptr, nullptr, guard));
  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->small_allocator(), ptr, BlockType::kActive));

  allocation.reset();
}

// --- P1: FreeImpl release path ---

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     FreeImplErasesRouteAndDelegates) {
  auto allocator = CreateAllocator();

  auto allocation = allocator->Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  auto* ptr = allocation->ptr();

  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->small_allocator(), ptr, BlockType::kActive));

  allocation.reset();

  auto reused = allocator->Allocate(256UL);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), ptr);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, CrossPoolAllocFree) {
  auto allocator = CreateAllocator();

  // Allocate across both routed pools.
  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);
  auto* small_ptr = small->ptr();
  auto* large_ptr = large->ptr();

  // Free in a different order than allocation.
  large.reset();
  auto large_reused = allocator->Allocate(2UL << 20);
  ASSERT_NE(large_reused, nullptr);
  EXPECT_EQ(large_reused->ptr(), large_ptr);

  small.reset();
  auto small_reused = allocator->Allocate(256UL);
  ASSERT_NE(small_reused, nullptr);
  EXPECT_EQ(small_reused->ptr(), small_ptr);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     SetBlockRemapEventReturnsFalseForUnknownPtr) {
  auto allocator = CreateAllocator();

  EXPECT_FALSE(allocator->SetBlockRemapEvent(
      reinterpret_cast<void*>(0x1), nullptr, nullptr));
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     VmmTensorPartsVisitorFindsRoutedV2Blocks) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  paddle::memory::VmmTensorPartsVisitor small_visitor(small->ptr());
  allocator->Accept(&small_visitor);

  ASSERT_TRUE(small_visitor.Found());
  ASSERT_EQ(small_visitor.Parts().size(), 1UL);
  EXPECT_EQ(small_visitor.Parts()[0].chunk_rel_off, 0UL);
  EXPECT_EQ(small_visitor.Parts()[0].len, small->size());

  paddle::memory::VmmTensorPartsVisitor large_visitor(large->ptr());
  allocator->Accept(&large_visitor);

  ASSERT_TRUE(large_visitor.Found());
  ASSERT_EQ(large_visitor.Parts().size(), 1UL);
  EXPECT_EQ(large_visitor.Parts()[0].chunk_rel_off, 0UL);
  EXPECT_EQ(large_visitor.Parts()[0].len, large->size());

  auto* small_ptr = small->ptr();
  auto* large_ptr = large->ptr();
  small.reset();
  large.reset();

  auto next_small = allocator->Allocate(256UL);
  auto next_large = allocator->Allocate(2UL << 20);
  ASSERT_NE(next_small, nullptr);
  ASSERT_NE(next_large, nullptr);
  EXPECT_NE(next_small->ptr(), small_ptr);
  EXPECT_NE(next_large->ptr(), large_ptr);
}

// --- P2: threshold boundary tests ---

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     SmallAllocationThresholdBoundary) {
  auto allocator = CreateAllocator();
  // CreateAllocator uses small_allocation_threshold = 2MB.

  // size = threshold - 1 → Small.
  auto just_below = allocator->Allocate((2UL << 20) - 1);
  ASSERT_NE(just_below, nullptr);
  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->small_allocator(), just_below->ptr(), BlockType::kActive));

  // size = threshold → Large (>= threshold goes to large).
  auto exact = allocator->Allocate(2UL << 20);
  ASSERT_NE(exact, nullptr);
  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->large_allocator(), exact->ptr(), BlockType::kActive));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
