// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

#include <algorithm>
#include <iterator>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

class VMMAutoGrowthBestFitBlockAllocationV2 : public Allocation {
 public:
  explicit VMMAutoGrowthBestFitBlockAllocationV2(const BlockListIt& block_it)
      : Allocation(block_it->ptr_,
                   block_it->chunk_->underlying_allocation_->base_ptr(),
                   block_it->size_,
                   block_it->chunk_->underlying_allocation_->place()),
        block_it_(block_it) {}

  const BlockListIt& block_it() const { return block_it_; }

 private:
  BlockListIt block_it_;
};

bool TryConcatAdjacent(BlockPart* a, const BlockPart& b) {
  if (a == nullptr) {
    return false;
  }
  if (a->chunk.get() != b.chunk.get()) {
    return false;
  }
  if (a->chunk_rel_off + a->len != b.chunk_rel_off) {
    return false;
  }
  a->len += b.len;
  return true;
}

std::vector<BlockPart> SlicePartsForRange(const std::vector<BlockPart>& parts,
                                          size_t pick_off,
                                          size_t pick_len) {
  std::vector<BlockPart> out;
  size_t cursor = 0;
  size_t need = pick_len;
  for (const auto& p : parts) {
    if (need == 0) {
      break;
    }
    size_t left = cursor;
    size_t right = cursor + p.len;
    cursor = right;
    size_t slice_left = std::max(left, pick_off);
    size_t slice_right = std::min(right, pick_off + pick_len);
    if (slice_left >= slice_right) {
      continue;
    }
    BlockPart cut{p.chunk,
                  p.chunk_rel_off + (slice_left - left),
                  slice_right - slice_left};
    if (!out.empty() && TryConcatAdjacent(&out.back(), cut)) {
      need -= (slice_right - slice_left);
      continue;
    }
    out.push_back(std::move(cut));
    need -= (slice_right - slice_left);
  }
  return out;
}

void AppendPartsTail(std::vector<BlockPart>* dst,
                     const std::vector<BlockPart>& src) {
  if (src.empty()) {
    return;
  }
  if (!dst->empty() && TryConcatAdjacent(&dst->back(), src.front())) {
    dst->insert(dst->end(), std::next(src.begin()), src.end());
    return;
  }
  dst->insert(dst->end(), src.begin(), src.end());
}

}  // namespace

VMMAutoGrowthBestFitAllocatorV2::VMMAutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<Allocator>& underlying_allocator,
    size_t alignment,
    const GPUPlace& place,
    PoolType pool_type)
    : underlying_allocator_(underlying_allocator),
      alignment_(alignment),
      place_(place),
      pool_type_(pool_type) {}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocateImpl(size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size = AlignedSize(size, alignment_);
  if (auto* allocation = AllocFromFreeBlocks(size)) {
    return allocation;
  }

  auto allocation =
      static_unique_ptr_cast<Allocation>(underlying_allocator_->Allocate(size));
  auto* cuda_vmm_allocator =
      dynamic_cast<CUDAVirtualMemAllocatorV2*>(underlying_allocator_.get());
  PADDLE_ENFORCE_NOT_NULL(
      cuda_vmm_allocator,
      common::errors::InvalidArgument(
          "VMMAutoGrowthBestFitAllocatorV2 expects "
          "CUDAVirtualMemAllocatorV2 as underlying allocator."));

  std::vector<BlockPart> parts;
  PADDLE_ENFORCE_EQ(
      cuda_vmm_allocator->CollectAllocationParts(allocation->base_ptr(),
                                                 &parts),
      true,
      common::errors::NotFound("Can not collect VMM parts for allocation %p.",
                               allocation->base_ptr()));

  chunks_.emplace_back(std::move(allocation));
  auto* chunk = &chunks_.back();

  BlockV2 block;
  block.ptr_ = chunk->underlying_allocation_->ptr();
  block.size_ = chunk->underlying_allocation_->size();
  block.type_ = BlockType::kActive;
  block.chunk_ = chunk;
  block.parts_ = std::move(parts);
  block.pool_type_ = pool_type_;
  auto it = blocks_.insert(blocks_.end(), std::move(block));
  allocated_blocks_[it->ptr_] = it;
  return new VMMAutoGrowthBestFitBlockAllocationV2(it);
}

void VMMAutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto* wrapped_allocation =
      static_cast<VMMAutoGrowthBestFitBlockAllocationV2*>(allocation);
  auto it = wrapped_allocation->block_it();
  if (it == blocks_.end()) {
    delete wrapped_allocation;
    return;
  }
  allocated_blocks_.erase(it->ptr_);
  it->type_ = BlockType::kFree;
  TryMerge(it);
  delete wrapped_allocation;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                                                         gpuStream_t stream,
                                                         gpuEvent_t event
#else
                                                         void* stream,
                                                         void* event
#endif
) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  it->second->last_use_stream_ = stream;
  it->second->remap_safe_event_ = event;
#else
  (void)stream;
  (void)event;
#endif
  return true;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size) {
  auto it = free_blocks_.lower_bound({size, nullptr});
  if (it == free_blocks_.end()) {
    return nullptr;
  }

  auto block_it = it->second;
  EraseFreeBlock(block_it);

  if (block_it->size_ > size) {
    const size_t remaining_size = block_it->size_ - size;
    BlockV2 remaining_block;
    remaining_block.ptr_ = reinterpret_cast<uint8_t*>(block_it->ptr_) + size;
    remaining_block.size_ = remaining_size;
    remaining_block.type_ = BlockType::kFree;
    remaining_block.chunk_ = block_it->chunk_;
    remaining_block.parts_ =
        SlicePartsForRange(block_it->parts_, size, remaining_size);
    remaining_block.pool_type_ = block_it->pool_type_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    remaining_block.owning_stream_ = block_it->owning_stream_;
    remaining_block.last_use_stream_ = block_it->last_use_stream_;
#endif

    block_it->size_ = size;
    block_it->parts_ = SlicePartsForRange(block_it->parts_, 0, size);
    auto remain_it =
        blocks_.insert(std::next(block_it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  block_it->type_ = BlockType::kActive;
  allocated_blocks_[block_it->ptr_] = block_it;
  return new VMMAutoGrowthBestFitBlockAllocationV2(block_it);
}

BlockListIt VMMAutoGrowthBestFitAllocatorV2::FindBlockByPtr(void* ptr) {
  auto it = allocated_blocks_.find(ptr);
  if (it != allocated_blocks_.end()) {
    return it->second;
  }
  for (auto block_it = blocks_.begin(); block_it != blocks_.end(); ++block_it) {
    if (block_it->ptr_ == ptr) {
      return block_it;
    }
  }
  return blocks_.end();
}

void VMMAutoGrowthBestFitAllocatorV2::InsertFreeBlock(BlockListIt it) {
  free_blocks_[{it->size_, it->ptr_}] = it;
}

void VMMAutoGrowthBestFitAllocatorV2::EraseFreeBlock(BlockListIt it) {
  free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::TryMerge(BlockListIt it) {
  if (it != blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(prev->ptr_) + prev->size_ ==
            reinterpret_cast<uint8_t*>(it->ptr_)) {
      EraseFreeBlock(prev);
      AppendPartsTail(&prev->parts_, it->parts_);
      prev->size_ += it->size_;
      blocks_.erase(it);
      it = prev;
    }
  }

  auto next = std::next(it);
  if (next != blocks_.end() && next->type_ == BlockType::kFree &&
      reinterpret_cast<uint8_t*>(it->ptr_) + it->size_ ==
          reinterpret_cast<uint8_t*>(next->ptr_)) {
    EraseFreeBlock(next);
    AppendPartsTail(&it->parts_, next->parts_);
    it->size_ += next->size_;
    blocks_.erase(next);
  }

  InsertFreeBlock(it);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
