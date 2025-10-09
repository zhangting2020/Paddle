// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"

#include <mutex>

#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

bool NeedSplit(size_t block_size, size_t alignment, size_t alloc_size) {
  return block_size > (alloc_size * 2) || (block_size - alloc_size) > alignment;
}

// 折叠相邻且同段的两个 part（a 与 b 紧邻且同 seg，则把 b 并入 a）
static inline bool TryConcatAdjacent(BlockPart *a, const BlockPart &b) {
  if (!a) return false;
  if (a->chunk.get() != b.chunk.get()) return false;
  if (a->chunk_rel_off + a->len != b.chunk_rel_off) return false;
  a->len += b.len;
  return true;
}

// 从 parts 中切出 [pick_off, pick_off+pick_len) 对应的子片段序列
static std::vector<BlockPart> SlicePartsForRange(
    const std::vector<BlockPart> &parts, size_t pick_off, size_t pick_len) {
  std::vector<BlockPart> out;
  size_t cursor = 0, need = pick_len;
  for (const auto &p : parts) {
    if (!need) break;
    size_t L = cursor;
    size_t R = cursor + p.len;
    cursor = R;
    size_t l = std::max(L, pick_off);
    size_t r = std::min(R, pick_off + pick_len);
    if (l >= r) continue;
    BlockPart cut{p.chunk, p.chunk_rel_off + (l - L), r - l};
    // 与上一条尝试折叠
    if (!out.empty() && TryConcatAdjacent(&out.back(), cut)) {
      // 已并入
    } else {
      out.push_back(std::move(cut));
    }
    need -= (r - l);
  }
  return out;
}

// 把 src 的 parts 追加到 dst 的尾端；在边界尝试折叠（同段且相邻）
static inline void AppendPartsTail(std::vector<BlockPart> *dst,
                                   const std::vector<BlockPart> &src) {
  if (src.empty()) return;
  if (!dst->empty() && TryConcatAdjacent(&dst->back(), src.front())) {
    // 头尾已折叠，余下的直接追加
    dst->insert(dst->end(), std::next(src.begin()), src.end());
  } else {
    dst->insert(dst->end(), src.begin(), src.end());
  }
}

VirtualMemoryAutoGrowthBestFitAllocator::
    VirtualMemoryAutoGrowthBestFitAllocator(
        const std::shared_ptr<Allocator> &underlying_allocator,
        size_t alignment,
        const phi::GPUPlace &place)
    : underlying_allocator_(
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment)),
      alignment_(alignment),
      place_(place) {}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocateImpl(
    size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size = AlignedSize(size, alignment_);
  auto result = AllocFromFreeBlocks(size);

  if (!result) {
    ExtendAndMerge(size);
    result = AllocFromFreeBlocks(size);
  }

  return result;
}

void VirtualMemoryAutoGrowthBestFitAllocator::FreeImpl(
    phi::Allocation *allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  TryMergeBlock2Blocks(block_it);
  delete allocation;
}

void VirtualMemoryAutoGrowthBestFitAllocator::TryMergeBlock2Blocks(
    std::list<Block>::iterator block) {
  if (block->ptr_ == all_blocks_.front().ptr_ &&
      block->ptr_ == all_blocks_.back().ptr_) {
    block->is_free_ = true;
    free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
  } else if (block->ptr_ == all_blocks_.front().ptr_) {
    auto next = std::next(block);
    if (next->is_free_ &&
        reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ == next->ptr_) {
      AppendPartsTail(&block->parts_, next->parts_);
      // merge with next
      block->size_ += next->size_;
      block->is_free_ = true;
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else if (block->ptr_ == all_blocks_.back().ptr_) {
    auto pre = std::prev(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      AppendPartsTail(&pre->parts_, block->parts_);
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else {
    auto pre = std::prev(block);
    auto next = std::next(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_ &&
        !(next->is_free_ &&
          reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
              next->ptr_)) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      AppendPartsTail(&pre->parts_, block->parts_);
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else if (next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_ &&
               !(pre->is_free_ &&
                 reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                     block->ptr_)) {
      // merge with next
      block->size_ += next->size_;
      block->is_free_ = true;
      AppendPartsTail(&block->parts_, next->parts_);
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else if (pre->is_free_ &&
               reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                   block->ptr_ &&
               next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_) {
      // merge with pre and next
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      AppendPartsTail(&pre->parts_, block->parts_);
      AppendPartsTail(&pre->parts_, next->parts_);
      pre->size_ += (block->size_ + next->size_);
      all_blocks_.erase(block);
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  }
}

void VirtualMemoryAutoGrowthBestFitAllocator::ExtendAndMerge(size_t size) {
  void *ptr = nullptr;

  auto allocateptr = underlying_allocator_->Allocate(size);
  ptr = allocateptr->ptr();
  size = allocateptr->size();
  allocations_.push_back(std::move(allocateptr));  // hold allocation

  // 从底层 allocation 提取 VMM 段信息
  auto *raw = allocations_.back().get();
  auto *base_alloc = dynamic_cast<Allocation *>(raw);
  PADDLE_ENFORCE_NOT_NULL(base_alloc, "Underlying allocation null");
  auto handle = base_alloc->handle();
  PADDLE_ENFORCE_NE(handle, 0, "Underlying allocation is not VMM (handle=0)");

  auto chunk = std::make_shared<VmmChunkMeta>();
  chunk->base = reinterpret_cast<CUdeviceptr>(ptr);
  chunk->size = size;
  chunk->handle = handle;
  chunk->device = place_.device;

  // 新 free 块的 parts：整段一条
  std::vector<BlockPart> new_parts(1,
                                   BlockPart{chunk, /*chunk_rel_off=*/0, size});

  if (all_blocks_.empty()) {
    all_blocks_.emplace_back(ptr, size, true, std::move(new_parts));
    free_blocks_.emplace(std::make_pair(size, ptr), all_blocks_.begin());
    return;
  }

  // insert to back
  auto block_it = all_blocks_.end();
  block_it--;
  PADDLE_ENFORCE_LE(
      reinterpret_cast<uint8_t *>(block_it->ptr_),
      reinterpret_cast<uint8_t *>(ptr),
      "ExtendAndMerge expects monotonically increasing addresses from "
      "underlying_allocator_. Got new ptr before last block.");
  if (block_it->is_free_ &&
      reinterpret_cast<uint8_t *>(block_it->ptr_) + block_it->size_ == ptr) {
    // merge with pre
    free_blocks_.erase(std::make_pair(block_it->size_, block_it->ptr_));
    block_it->size_ += size;
    AppendPartsTail(&block_it->parts_, new_parts);
    free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                         block_it);
  } else {
    // do not merge
    all_blocks_.emplace_back(ptr, size, true, std::move(new_parts));
    auto block_it = all_blocks_.end();
    block_it--;
    free_blocks_.emplace(std::make_pair(size, ptr), block_it);
  }
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocFromFreeBlocks(
    size_t size) {
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  if (iter != free_blocks_.end()) {
    std::list<Block>::iterator block_it = iter->second;
    free_blocks_.erase(iter);
    if (NeedSplit(block_it->size_, alignment_, size)) {
      void *remaining_ptr = reinterpret_cast<uint8_t *>(block_it->ptr_) + size;
      size_t remaining_size = block_it->size_ - size;
      VLOG(10) << "[AllocFromFreeBlocks] Block size: " << block_it->size_
               << ", Request size: " << size
               << ", Remaining: " << remaining_size
               << ", Original parts count: " << block_it->parts_.size();
      std::vector<BlockPart> alloc_parts = SlicePartsForRange(
          block_it->parts_, /*pick_off=*/0, /*pick_len=*/size);
      std::vector<BlockPart> remaining_parts = SlicePartsForRange(
          block_it->parts_, /*pick_off=*/size, /*pick_len=*/remaining_size);
      VLOG(10) << "[AllocFromFreeBlocks] Alloc parts count: "
               << alloc_parts.size()
               << ", Remaining parts count: " << remaining_parts.size();
      block_it->size_ = size;
      block_it->is_free_ = false;
      block_it->parts_.swap(alloc_parts);  // 设置分配块的 parts

      // 创建剩余空闲块，并设置其 parts
      auto remaining_free_block = all_blocks_.insert(
          std::next(block_it),
          Block(remaining_ptr,
                remaining_size,
                true,
                std::move(remaining_parts)));  // 设置剩余块的 parts
      free_blocks_.emplace(std::make_pair(remaining_size, remaining_ptr),
                           remaining_free_block);
    } else {
      block_it->is_free_ = false;
    }
    return new BlockAllocation(block_it, place_);
  }

  return nullptr;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
