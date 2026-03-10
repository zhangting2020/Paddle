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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

struct ChunkV2;

enum class PoolType : uint8_t {
  kStable = 0,
  kLongLived = 1,
  kTransient = 2,
  kOversized = 3,
};

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kGap = 2,
};

struct PerPoolStats {
  std::atomic<size_t> alloc_count{0};
  std::atomic<size_t> alloc_bytes{0};
  std::atomic<size_t> free_count{0};
  std::atomic<size_t> remap_bytes{0};
  std::atomic<size_t> offload_candidate_bytes{0};

  std::atomic<size_t> live_bytes{0};
  std::atomic<size_t> free_bytes{0};
  std::atomic<size_t> gap_bytes{0};
};

struct BlockV2 {
  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kGap};
  ChunkV2* chunk_{nullptr};
  std::vector<BlockPart> parts_;
  PoolType pool_type_{PoolType::kTransient};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t owning_stream_{nullptr};
  gpuStream_t last_use_stream_{nullptr};
  gpuEvent_t remap_safe_event_{nullptr};
#endif
  bool ipc_exported_{false};
};

using BlockList = std::list<BlockV2>;
using BlockListIt = BlockList::iterator;
using PtrBlockMap = std::unordered_map<void*, BlockListIt>;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
