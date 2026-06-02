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

#include <cstdint>
#include <memory>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

#if defined(PADDLE_WITH_CUDA)
using VMMDevicePtr = CUdeviceptr;
using VMMAllocHandle = CUmemGenericAllocationHandle;
#else
using VMMDevicePtr = uintptr_t;
using VMMAllocHandle = uint64_t;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
struct CUDAEventGuard {
  gpuEvent_t event{nullptr};

  explicit CUDAEventGuard(gpuEvent_t e) : event(e) {}
  ~CUDAEventGuard() {
    if (event != nullptr) {
#ifdef PADDLE_WITH_CUDA
      cudaEventDestroy(event);
#else
      hipEventDestroy(event);
#endif
    }
  }

  CUDAEventGuard(const CUDAEventGuard&) = delete;
  CUDAEventGuard& operator=(const CUDAEventGuard&) = delete;
};
#endif

// V2 keeps the bottom-layer shared types independent from the best-fit layer
// so that CUDAVirtualMemAllocatorV2 can be reviewed and compiled separately.
enum class PoolType : uint8_t {
  kSmall = 0,
  kLarge = 1,
};

// Fixed-size handle metadata returned by the bottom VMM provider. Upper layers
// may later reference these handles from block-level views, remap metadata, or
// IPC export state.
struct VMMHandleMeta {
  VMMHandleMeta() = default;
  VMMHandleMeta(VMMDevicePtr base,
                size_t size,
                VMMAllocHandle handle,
                int device)
      : base(base), size(size), handle(handle), device(device) {}

  VMMDevicePtr Base() const { return base; }
  size_t Size() const { return size; }
  VMMAllocHandle AllocationHandle() const { return handle; }
  int Device() const { return device; }
  bool IsOwnedByRemapDestination() const { return owned_by_remap_destination; }
  void MarkOwnedByRemapDestination() { owned_by_remap_destination = true; }
  void RestoreOriginalOwnership() { owned_by_remap_destination = false; }

  VMMDevicePtr base;
  size_t size;
  VMMAllocHandle handle;
  int device;
  bool owned_by_remap_destination{false};
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is only used to bootstrap upper-layer block state or
// answer allocation-level IPC/export queries.
using HandleLayout = std::vector<std::shared_ptr<VMMHandleMeta>>;

struct IpcBlockPartDescriptor {
  VMMDevicePtr handle_base;
  size_t handle_size;
  VMMAllocHandle handle;
  int device;
  size_t handle_rel_off;
  size_t len;
};

// A logical slice of one fixed-size VMM handle. This is the block-level view
// owned by VMMAutoGrowthBestFitAllocatorV2 and is updated by split / merge /
// remap after the initial HandleLayout has been consumed. Future IPC export
// still exports whole handles at the driver layer; BlockPartV2 carries the
// slice metadata needed to rebuild the logical tensor view on import.
struct BlockPartV2 {
  std::shared_ptr<VMMHandleMeta> handle;
  size_t handle_rel_off;
  size_t len;

  // Return a sub-slice of this part in handle-relative coordinates.
  BlockPartV2 Slice(size_t offset, size_t slice_len) const {
    return {handle, handle_rel_off + offset, slice_len};
  }

  // Fold an adjacent slice from the same handle into this part.
  bool TryExtend(const BlockPartV2& next) {
    if (handle.get() != next.handle.get()) {
      return false;
    }
    if (handle_rel_off + len != next.handle_rel_off) {
      return false;
    }
    len += next.len;
    return true;
  }
};

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kGap = 2,
};

struct BlockV2 {
  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kGap};
  std::vector<BlockPartV2> parts_;
  PoolType pool_type_{PoolType::kLarge};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t owning_stream_{nullptr};
  gpuStream_t last_use_stream_{nullptr};
  gpuEvent_t remap_safe_event_{nullptr};
#endif
  bool ipc_exported_{false};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
