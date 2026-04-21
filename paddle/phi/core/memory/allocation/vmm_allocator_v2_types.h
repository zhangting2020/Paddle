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

#include <memory>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
using VmmDevicePtr = CUdeviceptr;
using VmmAllocHandle = CUmemGenericAllocationHandle;
#else
using VmmDevicePtr = uintptr_t;
using VmmAllocHandle = uint64_t;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

// RAII wrapper around gpuEvent_t so that multiple blocks can share
// ownership of the same event via shared_ptr.  The event is only
// destroyed when the last reference is dropped, preventing the
// double-destroy SIGSEGV that occurred when raw gpuEvent_t pointers
// were shallow-copied across split blocks and then independently
// destroyed during merge.
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
struct CudaEventGuard {
  gpuEvent_t event{nullptr};

  explicit CudaEventGuard(gpuEvent_t e) : event(e) {}
  ~CudaEventGuard() {
    if (event != nullptr) {
#ifdef PADDLE_WITH_CUDA
      cudaEventDestroy(event);
#else
      hipEventDestroy(event);
#endif
    }
  }

  // Non-copyable — ownership is shared via shared_ptr.
  CudaEventGuard(const CudaEventGuard&) = delete;
  CudaEventGuard& operator=(const CudaEventGuard&) = delete;
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
struct VmmHandleMeta {
  VmmDevicePtr base;
  size_t size;
  VmmAllocHandle handle;
  int device;
  // Set to true by the compactor after it unmaps this handle from its
  // original VA and remaps it to a new (tail/gap) VA.  FreeImpl must
  // skip cuMemUnmap+cuMemRelease for remapped handles — their
  // lifetime is now managed by the block that received them.
  bool remapped{false};
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is only used to bootstrap upper-layer block state or
// answer allocation-level IPC/export queries.
using HandleLayout = std::vector<std::shared_ptr<VmmHandleMeta>>;

// A logical slice of one fixed-size VMM handle. This is the block-level view
// owned by VMMAutoGrowthBestFitAllocatorV2 and is updated by split / merge /
// remap after the initial HandleLayout has been consumed. Future IPC export
// still exports whole handles at the driver layer; BlockPartV2 carries the
// slice metadata needed to rebuild the logical tensor view on import.
struct BlockPartV2 {
  std::shared_ptr<VmmHandleMeta> handle;
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
  // Shared ownership: split blocks share the same event; the event is
  // only destroyed when all blocks drop their reference.
  std::shared_ptr<CudaEventGuard> remap_safe_event_;
#endif
  bool ipc_exported_{false};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
