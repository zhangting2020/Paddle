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

#pragma once

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>

#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#include <mutex>  // NOLINT

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif

#if CUDA_VERSION >= 10020

namespace paddle {
namespace memory {
namespace allocation {

// Allocate memory using NVIDIA's virtual memory management technology

struct VmmShareInfo {
  int os_fd{-1};     // Linux: file descriptor
  size_t size{0};    // total mapped length (page aligned)
  size_t offset{0};  // byte offset inside the handle (page aligned)
  int device{-1};    // exporter device
};

struct ImportedVmmMulti {
  CUdeviceptr base{0};
  size_t total{0};
  std::vector<CUmemGenericAllocationHandle> hs;
  ~ImportedVmmMulti() {
    if (base && total) {
      phi::dynload::cuMemUnmap(base, total);
    }
    for (auto h : hs) {
      if (h) phi::dynload::cuMemRelease(h);
    }
    if (base && total) {
      phi::dynload::cuMemAddressFree(base, total);
    }
  }
};

class VmmImportedAllocation : public phi::Allocation {
 public:
  VmmImportedAllocation(void* p,
                        size_t n,
                        phi::Place plc,
                        std::shared_ptr<ImportedVmmMulti> keep)
      : Allocation(p, n, plc), keep_(std::move(keep)) {}

 private:
  std::shared_ptr<ImportedVmmMulti> keep_;
};

#pragma pack(push, 1)
struct VmmIpcHeader {
  uint8_t version;       // 固定 1
  uint8_t type;          // 固定 2: vmm-ipc
  uint16_t flags;        // bit0: 使用 pidfd 路线
  uint32_t pid;          // 导出进程 pid
  uint32_t num_entries;  // N（=1 即单句柄）
  uint64_t total_size;   // VA 总长度（= Block.size_）
};

struct VmmIpcEntry {
  uint8_t handle_type;  // 1: POSIX_FD（cuMemExportToShareableHandle FD）
  uint8_t reserved[7];
  uint64_t rel_offset;  // 该 part 映射到目标连续 VA 的相对偏移
  uint64_t seg_len;     // 该 part 的长度（需为 VMM granularity 倍数）
  uint64_t chunk_rel_off;  // ★ 该 part 在“底层 allocation handle”内的偏移
};
#pragma pack(pop)

// 可选：编译期校验，防止意外改动
static_assert(sizeof(VmmIpcHeader) == 20, "VmmIpcHeader size changed");
static_assert(sizeof(VmmIpcEntry) == 32, "VmmIpcEntry size changed");

class CUDAVirtualMemAllocator : public Allocator {
 public:
  explicit CUDAVirtualMemAllocator(const phi::GPUPlace& place);
  bool IsAllocThreadSafe() const override;
  static bool ExportShareHandleFromVA(CUdeviceptr va,
                                      CUdeviceptr base_ptr,
                                      CUmemGenericAllocationHandle handle,
                                      size_t size,
                                      int device_id,
                                      VmmShareInfo* out);
  size_t granularity() const { return granularity_; }

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

  bool ExportShareHandleFromVA(CUdeviceptr va, VmmShareInfo* out);

 private:
  phi::GPUPlace place_;
  std::once_flag once_flag_;

  CUdeviceptr virtual_mem_base_;
  size_t virtual_mem_size_;
  size_t virtual_mem_alloced_offset_;
  size_t granularity_;

  CUmemAllocationProp prop_;
  std::vector<CUmemAccessDesc> access_desc_;

  std::map<CUdeviceptr, std::pair<CUmemGenericAllocationHandle, size_t>>
      virtual_2_physical_map_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
