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
#include <string>
#include <unordered_map>

#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle::memory::allocation {

bool VMMV2StepStatsEnabled();

void RecordVMMV2Alloc(int device_id,
                      PoolType pool_type,
                      const char* path,
                      uint64_t request_size,
                      uint64_t elapsed_us,
                      uint64_t lock_wait_us,
                      uint64_t block_count,
                      uint64_t free_blocks,
                      uint64_t unmapped_free_blocks,
                      uint64_t tail_offset);

void RecordVMMV2Free(int device_id,
                     PoolType pool_type,
                     uint64_t allocation_size,
                     uint64_t elapsed_us,
                     uint64_t lock_wait_us,
                     uint64_t block_count,
                     uint64_t free_blocks,
                     uint64_t unmapped_free_blocks);

struct VMMV2MappedFreeDetailStats {
  uint64_t lower_bound_us{0};
  uint64_t stale_erase_count{0};
  uint64_t stale_erase_us{0};
  uint64_t erase_free_us{0};
  uint64_t split_count{0};
  uint64_t split_us{0};
  uint64_t insert_block_us{0};
  uint64_t insert_free_us{0};
  uint64_t mark_active_us{0};
  uint64_t wrapper_new_us{0};
};

struct VMMV2FreeDetailStats {
  uint64_t remap_safety_count{0};
  uint64_t remap_safety_us{0};
  uint64_t mark_free_us{0};
  uint64_t try_merge_us{0};
  uint64_t merge_prev_count{0};
  uint64_t merge_next_count{0};
  uint64_t erase_free_us{0};
  uint64_t absorb_us{0};
  uint64_t erase_block_us{0};
  uint64_t insert_free_us{0};
};

bool VMMV2DetailStatsEnabled();

void RecordVMMV2MappedFreeDetail(int device_id,
                                 const VMMV2MappedFreeDetailStats& detail);

void RecordVMMV2FreeDetail(int device_id, const VMMV2FreeDetailStats& detail);

void RecordStreamSafeProcess(int device_id,
                             uint64_t scanned,
                             uint64_t released,
                             uint64_t blocked,
                             uint64_t remaining,
                             uint64_t elapsed_us);

std::unordered_map<std::string, uint64_t> SnapshotAndResetVMMV2StepStats(
    int device_id);

}  // namespace paddle::memory::allocation
