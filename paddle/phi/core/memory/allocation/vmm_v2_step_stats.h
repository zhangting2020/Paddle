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
                      uint64_t block_count,
                      uint64_t free_blocks,
                      uint64_t unmapped_free_blocks,
                      uint64_t tail_offset);

void RecordVMMV2Free(int device_id,
                     PoolType pool_type,
                     uint64_t allocation_size,
                     uint64_t elapsed_us,
                     uint64_t block_count,
                     uint64_t free_blocks,
                     uint64_t unmapped_free_blocks);

void RecordStreamSafeProcess(int device_id,
                             uint64_t scanned,
                             uint64_t released,
                             uint64_t blocked,
                             uint64_t remaining,
                             uint64_t elapsed_us);

std::unordered_map<std::string, uint64_t> SnapshotAndResetVMMV2StepStats(
    int device_id);

}  // namespace paddle::memory::allocation
