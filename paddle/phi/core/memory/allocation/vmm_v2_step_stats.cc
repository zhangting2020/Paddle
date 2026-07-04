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

#include "paddle/phi/core/memory/allocation/vmm_v2_step_stats.h"

#include <algorithm>
#include <array>

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

PHI_DEFINE_EXPORTED_bool(
    vmm_v2_step_stats,
    false,
    "Collect low-overhead per-step VMM V2 allocator diagnostics. "
    "Use core._vmm_v2_step_stats_snapshot_and_reset(device_id) to fetch and "
    "reset counters.");

PHI_DEFINE_EXPORTED_bool(
    vmm_v2_detail_stats,
    false,
    "Collect detailed VMM V2 allocator sub-stage timings. This adds extra "
    "timers on allocator hot paths and is for short diagnostic runs only.");

PHI_DEFINE_EXPORTED_bool(
    vmm_v2_skip_remap_safety_hot_path,
    false,
    "Diagnostic-only VMM V2 mode: skip remap-safety state marking on normal "
    "allocator hot paths. This is unsafe when VMM remap/compaction may run "
    "and is only intended to isolate remap-safety metadata overhead.");

namespace paddle::memory::allocation {

namespace {

constexpr int kMaxDeviceCount = 16;

struct StepStats {
  uint64_t alloc_count{0};
  uint64_t alloc_total_us{0};
  uint64_t alloc_max_us{0};
  uint64_t alloc_lock_wait_total_us{0};
  uint64_t alloc_lock_wait_max_us{0};
  uint64_t alloc_request_bytes{0};
  uint64_t free_count{0};
  uint64_t free_total_us{0};
  uint64_t free_max_us{0};
  uint64_t free_lock_wait_total_us{0};
  uint64_t free_lock_wait_max_us{0};
  uint64_t free_bytes{0};
  uint64_t grow_count{0};
  uint64_t grow_total_us{0};
  uint64_t mapped_free_count{0};
  uint64_t mapped_free_total_us{0};
  uint64_t unmapped_free_count{0};
  uint64_t unmapped_free_total_us{0};
  uint64_t small_pool_alloc_count{0};
  uint64_t large_pool_alloc_count{0};
  uint64_t small_pool_free_count{0};
  uint64_t large_pool_free_count{0};
  uint64_t free_remap_safety_count{0};
  uint64_t free_remap_safety_us{0};
  uint64_t stream_process_count{0};
  uint64_t stream_process_total_us{0};
  uint64_t stream_process_max_us{0};
  uint64_t stream_scanned{0};
  uint64_t stream_released{0};
  uint64_t stream_blocked{0};
  uint64_t stream_remaining_max{0};
  uint64_t last_block_count{0};
  uint64_t last_free_blocks{0};
  uint64_t last_unmapped_free_blocks{0};
  uint64_t last_tail_offset{0};
};

std::array<StepStats, kMaxDeviceCount> g_step_stats;
std::array<SpinLock, kMaxDeviceCount> g_step_stats_locks;

bool IsValidDeviceId(int device_id) {
  return device_id >= 0 && device_id < kMaxDeviceCount;
}

void AddTiming(uint64_t elapsed_us, uint64_t* total_us, uint64_t* max_us) {
  *total_us += elapsed_us;
  *max_us = std::max(*max_us, elapsed_us);
}

}  // namespace

bool VMMV2StepStatsEnabled() { return FLAGS_vmm_v2_step_stats; }

bool VMMV2DetailStatsEnabled() {
  return FLAGS_vmm_v2_step_stats && FLAGS_vmm_v2_detail_stats;
}

void RecordVMMV2Alloc(int device_id,
                      PoolType pool_type,
                      const char* path,
                      uint64_t request_size,
                      uint64_t elapsed_us,
                      uint64_t lock_wait_us,
                      uint64_t block_count,
                      uint64_t free_blocks,
                      uint64_t unmapped_free_blocks,
                      uint64_t tail_offset) {
  if (!FLAGS_vmm_v2_step_stats || !IsValidDeviceId(device_id)) {
    return;
  }
  std::lock_guard<SpinLock> guard(g_step_stats_locks[device_id]);
  StepStats& stats = g_step_stats[device_id];
  ++stats.alloc_count;
  stats.alloc_request_bytes += request_size;
  AddTiming(elapsed_us, &stats.alloc_total_us, &stats.alloc_max_us);
  AddTiming(lock_wait_us,
            &stats.alloc_lock_wait_total_us,
            &stats.alloc_lock_wait_max_us);
  if (pool_type == PoolType::kSmall) {
    ++stats.small_pool_alloc_count;
  } else {
    ++stats.large_pool_alloc_count;
  }
  const std::string path_name(path);
  if (path_name == "grow") {
    ++stats.grow_count;
    stats.grow_total_us += elapsed_us;
  } else if (path_name == "mapped_free") {
    ++stats.mapped_free_count;
    stats.mapped_free_total_us += elapsed_us;
  } else if (path_name == "unmapped_free") {
    ++stats.unmapped_free_count;
    stats.unmapped_free_total_us += elapsed_us;
  }
  stats.last_block_count = block_count;
  stats.last_free_blocks = free_blocks;
  stats.last_unmapped_free_blocks = unmapped_free_blocks;
  stats.last_tail_offset = tail_offset;
}

void RecordVMMV2Free(int device_id,
                     PoolType pool_type,
                     uint64_t allocation_size,
                     uint64_t elapsed_us,
                     uint64_t lock_wait_us,
                     uint64_t block_count,
                     uint64_t free_blocks,
                     uint64_t unmapped_free_blocks) {
  if (!FLAGS_vmm_v2_step_stats || !IsValidDeviceId(device_id)) {
    return;
  }
  std::lock_guard<SpinLock> guard(g_step_stats_locks[device_id]);
  StepStats& stats = g_step_stats[device_id];
  ++stats.free_count;
  stats.free_bytes += allocation_size;
  AddTiming(elapsed_us, &stats.free_total_us, &stats.free_max_us);
  AddTiming(lock_wait_us,
            &stats.free_lock_wait_total_us,
            &stats.free_lock_wait_max_us);
  if (pool_type == PoolType::kSmall) {
    ++stats.small_pool_free_count;
  } else {
    ++stats.large_pool_free_count;
  }
  stats.last_block_count = block_count;
  stats.last_free_blocks = free_blocks;
  stats.last_unmapped_free_blocks = unmapped_free_blocks;
}

void RecordVMMV2FreeDetail(int device_id, const VMMV2FreeDetailStats& detail) {
  if (!VMMV2DetailStatsEnabled() || !IsValidDeviceId(device_id)) {
    return;
  }
  std::lock_guard<SpinLock> guard(g_step_stats_locks[device_id]);
  StepStats& stats = g_step_stats[device_id];
  stats.free_remap_safety_count += detail.remap_safety_count;
  stats.free_remap_safety_us += detail.remap_safety_us;
}

void RecordStreamSafeProcess(int device_id,
                             uint64_t scanned,
                             uint64_t released,
                             uint64_t blocked,
                             uint64_t remaining,
                             uint64_t elapsed_us) {
  if (!FLAGS_vmm_v2_step_stats || !IsValidDeviceId(device_id)) {
    return;
  }
  std::lock_guard<SpinLock> guard(g_step_stats_locks[device_id]);
  StepStats& stats = g_step_stats[device_id];
  ++stats.stream_process_count;
  AddTiming(
      elapsed_us, &stats.stream_process_total_us, &stats.stream_process_max_us);
  stats.stream_scanned += scanned;
  stats.stream_released += released;
  stats.stream_blocked += blocked;
  stats.stream_remaining_max = std::max(stats.stream_remaining_max, remaining);
}

std::unordered_map<std::string, uint64_t> SnapshotAndResetVMMV2StepStats(
    int device_id) {
  std::unordered_map<std::string, uint64_t> result;
  if (!IsValidDeviceId(device_id)) {
    return result;
  }
  StepStats snapshot;
  {
    std::lock_guard<SpinLock> guard(g_step_stats_locks[device_id]);
    snapshot = g_step_stats[device_id];
    g_step_stats[device_id] = StepStats{};
  }

  result["alloc_count"] = snapshot.alloc_count;
  result["alloc_total_us"] = snapshot.alloc_total_us;
  result["alloc_max_us"] = snapshot.alloc_max_us;
  result["alloc_lock_wait_total_us"] = snapshot.alloc_lock_wait_total_us;
  result["alloc_lock_wait_max_us"] = snapshot.alloc_lock_wait_max_us;
  result["alloc_request_bytes"] = snapshot.alloc_request_bytes;
  result["free_count"] = snapshot.free_count;
  result["free_total_us"] = snapshot.free_total_us;
  result["free_max_us"] = snapshot.free_max_us;
  result["free_lock_wait_total_us"] = snapshot.free_lock_wait_total_us;
  result["free_lock_wait_max_us"] = snapshot.free_lock_wait_max_us;
  result["free_bytes"] = snapshot.free_bytes;
  result["grow_count"] = snapshot.grow_count;
  result["grow_total_us"] = snapshot.grow_total_us;
  result["mapped_free_count"] = snapshot.mapped_free_count;
  result["mapped_free_total_us"] = snapshot.mapped_free_total_us;
  result["unmapped_free_count"] = snapshot.unmapped_free_count;
  result["unmapped_free_total_us"] = snapshot.unmapped_free_total_us;
  result["small_pool_alloc_count"] = snapshot.small_pool_alloc_count;
  result["large_pool_alloc_count"] = snapshot.large_pool_alloc_count;
  result["small_pool_free_count"] = snapshot.small_pool_free_count;
  result["large_pool_free_count"] = snapshot.large_pool_free_count;
  result["free_remap_safety_count"] = snapshot.free_remap_safety_count;
  result["free_remap_safety_us"] = snapshot.free_remap_safety_us;
  result["stream_process_count"] = snapshot.stream_process_count;
  result["stream_process_total_us"] = snapshot.stream_process_total_us;
  result["stream_process_max_us"] = snapshot.stream_process_max_us;
  result["stream_scanned"] = snapshot.stream_scanned;
  result["stream_released"] = snapshot.stream_released;
  result["stream_blocked"] = snapshot.stream_blocked;
  result["stream_remaining_max"] = snapshot.stream_remaining_max;
  result["last_block_count"] = snapshot.last_block_count;
  result["last_free_blocks"] = snapshot.last_free_blocks;
  result["last_unmapped_free_blocks"] = snapshot.last_unmapped_free_blocks;
  result["last_tail_offset"] = snapshot.last_tail_offset;
  return result;
}

}  // namespace paddle::memory::allocation
