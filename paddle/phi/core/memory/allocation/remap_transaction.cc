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

#include "paddle/phi/core/memory/allocation/remap_transaction.h"

#include <utility>

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace allocation {

void RemapTransaction::PrepareCandidates(const VaRanges& source_ranges,
                                         const VaRanges& target_ranges,
                                         size_t target_bytes) {
  candidates_ = vmm_allocator_->CollectBackingCompactCandidates(
      source_ranges, target_ranges, target_bytes);
}

bool RemapTransaction::ValidateSourcePages(const char* context) const {
  return vmm_allocator_->ValidateMappedBackingPages(candidates_.source_pages,
                                                    context);
}

bool RemapTransaction::ValidateTargetPages(const char* context) const {
  return vmm_allocator_->ValidateUnmappedBackingPages(candidates_.target_pages,
                                                      context);
}

RemapTransaction::CandidateValidation RemapTransaction::ValidateCandidates(
    const char* source_context, const char* target_context) const {
  CandidateValidation validation;
  validation.source_ok = ValidateSourcePages(source_context);
  validation.target_ok = ValidateTargetPages(target_context);
  return validation;
}

void RemapTransaction::SetSourceRollbackAction(std::function<void()> action) {
  source_rollback_action_ = std::move(action);
}

void RemapTransaction::MapHandlesToDestination(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  RecordDestinationRange(dst, handles.size());
  vmm_allocator_->MapHandlesToVA(dst, handles, metas);
}

void RemapTransaction::MapHandleRangeToDestination(
    VmmDevicePtr dst,
    const std::vector<VmmAllocHandle>& handles,
    size_t start,
    size_t count,
    const std::vector<std::shared_ptr<VmmHandleMeta>>* metas) {
  std::vector<VmmAllocHandle> handle_slice(handles.begin() + start,
                                           handles.begin() + start + count);
  if (metas == nullptr) {
    MapHandlesToDestination(dst, handle_slice, nullptr);
    return;
  }

  std::vector<std::shared_ptr<VmmHandleMeta>> meta_slice(metas->begin() + start,
                                                         metas->begin() + start +
                                                             count);
  MapHandlesToDestination(dst, handle_slice, &meta_slice);
}

void RemapTransaction::RecordDestinationRange(VmmDevicePtr dst,
                                              size_t handle_count) {
  pending_destination_ranges_.push_back({dst, handle_count});
}

void RemapTransaction::Commit() {
  ClearPendingDestinations();
  source_rollback_action_ = nullptr;
  completed_ = true;
}

void RemapTransaction::UnmapPartialDestination(VmmDevicePtr dst_base,
                                               size_t handle_count) {
  for (size_t i = 0; i < handle_count; ++i) {
    vmm_allocator_->TryUnmapHandle(dst_base + i * handle_size_, handle_size_);
  }
}

void RemapTransaction::Rollback() {
  if (completed_) {
    return;
  }
  RollbackPendingDestinations();
  if (source_rollback_action_) {
    source_rollback_action_();
  }
  source_rollback_action_ = nullptr;
  completed_ = true;
}

void RemapTransaction::RollbackPendingDestinations() {
  for (auto it = pending_destination_ranges_.rbegin();
       it != pending_destination_ranges_.rend();
       ++it) {
    VLOG(0) << "VMM V2 remap transaction: unmapping pending destination "
            << reinterpret_cast<void*>(it->dst)
            << " handles=" << it->handle_count;
    UnmapPartialDestination(it->dst, it->handle_count);
  }
  pending_destination_ranges_.clear();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
