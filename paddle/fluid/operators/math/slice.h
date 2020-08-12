/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <utility>
#include <vector>
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

using framework::To32BitIndex;

template <typename DeviceContext, typename T, size_t D>
void SliceFunction(const framework::ExecutionContext& context,
                   const std::vector<int>& offsets_vec,
                   const std::vector<int>& extents_vec,
                   const framework::Tensor& src, framework::Tensor* out) {
  auto src_tensor = framework::EigenTensor<T, D>::From(src);
  auto out_tensor = framework::EigenTensor<T, D>::From(*out);

  Eigen::array<int, D> offsets;
  Eigen::array<int, D> extents;

  for (int i = 0; i < D; ++i) {
    offsets[i] = offsets_vec[i];
    extents[i] = extents_vec[i];
  }

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  To32BitIndex(out_tensor).device(place) =
      To32BitIndex(src_tensor).slice(offsets, extents);
}

template <typename DeviceContext, typename T>
void SliceFunctor(int rank, const framework::ExecutionContext& context,
                  const std::vector<int>& offsets,
                  const std::vector<int>& extents, const framework::Tensor& src,
                  framework::Tensor* out) {
  switch (rank) {
    case 1:
      SliceFunction<DeviceContext, T, 1>(context, offsets, extents, src, out);
      break;
    case 2:
      SliceFunction<DeviceContext, T, 2>(context, offsets, extents, src, out);
      break;
    case 3:
      SliceFunction<DeviceContext, T, 3>(context, offsets, extents, src, out);
      break;
    case 4:
      SliceFunction<DeviceContext, T, 4>(context, offsets, extents, src, out);
      break;
    case 5:
      SliceFunction<DeviceContext, T, 5>(context, offsets, extents, src, out);
      break;
    case 6:
      SliceFunction<DeviceContext, T, 6>(context, offsets, extents, src, out);
      break;
    default:
      PADDLE_THROW(
          "PadOp only support tensors with no more than 6 dimensions.");
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
