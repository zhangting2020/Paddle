/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/solve_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/expand_as_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/lu_kernel.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/kernels/squeeze_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

namespace phi {

using Tensor = DenseTensor;

// check the input other is vector_case or not
static inline bool is_vector_rhs(const DenseTensor& input,
                                 const DenseTensor& other) {
  auto x_dim = input.dims();
  auto y_dim = other.dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();
  std::vector<int64_t> x_dims_vec = common::vectorize(x_dim);
  std::vector<int64_t> y_dims_vec = common::vectorize(y_dim);

  std::vector<int64_t>::const_iterator f = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l = x_dims_vec.end() - 1;
  std::vector<int64_t> x_dims_vec_cut(f, l);  // input.shape[:-1]

  std::vector<int64_t> expected_batched_rhs_shape(x_dims_vec_cut);
  bool vector_case =
      y_dim_size == 1 || (x_dim_size - 1 == y_dim_size &&
                          y_dims_vec == (expected_batched_rhs_shape));

  return vector_case;
}

// Prepared for the broadcast operation
static std::vector<int64_t> get_broadcast_batch_portion(
    std::vector<int64_t> x, std::vector<int64_t> y) {
  size_t size_x = x.size();
  size_t size_y = y.size();
  size_t size = std::max(size_x, size_y);
  std::vector<int64_t> batchPortion(size);
  ptrdiff_t i = (ptrdiff_t)size - 1;
  for (; i >= 0; --i) {
    ptrdiff_t offset = size - i - 1;
    ptrdiff_t dim_x = size_x - offset - 1;
    ptrdiff_t dim_y = size_y - offset - 1;
    int64_t x_size = (dim_x >= 0) ? x[dim_x] : 1;
    int64_t y_size = (dim_y >= 0) ? y[dim_y] : 1;
    PADDLE_ENFORCE_EQ(
        (x_size == y_size || x_size == 1 || y_size == 1),
        true,
        common::errors::PreconditionNotMet(
            "The size of tensor x (%d) must match the size of tensor y "
            "(%d) at non-singleton dimension %d.",
            x_size,
            y_size,
            i));

    batchPortion[i] = x_size != 1 ? x_size : y_size;
  }
  return batchPortion;
}

// broadcast the batch dimensions of tensor x and tensor y.
static inline std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_broadcast_dims(const Tensor& x, const Tensor& y) {
  std::vector<int64_t> x_dims_vec = common::vectorize(x.dims());
  std::vector<int64_t> y_dims_vec = common::vectorize(y.dims());
  std::vector<int64_t>::const_iterator f1 = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l1 = x_dims_vec.end() - 2;
  std::vector<int64_t> x_dims_vec_cut(f1, l1);

  std::vector<int64_t>::const_iterator f2 = y_dims_vec.begin();
  std::vector<int64_t>::const_iterator l2 = y_dims_vec.end() - 2;
  std::vector<int64_t> y_dims_vec_cut(f2, l2);

  std::vector<int64_t> expand_batch_portion =
      get_broadcast_batch_portion(x_dims_vec_cut, y_dims_vec_cut);
  std::vector<int64_t> x_expand_size({expand_batch_portion});
  x_expand_size.insert(x_expand_size.end(),
                       {x_dims_vec[static_cast<int>(x_dims_vec.size()) - 2],
                        x_dims_vec[static_cast<int>(x_dims_vec.size()) - 1]});
  std::vector<int64_t> y_expand_size({expand_batch_portion});
  y_expand_size.insert(y_expand_size.end(),
                       {y_dims_vec[static_cast<int>(y_dims_vec.size()) - 2],
                        y_dims_vec[static_cast<int>(y_dims_vec.size()) - 1]});

  return std::make_tuple(x_expand_size, y_expand_size);
}

template <typename Context, typename T>
static void linalg_solve(const GPUContext& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool left,
                         DenseTensor* out) {
  // Keep the same front-end shape handling as the generic path
  bool is_vector = is_vector_rhs(x, y);
  if (!left) {
    PADDLE_ENFORCE_EQ(is_vector,
                      false,
                      common::errors::InvalidArgument(
                          "Right solve expects matrix RHS, but got vector."));
  }

  // Shape checks to align with previous behavior and PyTorch
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();
  if (!left && y_dim_size < 2) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Right solve expects Y to be at least 2D."));
  }
  if (!is_vector) {
    PADDLE_ENFORCE_EQ(
        left ? x_dim[x_dim_size - 1] : x_dim[x_dim_size - 1],
        left ? y_dim[y_dim_size - 2] : y_dim[y_dim_size - 1],
        common::errors::InvalidArgument(
            "Incompatible shapes for solve. X shape = [%s], Y shape = [%s].",
            x_dim,
            y_dim));
  }
  Tensor x_in = x;
  Tensor y_in = y;
  Tensor x_tmp, y_tmp;
  if (!left) {
    const auto& new_dims_vec_x = phi::funcs::getNewDimsVec(x.dims());
    x_tmp.Resize(common::make_ddim(new_dims_vec_x));
    dev_ctx.template Alloc<T>(&x_tmp);
    phi::funcs::TransposeNormal<GPUContext, T> trans;
    std::vector<int> new_axis_x = phi::funcs::getNewAxis(x.dims().size());
    trans(dev_ctx, x, &x_tmp, new_axis_x);
    x_in = x_tmp;

    const auto& new_dims_vec_y = phi::funcs::getNewDimsVec(y.dims());
    y_tmp.Resize(common::make_ddim(new_dims_vec_y));
    dev_ctx.template Alloc<T>(&y_tmp);
    std::vector<int> new_axis_y = phi::funcs::getNewAxis(y.dims().size());
    trans(dev_ctx, y, &y_tmp, new_axis_y);
    y_in = y_tmp;
  }

  Tensor tmp_y;
  if (is_vector) {
    dev_ctx.Alloc(&tmp_y, y_in.dtype());
    phi::Unsqueeze<T, GPUContext>(dev_ctx, y_in, {-1}, &tmp_y, nullptr);
  } else {
    tmp_y.Resize(y_in.dims());
    dev_ctx.Alloc(&tmp_y, y_in.dtype());
    phi::Copy(dev_ctx, y_in, dev_ctx.GetPlace(), false, &tmp_y);
  }

  Tensor tmp_x;
  tmp_x.Resize(x_in.dims());
  dev_ctx.Alloc(&tmp_x, x_in.dtype());
  phi::Copy(dev_ctx, x_in, dev_ctx.GetPlace(), false, &tmp_x);

  std::vector<int64_t> x_broadcast_dims;
  std::vector<int64_t> y_broadcast_dims;
  std::tie(x_broadcast_dims, y_broadcast_dims) =
      get_broadcast_dims(tmp_x, tmp_y);

  Tensor tmp_x_bc;
  phi::ExpandAsKernel<T, GPUContext>(
      dev_ctx, tmp_x, nullptr, x_broadcast_dims, &tmp_x_bc);

  Tensor tmp_y_bc;
  phi::ExpandAsKernel<T, GPUContext>(
      dev_ctx, tmp_y, nullptr, y_broadcast_dims, &tmp_y_bc);

  // Use cusolver-based LU factorization + solve to align with PyTorch path
  Tensor lu;
  Tensor pivots;
  Tensor infos;
  lu.Resize(tmp_x_bc.dims());
  LUKernel<T, GPUContext>(
      dev_ctx, tmp_x_bc, /*pivot=*/true, &lu, &pivots, &infos);

  // check cusolver info to align with torch error reporting
  std::vector<int> info_vec;
  phi::TensorToVector(infos, dev_ctx, &info_vec);
  for (size_t i = 0; i < info_vec.size(); ++i) {
    PADDLE_ENFORCE_EQ(info_vec[i],
                      0,
                      common::errors::PreconditionNotMet(
                          "LU factorization failed at batch %zu: info = %d.",
                          i,
                          info_vec[i]));
  }

  Tensor solved;
  solved.Resize(tmp_y_bc.dims());
  std::string trans_flag = left ? "N" : "T";
  LuSolveKernel<T, GPUContext>(
      dev_ctx, tmp_y_bc, lu, pivots, trans_flag, &solved);

  if (is_vector) {
    Tensor squeezed;
    squeezed.Resize(solved.dims());
    phi::Squeeze<T, GPUContext>(dev_ctx, solved, {-1}, &squeezed);
    *out = std::move(squeezed);
  } else {
    *out = std::move(solved);
  }

  if (!left) {
    Tensor out_tmp = *out;
    const auto& new_dims_vec = phi::funcs::getNewDimsVec(out->dims());
    out->Resize(common::make_ddim(new_dims_vec));
    dev_ctx.template Alloc<T>(out);
    phi::funcs::TransposeNormal<GPUContext, T> trans;
    std::vector<int> new_axis = phi::funcs::getNewAxis(out_tmp.dims().size());
    trans(dev_ctx, out_tmp, out, new_axis);
  }
}

template <typename T, typename Context>
void SolveGPUKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    bool left,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    std::vector<int> out_dims;
    if (y_dims.size() == 1) {
      out_dims =
          std::vector<int>(x_dims.Get(), x_dims.Get() + x_dims.size() - 2);
      out_dims.push_back(y_dims[y_dims.size() - 1]);
    } else {
      // broadcast
      std::vector<int> x_shape(x_dims.Get(), x_dims.Get() + x_dims.size() - 2);
      std::vector<int> y_shape(y_dims.Get(), y_dims.Get() + y_dims.size() - 2);
      auto x_it = x_shape.rbegin();
      auto y_it = y_shape.rbegin();
      while (x_it != x_shape.rend() || y_it != y_shape.rend()) {
        int x_dim = (x_it != x_shape.rend()) ? *x_it : 1;
        int y_dim = (y_it != y_shape.rend()) ? *y_it : 1;
        if (x_dim == 0 || y_dim == 0) {
          out_dims.push_back(0);
        } else {
          out_dims.push_back(std::max(x_dim, y_dim));
        }
        if (x_it != x_shape.rend()) ++x_it;
        if (y_it != y_shape.rend()) ++y_it;
      }
      std::reverse(out_dims.begin(), out_dims.end());
      out_dims.insert(out_dims.end(),
                      y_dims.Get() + y_dims.size() - 2,
                      y_dims.Get() + y_dims.size());
    }
    out->Resize(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(out);
    return;
  }
  linalg_solve<Context, T>(dev_ctx, x, y, left, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(solve, GPU, ALL_LAYOUT, phi::SolveGPUKernel, float, double) {
}
