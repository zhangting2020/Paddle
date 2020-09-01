// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/matmul_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

__global__ void FillData(const int N, const half2* in_data, const int in_height,
                         const int in_width, const int out_height,
                         const int out_width, half2* out_data) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = index; i < N / 2; i += blockDim.x * gridDim.x) {
    int nc = i / in_width;
    const int in_w = i % in_width;
    const int in_h = nc % in_height;
    nc /= in_height;
    out_data[(nc * out_height + in_h) * out_width + in_w] = in_data[i];
  }
  // in only one thread, process final element (if there is one)
  if (index == N / 2 && N % 2 == 1) {
    int nc = N / in_width;
    const int in_w = N % in_width;
    const int in_h = nc % in_height;
    nc /= in_height;
    out_data[(nc * out_height + in_h) * out_width + in_w] = in_data[N - 1];
  }
}

__global__ void Slice(const int N, const half2* in_data, const int in_height,
                      const int in_width, const int out_height,
                      const int out_width, half2* out_data) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = index; i < N / 2; i += blockDim.x * gridDim.x) {
    int nc = i / out_width;
    const int out_w = i % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    out_data[i] = in_data[(nc * in_height + out_h) * in_width + out_w];
  }
  // in only one thread, process final element (if there is one)
  if (index == N / 2 && N % 2 == 1) {
    int nc = N / out_width;
    const int out_w = N % out_width;
    const int out_h = nc % out_height;
    nc /= out_height;
    out_data[N - 1] = in_data[(nc * in_height + out_h) * in_width + out_w];
  }
}

template <typename DeviceContext, typename T>
void PadFunction(const framework::ExecutionContext& context,
                 const framework::Tensor& in, framework::Tensor* out) {
  const int in_dim_size = in.dims().size();
  const int out_dim_size = out->dims().size();
  const T* in_data = in.data<T>();
  T* out_data = out->data<T>();
  const int in_height = in.dims()[in_dim_size - 2];
  const int in_width = in.dims()[in_dim_size - 1];
  const int out_height = out->dims()[out_dim_size - 2];
  const int out_width = out->dims()[out_dim_size - 1];
  cudaMemset(out_data, 0, out->numel() * sizeof(T));
  auto stream = context.cuda_device_context().stream();
  int block = PADDLE_CUDA_NUM_THREADS;
  const int in_size = in.numel();
  int grid = (in_size + block - 1) / block;
  const half2* in_data2 = reinterpret_cast<const half2*>(in_data);
  half2* out_data2 = reinterpret_cast<half2*>(out_data);
  FillData<<<grid, block, 0, stream>>>(in_size, in_data2, in_height, in_width,
                                       out_height, out_width, out_data2);
}

template <typename DeviceContext, typename T>
void SliceFunction(const framework::ExecutionContext& context,
                   const framework::Tensor& in, framework::Tensor* out) {
  const int in_dim_size = in.dims().size();
  const int out_dim_size = out->dims().size();
  const T* in_data = in.data<T>();
  T* out_data = out->data<T>();
  const int in_height = in.dims()[in_dim_size - 2];
  const int in_width = in.dims()[in_dim_size - 1];
  const int out_height = out->dims()[out_dim_size - 2];
  const int out_width = out->dims()[out_dim_size - 1];
  auto stream = context.cuda_device_context().stream();
  int block = PADDLE_CUDA_NUM_THREADS;
  const int out_size = out->numel();
  int grid = (out_size / 2 + block - 1) / block;
  const half2* in_data2 = reinterpret_cast<const half2*>(in_data);
  half2* out_data2 = reinterpret_cast<half2*>(out_data);
  Slice<<<grid, block, 0, stream>>>(out_size, in_data2, in_height, in_width,
                                    out_height, out_width, out_data2);
}

template <typename DeviceContext, typename T>
class MatMulFP16Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& x = GET_DATA_SAFELY(context.Input<framework::Tensor>("X"), "Input",
                              "X", "MatMul");
    auto& y = GET_DATA_SAFELY(context.Input<framework::Tensor>("Y"), "Input",
                              "Y", "MatMul");
    auto* out = context.Output<framework::Tensor>("Out");

    bool pad_x_row = false;
    bool pad_x_col = false;
    bool pad_y_row = false;
    bool pad_y_col = false;
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");
    IsPaddingDims(x.dims(), &pad_x_row, &pad_x_col, transpose_x, true);
    IsPaddingDims(y.dims(), &pad_y_row, &pad_y_col, transpose_y, false);
    framework::Tensor pad_x;
    framework::Tensor pad_y;
    framework::Tensor pad_out;
    framework::DDim pad_out_dim =
        framework::make_ddim(framework::vectorize(out->dims()));
    if (pad_x_row || pad_x_col) {
      framework::DDim pad_x_dim;
      GetNewDims(x.dims(), pad_x_row, pad_x_col, &pad_x_dim);
      pad_x.mutable_data<T>(pad_x_dim, context.GetPlace());
      PadFunction<DeviceContext, T>(context, x, &pad_x);
    } else {
      pad_x = x;
    }
    if (pad_y_row || pad_y_col) {
      framework::DDim pad_y_dim;
      GetNewDims(y.dims(), pad_y_row, pad_y_col, &pad_y_dim);
      pad_y.mutable_data<T>(pad_y_dim, context.GetPlace());
      PadFunction<DeviceContext, T>(context, y, &pad_y);
      // if padding dimension is N, the out dims will be changed.
      if (pad_y_col && !transpose_y) {
        pad_out_dim[out->dims().size() - 1] =
            pad_y.dims()[pad_y.dims().size() - 1];
        VLOG(3) << "pad_out: " << pad_out_dim;
      } else if (pad_y_row && transpose_y) {
        pad_out_dim[out->dims().size() - 1] =
            pad_y.dims()[pad_y.dims().size() - 2];
        VLOG(3) << "pad_out: " << pad_out_dim;
      } else {
        pad_out = *out;
      }
    } else {
      pad_y = y;
      pad_out = *out;
    }
    pad_out.mutable_data<T>(pad_out_dim, context.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a =
        math::CreateMatrixDescriptor(RowMatrixFromVector(pad_x.dims()), 0,
                                     context.Attr<bool>("transpose_X"));
    auto mat_dim_b =
        math::CreateMatrixDescriptor(ColumnMatrixFromVector(pad_y.dims()), 0,
                                     context.Attr<bool>("transpose_Y"));
    auto scale = static_cast<T>(context.Attr<float>("alpha"));

    int head_number = 1;
    const auto& x_dims = pad_x.dims();
    const auto& y_dims = pad_y.dims();
    if (head_number <= 1 && x_dims.size() == 3 && y_dims.size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!context.Attr<bool>("transpose_X")) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
    blas.MatMul(pad_x, mat_dim_a, pad_y, mat_dim_b, scale, &pad_out, T(0));
    // slice output
    if ((pad_y_col && !transpose_y) || (pad_y_row && transpose_y)) {
      std::vector<int> offsets(out->dims().size(), 0);
      std::vector<int> extents;
      for (int i = 0; i < out->dims().size(); ++i) {
        extents.push_back(out->dims()[i]);
      }
      out->mutable_data<T>(context.GetPlace());
      // math::SliceFunctor<DeviceContext, T>(out->dims().size(), context,
      // offsets,
      //                                      extents, pad_out, out);

      SliceFunction<DeviceContext, T>(context, pad_out, out);
    } else {
      *out = pad_out;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    matmul, ops::MatMulKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MatMulKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MatMulFP16Kernel<paddle::platform::CUDADeviceContext,
                          paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext,
                          paddle::platform::float16>);
