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

#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include <type_traits>
#include "glog/logging.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#ifndef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/cusolver.h"
#endif

namespace phi {
namespace funcs {

#ifndef PADDLE_WITH_HIP
/**
 * Transform pivot array to permutation by swapping perm[i] and perm[pivot[i]]
 * from 0 to n-1, where pivot and perm have shape [batch_size, n].
 * Example:
 *    Input pivot = [[6, 7, 4, 5, 5, 7, 8, 8]]
 *    Output perm = [[5, 6, 3, 4, 2, 1, 7, 0]]
 */
__global__ void UnpackPivot(const int* __restrict__ pivot,
                            int* __restrict__ perm,
                            int64_t batch_size,
                            int64_t n) {
  constexpr int warp_size = 32;
  int warps_per_block = blockDim.x / warp_size;
  int warp_id = threadIdx.x / warp_size;
  int warp_offset = threadIdx.x % warp_size;
  int64_t offset = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  int64_t stride = static_cast<int64_t>(gridDim.x) * warps_per_block;

  for (; offset < batch_size; offset += stride) {
    // init perm[*, n] with 0...n-1
    for (int64_t i = warp_offset; i < n; i += warp_size) {
      perm[offset * n + i] = offset * n + i;
    }
    __syncwarp();

    // Since the swapping makes entirely discrete access, we only use the first
    // thread in each warp to avoid warp divergence.
    if (warp_offset > 0) continue;

    // Swap perm[i] and perm[pivot[i]] for i in 0...n-1
    for (int64_t i = offset * n; i < offset * n + n; ++i) {
      int64_t j = pivot[i] - 1 + offset * n;  // cublas use 1-index
      int tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
  }
}

/**
 * Eliminate the L and U in equation:
 *    (U^T @ L^T @ P) @ X = B  (the U^T @ L^T @ P is stored in A)
 * by solving the inversion of L^T and U^T respectively. The result is:
 *    P @ X = L^T^-1 @ U^T^-1 @ B
 * and is stored in B.
 */
template <typename Context, typename T>
void SolveLU(const phi::funcs::BlasT<Context, T>& blas,
             int m,
             int n,
             const T* A,
             T* B,
             int batch_size) {
  constexpr T alpha = 1.0;
  for (int64_t i = 0; i < batch_size; ++i) {
    // Before: U^T @ L^T @ P @ X = B
    blas.TRSM(CblasRight,
              CblasLower,
              CblasTrans,
              CblasNonUnit,
              m,
              n,
              alpha,
              A + i * n * n,
              n,
              B + i * m * n,
              n);
    // After: L^T @ P @ X = U^T^-1 @ B
    blas.TRSM(CblasRight,
              CblasUpper,
              CblasTrans,
              CblasUnit,
              m,
              n,
              alpha,
              A + i * n * n,
              n,
              B + i * m * n,
              n);
    // After: P @ X = L^T^-1 @ U^T^-1 @ B
  }
}

// Batched version of SolveLU.
template <typename Context, typename T>
void BatchedSolveLU(const phi::funcs::BlasT<Context, T>& blas,
                    int m,
                    int n,
                    const T** A,
                    T** B,
                    int batch_size) {
  constexpr T alpha = 1.0;
  blas.BatchedTRSM(CblasRight,
                   CblasLower,
                   CblasTrans,
                   CblasNonUnit,
                   m,
                   n,
                   alpha,
                   A,
                   n,
                   B,
                   n,
                   batch_size);
  blas.BatchedTRSM(CblasRight,
                   CblasUpper,
                   CblasTrans,
                   CblasUnit,
                   m,
                   n,
                   alpha,
                   A,
                   n,
                   B,
                   n,
                   batch_size);
}
#endif

template <typename Context, typename T>
void MatrixSolveFunctor<Context, T>::operator()(const Context& dev_ctx,
                                                const DenseTensor& a,
                                                const DenseTensor& b,
                                                DenseTensor* out) {
#ifndef PADDLE_WITH_HIP

  // solve the equation: Ax = B,
  // use cuBlas cublas<S/D>getrfBatched function to performs the LU
  // factorization of each matrix A,
  // and then use cuBlas cublas<S/D>getrsBatched function to solve the
  // equation after LU factorization.
  // ref:
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
  const auto& a_dims = a.dims();
  const int a_rank = a_dims.size();
  int n = a_dims[a_rank - 1];
  int lda = n;
  int batch_size = a_rank > 2 ? static_cast<int>(a.numel() / (n * n)) : 1;
  CUDNN_ENFORCE_TENSOR_SIZE_SUPPORTED(a);

  const auto& b_dims = b.dims();
  const int b_rank = b_dims.size();
  int nrhs = b_dims[b_rank - 1];
  int ldb = b_dims[b_rank - 2];
  CUDNN_ENFORCE_TENSOR_SIZE_SUPPORTED(b);

  // make sure the out dims is right
  out->Resize(b_dims);
  dev_ctx.template Alloc<T>(out);

  // copy input A to a temporary tensor tmp_a,
  // LU factorization, written back to original matrix A, so in the beginning,
  // it's necessary to create a temporary tensor tmp_a.
  DenseTensor tmp_a(a.dtype());
  tmp_a.Resize(a.dims());
  dev_ctx.template Alloc<T>(&tmp_a);
  phi::Copy(dev_ctx, a, dev_ctx.GetPlace(), false, &tmp_a);

  // copy input B to a temporary tensor tmp_b, and transpose tmp_b,
  // because cuBlas assumes column-major while Paddle uses row-majar.
  DenseTensor tmp_b(b.type());
  const auto& new_dims_vec = getNewDimsVec(b_dims);
  tmp_b.Resize(common::make_ddim(new_dims_vec));
  dev_ctx.template Alloc<T>(&tmp_b);
  phi::funcs::TransposeNormal<Context, T> trans;
  std::vector<int> new_axis = getNewAxis(b_rank);
  trans(dev_ctx, b, &tmp_b, new_axis);

  const T* a_data_in_gpu = tmp_a.data<T>();
  const T* b_data_in_gpu = tmp_b.data<T>();

  std::vector<const T*> cpu_ptrs(batch_size * 2);
  for (int i = 0; i < batch_size; ++i) {
    cpu_ptrs[i] = a_data_in_gpu + i * n * n;
    cpu_ptrs[i + batch_size] = b_data_in_gpu + i * n * nrhs;
  }

  // Copy the addresses of A and tmp_b from host to device.
  phi::Allocator::AllocationPtr tmp_gpu_ptrs_data = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      cpu_ptrs.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  memory_utils::Copy(dev_ctx.GetPlace(),
                     tmp_gpu_ptrs_data->ptr(),
                     phi::CPUPlace(),
                     static_cast<void*>(cpu_ptrs.data()),
                     cpu_ptrs.size() * sizeof(T*),
                     dev_ctx.stream());

  T** gpu_tmp_b_ptrs =
      reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;

#ifndef PADDLE_WITH_HIP
  if (batch_size == 1) {
    // Mirror torch.linalg.solve CUDA path for single matrices: use cuSOLVER
    // non-batched GETRF/GETRS on column-major copies of A and B.
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "cuSOLVER path supports float/double only.");
    DenseTensor a_col(tmp_a.dtype());
    a_col.Resize(common::make_ddim(getNewDimsVec(a_dims)));
    dev_ctx.template Alloc<T>(&a_col);
    phi::funcs::TransposeNormal<Context, T> trans_a;
    trans_a(dev_ctx, tmp_a, &a_col, new_axis);

    DenseTensor& b_col = tmp_b;
    auto handle = dev_ctx.cusolver_dn_handle();
    int lwork = 0;
    if (std::is_same<T, float>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgetrf_bufferSize(
          handle, n, n, reinterpret_cast<float*>(a_col.data<T>()), n, &lwork));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgetrf_bufferSize(
          handle, n, n, reinterpret_cast<double*>(a_col.data<T>()), n, &lwork));
    }
    phi::Allocator::AllocationPtr work_ptr = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        lwork * sizeof(T),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    phi::Allocator::AllocationPtr piv_info_ptr = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        (n + 1) * sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    int* d_ipiv = reinterpret_cast<int*>(piv_info_ptr->ptr());
    int* d_info = d_ipiv + n;

    if (std::is_same<T, float>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgetrf(
          handle,
          n,
          n,
          reinterpret_cast<float*>(a_col.data<T>()),
          n,
          reinterpret_cast<float*>(work_ptr->ptr()),
          d_ipiv,
          d_info));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgetrf(
          handle,
          n,
          n,
          reinterpret_cast<double*>(a_col.data<T>()),
          n,
          reinterpret_cast<double*>(work_ptr->ptr()),
          d_ipiv,
          d_info));
    }
    int info_host = 0;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_host,
                       dev_ctx.GetPlace(),
                       d_info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_host,
        0,
        common::errors::PreconditionNotMet(
            "GETRF failed with info = %d when batch_size == 1.", info_host));

    if (std::is_same<T, float>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgetrs(
          handle,
          CUBLAS_OP_N,
          n,
          nrhs,
          reinterpret_cast<float*>(a_col.data<T>()),
          n,
          d_ipiv,
          reinterpret_cast<float*>(b_col.data<T>()),
          n,
          d_info));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgetrs(
          handle,
          CUBLAS_OP_N,
          n,
          nrhs,
          reinterpret_cast<double*>(a_col.data<T>()),
          n,
          d_ipiv,
          reinterpret_cast<double*>(b_col.data<T>()),
          n,
          d_info));
    }
    memory_utils::Copy(phi::CPUPlace(),
                       &info_host,
                       dev_ctx.GetPlace(),
                       d_info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_host,
        0,
        common::errors::InvalidArgument(
            "GETRS failed with info = %d when batch_size == 1.", info_host));

    phi::funcs::TransposeNormal<Context, T> trans_back;
    trans_back(dev_ctx, b_col, out, new_axis);
    return;
  }
#endif

  // Allocate device memory for BatchedGETRF's info and pivots.
  int num_ints = n < 32 ? batch_size : batch_size * (n + 1);
  phi::Allocator::AllocationPtr tmp_gpu_info_data = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      num_ints * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* gpu_info_ptr = reinterpret_cast<int*>(tmp_gpu_info_data->ptr());

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  // only for singular checking
  std::vector<int> info;
  info.resize(batch_size);

  int* gpu_pivot_ptr =
      reinterpret_cast<int*>(tmp_gpu_info_data->ptr()) + batch_size;

  VLOG(3) << "matrix_solve GETRF batch_size=" << batch_size << " n=" << n
          << " nrhs=" << nrhs << " lda=" << lda << " ldb=" << ldb
          << " d_A_array=" << tmp_gpu_ptrs_data->ptr()
          << " d_B_array=" << static_cast<void*>(gpu_tmp_b_ptrs)
          << " int_buf=" << static_cast<void*>(gpu_info_ptr)
          << " hA0=" << (cpu_ptrs.empty() ? nullptr : cpu_ptrs[0]) << " hB0="
          << (cpu_ptrs.size() <= static_cast<size_t>(batch_size)
                  ? nullptr
                  : cpu_ptrs[batch_size]);

  // This function performs the LU factorization of each matrix A by the
  // equation A = L * U. L and U are written back to original matrix A,
  // and diagonal elements of L are discarded.
  blas.BatchedGETRF(n,
                    reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                    gpu_pivot_ptr,
                    gpu_info_ptr,
                    batch_size);

  // check whether BatchedGETRF is executed successfully or not
  memory_utils::Copy(phi::CPUPlace(),
                     info.data(),
                     dev_ctx.GetPlace(),
                     gpu_info_ptr,
                     sizeof(int) * batch_size,
                     dev_ctx.stream());
  for (int i = 0; i < batch_size; ++i) {
    PADDLE_ENFORCE_EQ(info[i],
                      0,
                      common::errors::PreconditionNotMet(
                          "For batch [%d]: U(%d, %d) is zero, singular U. "
                          "Please check the matrix value and change it to a "
                          "non-singular matrix",
                          i,
                          info[i],
                          info[i]));
  }

  // hold the result code from BatchedGETRS
  int host_info = 0;

  VLOG(3) << "matrix_solve GETRS batch_size=" << batch_size << " n=" << n
          << " nrhs=" << nrhs << " lda=" << lda << " ldb=" << ldb
          << " d_A_array=" << tmp_gpu_ptrs_data->ptr()
          << " d_B_array=" << static_cast<void*>(gpu_tmp_b_ptrs)
          << " int_buf=" << static_cast<void*>(gpu_info_ptr)
          << " hA0=" << (cpu_ptrs.empty() ? nullptr : cpu_ptrs[0]) << " hB0="
          << (cpu_ptrs.size() <= static_cast<size_t>(batch_size)
                  ? nullptr
                  : cpu_ptrs[batch_size]);

  // to solve the equation after LU factorization
  CBLAS_TRANSPOSE transA = CblasTrans;
  blas.BatchedGETRS(transA,
                    n,
                    nrhs,
                    reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                    lda,
                    gpu_pivot_ptr,
                    gpu_tmp_b_ptrs,
                    ldb,
                    &host_info,
                    batch_size);

  // check whether BatchedGETRS is executed successfully or not
  PADDLE_ENFORCE_EQ(host_info,
                    0,
                    common::errors::InvalidArgument(
                        "The [%d]'th argument to cublas*getrsBatched had "
                        "an illegal value.",
                        -host_info));

  // transpose tmp_b to get the final result in row-major form.
  phi::funcs::TransposeNormal<Context, T> trans2;
  trans2(dev_ctx, tmp_b, out, new_axis);

#else
  compute_solve_eigen<Context, T>(dev_ctx, a, b, out);
#endif
}

template class MatrixSolveFunctor<GPUContext, float>;
template class MatrixSolveFunctor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
