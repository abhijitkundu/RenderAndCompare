/**
 * @file NumericDiff.cu
 * @brief NumericDiff
 *
 * @author Abhijit Kundu
 */

#include "NumericDiff.h"
#include "CuteGL/Utils/CudaUtils.h"

namespace RaC {

template<class T>
__global__ void central_diff_kernel(const int n,
                                    const T* const fplus,
                                    const T* const fminus,
                                    const T step_size,
                                    T* diff,
                                    const int diff_stride) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    diff[i*diff_stride] = (fplus[i] - fminus[i]) / (2 * step_size);
  }
}

template <class T>
void central_diff_gpu(const int n, const T* const fplus, const T* const fminus, const T step_size, T* diff, const int diff_stride) {
  central_diff_kernel<<< (n + 32 - 1) / 32, 32>>>(n, fplus, fminus, step_size, diff, diff_stride);
}

// explicit instantiation
template void central_diff_gpu<float>(const int n,
                                      const float* const fplus,
                                      const float* const fminus,
                                      const float step_size,
                                      float* diff,
                                      const int diff_stride);
// explicit instantiation
template void central_diff_gpu<double>(const int n,
                                       const double* const fplus,
                                       const double* const fminus,
                                       const double step_size,
                                       double* diff,
                                       const int diff_stride);

}  // namespace RaC
