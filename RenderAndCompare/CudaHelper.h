/**
 * @file CudaHelper.h
 * @brief CudaHelper
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_CUDAHELPER_H_
#define RENDERANDCOMPARE_CUDAHELPER_H_

#include <cuda_runtime.h>

namespace RaC {

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

}  // namespace RaC

#endif // end RENDERANDCOMPARE_CUDAHELPER_H_
