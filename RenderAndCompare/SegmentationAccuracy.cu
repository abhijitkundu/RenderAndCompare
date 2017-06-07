/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace RaC {

template<int NumBins, class ImageScalar, class HistVector = uint3>
__global__ void compute_seg_histograms_vector(const ImageScalar* const gt_image,
                                              const ImageScalar* const pred_image,
                                              std::size_t size,
                                              HistVector *hist) {

  // Initialize shared mem
  __shared__ HistVector sm_hist[NumBins];
  sm_hist[threadIdx.x].x = 0;
  sm_hist[threadIdx.x].y = 0;
  sm_hist[threadIdx.x].z = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);

    atomicAdd(&sm_hist[gt_label].x, 1);
    atomicAdd(&sm_hist[pred_label].z, 1);
    if (gt_label == pred_label)
      atomicAdd(&sm_hist[gt_label].y, 1);

    i += offset;
  }
  __syncthreads();


  atomicAdd(&(hist[threadIdx.x].x), sm_hist[threadIdx.x].x);
  atomicAdd(&(hist[threadIdx.x].y), sm_hist[threadIdx.x].y);
  atomicAdd(&(hist[threadIdx.x].z), sm_hist[threadIdx.x].z);
}

template<int NumBins, class ImageScalar, class HistVector = uint3>
__global__ void compute_seg_histograms_vector_private(const ImageScalar* const gt_image,
                                                      const ImageScalar* const pred_image,
                                                      std::size_t size,
                                                      HistVector *hist) {

  // Initialize shared mem
  __shared__ HistVector sm_hist[NumBins];
  sm_hist[threadIdx.x].x = 0;
  sm_hist[threadIdx.x].y = 0;
  sm_hist[threadIdx.x].z = 0;
  __syncthreads();

  // Initialize private mem
  HistVector pmem[NumBins];
#pragma unroll
  for (int i = 0; i < NumBins; ++i) {
    pmem[i].x = 0;
    pmem[i].y = 0;
    pmem[i].z = 0;
  }


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);

    ++pmem[gt_label].x;
    ++pmem[pred_label].z;
    if (gt_label == pred_label)
      ++pmem[gt_label].y;

    i += offset;
  }


#pragma unroll
  for (int i = 0; i < NumBins; ++i) {
    atomicAdd(&(sm_hist[i].x), pmem[i].x);
    atomicAdd(&(sm_hist[i].y), pmem[i].y);
    atomicAdd(&(sm_hist[i].z), pmem[i].z);
  }

  __syncthreads();


  atomicAdd(&(hist[threadIdx.x].x), sm_hist[threadIdx.x].x);
  atomicAdd(&(hist[threadIdx.x].y), sm_hist[threadIdx.x].y);
  atomicAdd(&(hist[threadIdx.x].z), sm_hist[threadIdx.x].z);
}

// Functor for computing class iou
struct class_iou {
  template <typename T>
  __host__ __device__
  float2 operator()(const T& a) const {
    float2 class_iou;
    int union_pixels = a.x + a.z - a.y;
    if (union_pixels > 0) {
      class_iou.x = float(a.y) / union_pixels;
      class_iou.y = 1;
    }
    else {
      class_iou.x = 0;
      class_iou.y = 0;
    }
    return class_iou;
  }
};

struct add_float2 {
  __host__ __device__
  float2 operator()(const float2& a, const float2& b) const {
    return make_float2(a.x + b.x, a.y + b.y);
  }
};

void compute_seg_histograms(const uint8_t* const gt_image,
                            const uint8_t* const pred_image,
                            int width, int height) {
  using ImageScalar = uint8_t;
  using HistVector = int3;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;
  thrust::device_vector<HistVector> d_hist(NumBins);
  thrust::device_vector<float2> d_mean_ious(1);

  GpuTimer gpu_timer;
  gpu_timer.Start();

  // Initialize histogram to zeroes
  thrust::fill(d_hist.begin(), d_hist.end(), make_int3(0, 0, 0));

  compute_seg_histograms_vector<NumBins><<<28*8, NumBins>>>(d_gt_image, d_pred_image , width * height, d_hist.data().get());

//  d_mean_ious[0] = thrust::transform_reduce(d_hist.begin(), d_hist.end(), class_iou(), make_float2(0, 0), add_float2());
//  float2 mean_iou = thrust::transform_reduce(d_hist.begin(), d_hist.end(), class_iou(), make_float2(0, 0), add_float2());

  thrust::host_vector<HistVector> h_hist = d_hist;
  float2 mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());


  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms.  ";
//  float2 mean_iou = d_mean_ious[0];
  std::cout << "mean_iou = " << mean_iou.x /  mean_iou.y << "\n";

  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
}



void computeIoUwithCUDAseq(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                            const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images,
                            const int trials) {
  using ImageScalar = uint8_t;
  using HistVector = int3;

  if (gt_images.size() != pred_images.size())
     throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const std::size_t gt_images_bytes = gt_images.size()  * sizeof(uint8_t);
  const std::size_t pred_images_bytes = pred_images.size()  * sizeof(uint8_t);

  uint8_t* d_gt_images;
  uint8_t* d_pred_images;

  cudaCheckError(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  cudaCheckError(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  cudaCheckError(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumBins = 25;

    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    GpuTimer gpu_timer;
    gpu_timer.Start();

    thrust::device_vector<HistVector> d_hist(NumBins);
    thrust::host_vector<HistVector> h_hist(NumBins);

    float mean_iou = 0;
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      // Initialize histogram to zeroes
      thrust::fill(d_hist.begin(), d_hist.end(), make_int3(0, 0, 0));

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumBins><<<28*8, NumBins>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist.data().get());

      h_hist = d_hist;
      float2 f2_mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  cudaCheckError(cudaFree((void*)d_gt_images));
  cudaCheckError(cudaFree((void*)d_pred_images));
}

void computeIoUwithCUDApar(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                            const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images,
                            const int trials) {
  using ImageScalar = uint8_t;
  using HistVector = int3;

  if (gt_images.size() != pred_images.size())
     throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const std::size_t gt_images_bytes = gt_images.size()  * sizeof(uint8_t);
  const std::size_t pred_images_bytes = pred_images.size()  * sizeof(uint8_t);

  uint8_t* d_gt_images;
  uint8_t* d_pred_images;

  cudaCheckError(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  cudaCheckError(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  cudaCheckError(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumBins = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    HistVector* d_hist;
    const std::size_t d_hist_bytes = images_per_blob * NumBins * sizeof(HistVector);
    cudaCheckError(cudaMalloc((void**)(&d_hist), d_hist_bytes));
    cudaCheckError(cudaMemset(d_hist, 0, d_hist_bytes));

    thrust::host_vector<HistVector, thrust::cuda::experimental::pinned_allocator<HistVector> > h_hist(images_per_blob * NumBins);

    GpuTimer gpu_timer;
    gpu_timer.Start();



    float mean_iou = 0;
  #pragma omp parallel for reduction(+:mean_iou)
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      thrust::device_ptr<HistVector> d_hist_ptr(d_hist + NumBins * i);

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumBins><<<28*8, NumBins>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist_ptr.get());

      auto h_hist_it = h_hist.begin() + NumBins * i;
      thrust::copy(d_hist_ptr, d_hist_ptr + NumBins, h_hist_it);
      float2 f2_mean_iou = thrust::transform_reduce(h_hist_it, h_hist_it + NumBins, class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  cudaCheckError(cudaFree((void*)d_gt_images));
  cudaCheckError(cudaFree((void*)d_pred_images));
}

void computeIoUwithCUDAstreams(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                                const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images,
                                const int trials) {
  using ImageScalar = uint8_t;
  using HistVector = int3;

  if (gt_images.size() != pred_images.size())
     throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const std::size_t gt_images_bytes = gt_images.size()  * sizeof(uint8_t);
  const std::size_t pred_images_bytes = pred_images.size()  * sizeof(uint8_t);

  uint8_t* d_gt_images;
  uint8_t* d_pred_images;

  cudaCheckError(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  cudaCheckError(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  cudaCheckError(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumBins = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    HistVector* d_hist;
    const std::size_t d_hist_bytes = images_per_blob * NumBins * sizeof(HistVector);
    cudaCheckError(cudaMalloc((void**)(&d_hist), d_hist_bytes));
    cudaCheckError(cudaMemset(d_hist, 0, d_hist_bytes));

    GpuTimer gpu_timer;
    gpu_timer.Start();

    float mean_iou = 0;

    const int num_streams = 16;
    cudaStream_t streams[num_streams];

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
  #pragma omp parallel for reduction(+:mean_iou) num_threads(num_streams)
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      const auto threadId = omp_get_thread_num();
      cudaStreamCreate(&streams[threadId]);

      thrust::device_ptr<HistVector> d_hist_ptr(d_hist + NumBins * i);

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumBins><<<28*8, NumBins, 0, streams[threadId]>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist_ptr.get());

      thrust::host_vector<HistVector> h_hist(NumBins);
      thrust::copy(d_hist_ptr, d_hist_ptr + NumBins, h_hist.begin());
      float2 f2_mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  cudaCheckError(cudaFree((void*)d_gt_images));
  cudaCheckError(cudaFree((void*)d_pred_images));
}

template <typename ImageScalar, typename HistScalar>
__global__ void histogram_atomics(const ImageScalar* const image, int width, int height, HistScalar *hist) {
  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  for (int col = x; col < width; col += nx) {
    for (int row = y; row < height; row += ny) {
      int label = static_cast<int>(image[row * width + col]);
      atomicAdd(&hist[label] , 1 );
    }
  }
}

void computeHistogramWithAtomics(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));

  HistScalar *d_hist;
  cudaCheckError(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));


  GpuTimer gpu_timer;
  gpu_timer.Start();

  cudaCheckError(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));

  dim3 block(16, 16);
  dim3 grid((width + 16 - 1) / 16 , (height + 16 - 1) / 16 ) ;

  histogram_atomics<<<grid, block>>>(d_image, width, height, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree((void*)d_image));
  cudaCheckError(cudaFree((void*)d_hist));
}

template <int NumBins, int NumParts, typename ImageScalar, typename HistScalar>
__global__ void histogram_smem_atomics(const ImageScalar* const image, int width, int height, HistScalar *out) {
  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;


  // linear thread index within 2D block
  int t = threadIdx.x + threadIdx.y * blockDim.x;

  // total threads in 2D block
  int nt = blockDim.x * blockDim.y;

  // linear block index within 2D grid
  int g = blockIdx.x + blockIdx.y * gridDim.x;

  // initialize temporary hist array in shared memory
  __shared__ HistScalar smem[NumBins + 1];
  for (int i = t; i < NumBins + 1; i += nt)
    smem[i] = 0;
  __syncthreads();

  for (int col = x; col < width; col += nx) {
    for (int row = y; row < height; row += ny) {
      int label = static_cast<int>(image[row * width + col]);
      atomicAdd(&smem[label] , 1 );
    }
  }
  __syncthreads();

  // write partial histogram into the global memory
  out += g * NumParts;
  for (int i = t; i < NumBins; i += nt) {
    out[i] = smem[i];
  }
}

template <int NumBins, int NumParts, typename HistScalar>
__global__ void histogram_smem_accum(const HistScalar *in, int n, HistScalar *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NumBins) {
    HistScalar total = 0;
    for (int j = 0; j < n; j++)
      total += in[i + NumParts * j];
    out[i] = total;
  }
}

void computeHistogramWithSharedAtomics(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));

  HistScalar *d_hist;
  cudaMalloc(&d_hist, num_labels * sizeof(HistScalar));

  static const int NumBins = 25;
  static const int NumParts = 1024;

  dim3 block(32, 4);
  dim3 grid(16, 16);
  int total_blocks = grid.x * grid.y;

  // allocate partial histogram
  HistScalar *d_part_hist;
  cudaMalloc(&d_part_hist, total_blocks * NumParts * sizeof(HistScalar));

  dim3 block2(128);
  dim3 grid2((NumBins + block.x - 1) / block.x);

  GpuTimer gpu_timer;
  gpu_timer.Start();

  histogram_smem_atomics<NumBins, NumParts><<<grid, block>>>(d_image, width, height, d_part_hist);

  histogram_smem_accum<NumBins, NumParts><<<grid2, block2>>>(d_part_hist, total_blocks, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, d_hist, NumBins * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree((void*)d_image));
  cudaCheckError(cudaFree((void*)d_hist));
  cudaCheckError(cudaFree((void*)d_part_hist));
}



template <int NumBins, typename ImageScalar, typename HistScalar>
__global__ void histogram_shared_bins(const ImageScalar* const image, std::size_t size, HistScalar *hist) {

  // Initialize shared mem
  __shared__ HistScalar smem[NumBins];
  smem[threadIdx.x] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    int label = static_cast<int>(image[i]);
    atomicAdd(&smem[label] , 1 );
    i += offset;
  }
  __syncthreads();
  atomicAdd(&(hist[threadIdx.x]), smem[threadIdx.x]);
}


void computeHistogramWithSharedBins(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  if (num_labels != 25)
    throw std::runtime_error("Only support 25 labels");

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));


  int numSMs;
  cudaCheckError(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  HistScalar *d_hist;
  cudaCheckError(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));

  GpuTimer gpu_timer;
  gpu_timer.Start();

  cudaCheckError(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));

  static const int NumBins = 25;

  histogram_shared_bins<NumBins><<<numSMs*8, NumBins>>>(d_image, width * height, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree((void*)d_image));
  cudaCheckError(cudaFree((void*)d_hist));
}

template <int NumBins, typename ImageScalar, typename HistScalar>
__global__ void histogram_private_bins(const ImageScalar* const image, std::size_t size, HistScalar *hist) {

  // Initialize private mem
  HistScalar smem[NumBins];
#pragma unroll
  for (int i = 0; i < NumBins; ++i)
    smem[i] = 0;

  const int offset = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x ;i < size; i += offset) {
    int label = static_cast<int>(image[i]);
    ++smem[label];
  }

#pragma unroll
  for (int i = 0; i < NumBins; ++i)
    atomicAdd(&(hist[i]), smem[i]);
}

void computeHistogramWithPrivateBins(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  if (num_labels != 25)
    throw std::runtime_error("Only support 25 labels");

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));


  int numSMs;
  cudaCheckError(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  HistScalar *d_hist;
  cudaCheckError(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));

  GpuTimer gpu_timer;
  gpu_timer.Start();

  cudaCheckError(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));
  static const int NumBins = 25;

  histogram_private_bins<NumBins><<<numSMs, 256>>>(d_image, width * height, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree((void*)d_image));
  cudaCheckError(cudaFree((void*)d_hist));
}

void computeHistogramWithThrust(const uint8_t* const image, int width, int height, int *hist, int num_bins) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  thrust::device_vector<ImageScalar> d_image(width * height);
  cudaCheckError(cudaMemcpy(d_image.data().get(), image, width * height * sizeof(ImageScalar), cudaMemcpyHostToDevice));

  thrust::device_vector<HistScalar> histogram(num_bins);

  GpuTimer gpu_timer;
  gpu_timer.Start();

  // sort data to bring equal elements together
  thrust::sort(d_image.begin(), d_image.end());

//  // number of histogram bins is equal to the maximum value plus one
//  HistScalar num_bins = d_image.back() + 1;

  // compute cumulative histogram by finding the end of each bin of values
  thrust::counting_iterator<HistScalar> search_begin(0);
  thrust::upper_bound(d_image.begin(), d_image.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, histogram.data().get(), num_bins * sizeof(HistScalar), cudaMemcpyDeviceToHost));
}

}  // namespace RaC



