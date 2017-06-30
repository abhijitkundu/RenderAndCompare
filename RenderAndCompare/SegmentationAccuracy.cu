/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#define EIGEN_USE_GPU

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

template <typename Scalar>
__inline__ __device__
Scalar warpReduceSum(Scalar val) {
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

template <typename Scalar>
__inline__ __device__
Scalar warpAllReduceSum(Scalar val) {
  for (unsigned int mask = warpSize/2; mask > 0; mask /= 2)
    val += __shfl_xor(val, mask);
  return val;
}

template<int NumLabels, class ImageScalar, class HistVector = uint3>
__global__ void compute_seg_histograms_vector(const ImageScalar* const gt_image,
                                              const ImageScalar* const pred_image,
                                              std::size_t size,
                                              HistVector *hist) {

  // Initialize shared mem
  __shared__ HistVector sm_hist[NumLabels];
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

template<int NumLabels, class ImageScalar, class HistVector = uint3>
__global__ void compute_seg_histograms_vector_private(const ImageScalar* const gt_image,
                                                      const ImageScalar* const pred_image,
                                                      std::size_t size,
                                                      HistVector *hist) {

  // Initialize shared mem
  __shared__ HistVector sm_hist[NumLabels];
  sm_hist[threadIdx.x].x = 0;
  sm_hist[threadIdx.x].y = 0;
  sm_hist[threadIdx.x].z = 0;
  __syncthreads();

  // Initialize private mem
  HistVector pmem[NumLabels];
#pragma unroll
  for (int i = 0; i < NumLabels; ++i) {
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
  for (int i = 0; i < NumLabels; ++i) {
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
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;
  thrust::device_vector<HistVector> d_hist(NumLabels);
  thrust::device_vector<float2> d_mean_ious(1);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  // Initialize histogram to zeroes
  thrust::fill(d_hist.begin(), d_hist.end(), make_int3(0, 0, 0));

  compute_seg_histograms_vector<NumLabels><<<28*8, NumLabels>>>(d_gt_image, d_pred_image , width * height, d_hist.data().get());

//  d_mean_ious[0] = thrust::transform_reduce(d_hist.begin(), d_hist.end(), class_iou(), make_float2(0, 0), add_float2());
//  float2 mean_iou = thrust::transform_reduce(d_hist.begin(), d_hist.end(), class_iou(), make_float2(0, 0), add_float2());

  thrust::host_vector<HistVector> h_hist = d_hist;
  float2 mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());


  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
//  float2 mean_iou = d_mean_ious[0];
  std::cout << "mean_iou = " << mean_iou.x /  mean_iou.y << "\n";

  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
}

template <int NumLabels, typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_shared_bins(const ImageScalar* const gt_image,
                                             const ImageScalar* const pred_image,
                                             int width, int height,
                                             CMScalar* d_conf_mat) {
  // Initialize shared mem
  __shared__ CMScalar smem[NumLabels][NumLabels];
  smem[threadIdx.y][threadIdx.x] = 0;
  __syncthreads();

  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  for (int col = x; col < width; col += nx)
    for (int row = y; row < height; row += ny) {
      int pixel_index = row * width + col;
      int gt_label = static_cast<int>(gt_image[pixel_index]);
      int pred_label = static_cast<int>(pred_image[pixel_index]);
      atomicAdd(&smem[gt_label][pred_label] , 1 );
    }
  __syncthreads();

  atomicAdd(&(d_conf_mat[threadIdx.x + threadIdx.y * blockDim.x]), smem[threadIdx.x][threadIdx.y]);
}

template<int NumLabels, typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_shared_strided_atomics(const ImageScalar* const gt_image,
                                                        const ImageScalar* const pred_image,
                                                        int width, int height,
                                                        CMScalar* d_conf_mat) {
  // Initialize shared mem
  __shared__ CMScalar smem[NumLabels][NumLabels];
  if ((threadIdx.y < NumLabels) && (threadIdx.x < NumLabels))
    smem[threadIdx.y][threadIdx.x] = 0;
  __syncthreads();

  // pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // grid dimensions
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;

  for (int col = x; col < width; col += nx)
    for (int row = y; row < height; row += ny) {
      int pixel_index = row * width + col;
      int gt_label = static_cast<int>(gt_image[pixel_index]);
      int pred_label = static_cast<int>(pred_image[pixel_index]);
      atomicAdd(&smem[gt_label][pred_label] , 1 );
    }
  __syncthreads();

  if ((threadIdx.y < NumLabels) && (threadIdx.x < NumLabels))
    atomicAdd(&(d_conf_mat[threadIdx.x + threadIdx.y * NumLabels]), smem[threadIdx.y][threadIdx.x]);
}

template<int NumLabels, typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_shared_1d_strided_atomics(const ImageScalar* const gt_image,
                                                           const ImageScalar* const pred_image,
                                                           int size,
                                                           CMScalar* d_conf_mat) {
  static const int NumBins = NumLabels * NumLabels;
  // Initialize shared mem
  __shared__ CMScalar smem[NumBins];
  if (threadIdx.x < NumBins)
    smem[threadIdx.x] = 0;
  __syncthreads();

  // stride length
  const int stride = blockDim.x * gridDim.x;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);
    atomicAdd(&smem[gt_label * NumLabels + pred_label] , 1 );
  }
  __syncthreads();

  if (threadIdx.x < NumBins)
    atomicAdd(&(d_conf_mat[threadIdx.x]), smem[threadIdx.x]);
}

void compute_confusion_matrix(const uint8_t* const gt_image,
                              const uint8_t* const pred_image,
                              int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;

  CMScalar* d_conf_mat;
  const std::size_t conf_mat_bytes = NumLabels * NumLabels * sizeof(CMScalar);
  CUDA_CHECK(cudaMalloc((void**)(&d_conf_mat), conf_mat_bytes));
  CUDA_CHECK(cudaMemset(d_conf_mat, 0, conf_mat_bytes));
  Eigen::Matrix<CMScalar, NumLabels, NumLabels, Eigen::RowMajor> h_conf_mat;

  using Vector = Eigen::Matrix<CMScalar, NumLabels, 1>;

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  dim3 gridDim(16, 4);
  dim3 blockDim(NumLabels, NumLabels);

  confusion_matrix_shared_bins<NumLabels><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat);

  CUDA_CHECK(cudaMemcpy(h_conf_mat.data(), d_conf_mat, h_conf_mat.size() * sizeof(CMScalar), cudaMemcpyDeviceToHost));

  Vector intersection_hist = h_conf_mat.diagonal();
  Vector union_hist = h_conf_mat.rowwise().sum() + h_conf_mat.colwise().sum().transpose() - intersection_hist;

  float mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < NumLabels; ++i) {
    if (union_hist[i] > 0) {
      mean_iou += float(intersection_hist[i]) / union_hist[i];
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;


  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
  std::cout << "mean_iou = " << mean_iou << "\n";

  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
}

template <typename Scalar>
__global__ void get_diagonal(const Scalar* const matrix, int length, Scalar *diag) {
    diag[threadIdx.x] = matrix[threadIdx.x * length + threadIdx.x];
}

__global__ void sum_reduce(const int* const matrix, int length, int* rowwise_sum, int* colwise_sum) {
  int row_sum = 0;
  int col_sum = 0;

  if ((threadIdx.x < length) && (threadIdx.y < length)) {
    row_sum = matrix[threadIdx.y * length + threadIdx.x];
    col_sum = matrix[threadIdx.x * length + threadIdx.y];
  }

  row_sum = warpReduceSum(row_sum);
  col_sum = warpReduceSum(col_sum);

  if (threadIdx.x==0 && threadIdx.y < length) {
    rowwise_sum[threadIdx.y] = row_sum;
    colwise_sum[threadIdx.y] = col_sum;
  }
}

template <int NumLabels>
__global__ void warp_iou_kernel(const int* const matrix, float* mean_iou) {
  int row_sum = 0;
  int col_sum = 0;

  __shared__ int tp[32];
  __shared__ int rs[32];
  __shared__ int cs[32];

  if ((threadIdx.x < NumLabels) && (threadIdx.y < NumLabels)) {
    row_sum = matrix[threadIdx.y * NumLabels + threadIdx.x];
    col_sum = matrix[threadIdx.x * NumLabels + threadIdx.y];

    if (threadIdx.x == threadIdx.y)
      tp[threadIdx.x] = col_sum;
  }
  __syncthreads();              // Wait for all partial reductions

  row_sum = warpReduceSum(row_sum);
  col_sum = warpReduceSum(col_sum);

  if (threadIdx.x==0) {
    rs[threadIdx.y] = row_sum;
    cs[threadIdx.y] = col_sum;
  }
  __syncthreads();              // Wait for all partial reductions

  if (threadIdx.y == 0) {
    float c_iou = 0;
    int c_valid = 0;

    if (threadIdx.x < NumLabels) {
      int c_intersection = tp[threadIdx.x];
      int c_union = rs[threadIdx.x] + cs[threadIdx.x] - c_intersection;

      if (c_union > 0) {
        c_iou = float(c_intersection) / c_union;
        c_valid = 1;
      }
    }

    c_iou = warpReduceSum(c_iou);
    c_valid = warpReduceSum(c_valid);

    if (threadIdx.x==0)
      mean_iou[0] = c_iou / c_valid;
  }
}

// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << name << " = [";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, ", "));
  std::cout << "\b\b]" << std::endl;
}

void compute_confusion_tensor(const uint8_t* const gt_image,
                              const uint8_t* const pred_image,
                              int width, int height) {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;


  const std::size_t conf_mat_bytes = NumLabels * NumLabels * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr = static_cast<CMScalar*>(gpu_device.allocate(conf_mat_bytes));
  CUDA_CHECK(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  const std::size_t hist_bytes = NumLabels * sizeof(CMScalar);
  CMScalar* d_intersection_hist_ptr = static_cast<CMScalar*>(gpu_device.allocate(hist_bytes));
  CMScalar* d_union_hist_ptr = static_cast<CMScalar*>(gpu_device.allocate(hist_bytes));

  using Tensor2 = Eigen::Tensor<CMScalar, 2, Eigen::RowMajor>;
  using Tensor1 = Eigen::Tensor<CMScalar, 1, Eigen::RowMajor>;

  Eigen::TensorMap<Tensor2> d_conf_mat(d_conf_mat_ptr, NumLabels, NumLabels);

  Eigen::TensorMap<Tensor1> d_intersection_hist(d_intersection_hist_ptr, NumLabels);
  Eigen::TensorMap<Tensor1> d_union_hist(d_union_hist_ptr, NumLabels);

  Tensor1 h_intersection_hist(NumLabels);
  Tensor1 h_union_hist(NumLabels);

  const Eigen::array<int, 1> red_axis({0});
  const Eigen::array<int, 1> green_axis({1});

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  dim3 gridDim(16, 4);
  dim3 blockDim(NumLabels, NumLabels);

  confusion_matrix_shared_bins<NumLabels><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat_ptr);
  get_diagonal<<<1, NumLabels>>>(d_conf_mat_ptr, NumLabels, d_intersection_hist_ptr);

  d_union_hist.device(gpu_device) = d_conf_mat.sum(red_axis) + d_conf_mat.sum(green_axis) - d_intersection_hist;

//  assert(cudaMemcpyAsync(h_intersection_hist.data(), d_intersection_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
//  assert(cudaMemcpyAsync(h_union_hist.data(), d_union_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
//  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  CUDA_CHECK(cudaMemcpy(h_intersection_hist.data(), d_intersection_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_union_hist.data(), d_union_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost));

  float mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < NumLabels; ++i) {
    if (h_union_hist[i] > 0) {
      mean_iou += float(h_intersection_hist[i]) / h_union_hist[i];
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
  std::cout << "mean_iou = " << mean_iou << "\n";

  gpu_device.deallocate(d_conf_mat_ptr);
  gpu_device.deallocate(d_intersection_hist_ptr);
  gpu_device.deallocate(d_union_hist_ptr);
  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
}

void compute_cmat_warped_iou(const uint8_t* const gt_image, const uint8_t* const pred_image, int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;


  const std::size_t conf_mat_bytes = NumLabels * NumLabels * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr;
  CUDA_CHECK(cudaMalloc(&d_conf_mat_ptr, conf_mat_bytes));
  CUDA_CHECK(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  thrust::device_vector<float>mean_ious(1);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  {
    dim3 blockDim(NumLabels, NumLabels);
    dim3 gridDim((width + NumLabels - 1) / NumLabels, (height + NumLabels - 1) / NumLabels);
    confusion_matrix_shared_bins<NumLabels><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat_ptr);
  }
  {
    dim3 blockdim(32, 32);
    warp_iou_kernel<NumLabels><<<1, blockdim>>>(d_conf_mat_ptr, mean_ious.data().get());
  }

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
  print_vector("mean_iou =  ", mean_ious);

  CUDA_CHECK(cudaFree((void*)d_conf_mat_ptr));
  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
}

void compute_ssa_cmat_warped_iou(const uint8_t* const gt_image, const uint8_t* const pred_image, int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;
  const std::size_t conf_mat_bytes = NumLabels * NumLabels * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr;
  CUDA_CHECK(cudaMalloc(&d_conf_mat_ptr, conf_mat_bytes));
  CUDA_CHECK(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  thrust::device_vector<float>mean_ious(1);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  {
    int block_length = 25;
    dim3 blockDim(block_length, block_length);
    dim3 gridDim((width + block_length - 1) / block_length, (height + block_length - 1) / block_length);
    confusion_matrix_shared_strided_atomics<NumLabels><<<gridDim, blockDim>>>(d_gt_image, d_pred_image ,width, height, d_conf_mat_ptr);
  }
  {
    dim3 blockdim(32, 32);
    warp_iou_kernel<NumLabels><<<1, blockdim>>>(d_conf_mat_ptr, mean_ious.data().get());
  }

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
  print_vector("mean_iou =  ", mean_ious);

  CUDA_CHECK(cudaFree((void*)d_conf_mat_ptr));
  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
}

void compute_ssa1d_cmat_warped_iou(const uint8_t* const gt_image, const uint8_t* const pred_image, int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_gt_image, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_pred_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumLabels = 25;
  const std::size_t conf_mat_bytes = NumLabels * NumLabels * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr;
  CUDA_CHECK(cudaMalloc(&d_conf_mat_ptr, conf_mat_bytes));
  CUDA_CHECK(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  thrust::device_vector<float>mean_ious(1);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  {
    int size  = width * height;
    int blocksize = 1024; // Minimum: NumLabels * NumLabels
    int gridsize = (size + blocksize - 1) / blocksize;
    assert(blocksize >= NumLabels * NumLabels);
    confusion_matrix_shared_1d_strided_atomics<NumLabels><<<gridsize, blocksize>>>(d_gt_image, d_pred_image ,size, d_conf_mat_ptr);
  }
  {
    dim3 blockdim(32, 32);
    warp_iou_kernel<NumLabels><<<1, blockdim>>>(d_conf_mat_ptr, mean_ious.data().get());
  }

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms.  ";
  print_vector("mean_iou =  ", mean_ious);

  CUDA_CHECK(cudaFree((void*)d_conf_mat_ptr));
  CUDA_CHECK(cudaFree((void*)d_gt_image));
  CUDA_CHECK(cudaFree((void*)d_pred_image));
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
  CUDA_CHECK(cudaMalloc(&d_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));

  HistScalar *d_hist;
  CUDA_CHECK(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  CUDA_CHECK(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));

  dim3 block(16, 16);
  dim3 grid((width + 16 - 1) / 16 , (height + 16 - 1) / 16 ) ;

  histogram_atomics<<<grid, block>>>(d_image, width, height, d_hist);

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms\n";

  CUDA_CHECK(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree((void*)d_image));
  CUDA_CHECK(cudaFree((void*)d_hist));
}

template <int NumLabels, int NumParts, typename ImageScalar, typename HistScalar>
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
  __shared__ HistScalar smem[NumLabels + 1];
  for (int i = t; i < NumLabels + 1; i += nt)
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
  for (int i = t; i < NumLabels; i += nt) {
    out[i] = smem[i];
  }
}

template <int NumLabels, int NumParts, typename HistScalar>
__global__ void histogram_smem_accum(const HistScalar *in, int n, HistScalar *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NumLabels) {
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
  CUDA_CHECK(cudaMalloc(&d_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));

  HistScalar *d_hist;
  cudaMalloc(&d_hist, num_labels * sizeof(HistScalar));

  static const int NumLabels = 25;
  static const int NumParts = 1024;

  dim3 block(32, 4);
  dim3 grid(16, 16);
  int total_blocks = grid.x * grid.y;

  // allocate partial histogram
  HistScalar *d_part_hist;
  cudaMalloc(&d_part_hist, total_blocks * NumParts * sizeof(HistScalar));

  dim3 block2(128);
  dim3 grid2((NumLabels + block.x - 1) / block.x);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  histogram_smem_atomics<NumLabels, NumParts><<<grid, block>>>(d_image, width, height, d_part_hist);

  histogram_smem_accum<NumLabels, NumParts><<<grid2, block2>>>(d_part_hist, total_blocks, d_hist);

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms\n";

  CUDA_CHECK(cudaMemcpy(hist, d_hist, NumLabels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree((void*)d_image));
  CUDA_CHECK(cudaFree((void*)d_hist));
  CUDA_CHECK(cudaFree((void*)d_part_hist));
}

template <int NumLabels, typename ImageScalar, typename HistScalar>
__global__ void histogram_shared_bins(const ImageScalar* const image, std::size_t size, HistScalar *hist) {

  // Initialize shared mem
  __shared__ HistScalar smem[NumLabels];
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
  CUDA_CHECK(cudaMalloc(&d_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));


  int numSMs;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  HistScalar *d_hist;
  CUDA_CHECK(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  CUDA_CHECK(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));

  static const int NumLabels = 25;

  histogram_shared_bins<NumLabels><<<numSMs*8, NumLabels>>>(d_image, width * height, d_hist);

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms\n";

  CUDA_CHECK(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree((void*)d_image));
  CUDA_CHECK(cudaFree((void*)d_hist));
}

template <int NumLabels, typename ImageScalar, typename HistScalar>
__global__ void histogram_private_bins(const ImageScalar* const image, std::size_t size, HistScalar *hist) {

  // Initialize private mem
  HistScalar smem[NumLabels];
#pragma unroll
  for (int i = 0; i < NumLabels; ++i)
    smem[i] = 0;

  const int offset = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x ;i < size; i += offset) {
    int label = static_cast<int>(image[i]);
    ++smem[label];
  }

#pragma unroll
  for (int i = 0; i < NumLabels; ++i)
    atomicAdd(&(hist[i]), smem[i]);
}

void computeHistogramWithPrivateBins(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  if (num_labels != 25)
    throw std::runtime_error("Only support 25 labels");

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  CUDA_CHECK(cudaMalloc(&d_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));


  int numSMs;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  HistScalar *d_hist;
  CUDA_CHECK(cudaMalloc(&d_hist, num_labels * sizeof(HistScalar)));

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

  CUDA_CHECK(cudaMemset(d_hist, 0, num_labels * sizeof(HistScalar)));
  static const int NumLabels = 25;

  histogram_private_bins<NumLabels><<<numSMs, 256>>>(d_image, width * height, d_hist);

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms\n";

  CUDA_CHECK(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree((void*)d_image));
  CUDA_CHECK(cudaFree((void*)d_hist));
}

void computeHistogramWithThrust(const uint8_t* const image, int width, int height, int *hist, int num_bins) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  thrust::device_vector<ImageScalar> d_image(width * height);
  CUDA_CHECK(cudaMemcpy(d_image.data().get(), image, width * height * sizeof(ImageScalar), cudaMemcpyHostToDevice));

  thrust::device_vector<HistScalar> histogram(num_bins);

  using CuteGL::GpuTimer;
  GpuTimer gpu_timer;
  gpu_timer.start();

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

  gpu_timer.stop();
  std::cout << "GPU Time = " << gpu_timer.elapsed_in_ms() << " ms\n";

  CUDA_CHECK(cudaMemcpy(hist, histogram.data().get(), num_bins * sizeof(HistScalar), cudaMemcpyDeviceToHost));
}


void computeIoUseq(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                   const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images, const int trials) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  if (gt_images.size() != pred_images.size())
     throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const std::size_t gt_images_bytes = gt_images.size()  * sizeof(uint8_t);
  const std::size_t pred_images_bytes = pred_images.size()  * sizeof(uint8_t);

  uint8_t* d_gt_images;
  uint8_t* d_pred_images;

  CUDA_CHECK(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  CUDA_CHECK(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {
    static const int NumLabels = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    thrust::device_vector<CMScalar>d_conf_mat(NumLabels * NumLabels);
    thrust::device_vector<float>mean_ious(images_per_blob);

    using CuteGL::GpuTimer;
    GpuTimer gpu_timer;
    gpu_timer.start();

    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      const Eigen::Index offset = image_size * i;
      {
        thrust::fill(d_conf_mat.begin(), d_conf_mat.end(), 0);
        const int blocksize = 1024; // Minimum: NumLabels * NumLabels
        const int gridsize = (image_size + blocksize - 1) / blocksize;
        assert(blocksize >= NumLabels * NumLabels);
        confusion_matrix_shared_1d_strided_atomics<NumLabels><<<gridsize, blocksize>>>(d_gt_images + offset, d_pred_images + offset , image_size, d_conf_mat.data().get());
      }
      {
        dim3 blockdim(32, 32);
        warp_iou_kernel<NumLabels><<<1, blockdim>>>(d_conf_mat.data().get(), mean_ious.data().get() + i);
      }
    }
    float mean_iou = thrust::reduce(mean_ious.begin(), mean_ious.end(), 0.0f, thrust::plus<float>());
    mean_iou /= images_per_blob;

    gpu_timer.stop();
    float elapsed_millis = gpu_timer.elapsed_in_ms();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }
  CUDA_CHECK(cudaFree((void*)d_gt_images));
  CUDA_CHECK(cudaFree((void*)d_pred_images));

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

  CUDA_CHECK(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  CUDA_CHECK(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumLabels = 25;

    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    using CuteGL::GpuTimer;
    GpuTimer gpu_timer;
    gpu_timer.start();

    thrust::device_vector<HistVector> d_hist(NumLabels);
    thrust::host_vector<HistVector> h_hist(NumLabels);

    float mean_iou = 0;
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      // Initialize histogram to zeroes
      thrust::fill(d_hist.begin(), d_hist.end(), make_int3(0, 0, 0));

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumLabels><<<28*8, NumLabels>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist.data().get());

      h_hist = d_hist;
      float2 f2_mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.stop();
    float elapsed_millis = gpu_timer.elapsed_in_ms();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  CUDA_CHECK(cudaFree((void*)d_gt_images));
  CUDA_CHECK(cudaFree((void*)d_pred_images));
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

  CUDA_CHECK(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  CUDA_CHECK(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumLabels = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    HistVector* d_hist;
    const std::size_t d_hist_bytes = images_per_blob * NumLabels * sizeof(HistVector);
    CUDA_CHECK(cudaMalloc((void**)(&d_hist), d_hist_bytes));
    CUDA_CHECK(cudaMemset(d_hist, 0, d_hist_bytes));

    thrust::host_vector<HistVector, thrust::cuda::experimental::pinned_allocator<HistVector> > h_hist(images_per_blob * NumLabels);

    using CuteGL::GpuTimer;
    GpuTimer gpu_timer;
    gpu_timer.start();



    float mean_iou = 0;
  #pragma omp parallel for reduction(+:mean_iou)
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      thrust::device_ptr<HistVector> d_hist_ptr(d_hist + NumLabels * i);

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumLabels><<<28*8, NumLabels>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist_ptr.get());

      auto h_hist_it = h_hist.begin() + NumLabels * i;
      thrust::copy(d_hist_ptr, d_hist_ptr + NumLabels, h_hist_it);
      float2 f2_mean_iou = thrust::transform_reduce(h_hist_it, h_hist_it + NumLabels, class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.stop();
    float elapsed_millis = gpu_timer.elapsed_in_ms();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  CUDA_CHECK(cudaFree((void*)d_gt_images));
  CUDA_CHECK(cudaFree((void*)d_pred_images));
}

void computeIoUpar(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                   const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images, const int trials) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  if (gt_images.size() != pred_images.size())
     throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const std::size_t gt_images_bytes = gt_images.size()  * sizeof(uint8_t);
  const std::size_t pred_images_bytes = pred_images.size()  * sizeof(uint8_t);

  uint8_t* d_gt_images;
  uint8_t* d_pred_images;

  CUDA_CHECK(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  CUDA_CHECK(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {
    static const int NumLabels = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    thrust::device_vector<CMScalar>d_conf_mats(NumLabels * NumLabels * images_per_blob, 0);
    thrust::device_vector<float>mean_ious(images_per_blob);

    using CuteGL::GpuTimer;
    GpuTimer gpu_timer;
    gpu_timer.start();

#pragma omp parallel for
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      CMScalar* d_conf_mat_ptr = d_conf_mats.data().get() + NumLabels * NumLabels * i;
      {

        const int blocksize = 1024; // Minimum: NumLabels * NumLabels
        const int gridsize = (image_size + blocksize - 1) / blocksize;
        assert(blocksize >= NumLabels * NumLabels);
        const Eigen::Index image_data_offset = image_size * i;
        confusion_matrix_shared_1d_strided_atomics<NumLabels><<<gridsize, blocksize>>>(d_gt_images + image_data_offset,
            d_pred_images + image_data_offset ,
            image_size, d_conf_mat_ptr);
      }
      {
        dim3 blockdim(32, 32);
        warp_iou_kernel<NumLabels><<<1, blockdim>>>(d_conf_mat_ptr, mean_ious.data().get() + i);
      }
    }
    float mean_iou = thrust::reduce(mean_ious.begin(), mean_ious.end(), 0.0f, thrust::plus<float>());
    mean_iou /= images_per_blob;

    gpu_timer.stop();
    float elapsed_millis = gpu_timer.elapsed_in_ms();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }
  CUDA_CHECK(cudaFree((void*)d_gt_images));
  CUDA_CHECK(cudaFree((void*)d_pred_images));

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

  CUDA_CHECK(cudaMalloc((void**)(&d_gt_images), gt_images_bytes));
  CUDA_CHECK(cudaMalloc((void**)(&d_pred_images), pred_images_bytes));
  CUDA_CHECK(cudaMemcpy(d_gt_images, gt_images.data(), gt_images_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pred_images, pred_images.data(), pred_images_bytes, cudaMemcpyHostToDevice));

  for (int trial = 0; trial < trials; ++trial) {

    static const int NumLabels = 25;
    const Eigen::Index images_per_blob = gt_images.dimension(0);
    const Eigen::Index height = gt_images.dimension(2);
    const Eigen::Index width = gt_images.dimension(3);
    const Eigen::Index image_size = width * height;

    HistVector* d_hist;
    const std::size_t d_hist_bytes = images_per_blob * NumLabels * sizeof(HistVector);
    CUDA_CHECK(cudaMalloc((void**)(&d_hist), d_hist_bytes));
    CUDA_CHECK(cudaMemset(d_hist, 0, d_hist_bytes));

    using CuteGL::GpuTimer;
    GpuTimer gpu_timer;
    gpu_timer.start();

    float mean_iou = 0;

    const int num_streams = 16;
    cudaStream_t streams[num_streams];

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
  #pragma omp parallel for reduction(+:mean_iou) num_threads(num_streams)
    for (Eigen::Index i = 0; i < images_per_blob; ++i) {
      const auto threadId = omp_get_thread_num();
      cudaStreamCreate(&streams[threadId]);

      thrust::device_ptr<HistVector> d_hist_ptr(d_hist + NumLabels * i);

      const Eigen::Index offset = image_size * i;
      compute_seg_histograms_vector<NumLabels><<<28*8, NumLabels, 0, streams[threadId]>>>(d_gt_images + offset, d_pred_images + offset, image_size, d_hist_ptr.get());

      thrust::host_vector<HistVector> h_hist(NumLabels);
      thrust::copy(d_hist_ptr, d_hist_ptr + NumLabels, h_hist.begin());
      float2 f2_mean_iou = thrust::transform_reduce(h_hist.begin(), h_hist.end(), class_iou(), make_float2(0, 0), add_float2());
      mean_iou += f2_mean_iou.x / f2_mean_iou.y;
    }
    mean_iou /= images_per_blob;

    gpu_timer.stop();
    float elapsed_millis = gpu_timer.elapsed_in_ms();
    std::cout << "GPU Time = " << elapsed_millis << " ms.  ";
    std::cout << "mean_iou = " << mean_iou << "\n";

  }

  CUDA_CHECK(cudaFree((void*)d_gt_images));
  CUDA_CHECK(cudaFree((void*)d_pred_images));
}


}  // namespace RaC



