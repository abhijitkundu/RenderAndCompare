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

template <int NumBins, typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_shared_bins(const ImageScalar* const gt_image,
                                             const ImageScalar* const pred_image,
                                             int width, int height,
                                             CMScalar* d_conf_mat) {
  // Initialize shared mem
  __shared__ CMScalar smem[NumBins][NumBins];
  smem[threadIdx.x][threadIdx.y] = 0;
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

void compute_confusion_matrix(const uint8_t* const gt_image,
                              const uint8_t* const pred_image,
                              int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;

  CMScalar* d_conf_mat;
  const std::size_t conf_mat_bytes = NumBins * NumBins * sizeof(CMScalar);
  cudaCheckError(cudaMalloc((void**)(&d_conf_mat), conf_mat_bytes));
  cudaCheckError(cudaMemset(d_conf_mat, 0, conf_mat_bytes));
  Eigen::Matrix<CMScalar, NumBins, NumBins, Eigen::RowMajor> h_conf_mat;

  using Vector = Eigen::Matrix<CMScalar, NumBins, 1>;

  GpuTimer gpu_timer;
  gpu_timer.Start();

  dim3 gridDim(16, 4);
  dim3 blockDim(NumBins, NumBins);

  confusion_matrix_shared_bins<NumBins><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat);

  cudaCheckError(cudaMemcpy(h_conf_mat.data(), d_conf_mat, h_conf_mat.size() * sizeof(CMScalar), cudaMemcpyDeviceToHost));

  Vector intersection_hist = h_conf_mat.diagonal();
  Vector union_hist = h_conf_mat.rowwise().sum() + h_conf_mat.colwise().sum().transpose() - intersection_hist;

  float mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < NumBins; ++i) {
    if (union_hist[i] > 0) {
      mean_iou += float(intersection_hist[i]) / union_hist[i];
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;


  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms.  ";
  std::cout << "mean_iou = " << mean_iou << "\n";

  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
}


template <typename Scalar>
__global__ void get_diagonal(const Scalar* const matrix, int length, Scalar *diag) {
    diag[threadIdx.x] = matrix[threadIdx.x * length + threadIdx.x];
}

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



template <int NumBins>
__global__ void warp_iou_kernel(const int* const matrix, float* mean_iou) {
  int row_sum = 0;
  int col_sum = 0;

  __shared__ int tp[32];
  __shared__ int rs[32];
  __shared__ int cs[32];

  if ((threadIdx.x < NumBins) && (threadIdx.y < NumBins)) {
    row_sum = matrix[threadIdx.y * NumBins + threadIdx.x];
    col_sum = matrix[threadIdx.x * NumBins + threadIdx.y];

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

    if (threadIdx.x < NumBins) {
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
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;


  const std::size_t conf_mat_bytes = NumBins * NumBins * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr = static_cast<CMScalar*>(gpu_device.allocate(conf_mat_bytes));
  cudaCheckError(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  const std::size_t hist_bytes = NumBins * sizeof(CMScalar);
  CMScalar* d_intersection_hist_ptr = static_cast<CMScalar*>(gpu_device.allocate(hist_bytes));
  CMScalar* d_union_hist_ptr = static_cast<CMScalar*>(gpu_device.allocate(hist_bytes));

  using Tensor2 = Eigen::Tensor<CMScalar, 2, Eigen::RowMajor>;
  using Tensor1 = Eigen::Tensor<CMScalar, 1, Eigen::RowMajor>;

  Eigen::TensorMap<Tensor2> d_conf_mat(d_conf_mat_ptr, NumBins, NumBins);

  Eigen::TensorMap<Tensor1> d_intersection_hist(d_intersection_hist_ptr, NumBins);
  Eigen::TensorMap<Tensor1> d_union_hist(d_union_hist_ptr, NumBins);

  Tensor1 h_intersection_hist(NumBins);
  Tensor1 h_union_hist(NumBins);

  const Eigen::array<int, 1> red_axis({0});
  const Eigen::array<int, 1> green_axis({1});

  GpuTimer gpu_timer;
  gpu_timer.Start();

  dim3 gridDim(16, 4);
  dim3 blockDim(NumBins, NumBins);

  confusion_matrix_shared_bins<NumBins><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat_ptr);
  get_diagonal<<<1, NumBins>>>(d_conf_mat_ptr, NumBins, d_intersection_hist_ptr);

  d_union_hist.device(gpu_device) = d_conf_mat.sum(red_axis) + d_conf_mat.sum(green_axis) - d_intersection_hist;

//  assert(cudaMemcpyAsync(h_intersection_hist.data(), d_intersection_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
//  assert(cudaMemcpyAsync(h_union_hist.data(), d_union_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
//  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  cudaCheckError(cudaMemcpy(h_intersection_hist.data(), d_intersection_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(h_union_hist.data(), d_union_hist_ptr, hist_bytes, cudaMemcpyDeviceToHost));

  float mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < NumBins; ++i) {
    if (h_union_hist[i] > 0) {
      mean_iou += float(h_intersection_hist[i]) / h_union_hist[i];
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms.  ";
  std::cout << "mean_iou = " << mean_iou << "\n";

  gpu_device.deallocate(d_conf_mat_ptr);
  gpu_device.deallocate(d_intersection_hist_ptr);
  gpu_device.deallocate(d_union_hist_ptr);
  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
}

void compute_cmat_warped_iou(const uint8_t* const gt_image, const uint8_t* const pred_image, int width, int height) {
  using ImageScalar = uint8_t;
  using CMScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;


  const std::size_t conf_mat_bytes = NumBins * NumBins * sizeof(CMScalar);
  CMScalar* d_conf_mat_ptr;
  cudaCheckError(cudaMalloc(&d_conf_mat_ptr, conf_mat_bytes));
  cudaCheckError(cudaMemset(d_conf_mat_ptr, 0, conf_mat_bytes));

  thrust::device_vector<float>mean_ious(1);

  GpuTimer gpu_timer;
  gpu_timer.Start();

  {
    dim3 gridDim(16, 4);
    dim3 blockDim(NumBins, NumBins);
    confusion_matrix_shared_bins<NumBins><<<gridDim, blockDim>>>(d_gt_image, d_pred_image , width, height, d_conf_mat_ptr);
  }
  {
    dim3 blockdim(32, 32);
    warp_iou_kernel<NumBins><<<1, blockdim>>>(d_conf_mat_ptr, mean_ious.data().get());
  }

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms.  ";
  print_vector("mean_iou =  ", mean_ious);

  cudaCheckError(cudaFree((void*)d_conf_mat_ptr));
  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
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



