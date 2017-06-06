/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"

namespace RaC {


template<int NumBins, typename ImageScalar, typename HistScalar>
__global__ void compute_seg_histograms_scalar(const ImageScalar* const gt_image,
                                              const ImageScalar* const pred_image,
                                              std::size_t size,
                                              HistScalar *total_pixels_class,
                                              HistScalar *ok_pixels_class,
                                              HistScalar *label_pixels) {

  // Initialize shared mem
  __shared__ HistScalar sm_total_pixels_class[NumBins];
  __shared__ HistScalar sm_ok_pixels_class[NumBins];
  __shared__ HistScalar sm_label_pixels[NumBins];
  sm_total_pixels_class[threadIdx.x] = 0;
  sm_ok_pixels_class[threadIdx.x] = 0;
  sm_label_pixels[threadIdx.x] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);

    atomicAdd(&sm_total_pixels_class[gt_label], 1);
    atomicAdd(&sm_label_pixels[pred_label], 1);
    if (gt_label == pred_label)
      atomicAdd(&sm_ok_pixels_class[gt_label], 1);

    i += offset;
  }
  __syncthreads();

  atomicAdd(&(total_pixels_class[threadIdx.x]), sm_total_pixels_class[threadIdx.x]);
  atomicAdd(&(ok_pixels_class[threadIdx.x]), sm_ok_pixels_class[threadIdx.x]);
  atomicAdd(&(label_pixels[threadIdx.x]), sm_label_pixels[threadIdx.x]);
}


void compute_seg_histograms_sep(const uint8_t* const gt_image,
                                const uint8_t* const pred_image,
                                int width, int height) {
  using ImageScalar = uint8_t;
  using HistScalar = int;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;
  const std::size_t hist_bytes = NumBins * sizeof(HistScalar);

  HistScalar *d_total_pixels_class;
  HistScalar *d_ok_pixels_class;
  HistScalar *d_label_pixels;
  cudaCheckError(cudaMalloc(&d_total_pixels_class, hist_bytes));
  cudaCheckError(cudaMalloc(&d_ok_pixels_class, hist_bytes));
  cudaCheckError(cudaMalloc(&d_label_pixels, hist_bytes));


  GpuTimer gpu_timer;
  gpu_timer.Start();

  cudaCheckError(cudaMemset(d_total_pixels_class, 0, hist_bytes));
  cudaCheckError(cudaMemset(d_ok_pixels_class, 0, hist_bytes));
  cudaCheckError(cudaMemset(d_label_pixels, 0, hist_bytes));


  compute_seg_histograms_scalar<NumBins><<<28*8, NumBins>>>(d_gt_image, d_pred_image , width * height, d_total_pixels_class, d_ok_pixels_class, d_label_pixels);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  Eigen::VectorXi total_pixels_class(NumBins);
  Eigen::VectorXi ok_pixels_class(NumBins);
  Eigen::VectorXi label_pixels(NumBins);

  cudaCheckError(cudaMemcpy(total_pixels_class.data(), d_total_pixels_class, hist_bytes, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(ok_pixels_class.data(), d_ok_pixels_class, hist_bytes, cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(label_pixels.data(), d_label_pixels, hist_bytes, cudaMemcpyDeviceToHost));

  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
  std::cout << "total_pixels_class = " << total_pixels_class.format(fmt) << "\n";
  std::cout << "ok_pixels_class = " << ok_pixels_class.format(fmt) << "\n";
  std::cout << "label_pixels = " << label_pixels.format(fmt) << "\n";

  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
  cudaCheckError(cudaFree((void*)d_total_pixels_class));
  cudaCheckError(cudaFree((void*)d_ok_pixels_class));
  cudaCheckError(cudaFree((void*)d_label_pixels));
}

template<int NumBins, typename ImageScalar>
__global__ void compute_seg_histograms_vector(const ImageScalar* const gt_image,
                                              const ImageScalar* const pred_image,
                                              std::size_t size,
                                              uint3 *hist) {

  // Initialize shared mem
  __shared__ uint3 sm_hist[NumBins];
  sm_hist[threadIdx.x] = make_uint3(0, 0, 0);
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

  uint3& smh = sm_hist[threadIdx.x];
  uint3& hist_bin = hist[threadIdx.x];

  atomicAdd(&(hist_bin.x), smh.x);
  atomicAdd(&(hist_bin.y), smh.y);
  atomicAdd(&(hist_bin.z), smh.z);

//  atomicAdd(&(hist[threadIdx.x].x), sm_hist[threadIdx.x].x);
//  atomicAdd(&(hist[threadIdx.x].y), sm_hist[threadIdx.x].y);
//  atomicAdd(&(hist[threadIdx.x].z), sm_hist[threadIdx.x].z);
}

void compute_seg_histograms(const uint8_t* const gt_image,
                            const uint8_t* const pred_image,
                            int width, int height) {
  using ImageScalar = uint8_t;
  using HistVector = uint3;

  ImageScalar* d_gt_image;
  ImageScalar* d_pred_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_gt_image, image_bytes));
  cudaCheckError(cudaMalloc(&d_pred_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_gt_image, gt_image, image_bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_pred_image, pred_image, image_bytes, cudaMemcpyHostToDevice));

  static const int NumBins = 25;
  const std::size_t hist_bytes = NumBins * sizeof(uint3);

  HistVector * d_hist;
  cudaCheckError(cudaMalloc(&d_hist, hist_bytes));

  GpuTimer gpu_timer;
  gpu_timer.Start();

  cudaCheckError(cudaMemset(d_hist, 0, hist_bytes));

  compute_seg_histograms_vector<NumBins><<<28*8, NumBins>>>(d_gt_image, d_pred_image , width * height, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  Eigen::VectorXi total_pixels_class(NumBins);
  Eigen::VectorXi ok_pixels_class(NumBins);
  Eigen::VectorXi label_pixels(NumBins);


  HistVector hist[NumBins];
  cudaCheckError(cudaMemcpy(hist, d_hist, hist_bytes, cudaMemcpyDeviceToHost));

  std::cout << "total_pixels_class = [";
  for (int i= 0; i< NumBins; ++i) {
    std::cout << hist[i].x << ", ";
  }
  std::cout << "\b\b]\n";

  std::cout << "ok_pixels_class = [";
  for (int i= 0; i< NumBins; ++i) {
    std::cout << hist[i].y << ", ";
  }
  std::cout << "\b\b]\n";

  std::cout << "label_pixels = [";
  for (int i= 0; i< NumBins; ++i) {
    std::cout << hist[i].z << ", ";
  }
  std::cout << "\b\b]\n";

  cudaCheckError(cudaFree((void*)d_gt_image));
  cudaCheckError(cudaFree((void*)d_pred_image));
  cudaCheckError(cudaFree((void*)d_hist));
}


float computeIoUwithCUDA(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images,
                         const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images) {
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

  GpuTimer gpu_timer;
  gpu_timer.Start();

  float mean_iou = 0;


  gpu_timer.Stop();
  float elapsed_millis = gpu_timer.ElapsedMillis();
  std::cout << "GPU Time = " << elapsed_millis << " ms\n";

  cudaCheckError(cudaFree((void*)d_gt_images));
  cudaCheckError(cudaFree((void*)d_pred_images));

  return mean_iou;
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

}  // namespace RaC



