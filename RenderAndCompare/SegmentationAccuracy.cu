/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"
#include "CudaHelper.h"

#define CUDA_DEBUG

#ifdef CUDA_DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

namespace RaC {

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
__global__
void histogramDumb(const ImageScalar* const image, int width, int height, HistScalar *hist) {
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

template <class Scalar>
void computeHistogramWithCudaDumb(const Scalar* const image, int width, int height, int *hist, int num_labels) {
  using ImageScalar = Scalar;
  using HistScalar = int;

  ImageScalar* d_image;
  const std::size_t image_bytes = width * height * sizeof(ImageScalar);
  cudaCheckError(cudaMalloc(&d_image, image_bytes));
  cudaCheckError(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice));

  GpuTimer gpu_timer;
  gpu_timer.Start();

  HistScalar *d_hist;
  cudaMalloc(&d_hist, num_labels * sizeof(HistScalar));

  dim3 block(16, 16);
  dim3 grid((width + 16 - 1) / 16 , (height + 16 - 1) / 16 ) ;

  histogramDumb<<<grid, block>>>(d_image, width, height, d_hist);

  gpu_timer.Stop();
  std::cout << "GPU Time = " << gpu_timer.ElapsedMillis() << " ms\n";

  cudaCheckError(cudaMemcpy(hist, d_hist, num_labels * sizeof(HistScalar), cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree((void*)d_image));
  cudaCheckError(cudaFree((void*)d_hist));
}

void computeHistogramWithCuda(const uint8_t* const image, int width, int height, int *hist, int num_labels) {
  computeHistogramWithCudaDumb(image, width, height, hist, num_labels);
}

namespace histogram_smem_atomics
{
    // Decode float4 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        float* samples = reinterpret_cast<float*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
    }

    // Decode uchar4 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
    }

    // Decode uchar1 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        bins[0] = (unsigned int) pixel.x;
    }

    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    __global__ void histogram_smem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
    {
        // pixel coordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // grid dimensions
        int nx = blockDim.x * gridDim.x;
        int ny = blockDim.y * gridDim.y;

        // threads in workgroup
        int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
        int nt = blockDim.x * blockDim.y; // total threads in workgroup

        // linear group index in 0..ngroups-1
        int g = blockIdx.x + blockIdx.y * gridDim.x;

        // initialize temporary accumulation array in shared memory
        __shared__ unsigned int smem[ACTIVE_CHANNELS * NUM_BINS + 3];
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
            smem[i] = 0;
        __syncthreads();

        // process pixels
        // updates our group's partial histogram in smem
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                    atomicAdd(&smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL], 1);
            }
        }

        __syncthreads();

        // write partial histogram into the global memory
        // move to our workgroup's slice of output
        out += g * NUM_PARTS;

        // store local output to global
        for (int i = t; i < NUM_BINS; i += nt)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    __global__ void histogram_smem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range
        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];
        out[i] = total;
    }

}   // namespace histogram_smem_atomics

template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_smem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block.x - 1) / block.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    histogram_smem_atomics::histogram_smem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist);

    histogram_smem_atomics::histogram_smem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_part_hist);

    return elapsed_millis;
}


//__global__ void histogram_smem_atomics(const IN_TYPE *in, int width, int height, unsigned int *out)
//{
//  // pixel coordinates
//  int x = blockIdx.x * blockDim.x + threadIdx.x;
//  int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//  // grid dimensions
//  int nx = blockDim.x * gridDim.x;
//  int ny = blockDim.y * gridDim.y;
//
//  // linear thread index within 2D block
//  int t = threadIdx.x + threadIdx.y * blockDim.x;
//
//  // total threads in 2D block
//  int nt = blockDim.x * blockDim.y;
//
//  // linear block index within 2D grid
//  int g = blockIdx.x + blockIdx.y * gridDim.x;
//
//  // initialize temporary accumulation array in shared memory
//  __shared__ unsigned int smem[3 * NUM_BINS + 3];
//  for (int i = t; i < 3 * NUM_BINS + 3; i += nt) smem[i] = 0;
//  __syncthreads();
//
//  // process pixels
//  // updates our block's partial histogram in shared memory
//  for (int col = x; col < width; col += nx)
//    for (int row = y; row < height; row += ny) {
//      unsigned int r = (unsigned int)(256 * in[row * width + col].x);
//      unsigned int g = (unsigned int)(256 * in[row * width + col].y);
//      unsigned int b = (unsigned int)(256 * in[row * width + col].z);
//      atomicAdd(&smem[NUM_BINS * 0 + r + 0], 1);
//      atomicAdd(&smem[NUM_BINS * 1 + g + 1], 1);
//      atomicAdd(&smem[NUM_BINS * 2 + b + 2], 1);
//    }
//  __syncthreads();
//
//  // write partial histogram into the global memory
//  out += g * NUM_PARTS;
//  for (int i = t; i < NUM_BINS; i += nt) {
//    out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
//    out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1 + 1];
//    out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2 + 2];
//  }
//}
//
//__global__ void histogram_final_accum(const unsigned int *in, int n, unsigned int *out)
//{
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < 3 * NUM_BINS) {
//    unsigned int total = 0;
//    for (int j = 0; j < n; j++)
//      total += in[i + NUM_PARTS * j];
//    out[i] = total;
//  }
//}

}  // namespace RaC



