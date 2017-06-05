/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"

#include <cuda_runtime.h>


namespace RaC {

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

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

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



