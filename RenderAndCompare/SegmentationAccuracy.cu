/**
 * @file SegmentationAccuracy.cu
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"

namespace RaC {

template <typename Scalar>
__inline__ __device__
Scalar warpReduceSum(Scalar val) {
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

template<typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_ssa1d(const int num_of_pixels,
                                       const ImageScalar* const gt_image,
                                       const ImageScalar* const pred_image,
                                       const int num_of_labels,
                                       CMScalar* d_conf_mat) {
  const int num_of_bins = num_of_labels * num_of_labels;
  // Initialize shared mem
  extern __shared__ CMScalar smem[];
  if (threadIdx.x < num_of_bins)
    smem[threadIdx.x] = 0;
  __syncthreads();

  // stride length
  const int stride = blockDim.x * gridDim.x;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_of_pixels; i += stride) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);
    atomicAdd(&smem[gt_label * num_of_labels + pred_label] , 1 );
  }
  __syncthreads();

  if (threadIdx.x < num_of_bins)
    atomicAdd(&(d_conf_mat[threadIdx.x]), smem[threadIdx.x]);
}

template <typename CMScalar, typename IoUScalar>
__global__ void mean_class_iou_from_confidence_matrix(const int num_of_labels, const CMScalar* const matrix, IoUScalar* mean_iou) {
  CMScalar row_sum = 0;
  CMScalar col_sum = 0;

  __shared__ CMScalar tp[32];
  __shared__ CMScalar rs[32];
  __shared__ CMScalar cs[32];

  if ((threadIdx.x < num_of_labels) && (threadIdx.y < num_of_labels)) {
    row_sum = matrix[threadIdx.y * num_of_labels + threadIdx.x];
    col_sum = matrix[threadIdx.x * num_of_labels + threadIdx.y];

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
    IoUScalar c_iou = 0;
    CMScalar c_valid = 0;

    if (threadIdx.x < num_of_labels) {
      CMScalar c_intersection = tp[threadIdx.x];
      CMScalar c_union = rs[threadIdx.x] + cs[threadIdx.x] - c_intersection;

      if (c_union > 0) {
        c_iou = IoUScalar(c_intersection) / c_union;
        c_valid = 1;
      }
    }

    c_iou = warpReduceSum(c_iou);
    c_valid = warpReduceSum(c_valid);

    if (threadIdx.x==0)
      mean_iou[0] = c_iou / c_valid;
  }
}

template <typename CMScalar, typename IoUScalar>
__global__ void iou_loss_from_confidence_matrix(const int num_of_labels, const CMScalar* const matrix, IoUScalar* iou_loss) {
  CMScalar row_sum = 0;
  CMScalar col_sum = 0;

  __shared__ CMScalar tp[32];
  __shared__ CMScalar rs[32];
  __shared__ CMScalar cs[32];

  if ((threadIdx.x < num_of_labels) && (threadIdx.y < num_of_labels)) {
    row_sum = matrix[threadIdx.y * num_of_labels + threadIdx.x];
    col_sum = matrix[threadIdx.x * num_of_labels + threadIdx.y];

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
    IoUScalar c_iou = 0;
    CMScalar c_valid = 0;

    if (threadIdx.x < num_of_labels) {
      CMScalar c_intersection = tp[threadIdx.x];
      CMScalar c_union = rs[threadIdx.x] + cs[threadIdx.x] - c_intersection;

      if (c_union > 0) {
        c_iou = IoUScalar(c_intersection) / c_union;
        c_valid = 1;
      }
    }

    c_iou = warpReduceSum(c_iou);
    c_valid = warpReduceSum(c_valid);

    if (threadIdx.x==0)
      iou_loss[0] = IoUScalar(1) - (c_iou / c_valid);
  }
}

template<class ImageScalar, class CMScalar>
void cudaSegmConfusionMatrix(const int num_of_pixels,
                             const ImageScalar* const d_gt_image,
                             const ImageScalar* const d_pred_image,
                             const int num_of_labels,
                             CMScalar* d_conf_mat,
                             cudaStream_t stream) {
  assert(num_of_labels < 32);
  const int blocksize = 1024; // Minimum: num_of_labels_ * num_of_labels_
  assert(blocksize >= num_of_labels * num_of_labels);
  const int gridsize = (num_of_pixels + blocksize - 1) / blocksize;
  const int shared_mem_size = num_of_labels * num_of_labels * sizeof(CMScalar);

  confusion_matrix_ssa1d<<<gridsize, blocksize, shared_mem_size, stream>>>(num_of_pixels, d_gt_image, d_pred_image, num_of_labels, d_conf_mat);
}

template <class CMScalar, class IoUScalar>
void cudaMeanClassIoU(const int num_of_labels, const CMScalar* const d_conf_mat, IoUScalar* mean_io, cudaStream_t stream) {
  assert(num_of_labels < 32);
  dim3 blockdim(32, 32);
  const int smem_size = 32 * 3 * sizeof(CMScalar);
  mean_class_iou_from_confidence_matrix<<<1, blockdim, smem_size, stream>>>(num_of_labels, d_conf_mat, mean_io);
}

template<class ImageScalar, class CMScalar, class IoUScalar>
void cudaIoULoss(const int num_of_pixels,
                 const ImageScalar* const d_gt_image,
                 const ImageScalar* const d_pred_image,
                 const int num_of_labels,
                 CMScalar* d_conf_mat,
                 IoUScalar* iou_loss,
                 cudaStream_t stream) {
  assert(num_of_labels < 32);

  const size_t conf_mat_size = num_of_labels * num_of_labels * sizeof(CMScalar);

  {
    // Set Confusion Matrix to zero
    CHECK_CUDA(cudaMemsetAsync(d_conf_mat, 0, conf_mat_size, stream));
  }

  {
    // Compute Confusion Matrix
    const int blocksize = 1024; // Minimum: num_of_labels_ * num_of_labels_
    assert(blocksize >= num_of_labels * num_of_labels);
    const int gridsize = (num_of_pixels + blocksize - 1) / blocksize;
    confusion_matrix_ssa1d<<<gridsize, blocksize, conf_mat_size, stream>>>(num_of_pixels, d_gt_image, d_pred_image, num_of_labels, d_conf_mat);
  }

  {
    // Compute loss (1 - mean_iou)
    dim3 blockdim(32, 32);
    const int smem_size = 32 * 3 * sizeof(CMScalar);
    iou_loss_from_confidence_matrix<<<1, blockdim, smem_size, stream>>>(num_of_labels, d_conf_mat, iou_loss);
  }
}

// explicit instantiation.
template void cudaIoULoss<float, int, float>(const int num_of_pixels,
                                             const float* const d_gt_image,
                                             const float* const d_pred_image,
                                             const int num_of_labels,
                                             int* d_conf_mat,
                                             float* iou_loss,
                                             cudaStream_t stream);

template void cudaIoULoss<double, int, double>(const int num_of_pixels,
                                               const double* const d_gt_image,
                                               const double* const d_pred_image,
                                               const int num_of_labels,
                                               int* d_conf_mat,
                                               double* iou_loss,
                                               cudaStream_t stream);


}  // namespace RaC


