/**
 * @file SegmAccuracyLayer.cu
 * @brief SegmAccuracyLayer
 *
 * @author Abhijit Kundu
 */

#include "SegmAccuracyLayer.h"
#include "SegmAccuracyLayer.h"

namespace caffe {

template<typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_ssa1d(const int num_of_labels,
                                       const ImageScalar* const gt_image,
                                       const ImageScalar* const pred_image,
                                       const int image_data_size,
                                       CMScalar* d_conf_mat) {
  const int num_of_bins = num_of_labels * num_of_labels;
  // Initialize shared mem
  extern __shared__ CMScalar smem[];
  if (threadIdx.x < num_of_bins)
    smem[threadIdx.x] = 0;
  __syncthreads();

  // stride length
  const int stride = blockDim.x * gridDim.x;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_data_size; i += stride) {
    int gt_label = static_cast<int>(gt_image[i]);
    int pred_label = static_cast<int>(pred_image[i]);
    atomicAdd(&smem[gt_label * num_of_labels + pred_label] , 1 );
  }
  __syncthreads();

  if (threadIdx.x < num_of_bins)
    atomicAdd(&(d_conf_mat[threadIdx.x]), smem[threadIdx.x]);
}

template<typename ImageScalar, typename CMScalar>
__global__ void confusion_matrix_ssa1d(const int num_of_labels,
                                       const ImageScalar* const gt_image,
                                       const ImageScalar* const pred_image,
                                       const int image_data_size,
                                       const CMScalar* const label_map,
                                       CMScalar* d_conf_mat) {
  const int num_of_bins = num_of_labels * num_of_labels;
  // Initialize shared mem
  extern __shared__ CMScalar smem[];
  if (threadIdx.x < num_of_bins)
    smem[threadIdx.x] = 0;
  __syncthreads();

  // stride length
  const int stride = blockDim.x * gridDim.x;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_data_size; i += stride) {
    int gt_label = label_map[static_cast<int>(gt_image[i])];
    int pred_label = label_map[static_cast<int>(pred_image[i])];
    atomicAdd(&smem[gt_label * num_of_labels + pred_label] , 1 );
  }
  __syncthreads();

  if (threadIdx.x < num_of_bins)
    atomicAdd(&(d_conf_mat[threadIdx.x]), smem[threadIdx.x]);
}

template <typename Scalar>
__inline__ __device__
Scalar warpReduceSum(Scalar val) {
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

__global__ void histogram_from_confidence_matrix(const int num_of_labels,
                                                 const int* const conf_mat,
                                                 int* tp_hist,
                                                 int* gt_hist,
                                                 int* pred_hist) {
  int row_sum = 0;
  int col_sum = 0;
  if ((threadIdx.x < num_of_labels) && (threadIdx.y < num_of_labels)) {
    row_sum = conf_mat[threadIdx.y * num_of_labels + threadIdx.x];
    col_sum = conf_mat[threadIdx.x * num_of_labels + threadIdx.y];

    if (threadIdx.x == threadIdx.y) {
      tp_hist[threadIdx.x] = col_sum;
    }
  }

  row_sum = warpReduceSum(row_sum);
  col_sum = warpReduceSum(col_sum);

  if (threadIdx.x == 0 && (threadIdx.y < num_of_labels)) {
    gt_hist[threadIdx.y] = row_sum;
    pred_hist[threadIdx.y] = col_sum;
  }
}

template <typename HistScalar, typename Scalar>
__global__ void mean_class_iou_from_histogram(const int num_of_labels,
                                              const HistScalar* const tp_hist,
                                              const HistScalar* const gt_hist,
                                              const HistScalar* const pred_hist,
                                              Scalar* mean_class_iou) {
  Scalar c_iou = 0;
  if (threadIdx.x < num_of_labels) {
    c_iou = Scalar(tp_hist[threadIdx.x]) / (gt_hist[threadIdx.x] + pred_hist[threadIdx.x]- tp_hist[threadIdx.x]);
  }
  c_iou = warpReduceSum(c_iou);

  if (threadIdx.x==0)
    mean_class_iou[0] = c_iou / num_of_labels;
}

template <typename HistScalar, typename Scalar>
__global__ void mean_class_acc_from_histogram(const int num_of_labels,
                                              const HistScalar* const tp_hist,
                                              const HistScalar* const pred_hist,
                                              Scalar* mean_class_acc) {
  float c_acc = 0;
  if (threadIdx.x < num_of_labels) {
    c_acc = Scalar(tp_hist[threadIdx.x]) / pred_hist[threadIdx.x];
  }
  c_acc = warpReduceSum(c_acc);

  if (threadIdx.x==0)
    mean_class_acc[0] = c_acc / num_of_labels;
}

template <typename HistScalar, typename Scalar>
__global__ void global_pixel_acc_from_histogram(const int num_of_labels,
                                                const HistScalar* const tp_hist,
                                                const HistScalar* const gt_hist,
                                                Scalar* global_pixel_acc) {
  int tp_sum = 0;
  int gt_sum = 0;
  if (threadIdx.x < num_of_labels) {
    tp_sum = tp_hist[threadIdx.x];
    gt_sum = gt_hist[threadIdx.x];
  }
  tp_sum = warpReduceSum(tp_sum);
  gt_sum = warpReduceSum(gt_sum);

  if (threadIdx.x==0)
    global_pixel_acc[0] = Scalar(tp_sum) / gt_sum;
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


template <typename Dtype>
void SegmAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (reset_) {
    caffe_gpu_set(confidence_matrix_.count(), 0, confidence_matrix_.mutable_gpu_data());
  }

  CHECK_LE(num_of_labels_, 32) << "GPU implementation does not support more than 32 labels";

  const Dtype* pred_labels_data = bottom[0]->gpu_data();
  const Dtype* gt_labels_data = bottom[1]->gpu_data();
  const int* label_map_data = label_map_.gpu_data();

  int* confidence_matrix_data = confidence_matrix_.mutable_gpu_data();

  // Since confidence_matrix_.mutable_gpu_diff() is not used we repurpose it for histogram
  int* tp_hist = confidence_matrix_.mutable_gpu_diff();
  int* gt_hist = tp_hist + num_of_labels_;
  int* pred_hist = tp_hist + 2 * num_of_labels_;

  // Update confidence matrix
  {
    const int image_blob_size = bottom[0]->count();

    const int blocksize = 1024; // Minimum: num_of_labels_ * num_of_labels_
    const int gridsize = (image_blob_size + blocksize - 1) / blocksize;
    assert(blocksize >= num_of_labels_ * num_of_labels_);
    const int shared_mem_size = num_of_labels_ * num_of_labels_ * sizeof(int);

    if (label_map_.count())
      confusion_matrix_ssa1d<<<gridsize, blocksize, shared_mem_size>>>(num_of_labels_, gt_labels_data, pred_labels_data, image_blob_size, label_map_data, confidence_matrix_data);
    else
      confusion_matrix_ssa1d<<<gridsize, blocksize, shared_mem_size>>>(num_of_labels_, gt_labels_data, pred_labels_data, image_blob_size, confidence_matrix_data);
  }

  // Compute histograms from conf matrix
  {
    dim3 blockdim(32, 32);
    histogram_from_confidence_matrix<<<1, blockdim>>>(num_of_labels_, confidence_matrix_data, tp_hist, gt_hist, pred_hist);
  }

  for (std::size_t i =0; i < metrics_.size(); ++i) {
    switch (metrics_[i]) {
      case SegmAccuracyParameter_AccuracyMetric_PixelAccuracy:
        global_pixel_acc_from_histogram<<<1, 32>>>(num_of_labels_, tp_hist, gt_hist, top[i]->mutable_gpu_data());
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassAccuracy:
        mean_class_acc_from_histogram<<<1, 32>>>(num_of_labels_, tp_hist, pred_hist, top[i]->mutable_gpu_data());
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassIoU:
        mean_class_iou_from_histogram<<<1, 32>>>(num_of_labels_, tp_hist, gt_hist, pred_hist, top[i]->mutable_gpu_data());
        break;
      default:
          LOG(FATAL) << "Unknown Accuracy metric.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SegmAccuracyLayer);

}  // namespace caffe
