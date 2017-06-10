/**
 * @file SegmAccuracyLayer.cu
 * @brief SegmAccuracyLayer
 *
 * @author Abhijit Kundu
 */

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


template <typename Dtype>
void SegmAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (reset_) {
    caffe_gpu_set(confidence_matrix_.count(), 0, confidence_matrix_.mutable_gpu_data());
  }

  const Dtype* pred_labels_data = bottom[0]->gpu_data();
  const Dtype* gt_labels_data = bottom[1]->gpu_data();
  const int* label_map_data = label_map_.gpu_data();

  int* confidence_matrix_data = confidence_matrix_.mutable_gpu_data();

  // Update confidence matrix
  {
    const int image_blob_size = bottom[0]->count();

    const int blocksize = 1024; // Minimum: num_of_labels_ * num_of_labels_
    const int gridsize = (image_blob_size + blocksize - 1) / blocksize;
    assert(blocksize >= num_of_labels_ * num_of_labels_);

    assert(num_of_labels_ <= 32);
    const int shared_mem_size = num_of_labels_ * num_of_labels_ * sizeof(int);

    if (label_map_.count())
      confusion_matrix_ssa1d<<<gridsize, blocksize, shared_mem_size>>>(num_of_labels_, gt_labels_data, pred_labels_data, image_blob_size, label_map_data, confidence_matrix_data);
    else
      confusion_matrix_ssa1d<<<gridsize, blocksize, shared_mem_size>>>(num_of_labels_, gt_labels_data, pred_labels_data, image_blob_size, confidence_matrix_data);
  }

  // TODO: Make Full GPU

  using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<RowMajorMatrixXi> confidence_matrix(confidence_matrix_.mutable_cpu_data(), num_of_labels_, num_of_labels_);

  const Eigen::ArrayXi true_positives = confidence_matrix.diagonal();
  const Eigen::ArrayXi gt_pixels = confidence_matrix.rowwise().sum();
  const Eigen::ArrayXi pred_pixels = confidence_matrix.colwise().sum();


  for (std::size_t i =0; i < metrics_.size(); ++i) {
    switch (metrics_[i]) {
      case SegmAccuracyParameter_AccuracyMetric_PixelAccuracy:
        top[i]->mutable_cpu_data()[0] = Dtype(true_positives.sum()) / gt_pixels.sum();
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassAccuracy:
        top[i]->mutable_cpu_data()[0] = (true_positives.cast<Dtype>() / pred_pixels.cast<Dtype>()).mean();
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassIoU:
        top[i]->mutable_cpu_data()[0] = (true_positives.cast<Dtype>() / (gt_pixels + pred_pixels - true_positives).cast<Dtype>()).mean();
        break;
      default:
          LOG(FATAL) << "Unknown Accuracy metric.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SegmAccuracyLayer);

}  // namespace caffe
