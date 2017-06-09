/**
 * @file SegmAccuracyLayer.cpp
 * @brief SegmAccuracyLayer
 *
 * @author Abhijit Kundu
 */

#include "SegmAccuracyLayer.h"
#include "caffe/util/math_functions.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace caffe {

template <typename Dtype>
void SegmAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const SegmAccuracyParameter segm_accuracy_param = this->layer_param_.segm_accuracy_param();
  num_of_labels_ = segm_accuracy_param.num_of_labels();
  reset_ = segm_accuracy_param.reset();

  label_map_.Reshape({segm_accuracy_param.label_map_size()});
  {
    int* label_map_data = label_map_.mutable_cpu_data();
    for (int i = 0; i < segm_accuracy_param.label_map_size(); ++i)
      label_map_data[i] = segm_accuracy_param.label_map(i);
  }

  ignored_labels_.resize(segm_accuracy_param.ignored_labels_size());
  for (int i = 0; i < segm_accuracy_param.ignored_labels_size(); ++i) {
    ignored_labels_[i] = segm_accuracy_param.ignored_labels(i);
  }

  if (label_map_.count())  {
    Eigen::Map<const Eigen::VectorXi> label_map(label_map_.cpu_data(), label_map_.count());
    CHECK_EQ(label_map.minCoeff(), 0);
    CHECK_EQ(label_map.maxCoeff(), num_of_labels_ -  1);
  }

  CHECK_EQ(top.size(), segm_accuracy_param.metrics_size()) << "Number of tops should be same as number of metrics";
  metrics_.resize(segm_accuracy_param.metrics_size());
  for (int i = 0; i < segm_accuracy_param.metrics_size(); ++i) {
    metrics_[i] = segm_accuracy_param.metrics(i);
  }
  CHECK_EQ(top.size(), metrics_.size());

  // Initialize confidence matrix
  confidence_matrix_.Reshape({num_of_labels_, num_of_labels_});
  caffe_set(confidence_matrix_.count(), 0, confidence_matrix_.mutable_cpu_data());

  LOG(INFO) << "num_of_labels = " << num_of_labels_;
  LOG(INFO) << "Reset every pass = " << std::boolalpha << reset_;
  LOG(INFO) << "label_map_.shape() = " << label_map_.shape_string();
  LOG(INFO) << "confidence_matrix_.shape() = " << confidence_matrix_.shape_string();
  LOG(INFO) << "ignored_labels_.size() = " << ignored_labels_.size();
}

template <typename Dtype>
void SegmAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes());
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());

  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[1]->num_axes(), 4);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());


  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  for (std::size_t i =0; i < top.size(); ++i)
    top[i]->Reshape(top_shape);
}

template <typename Dtype>
void SegmAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (reset_) {
    caffe_set(confidence_matrix_.count(), 0, confidence_matrix_.mutable_cpu_data());
  }

  const Dtype* pred_labels_data = bottom[0]->cpu_data();
  const Dtype* gt_labels_data = bottom[1]->cpu_data();
  const int* label_map_data = label_map_.cpu_data();

  const int num = bottom[0]->num();
  assert(bottom[0]->channels() == 1);
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<RowMajorMatrixXi> confidence_matrix(confidence_matrix_.mutable_cpu_data(), num_of_labels_, num_of_labels_);

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h)
      for (int w = 0; w < width; ++w) {
        const int index = h * width + w;

        int pred_label;
        int gt_label;

        if (label_map_.count()) {
          // TODO assert bound check
          pred_label = label_map_data[static_cast<int>(pred_labels_data[index])];
          gt_label = label_map_data[static_cast<int>(gt_labels_data[index])];
        }
        else {
          // TODO assert bound check
          pred_label = static_cast<int>(pred_labels_data[index]);
          gt_label = static_cast<int>(gt_labels_data[index]);
        }


        if (gt_label < 0 || gt_label >= num_of_labels_) {
          LOG(FATAL) << "Unexpected GT label " << gt_label
              << ". num: " << i  << ". row: " << h << ". col: " << w;
        }
        if (pred_label < 0 || pred_label >= num_of_labels_) {
          LOG(FATAL) << "Unexpected pred label " << pred_label
              << ". num: " << i  << ". row: " << h << ". col: " << w;
        }

        ++confidence_matrix(gt_label, pred_label);
      }
    pred_labels_data  += bottom[0]->offset(1);
    gt_labels_data += bottom[1]->offset(1);
  }

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

INSTANTIATE_CLASS(SegmAccuracyLayer);

}  // namespace caffe


