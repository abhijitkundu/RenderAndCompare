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

  label_map_.resize(segm_accuracy_param.label_map_size());
  for (int i = 0; i < segm_accuracy_param.label_map_size(); ++i) {
    label_map_[i] = segm_accuracy_param.label_map(i);
  }

  ignored_labels_.resize(segm_accuracy_param.ignored_labels_size());
  for (int i = 0; i < segm_accuracy_param.ignored_labels_size(); ++i) {
    ignored_labels_[i] = segm_accuracy_param.ignored_labels(i);
  }

  if (label_map_.size())  {
    CHECK_EQ(label_map_.minCoeff(), 0);
    CHECK_EQ(label_map_.maxCoeff(), num_of_labels_ -  1);
  }

  CHECK_EQ(top.size(), segm_accuracy_param.metrics_size()) << "Number of tops should be same as number of metrics";
  metrics_.resize(segm_accuracy_param.metrics_size());
  for (int i = 0; i < segm_accuracy_param.metrics_size(); ++i) {
    metrics_[i] = segm_accuracy_param.metrics(i);
  }
  CHECK_EQ(top.size(), metrics_.size());

  LOG(INFO) << "num_of_labels = " << num_of_labels_;
  LOG(INFO) << "Reset every pass = " << std::boolalpha << reset_;
  LOG(INFO) << "label_map_.size() = " << label_map_.size();
  LOG(INFO) << "ignored_labels_.size() = " << ignored_labels_.size();

  total_pixels_class_.resize(num_of_labels_);
  ok_pixels_class_.resize(num_of_labels_);
  label_pixels_.resize(num_of_labels_);

  total_pixels_class_.setZero();
  ok_pixels_class_.setZero();
  label_pixels_.setZero();

  total_pixels_ = 0;
  ok_pixels_ = 0;
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
  const Dtype* pred_labels_data = bottom[0]->cpu_data();
  const Dtype* gt_labels_data = bottom[1]->cpu_data();


  if (reset_) {
    total_pixels_class_.setZero();
    ok_pixels_class_.setZero();
    label_pixels_.setZero();

    total_pixels_ = 0;
    ok_pixels_ = 0;
  }

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h)
      for (int w = 0; w < width; ++w) {
        const int index = h * width + w;

        int pred_label;
        int gt_label;

        if (label_map_.size()) {
          pred_label = label_map_[static_cast<int>(pred_labels_data[index])];
          gt_label = label_map_[static_cast<int>(gt_labels_data[index])];
        }
        else {
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

        ++total_pixels_;
        ++total_pixels_class_[gt_label];

        ++label_pixels_[pred_label];

        if (gt_label == pred_label) {
          ++ok_pixels_;
          ++ok_pixels_class_[gt_label];
        }

      }
    pred_labels_data  += bottom[0]->offset(1);
    gt_labels_data += bottom[1]->offset(1);
  }

//  { // For Debug purposes
//    using Vector = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
//    Vector class_ious = 100.0 * ok_pixels_class_.cast<Dtype>().array()
//          / (total_pixels_class_ + label_pixels_ - ok_pixels_class_).cast<Dtype>().array();
//    Dtype mean_class_iou = class_ious.mean();
//
//    Vector class_pixel_accs = 100.0 * ok_pixels_class_.cast<Dtype>().array() / label_pixels_.cast<Dtype>().array();
//    Dtype mean_class_pixel_acc = class_pixel_accs.mean();
//
//    Dtype global_pixel_acc = Dtype(ok_pixels_ * 100.0) / total_pixels_;
//
//    LOG(INFO) << "class_iou= " << mean_class_iou << " class_pixel_acc= " << mean_class_pixel_acc << " global_pixel_acc= " << global_pixel_acc;
//  }

  for (std::size_t i =0; i < metrics_.size(); ++i) {
    switch (metrics_[i]) {
      case SegmAccuracyParameter_AccuracyMetric_PixelAccuracy:
        top[i]->mutable_cpu_data()[0] = Dtype(ok_pixels_) / total_pixels_;
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassAccuracy:
        top[i]->mutable_cpu_data()[0] = (ok_pixels_class_.cast<Dtype>().array() / label_pixels_.cast<Dtype>().array()).mean();
        break;
      case SegmAccuracyParameter_AccuracyMetric_ClassIoU:
        top[i]->mutable_cpu_data()[0] = (ok_pixels_class_.cast<Dtype>().array()
            / (total_pixels_class_ + label_pixels_ - ok_pixels_class_).cast<Dtype>().array()).mean();
        break;
      default:
          LOG(FATAL) << "Unknown Accuracy metric.";
    }
  }

}

INSTANTIATE_CLASS(SegmAccuracyLayer);

}  // namespace caffe


