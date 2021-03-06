#ifndef CAFFE_SEGM_ACCURACY_LAYER_H_
#define CAFFE_SEGM_ACCURACY_LAYER_H_

#include <thrust/device_vector.h>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include <Eigen/Core>
#include <vector>

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class SegmAccuracyLayer : public Layer<Dtype> {
 public:

  explicit SegmAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SegmAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- SegmAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    for (const auto prop_down : propagate_down) {
      if (prop_down) {
        NOT_IMPLEMENTED;
      }
    }
  }

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    for (const auto prop_down : propagate_down) {
      if (prop_down) {
        NOT_IMPLEMENTED;
      }
    }
  }

  int num_of_labels_;
  bool reset_;


  Eigen::VectorXi ignored_labels_;
  std::vector<SegmAccuracyParameter_AccuracyMetric> metrics_;

  // Label Map used
  Blob<int> label_map_;

  // TODO use unsigned int type
  Blob<int> confidence_matrix_;
};

}  // namespace caffe

#endif  // CAFFE_SEGM_ACCURACY_LAYER_H_
