/**
 * @file DataLayer.h
 * @brief DataLayer
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_ARTICULATEDOBJECTSDATALAYER_H_
#define RENDERANDCOMPARE_ARTICULATEDOBJECTSDATALAYER_H_

#include "RenderAndCompare/EigenTypedefs.h"
#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ImageLoaders.h"
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ArticulatedObjectsDataLayer : public Layer<Dtype> {
 public:
  using Vector10 = Eigen::Matrix<Dtype, 10, 1>;
  using Matrix4 = Eigen::Matrix<Dtype, 4, 4>;

  explicit ArticulatedObjectsDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  void addDataset(const RaC::Dataset& dataset);

 protected:

  RaC::BatchImageLoader image_loader_;
  Eigen::AlignedStdVector<Vector10> shape_params_;
  Eigen::AlignedStdVector<Vector10> pose_params_;
  Eigen::AlignedStdVector<Matrix4> camera_extrinsics_;
  Eigen::AlignedStdVector<Matrix4> model_poses_;
};

}  // end namespace caffe

#endif // end RENDERANDCOMPARE_ARTICULATEDOBJECTSDATALAYER_H_
