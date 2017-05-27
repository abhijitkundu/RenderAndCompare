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
#include <random>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ArticulatedObjectsDataLayer : public Layer<Dtype> {
 public:
  using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
  using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
  using Vector3 = Eigen::Matrix<Dtype, 3, 1>;

  explicit ArticulatedObjectsDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        batch_size_(-1),
        shape_param_size_(-1),
        pose_param_size_(-1),
        curr_data_idx_(0),
        rand_engine_() {
    std::random_device rd;
    rand_engine_.seed(rd());
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  void addDataset(const RaC::Dataset& dataset);

  void generateDatumIds();

  const Vector3& mean_bgr() const {return mean_bgr_;}

 protected:

  RaC::BatchImageLoader<uint8_t, 3> input_image_loader_;
  RaC::BatchImageLoader<uint8_t, 1> segm_image_loader_;
  Eigen::AlignedStdVector<VectorX> shape_params_;
  Eigen::AlignedStdVector<VectorX> pose_params_;
  Eigen::AlignedStdVector<Matrix4> camera_extrinsics_;
  Eigen::AlignedStdVector<Matrix4> model_poses_;

  int batch_size_;
  int shape_param_size_;
  int pose_param_size_;
  Vector3 mean_bgr_;
  std::vector<std::string> top_names_;
  std::size_t curr_data_idx_;
  std::vector<std::size_t> data_ids_;
  std::mt19937 rand_engine_;
};

}  // end namespace caffe

#endif // end RENDERANDCOMPARE_ARTICULATEDOBJECTSDATALAYER_H_
