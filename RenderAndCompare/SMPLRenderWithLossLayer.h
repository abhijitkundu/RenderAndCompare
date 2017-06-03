/**
 * @file SMPLRenderWithLossLayer.h
 * @brief SMPLRenderWithLossLayer
 *
 * @author Abhijit Kundu
 */

#ifndef CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_
#define CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_

#include "CuteGL/Renderer/SMPLRenderer.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include <vector>
#include <memory>

namespace caffe {

template<typename Dtype>
class SMPLRenderWithLossLayer : public LossLayer<Dtype> {
 public:
  using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
  using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
  using Image = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  SMPLRenderWithLossLayer();

  explicit SMPLRenderWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "SMPLRenderWithLoss";
  }
  virtual inline int ExactNumBottomBlobs() const {return 5;}
  virtual inline int ExactNumTopBlobs() const {return 1;}

  /// We can force backward for only shape_param and pose_param layers
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return (bottom_index == 0) || (bottom_index == 1);
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  template <class DS, class DP, class DC, class DM, class DI>
  Dtype renderAndCompare(const Eigen::MatrixBase<DS>& shape_param,
                         const Eigen::MatrixBase<DP>& pose_param,
                         const Eigen::MatrixBase<DC>& camera_extrinsic,
                         const Eigen::MatrixBase<DM>& model_pose,
                         const Eigen::MatrixBase<DI>& gt_image);

  std::unique_ptr<CuteGL::SMPLRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;

  VectorX losses_;
  Image rendered_image_;
};

}  // end namespace caffe

#endif // end CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_
