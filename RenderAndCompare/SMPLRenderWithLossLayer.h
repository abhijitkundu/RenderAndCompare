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
  using Tensor4 = Eigen::Tensor<Dtype, 4, Eigen::RowMajor>;

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

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


  std::unique_ptr<CuteGL::SMPLRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;

  Tensor4 rendered_images_;
};

}  // end namespace caffe

#endif // end CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_
