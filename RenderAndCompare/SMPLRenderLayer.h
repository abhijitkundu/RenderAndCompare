/**
 * @file SMPLRenderLayer.h
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_
#define RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_

#include "RenderAndCompare/SMPLMeshRenderer.h" // Should be included 1st
#include "RenderAndCompare/SMPLmodel.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"

#include <vector>
#include <memory>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class SMPLRenderLayer : public Layer<Dtype> {
 public:
  explicit SMPLRenderLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "SMPLRender";
  }
  virtual inline int MinBottomBlobs() const {return 4;}
  virtual inline int ExactNumTopBlobs() const {return 1;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  std::unique_ptr<CuteGL::SMPLMeshRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;

  using SMPL = RaC::SMPLmodel<float>;
  SMPL smpl_;
};

}  // end namespace caffe

#endif // end RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_
