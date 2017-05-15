#ifndef CAFFE_RENDER_LAYER_HPP_
#define CAFFE_RENDER_LAYER_HPP_

#include <vector>
#include <memory>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "CuteGL/Surface/OffScreenRenderViewer.h"
#include "CuteGL/Renderer/MultiObjectRenderer.h"


namespace caffe {

template <typename Dtype>
class RenderLayer : public Layer<Dtype> {
 public:

  explicit RenderLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Render"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  std::unique_ptr<CuteGL::MultiObjectRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;
};

}  // namespace caffe

#endif  // CAFFE_RENDER_LAYER_HPP_
