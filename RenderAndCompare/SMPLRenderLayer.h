/**
 * @file SMPLRenderLayer.h
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_
#define RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_

#include "CuteGL/Renderer/BatchSMPLRenderer.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"
#include "CuteGL/Utils/CudaUtils.h"

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
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    for (std::size_t i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
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

 private:
  std::unique_ptr<CuteGL::BatchSMPLRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;

  vector<cudaGraphicsResource*> cuda_pbo_resources_;
  vector<GLuint> pbo_ids_;
  vector<Dtype*> pbo_ptrs_;
};

}  // end namespace caffe

#endif // end RENDERANDCOMPARE_SMPL_RENDER_LAYER_H_
