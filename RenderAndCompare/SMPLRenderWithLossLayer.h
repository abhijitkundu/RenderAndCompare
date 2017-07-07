/**
 * @file SMPLRenderWithLossLayer.h
 * @brief SMPLRenderWithLossLayer
 *
 * @author Abhijit Kundu
 */

#ifndef CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_
#define CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_

#include "CuteGL/Renderer/BatchSMPLRenderer.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"
#include "CuteGL/Utils/CudaUtils.h"

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
  explicit SMPLRenderWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
  }

  virtual ~SMPLRenderWithLossLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {return "BatchSMPLRenderWithLoss";}
  virtual inline int ExactNumBottomBlobs() const {return 5;}

  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  /// We can force backward for only shape_param and pose_param layers
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return (bottom_index == 0) || (bottom_index == 1);
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  void renderToPBOs(const Blob<Dtype>& camera_extrisics, const Blob<Dtype>& model_poses);

  void renderAndCompareCPU(const Blob<Dtype>& camera_extrisics,
                           const Blob<Dtype>& model_poses,
                           const Blob<Dtype>& gt_segm_images,
                           Dtype* losses);

  void renderAndCompareGPU(const Blob<Dtype>& camera_extrisics,
                           const Blob<Dtype>& model_poses,
                           const Blob<Dtype>& gt_segm_images,
                           Dtype* losses, bool copy_rendered_images = false);

  std::unique_ptr<CuteGL::BatchSMPLRenderer> renderer_;
  std::unique_ptr<CuteGL::OffScreenRenderViewer> viewer_;

  vector<GLuint> pbo_ids_;
  vector<cudaGraphicsResource*> cuda_pbo_resources_;
  vector<Dtype*> pbo_ptrs_;

  Blob<int> confusion_matrices_;
  Blob<Dtype> losses_;
  Blob<Dtype> deltas_;
  Blob<Dtype> rendered_images_;
};

}  // end namespace caffe

#endif // end CAFFE_SMPL_RENDER_WITH_LOSS_LAYER_H_
