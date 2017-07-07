/**
 * @file SMPLRenderWithLossLayer.cpp
 * @brief SMPLRenderWithLossLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderWithLossLayer.h"
#include "SegmentationAccuracy.h"
#include "NumericDiff.h"
#include <CuteGL/Core/PoseUtils.h>
#include <glog/stl_logging.h>
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
SMPLRenderWithLossLayer<Dtype>::~SMPLRenderWithLossLayer() {
  for (std::size_t i = 0; i < cuda_pbo_resources_.size(); ++i) {
    CHECK_CUDA(cudaGraphicsUnregisterResource (cuda_pbo_resources_[i]));
  }
  viewer_->glFuncs()->glDeleteBuffers(pbo_ids_.size(), pbo_ids_.data());
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<< "---------- Setting up BatchSMPLRenderWithLoss ------------";
  using namespace CuteGL;

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (top.size() == 2 && this->layer_param_.loss_weight_size() == 1) {
    this->layer_param_.add_loss_weight(Dtype(0));
  }


  CHECK_LE(top.size(), 2);
  CHECK_GT(top.size(), 0);
  CHECK_GE(bottom.size(), 5) << "Need 5 bottom layers: shape_param pose_param camera_extrinsic model_pose gt_segm_image";

  const int num_frames = bottom[0]->shape(0);
  LOG(INFO)<< "Number of frames = " << num_frames;

  {
    std::vector<int> blob_shape = {num_frames, 10};
    CHECK_EQ(bottom[0]->shape(), blob_shape) << "bottom[0] is expected to be shape params with shape (num_frames, 10)";
  }
  {
    std::vector<int> blob_shape = {num_frames, 69};
    CHECK_EQ(bottom[1]->shape(), blob_shape) << "bottom[1] is expected to be pose params with shape (num_frames, 69)";
  }
  {
    std::vector<int> blob_shape = {num_frames, 4, 4};
    CHECK_EQ(bottom[2]->shape(), blob_shape) << "bottom[2] is expected to be camera extrinsics with shape (num_frames, 4, 4)";
  }
  {
    std::vector<int> blob_shape = {num_frames, 4, 4};
    CHECK_EQ(bottom[3]->shape(), blob_shape) << "bottom[3] is expected to be model poses with shape (num_frames, 4, 4)";
  }

  const Eigen::Array2i image_size(320, 240);
  const Eigen::Matrix3f K ((Eigen::Matrix3f() << 600.0, 0.0, 160.0, 0.0, 600.0, 120.0, 0.0, 0.0, 1.0).finished());

  {
    std::vector<int> blob_shape = {num_frames, 1, image_size.y(), image_size.x()};
    CHECK_EQ(bottom[4]->shape(), blob_shape) << "bottom[4] is expected to gt_segm_image does not match render image size";
  }

  {
    // Allocate confusion matrices
    const int num_of_labels =  25;
    confusion_matrices_.Reshape({num_frames, num_of_labels, num_of_labels});
    CHECK_EQ(confusion_matrices_.count(), num_frames * num_of_labels * num_of_labels);
  }

  {
    losses_.Reshape({num_frames});
    CHECK_EQ(losses_.num_axes(), 1) << "losses_ is expected to be a vector of size same as batch size (num_frames)";
    CHECK_EQ(losses_.count(), num_frames) << "losses_ is expected to be a vector of size same as batch size (num_frames)";
  }

  {
    deltas_.Reshape({num_frames});
    CHECK_EQ(deltas_.num_axes(), 1) << "deltas_ is expected to be a vector of size same as batch size (num_frames)";
    CHECK_EQ(deltas_.count(), num_frames) << "deltas_ is expected to be a vector of size same as batch size (num_frames)";
  }

  {
    // Allocate Rendered image data
    rendered_images_.Reshape(num_frames, 1, image_size.y(), image_size.x());
    CHECK_EQ(rendered_images_.shape(), bottom[4]->shape()) << "rendered_images_ is expected to {num_frames, 1, H, W}";
  }

  renderer_.reset(new CuteGL::BatchSMPLRenderer);
  renderer_->setDisplayGrid(false);
  renderer_->setDisplayAxis(false);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(image_size.x(), image_size.y());

  viewer_->camera().intrinsics() = getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.01f, 100.0f);

  LOG(INFO)<< "image_size= " << image_size.x() <<" x " << image_size.y();

  LOG(INFO)<< "Creating offscreen render surface";
  viewer_->create();
  viewer_->makeCurrent();

  LOG(INFO)<< "Adding SMPL data to renderer";
  renderer_->setSMPLData(num_frames, "smpl_neutral_lbs_10_207_0.h5", "vertex_segm24_col24_14.h5");

  LOG(INFO)<< "Generating PBOs";
  {
    const size_t pbo_size = image_size.x() * image_size.y() * sizeof(Dtype);

    // Create PBOS
    pbo_ids_ = createGLBuffers(viewer_->glFuncs(), num_frames, GL_PIXEL_PACK_BUFFER, pbo_size, GL_DYNAMIC_READ);
    // register PBOS with CUDA
    cuda_pbo_resources_ = createCudaGLBufferResources(pbo_ids_, cudaGraphicsRegisterFlagsReadOnly);
    // get device pointers to the PBOS
    pbo_ptrs_ = createCudaGLBufferPointers<Dtype>(cuda_pbo_resources_, pbo_size);
  }

  {
    // Make sure cublas_pointer_mode set to CUBLAS_POINTER_MODE_HOST
    // See http://docs.nvidia.com/cuda/cublas/index.html#unique_1881057720
    cublasPointerMode_t cublas_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(Caffe::cublas_handle(), &cublas_pointer_mode));
    CHECK_EQ(cublas_pointer_mode, CUBLAS_POINTER_MODE_HOST);
  }

  LOG(INFO) << "------ Done Setting up BatchSMPLRenderWithLoss -------";
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom,
                                             const vector<Blob<Dtype> *>& top) {
  // bottom reshapes are done in LayerSetUp
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  // Rendered image output
  if (top.size() > 1) {
    top[1]->ReshapeLike(*bottom[4]);
  }
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom,
                                                 const vector<Blob<Dtype> *>& top) {
  const int num_frames = bottom[4]->num();

  // Set shape and pose params
  {
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();
    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;

    assert(bottom[1]->shape(1) == 69);
    assert(renderer_->smplDrawer().pose_params().rows() == 72);

    renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(shape_param_data_ptr, bottom[0]->shape(1), num_frames).template cast<float>();
    renderer_->smplDrawer().pose_params().topRows(3).setZero();
    renderer_->smplDrawer().pose_params().bottomRows(69) = Eigen::Map<const Params>(pose_param_data_ptr, bottom[1]->shape(1), num_frames).template cast<float>();
  }

  // update VBOS
  renderer_->smplDrawer().updateShapeAndPose();

  // Render and Compare (compute IoU loss). Also copy rendered images from PBOs
  renderAndCompareCPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_cpu_data());

  // Compute total Loss
  top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(num_frames, losses_.cpu_data()) / num_frames;

  if (top.size() > 1) {
    top[1]->ShareData(rendered_images_);
  }
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *>& bottom,
                                                 const vector<Blob<Dtype> *>& top) {
  const int num_frames = bottom[4]->num();

  // Set shape and pose params
  {
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();
    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;

    assert(bottom[1]->shape(1) == 69);
    assert(renderer_->smplDrawer().pose_params().rows() == 72);

    renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(shape_param_data_ptr, bottom[0]->shape(1), num_frames).template cast<float>();
    renderer_->smplDrawer().pose_params().topRows(3).setZero();
    renderer_->smplDrawer().pose_params().bottomRows(69) = Eigen::Map<const Params>(pose_param_data_ptr, bottom[1]->shape(1), num_frames).template cast<float>();
  }

  // update VBOS
  renderer_->smplDrawer().updateShapeAndPose();

  // Render and Compare (compute IoU loss). Also copy rendered images from PBOs if top.size() > 1
  renderAndCompareGPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_gpu_data(), top.size() > 1);

  // Wait for losses_ to be computed on GPU
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute total Loss
  Dtype loss_sum;
  caffe_gpu_asum(num_frames, losses_.gpu_data(), &loss_sum);
  top[0]->mutable_cpu_data()[0] = loss_sum / num_frames;

  if (top.size() > 1) {
    top[1]->ShareData(rendered_images_);
  }
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::renderToPBOs(const Blob<Dtype>& camera_extrisics, const Blob<Dtype>& model_poses) {
  const int num_frames = camera_extrisics.num();
  assert(model_poses.num() == num_frames);
  assert(model_poses.count() == num_frames * 16);
  assert(camera_extrisics.count() == num_frames * 16);
  assert(pbo_ids_.size() == num_frames);

  const Dtype* camera_extrinsic_data_ptr = camera_extrisics.cpu_data();
  const Dtype* model_pose_data_ptr = model_poses.cpu_data();
  for (int i = 0; i < num_frames; ++i) {
    using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
    viewer_->camera().extrinsics() = Eigen::Map<const Matrix4>(camera_extrinsic_data_ptr + i * 16, 4, 4).template cast<float>();
    renderer_->modelPose() = Eigen::Map<const Matrix4>(model_pose_data_ptr + i * 16, 4, 4).template cast<float>();
    renderer_->smplDrawer().batchId() = i;

    viewer_->render();
    viewer_->readLabelBuffer(CuteGL::GLTraits<Dtype>::type, pbo_ids_[i]);
  }
  viewer_->glFuncs()->glFinish();
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::renderAndCompareGPU(const Blob<Dtype>& camera_extrisics,
                                                         const Blob<Dtype>& model_poses,
                                                         const Blob<Dtype>& gt_segm_images,
                                                         Dtype* losses,
                                                         bool copy_rendered_images) {
  const int num_frames = gt_segm_images.num();
  const int height = gt_segm_images.height();
  const int width = gt_segm_images.width();

  // Do the Rendering
  renderToPBOs(camera_extrisics, model_poses);

  // compute IoU losses and if required Copy from PBOS to CUDA data
  {
    const Dtype* gt_segm_image_data_ptr = gt_segm_images.gpu_data();
    int* confusion_matrices_data_ptr = confusion_matrices_.mutable_gpu_data();
    Dtype* rendered_images_data_ptr = copy_rendered_images ? rendered_images_.mutable_gpu_data() : nullptr;

    const int image_size = width * height;
    const size_t pbo_size = image_size * sizeof(Dtype);
    const int num_of_labels = 25;

    std::vector<cudaStream_t> streams(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      CHECK_CUDA(cudaStreamCreate(&streams[i]));

      RaC::cudaIoULoss(image_size,
                       gt_segm_image_data_ptr + i * image_size,
                       pbo_ptrs_[i],
                       num_of_labels,
                       confusion_matrices_data_ptr + i * num_of_labels * num_of_labels,
                       losses + i,
                       streams[i]);

      if (rendered_images_data_ptr)
        CHECK_CUDA(cudaMemcpyAsync(rendered_images_data_ptr + i * image_size, pbo_ptrs_[i], pbo_size, cudaMemcpyDeviceToDevice, streams[i]));

      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
  }
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::renderAndCompareCPU(const Blob<Dtype>& camera_extrisics,
                                                         const Blob<Dtype>& model_poses,
                                                         const Blob<Dtype>& gt_segm_images,
                                                         Dtype* loss) {

  const int num_frames = gt_segm_images.num();
  const int height = gt_segm_images.height();
  const int width = gt_segm_images.width();

  // Do the Rendering
  renderToPBOs(camera_extrisics, model_poses);

  // Copy from PBOS to CPU
  {
    Dtype* rendered_images_data_ptr = rendered_images_.mutable_cpu_data();
    std::vector<cudaStream_t> streams(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      CHECK_CUDA(cudaStreamCreate(&streams[i]));
      const size_t pbo_size = width * height * sizeof(Dtype);
      CHECK_CUDA(cudaMemcpyAsync(rendered_images_data_ptr + rendered_images_.offset(i, 0), pbo_ptrs_[i], pbo_size, cudaMemcpyDeviceToHost, streams[i]));
      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // compute IoU losses
  {
    const Dtype* gt_segm_image_data_ptr = gt_segm_images.cpu_data();
    const Dtype* rendered_images_data_ptr = rendered_images_.cpu_data();
    const int image_data_stride = width * height;

#pragma omp parallel for
    for (int i = 0; i < num_frames; ++i) {
      using Image = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<const Image>gt_segm_image(gt_segm_image_data_ptr + i * image_data_stride, height, width);
      Eigen::Map<const Image>rendered_segm_image(rendered_images_data_ptr + i * image_data_stride, height, width);
      loss[i] = Dtype(1.0) - RaC::computeIoU(gt_segm_image, rendered_segm_image);
    }
  }

}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[4]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to gt_segm_image";
  if (propagate_down[3]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to model_pose";
  if (propagate_down[2]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to camera_extrinsic";

  const int num_frames = bottom[4]->num();
  const Dtype gradient_scale = top[0]->cpu_diff()[0] / num_frames;

  // backpropagate to shape params
  if (propagate_down[0]) {
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();

    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
    using RowVectorX = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;

    Eigen::Map<const Params> shape_params(shape_param_data_ptr, bottom[0]->shape(1), num_frames);

    renderer_->smplDrawer().shape_params() = shape_params.template cast<float>();
    for (int i = 0; i < num_frames; ++i) {
      VectorX full_pose(72);
      full_pose.setZero();
      full_pose.tail(69) = Eigen::Map<const VectorX>(pose_param_data_ptr + bottom[1]->offset(i), bottom[1]->shape(1));
      renderer_->smplDrawer().pose_params().col(i) = full_pose.template cast<float>();
    }


    Eigen::Map<Params> gradient(bottom[0]->mutable_cpu_diff(), bottom[0]->shape(1), bottom[0]->shape(0));
    for (Eigen::Index j = 0; j < gradient.rows(); ++j) {

//      const Dtype min_step_size = 1e-4;
//      const Dtype relative_step_size = 1e-1;
//      const RowVectorX hvec = (shape_params.row(j).cwiseAbs() * relative_step_size).cwiseMax(min_step_size);
      const RowVectorX hvec = RowVectorX::Constant(num_frames, 1e-2);

      RowVectorX f_plus(num_frames);
      // Compute F(X+h)
      {
        renderer_->smplDrawer().shape_params().row(j) += hvec.template cast<float>();
        renderer_->smplDrawer().updateShapeAndPose(); // update VBOS
        renderAndCompareCPU(*bottom[2], *bottom[3], *bottom[4], f_plus.data());
      }

      RowVectorX f_minus(num_frames);
      // Compute F(X-h)
      {
        renderer_->smplDrawer().shape_params().row(j) -= 2 * hvec.template cast<float>();
        renderer_->smplDrawer().updateShapeAndPose(); // update VBOS
        renderAndCompareCPU(*bottom[2], *bottom[3], *bottom[4], f_minus.data());  // TODO: Only perform shape update
      }

      // Compute central difference derivative
      gradient.row(j) = (f_plus - f_minus).cwiseQuotient(2 * hvec);

      // Reset shape param
      renderer_->smplDrawer().shape_params().row(j) = shape_params.row(j).template cast<float>();
    }

    // Scale gradient
    gradient *= gradient_scale;

//    {
//      LOG(INFO) << "shape.diff.cwiseAbs.mean() = " << gradient.cwiseAbs().mean();
//    }
  }

  // backpropagate to pose params
  if (propagate_down[1]) {
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();

    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVectorX = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;

    renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(shape_param_data_ptr, bottom[0]->shape(1), num_frames).template cast<float>();
    Eigen::Map<const Params> pose_params(pose_param_data_ptr, bottom[1]->shape(1), num_frames);
    renderer_->smplDrawer().pose_params().topRows(3).setZero();
    renderer_->smplDrawer().pose_params().bottomRows(69) = pose_params.template cast<float>();

    Eigen::Map<Params> gradient(bottom[1]->mutable_cpu_diff(), bottom[1]->shape(1), bottom[1]->shape(0));
    for (Eigen::Index j = 0; j < gradient.rows(); ++j) {

//      const Dtype min_step_size = 1e-4;
//      const Dtype relative_step_size = 1e-1;
//      const RowVectorX hvec = (pose_params.row(j).cwiseAbs() * relative_step_size).cwiseMax(min_step_size);
      const RowVectorX hvec = RowVectorX::Constant(num_frames, 1e-2);

      RowVectorX f_plus(num_frames);
      // Compute F(X+h)
      {
        renderer_->smplDrawer().pose_params().row(j+3) += hvec.template cast<float>();
        renderer_->smplDrawer().updateShapeAndPose(); // update VBOS
        renderAndCompareCPU(*bottom[2], *bottom[3], *bottom[4], f_plus.data());
      }

      RowVectorX f_minus(num_frames);
      // Compute F(X-h)
      {
        renderer_->smplDrawer().pose_params().row(j+3) -= 2 * hvec.template cast<float>();
        renderer_->smplDrawer().updateShapeAndPose(); // update VBOS
        renderAndCompareCPU(*bottom[2], *bottom[3], *bottom[4], f_minus.data());  // TODO: Only perform pose update
      }

      // Compute central difference derivative
      gradient.row(j) = (f_plus - f_minus).cwiseQuotient(2 * hvec);

      // Reset shape param
      renderer_->smplDrawer().pose_params().row(j+3) = pose_params.row(j).template cast<float>();
    }

    // Scale gradient
    gradient *= gradient_scale;

//    {
//      LOG(INFO) << "pose.diff.cwiseAbs.mean() = " << gradient.cwiseAbs().mean();
//    }
  }
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[4]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to gt_segm_image";
  if (propagate_down[3]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to model_pose";
  if (propagate_down[2]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to camera_extrinsic";

  const int num_frames = bottom[4]->num();
  const Dtype gradient_scale = top[0]->cpu_diff()[0] / num_frames;

  // backpropagate to shape params
  if (propagate_down[0]) {
    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVectorX = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;

    Eigen::Map<const Params> shape_params(bottom[0]->cpu_data(), bottom[0]->shape(1), num_frames);
    renderer_->smplDrawer().shape_params() = shape_params.template cast<float>();
    renderer_->smplDrawer().pose_params().topRows(3).setZero();
    renderer_->smplDrawer().pose_params().bottomRows(69) = Eigen::Map<const Params>(bottom[1]->cpu_data(), bottom[1]->shape(1), num_frames).template cast<float>();

    Dtype* shape_diff_ptr = bottom[0]->mutable_gpu_diff();
    Eigen::Map<RowVectorX> hvec(deltas_.mutable_cpu_data(), deltas_.shape(0));

    const int num_of_params = shape_params.rows();
    for (int j = 0; j < num_of_params; ++j) {

      const Dtype min_step_size = 1e-4;
      const Dtype relative_step_size = 1e-1;
      hvec = (shape_params.row(j).cwiseAbs() * relative_step_size).cwiseMax(min_step_size);

      // Compute F(X+h)
      {
        renderer_->smplDrawer().shape_params().row(j) += hvec.template cast<float>();

        if (j == 0)
          renderer_->smplDrawer().updateShapeAndPose(); // update ShapeAndPose only for 1st time
        else
          renderer_->smplDrawer().updateShape(); // only update shape for the rest of the time

        renderAndCompareGPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_gpu_data(), false);
      }

      // Compute F(X-h)
      {
        renderer_->smplDrawer().shape_params().row(j) -= 2 * hvec.template cast<float>();
        renderer_->smplDrawer().updateShape(); //  shape-only update is good enough
        renderAndCompareGPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_gpu_diff(), false);
      }

      const Dtype* hvec_data_ptr = deltas_.gpu_data();

      // Wait for losses_ to be computed on GPU
      CHECK_CUDA(cudaDeviceSynchronize());

      // Compute central difference derivative
      RaC::central_diff_gpu(num_frames, losses_.gpu_data(), losses_.gpu_diff(), hvec_data_ptr, shape_diff_ptr + j, num_of_params);

      // Reset shape param
      renderer_->smplDrawer().shape_params().row(j) = shape_params.row(j).template cast<float>();
    }

    // Scale gradient
    caffe_gpu_scal(num_frames * num_of_params, gradient_scale, shape_diff_ptr);

//    {
//      CHECK_CUDA(cudaDeviceSynchronize());
//      Eigen::Map<Params> gradient(bottom[0]->mutable_cpu_diff(), bottom[0]->shape(1), bottom[0]->shape(0));
//      LOG(INFO) << "shape.diff.cwiseAbs.mean() = " << gradient.cwiseAbs().mean();
//    }
  }

  // backpropagate to pose params
  if (propagate_down[1]) {
    using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    using RowVectorX = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;

    renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(bottom[0]->cpu_data(), bottom[0]->shape(1), num_frames).template cast<float>();
    Eigen::Map<const Params> pose_params(bottom[1]->cpu_data(), bottom[1]->shape(1), num_frames);
    renderer_->smplDrawer().pose_params().topRows(3).setZero();
    renderer_->smplDrawer().pose_params().bottomRows(69) = pose_params.template cast<float>();

    Dtype* pose_diff_ptr = bottom[1]->mutable_gpu_diff();
    Eigen::Map<RowVectorX> hvec(deltas_.mutable_cpu_data(), deltas_.shape(0));

    const int num_of_params = pose_params.rows();
    for (int j = 0; j < num_of_params; ++j) {

      const Dtype min_step_size = 1e-4;
      const Dtype relative_step_size = 1e-1;
      hvec = (pose_params.row(j).cwiseAbs() * relative_step_size).cwiseMax(min_step_size);

      // Compute F(X+h)
      {
        renderer_->smplDrawer().pose_params().row(j+3) += hvec.template cast<float>();
        if (j == 0)
           renderer_->smplDrawer().updateShapeAndPose(); // update ShapeAndPose only for 1st time
         else
           renderer_->smplDrawer().updatePose(); // only update pose for the rest of the time
        renderAndCompareGPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_gpu_data(), false);
      }

      // Compute F(X-h)
      {
        renderer_->smplDrawer().pose_params().row(j+3) -= 2 * hvec.template cast<float>();
        renderer_->smplDrawer().updatePose();   // pose-only update is good enough
        renderAndCompareGPU(*bottom[2], *bottom[3], *bottom[4], losses_.mutable_gpu_diff(), false);
      }

      const Dtype* hvec_data_ptr = deltas_.gpu_data();

      // Wait for losses_ to be computed on GPU
      CHECK_CUDA(cudaDeviceSynchronize());

      // Compute central difference derivative
      RaC::central_diff_gpu(num_frames, losses_.gpu_data(), losses_.gpu_diff(), hvec_data_ptr, pose_diff_ptr + j, num_of_params);

      // Reset shape param
      renderer_->smplDrawer().pose_params().row(j+3) = pose_params.row(j).template cast<float>();
    }

    // Scale gradient
    caffe_gpu_scal(num_frames * num_of_params, gradient_scale, pose_diff_ptr);

//    {
//      CHECK_CUDA(cudaDeviceSynchronize());
//      Eigen::Map<Params> gradient(bottom[1]->mutable_cpu_diff(), bottom[1]->shape(1), bottom[1]->shape(0));
//      LOG(INFO) << "pose.diff.cwiseAbs.mean() = " << gradient.cwiseAbs().mean();
//    }
  }

  // TODO: Do we really need this sync?
  CHECK_CUDA(cudaDeviceSynchronize());
}

INSTANTIATE_CLASS(SMPLRenderWithLossLayer);

}  // end namespace caffe


