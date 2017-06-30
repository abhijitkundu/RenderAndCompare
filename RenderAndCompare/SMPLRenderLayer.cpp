/**
 * @file SMPLRenderLayer.cpp
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderLayer.h"
#include "CuteGL/Core/Config.h"
#include "CuteGL/Core/PoseUtils.h"
#include <glog/stl_logging.h>

namespace caffe {

template<typename Dtype>
void SMPLRenderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<< "---------- Setting up SMPLRenderLayer ------------";
  using namespace CuteGL;

  CHECK_EQ(top.size(), 1);
  CHECK_GE(bottom.size(), 4);

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
    CHECK_EQ(bottom[3]->shape(), blob_shape) << "bottom[2] is expected to be model poses with shape (num_frames, 4, 4)";
  }

  const Eigen::Array2i image_size(320, 240);
  const Eigen::Matrix3f K ((Eigen::Matrix3f() << 600.0, 0.0, 160.0, 0.0, 600.0, 120.0, 0.0, 0.0, 1.0).finished());

  std::vector<int> shape = {num_frames, 1, image_size.y(), image_size.x()};
  top[0]->Reshape(shape);

  renderer_.reset(new CuteGL::BatchSMPLRenderer);
  renderer_->setDisplayGrid(false);
  renderer_->setDisplayAxis(false);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(image_size.x(), image_size.y());

  viewer_->camera().intrinsics() = getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.01f, 100.0f);

  // Set camera pose via lookAt
  viewer_->setCameraToLookAt(Eigen::Vector3f(0.0f, 0.85f, 2.6f),
                             Eigen::Vector3f(0.0f, 0.85f, 0.0f),
                             Eigen::Vector3f::UnitY());

  LOG(INFO)<< "image_size= " << image_size.x() <<" x " << image_size.y();

  LOG(INFO)<< "Creating offscreen render surface";
  viewer_->create();
  viewer_->makeCurrent();

  LOG(INFO)<< "Adding SMPL data to renderer";
  renderer_->setSMPLData(num_frames, "smpl_neutral_lbs_10_207_0.h5", "vertex_segm24_col24_14.h5");

//  {
//    viewer_->fbo().bind();
//    GLuint rbo_id = viewer_->fbo().getRenderBufferObjectName(GL_COLOR_ATTACHMENT3);
//    viewer_->fbo().release();
//    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_gl_resource_, rbo_id, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
//  }

  LOG(INFO)<< "Generating PBOs";
  {
    cuda_pbo_resources_.resize(num_frames);
    pbo_ids_.resize(num_frames);
    pbo_ptrs_.resize(num_frames);

    const size_t pbo_size = image_size.x() * image_size.y() * sizeof(Dtype);

    // Create PBOS and register with CUDA
    viewer_->glFuncs()->glGenBuffers(num_frames, pbo_ids_.data());
    for (int p = 0; p < num_frames; ++p) {
      viewer_->glFuncs()->glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_ids_[p]);
      viewer_->glFuncs()->glBufferData(GL_PIXEL_PACK_BUFFER, pbo_size, 0, GL_DYNAMIC_READ);
      CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resources_[p], pbo_ids_[p], cudaGraphicsRegisterFlagsReadOnly));
      viewer_->glFuncs()->glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    // get device pointers to the pbos
    CHECK_CUDA(cudaGraphicsMapResources(num_frames, cuda_pbo_resources_.data(), 0));
    for (int p = 0; p < num_frames; ++p) {
      size_t buffer_size;
      CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void **)&pbo_ptrs_[p], &buffer_size, cuda_pbo_resources_[p]));
      assert(buffer_size == pbo_size);

    }
    CHECK_CUDA(cudaGraphicsUnmapResources(num_frames, cuda_pbo_resources_.data(), 0));
  }


  LOG(INFO)<< "------ Done Setting up SMPLRenderLayer -------";
}

template <typename Dtype>
void SMPLRenderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
void SMPLRenderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const int num_frames = top[0]->shape(0);

  const Dtype* shape_param_data = bottom[0]->cpu_data();
  const Dtype* pose_param_data = bottom[1]->cpu_data();
  const Dtype* camera_extrinsic_data = bottom[2]->cpu_data();
  const Dtype* model_pose_data = bottom[3]->cpu_data();
  Dtype* image_top_data = top[0]->mutable_cpu_data();

  using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;

  // Set shape and pose params
  renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(shape_param_data, bottom[0]->shape(1), num_frames).template cast<float>();
  for (int i = 0; i < num_frames; ++i) {
    using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
    VectorX full_pose(72);
    full_pose.setZero();
    full_pose.tail(69) = Eigen::Map<const VectorX>(pose_param_data + bottom[1]->offset(i), bottom[1]->shape(1));
    renderer_->smplDrawer().pose_params().col(i) = full_pose.template cast<float>();
  }

  // update VBOS
  renderer_->smplDrawer().updateShapeAndPose(true);

  // Do the rendering and copy back to CPU
  for (int i = 0; i < num_frames; ++i) {
    using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;

    viewer_->camera().extrinsics() = Eigen::Map<const Matrix4>(camera_extrinsic_data + bottom[2]->offset(i), 4, 4).template cast<float>();
    renderer_->modelPose() = Eigen::Map<const Matrix4>(model_pose_data + bottom[3]->offset(i), 4, 4).template cast<float>();
    renderer_->smplDrawer().batchId() = i;

    viewer_->render();

    viewer_->grabLabelBuffer((float*) (image_top_data + top[0]->offset(i, 0)));
  }
}

template<typename Dtype>
void SMPLRenderLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const int num_frames = top[0]->shape(0);
  const int height = top[0]->height();
  const int width = top[0]->width();

  const Dtype* shape_param_data = bottom[0]->cpu_data();
  const Dtype* pose_param_data = bottom[1]->cpu_data();
  const Dtype* camera_extrinsic_data = bottom[2]->cpu_data();
  const Dtype* model_pose_data = bottom[3]->cpu_data();

  Dtype* image_top_data = top[0]->mutable_gpu_data();

  using Params = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;

  // Set shape and pose params
  renderer_->smplDrawer().shape_params() = Eigen::Map<const Params>(shape_param_data, bottom[0]->shape(1), num_frames).template cast<float>();
  for (int i = 0; i < num_frames; ++i) {
    using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
    VectorX full_pose(72);
    full_pose.setZero();
    full_pose.tail(69) = Eigen::Map<const VectorX>(pose_param_data + bottom[1]->offset(i), bottom[1]->shape(1));
    renderer_->smplDrawer().pose_params().col(i) = full_pose.template cast<float>();
  }

  // update VBOS
  renderer_->smplDrawer().updateShapeAndPose();

  // Do the rendering
  {
    //  viewer_->makeCurrent();
    for (int i = 0; i < num_frames; ++i) {
      using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
      viewer_->camera().extrinsics() = Eigen::Map<const Matrix4>(camera_extrinsic_data + bottom[2]->offset(i), 4, 4).template cast<float>();
      renderer_->modelPose() = Eigen::Map<const Matrix4>(model_pose_data + bottom[3]->offset(i), 4, 4).template cast<float>();

      renderer_->smplDrawer().batchId() = i;

      viewer_->render();

      viewer_->glFuncs()->glBindFramebuffer(GL_READ_FRAMEBUFFER, viewer_->fbo().handle());
      viewer_->glFuncs()->glReadBuffer(GL_COLOR_ATTACHMENT3);
      viewer_->glFuncs()->glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_ids_[i]);
      viewer_->glFuncs()->glReadPixels(0, 0, width, height, GL_RED, CuteGL::GLTraits<Dtype>::type, 0);
      viewer_->glFuncs()->glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }
    viewer_->glFuncs()->glFinish();
    //  viewer_->doneCurrent();
  }

  // Copy from PBOS to CUDA data
  {
    std::vector<cudaStream_t> streams(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      CHECK_CUDA(cudaStreamCreate(&streams[i]));
      const size_t pbo_size = width * height * sizeof(Dtype);
      CHECK_CUDA(cudaMemcpyAsync(image_top_data + top[0]->offset(i, 0), pbo_ptrs_[i], pbo_size, cudaMemcpyDeviceToDevice, streams[i]));
      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
  }
}

INSTANTIATE_CLASS(SMPLRenderLayer);

}  // end namespace caffe
