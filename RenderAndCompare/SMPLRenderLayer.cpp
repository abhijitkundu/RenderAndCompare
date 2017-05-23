/**
 * @file SMPLRenderLayer.cpp
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderLayer.h"
#include "H5EigenDense.h"
#include "H5EigenTensor.h"

#include "CuteGL/Core/Config.h"
#include <CuteGL/Core/PoseUtils.h>

namespace caffe {

template<typename Dtype>
void SMPLRenderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<< "Setting up SMPLRenderLayer";
  using namespace CuteGL;

  CHECK_EQ(top.size(), 1);
  CHECK_GE(bottom.size(), 4);

  const int num_frames = bottom[0]->shape(0);
  for (const auto& blob : bottom) {
    CHECK_EQ(blob->shape(0), num_frames);
  }

  const Eigen::Array2i image_size(320, 240);
  const Eigen::Matrix3f K ((Eigen::Matrix3f() << 600.0, 0.0, 160.0, 0.0, 600.0, 120.0, 0.0, 0.0, 1.0).finished());

  std::vector<int> shape = {num_frames, 1, image_size.y(), image_size.x()};
  top[0]->Reshape(shape);

  renderer_.reset(new CuteGL::SMPLRenderer);
  renderer_->setDisplayGrid(false);
  renderer_->setDisplayAxis(false);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(image_size.x(), image_size.y());

  viewer_->camera().intrinsics() = getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.01f, 100.0f);

  LOG(INFO)<< "image_size= " << image_size;
  LOG(INFO)<< "K=\n" << K;

  LOG(INFO)<< "Creating offscreen render surface";
  viewer_->create();
  viewer_->makeCurrent();

  LOG(INFO)<< "Adding SMPL data to renderer";
  renderer_->setSMPLData("smpl_neutral_lbs_10_207_0.h5");

  LOG(INFO)<< "Done Setting up SMPLRenderLayer";
}

template <typename Dtype>
void SMPLRenderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SMPLRenderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num_frames = top[0]->shape(0);
  const int height = top[0]->height();
  const int width = top[0]->width();

   const Dtype* shape_param_data = bottom[0]->cpu_data();
   const Dtype* pose_param_data = bottom[1]->cpu_data();
   const Dtype* camera_extrinsic_data = bottom[2]->cpu_data();
   const Dtype* model_pose_data = bottom[3]->cpu_data();
   Dtype* image_top_data = top[0]->mutable_cpu_data();

   viewer_->makeCurrent();
   for (int i = 0; i < num_frames; ++i) {
     using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
     using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;

     viewer_->camera().extrinsics() = Eigen::Map<const Matrix4>(camera_extrinsic_data + bottom[2]->offset(i), 4, 4).template cast<float>();
     renderer_->modelPose() = Eigen::Map<const Matrix4>(model_pose_data + bottom[3]->offset(i), 4, 4).template cast<float>();

     renderer_->smplDrawer().shape() = Eigen::Map<const VectorX>(shape_param_data + bottom[0]->offset(i), bottom[0]->shape(1)).template cast<float>();

     VectorX full_pose (72);
     full_pose.setZero();
     full_pose.tail(69) = Eigen::Map<const VectorX>(pose_param_data + bottom[1]->offset(i), bottom[1]->shape(1));
     renderer_->smplDrawer().pose() = full_pose.template cast<float>();

     renderer_->smplDrawer().updateShapeAndPose();

     viewer_->render();

    using Image = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Image> image(image_top_data + top[0]->offset(i, 0), height, width);
    viewer_->grabDepthBuffer((float*) image.data());
   }
   viewer_->doneCurrent();
}

INSTANTIATE_CLASS(SMPLRenderLayer);

}  // end namespace caffe
