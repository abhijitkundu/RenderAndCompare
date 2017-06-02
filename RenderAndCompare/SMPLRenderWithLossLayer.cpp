/**
 * @file SMPLRenderWithLossLayer.cpp
 * @brief SMPLRenderWithLossLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderWithLossLayer.h"
#include "CuteGL/Core/PoseUtils.h"
#include <glog/stl_logging.h>

namespace caffe {

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {

  LOG(INFO)<< "Setting up SMPLRenderWithLoss";
  using namespace CuteGL;

  LossLayer<Dtype>::LayerSetUp(bottom, top);


  CHECK_EQ(top.size(), 1);
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

  // Allocate Rendered image data
  rendered_images_.resize(num_frames, 1, image_size.y(), image_size.x());

  renderer_.reset(new CuteGL::SMPLRenderer);
  renderer_->setDisplayGrid(false);
  renderer_->setDisplayAxis(false);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(image_size.x(), image_size.y());

  viewer_->camera().intrinsics() = getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.01f, 100.0f);

  LOG(INFO)<< "image_size= " << image_size.x() <<" x " << image_size.y();
  LOG(INFO)<< "K=\n" << K;

  LOG(INFO)<< "Creating offscreen render surface";
  viewer_->create();
  viewer_->makeCurrent();

  LOG(INFO)<< "Adding SMPL data to renderer";
  renderer_->setSMPLData("smpl_neutral_lbs_10_207_0.h5", "vertex_segm24_col24_14.h5");

  LOG(INFO)<< "Done Setting up SMPLRenderWithLoss";
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom,
                                             const vector<Blob<Dtype> *>& top) {
  // bottom reshapes are done in LayerSetUp
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom,
                                                 const vector<Blob<Dtype> *>& top) {
  const Eigen::Index num_frames = rendered_images_.dimension(0);
  const Eigen::Index height = rendered_images_.dimension(2);
  const Eigen::Index width = rendered_images_.dimension(3);

  // Render at current params
  {
    const Dtype* shape_param_data = bottom[0]->cpu_data();
    const Dtype* pose_param_data = bottom[1]->cpu_data();
    const Dtype* camera_extrinsic_data = bottom[2]->cpu_data();
    const Dtype* model_pose_data = bottom[3]->cpu_data();

    viewer_->makeCurrent();
    for (Eigen::Index i = 0; i < num_frames; ++i) {
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

      viewer_->grabLabelBuffer((float*) (&rendered_images_(i, 0, 0, 0)));
    }
    viewer_->doneCurrent();
  }
  // Compute Loss
  {
    const Dtype* gt_labels_data = bottom[4]->cpu_data();
    using Tensor4Const = Eigen::Tensor<const Dtype, 4, Eigen::RowMajor>;
    Eigen::TensorMap<Tensor4Const> gt_segm_images(gt_labels_data, num_frames, 1, height, width);

    const int num_of_labels = 25;
    Eigen::VectorXi total_pixels_class(num_of_labels);
    Eigen::VectorXi ok_pixels_class(num_of_labels);
    Eigen::VectorXi label_pixels(num_of_labels);

    total_pixels_class.setZero();
    ok_pixels_class.setZero();
    label_pixels.setZero();

    for (Eigen::Index index = 0; index < rendered_images_.size(); ++index) {
      const int pred_label = static_cast<int>(rendered_images_(index));
      const int gt_label = static_cast<int>(gt_segm_images(index));

      ++total_pixels_class[gt_label];
      ++label_pixels[pred_label];

      if (gt_label == pred_label) {
        ++ok_pixels_class[gt_label];
      }
    }

    Dtype mean_iou = 0;
    for (Eigen::Index i = 0; i < num_of_labels; ++i) {
      int union_pixels = total_pixels_class[i] + label_pixels[i] - ok_pixels_class[i];
      if (union_pixels > 0)  {
        Dtype class_iou = Dtype(ok_pixels_class[i]) / union_pixels;
        mean_iou += class_iou;
      }
    }
    mean_iou /= num_of_labels;

    top[0]->mutable_cpu_data()[0] = (Dtype(1.0) - mean_iou);
  }
}

INSTANTIATE_CLASS(SMPLRenderWithLossLayer);

}  // end namespace caffe
