/**
 * @file SMPLRenderWithLossLayer.cpp
 * @brief SMPLRenderWithLossLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderWithLossLayer.h"
#include "CuteGL/Core/PoseUtils.h"
#include <glog/stl_logging.h>
#include <Eigen/NumericalDiff>

namespace caffe {

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {

  LOG(INFO)<< "Setting up SMPLRenderWithLoss";
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

  // Allocate Rendered image data
  rendered_image_.resize(image_size.y(), image_size.x());
  losses_.resize(num_frames);

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

  // Rendered image output
  if (top.size() > 1) {
    top[1]->ReshapeLike(*bottom[4]);
  }
}

template<typename Dtype>
template <class DS, class DP, class DC, class DM, class DI>
Dtype SMPLRenderWithLossLayer<Dtype>::renderAndCompare(const Eigen::MatrixBase<DS>& shape_param,
                                                       const Eigen::MatrixBase<DP>& pose_param,
                                                       const Eigen::MatrixBase<DC>& camera_extrinsic,
                                                       const Eigen::MatrixBase<DM>& model_pose,
                                                       const Eigen::MatrixBase<DI>& gt_image) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DS);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DP);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DC, 4, 4);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DM, 4, 4);

  viewer_->camera().extrinsics() = camera_extrinsic.template cast<float>();
  renderer_->modelPose() = model_pose.template cast<float>();

  renderer_->smplDrawer().shape() = shape_param.template cast<float>();
  renderer_->smplDrawer().pose().tail(69) = pose_param.template cast<float>();

  renderer_->smplDrawer().updateShapeAndPose();

  viewer_->render();
  viewer_->grabLabelBuffer((float*) rendered_image_.data());

  Dtype loss;
  {
    // TODO: Move this to member data?
    const int num_of_labels = 25;
    Eigen::VectorXi total_pixels_class(num_of_labels);
    Eigen::VectorXi ok_pixels_class(num_of_labels);
    Eigen::VectorXi label_pixels(num_of_labels);

    total_pixels_class.setZero();
    ok_pixels_class.setZero();
    label_pixels.setZero();

    for (Eigen::Index index = 0; index < rendered_image_.size(); ++index) {
      const int pred_label = static_cast<int>(rendered_image_(index));
      const int gt_label = static_cast<int>(gt_image(index));

      ++total_pixels_class[gt_label];
      ++label_pixels[pred_label];

      if (gt_label == pred_label) {
        ++ok_pixels_class[gt_label];
      }
    }

    Dtype mean_iou = 0;
    int valid_labels = 0;
    for (Eigen::Index i = 0; i < num_of_labels; ++i) {
      int union_pixels = total_pixels_class[i] + label_pixels[i] - ok_pixels_class[i];
      if (union_pixels > 0)  {
        Dtype class_iou = Dtype(ok_pixels_class[i]) / union_pixels;
        mean_iou += class_iou;
        ++valid_labels;
      }
    }
    mean_iou /= valid_labels;
    loss = (Dtype(1.0) - mean_iou);
  }
  return loss;
}

template<typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom,
                                                 const vector<Blob<Dtype> *>& top) {
  const Eigen::Index num_frames = losses_.size();
//  const Eigen::Index height = rendered_image_.rows();
//  const Eigen::Index width = rendered_image_.cols();

  {
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();
    const Dtype* camera_extrinsic_data_ptr = bottom[2]->cpu_data();
    const Dtype* model_pose_data_ptr = bottom[3]->cpu_data();
    const Dtype* gt_segm_image_data_ptr = bottom[4]->cpu_data();

    Dtype* rendered_image_data_ptr = nullptr;
    if (top.size() > 1) {
      rendered_image_data_ptr = top[1]->mutable_cpu_data();
    }

    viewer_->makeCurrent();
    for (Eigen::Index i = 0; i < num_frames; ++i) {
      Eigen::Map<const VectorX>shape_param(shape_param_data_ptr + bottom[0]->offset(i), bottom[0]->shape(1));
      Eigen::Map<const VectorX>pose_param(pose_param_data_ptr + bottom[1]->offset(i), bottom[1]->shape(1));
      Eigen::Map<const Matrix4>camera_extrinsic(camera_extrinsic_data_ptr + bottom[2]->offset(i), 4, 4);
      Eigen::Map<const Matrix4>model_pose(model_pose_data_ptr + bottom[3]->offset(i), 4, 4);
      Eigen::Map<const Image>gt_segm_image(gt_segm_image_data_ptr + bottom[4]->offset(i), rendered_image_.rows(), rendered_image_.cols());

      losses_[i] = renderAndCompare(shape_param, pose_param, camera_extrinsic, model_pose, gt_segm_image);

      if (rendered_image_data_ptr) {
        Eigen::Map<Image>(rendered_image_data_ptr + top[1]->offset(i), rendered_image_.rows(), rendered_image_.cols()) = rendered_image_;
      }
    }
    viewer_->doneCurrent();
  }

  // Compute total Loss
  top[0]->mutable_cpu_data()[0] = losses_.mean();
}


template <typename Dtype>
void SMPLRenderWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[4]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to gt_segm_image";
  if (propagate_down[3]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to model_pose";
  if (propagate_down[2]) LOG(FATAL) << this->type() << " Layer cannot backpropagate to camera_extrinsic";

  const Eigen::Index num_frames = losses_.size();
  const Dtype gradient_scale = top[0]->cpu_diff()[0] / num_frames;

  // TODO Remove:
  CHECK_EQ(top[0]->cpu_diff()[0], 1.0);
  CHECK_EQ(bottom[0]->shape(0), num_frames);
  CHECK_EQ(bottom[1]->shape(0), num_frames);

  // backpropagate to shape params
  if (propagate_down[0]) {
    Dtype* shape_param_diff_ptr = bottom[0]->mutable_cpu_diff();
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();
    const Dtype* camera_extrinsic_data_ptr = bottom[2]->cpu_data();
    const Dtype* model_pose_data_ptr = bottom[3]->cpu_data();
    const Dtype* gt_segm_image_data_ptr = bottom[4]->cpu_data();

    viewer_->makeCurrent();
    for (Eigen::Index i = 0; i < num_frames; ++i) {
      // Compute shape gradients of frame i

      Eigen::Map<const VectorX>shape_param(shape_param_data_ptr + bottom[0]->offset(i), bottom[0]->shape(1));
      Eigen::Map<const VectorX>pose_param(pose_param_data_ptr + bottom[1]->offset(i), bottom[1]->shape(1));
      Eigen::Map<const Matrix4>camera_extrinsic(camera_extrinsic_data_ptr + bottom[2]->offset(i), 4, 4);
      Eigen::Map<const Matrix4>model_pose(model_pose_data_ptr + bottom[3]->offset(i), 4, 4);
      Eigen::Map<const Image>gt_segm_image(gt_segm_image_data_ptr + bottom[4]->offset(i), rendered_image_.rows(), rendered_image_.cols());

      Eigen::Map<VectorX>gradient(shape_param_diff_ptr + bottom[0]->offset(i), bottom[0]->shape(1));

      for (Eigen::Index j = 0; j < gradient.size(); ++j) {
        const Dtype min_step_size = 1e-4;
        const Dtype relative_step_size = 1e-1;

        const Dtype step_size = std::abs(shape_param[j]) * relative_step_size;
        const Dtype h = std::max(min_step_size, step_size);

        const Dtype two_h = 2 * h;

        VectorX parameters_plus_h = shape_param;
        parameters_plus_h[j] += h;

        const Dtype f_plus = renderAndCompare(parameters_plus_h, pose_param, camera_extrinsic, model_pose, gt_segm_image);

        parameters_plus_h[j] -= two_h;
        const Dtype f_minus = renderAndCompare(parameters_plus_h, pose_param, camera_extrinsic, model_pose, gt_segm_image);

        gradient[j] = (f_plus - f_minus) / two_h;
      }
    }
    viewer_->doneCurrent();

    // Scale gradient
    using MatrixX = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixX>shape_param_diff(shape_param_diff_ptr, bottom[0]->shape(0), bottom[0]->shape(1));
    shape_param_diff *= gradient_scale;
  }

  // backpropagate to pose params
  if (propagate_down[1]) {
    Dtype* pose_param_diff_ptr = bottom[1]->mutable_cpu_diff();
    const Dtype* shape_param_data_ptr = bottom[0]->cpu_data();
    const Dtype* pose_param_data_ptr = bottom[1]->cpu_data();
    const Dtype* camera_extrinsic_data_ptr = bottom[2]->cpu_data();
    const Dtype* model_pose_data_ptr = bottom[3]->cpu_data();
    const Dtype* gt_segm_image_data_ptr = bottom[4]->cpu_data();

    viewer_->makeCurrent();
    for (Eigen::Index i = 0; i < num_frames; ++i) {
      // Compute pose gradients of frame i

      Eigen::Map<const VectorX>shape_param(shape_param_data_ptr + bottom[0]->offset(i), bottom[0]->shape(1));
      Eigen::Map<const VectorX>pose_param(pose_param_data_ptr + bottom[1]->offset(i), bottom[1]->shape(1));
      Eigen::Map<const Matrix4>camera_extrinsic(camera_extrinsic_data_ptr + bottom[2]->offset(i), 4, 4);
      Eigen::Map<const Matrix4>model_pose(model_pose_data_ptr + bottom[3]->offset(i), 4, 4);
      Eigen::Map<const Image>gt_segm_image(gt_segm_image_data_ptr + bottom[4]->offset(i), rendered_image_.rows(), rendered_image_.cols());

      Eigen::Map<VectorX>gradient(pose_param_diff_ptr + bottom[1]->offset(i), bottom[1]->shape(1));

      for (Eigen::Index j = 0; j < gradient.size(); ++j) {
        const Dtype min_step_size = 1e-4;
        const Dtype relative_step_size = 1e-1;

        const Dtype step_size = std::abs(pose_param[j]) * relative_step_size;
        const Dtype h = std::max(min_step_size, step_size);

        const Dtype two_h = 2 * h;

        VectorX parameters_plus_h = pose_param;
        parameters_plus_h[j] += h;

        const Dtype f_plus = renderAndCompare(shape_param, parameters_plus_h, camera_extrinsic, model_pose, gt_segm_image);

        parameters_plus_h[j] -= two_h;
        const Dtype f_minus = renderAndCompare(shape_param, parameters_plus_h, camera_extrinsic, model_pose, gt_segm_image);

        gradient[j] = (f_plus - f_minus) / two_h;
      }
    }
    viewer_->doneCurrent();

    // Scale gradient
    using MatrixX = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatrixX>pose_param_diff(pose_param_diff_ptr, bottom[1]->shape(0), bottom[1]->shape(1));
    pose_param_diff *= gradient_scale;
  }
}


INSTANTIATE_CLASS(SMPLRenderWithLossLayer);

}  // end namespace caffe
