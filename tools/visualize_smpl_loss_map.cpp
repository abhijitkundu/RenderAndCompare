/**
 * @file visualize_smpl_loss_map.cpp
 * @brief visualize_smpl_loss_map
 *
 * @author Abhijit Kundu
 */

#include "CuteGL/Renderer/SMPLRenderer.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"
#include "CuteGL/Core/PoseUtils.h"
#include "RenderAndCompare/SegmentationAccuracy.h"
#include "RenderAndCompare/ImageUtils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QApplication>

int main(int argc, char **argv) {
  QApplication app(argc, argv);

  using namespace CuteGL;
  using Eigen::Vector3f;

  // Create the Renderer
  std::unique_ptr<SMPLRenderer> renderer(new SMPLRenderer);
  renderer->setDisplayGrid(false);
  renderer->setDisplayAxis(false);

  OffScreenRenderViewer viewer(renderer.get());
  viewer.setBackgroundColor(0, 0, 0);

  const int W = 320;
  const int H = 240;
  viewer.resize(W, H);

  const Eigen::Matrix3f K(
      (Eigen::Matrix3f() << 600.f, 0.f, 160.f, 0.f, 600.f, 120.f, 0.f, 0.f, 1.f).finished());
  viewer.camera().intrinsics() = getGLPerspectiveProjection(K, W, H, 0.01f, 100.0f);

  // Set camera pose via lookAt
  viewer.setCameraToLookAt(Vector3f(0.0f, 0.85f, 6.0f), Vector3f(0.0f, 0.85f, 0.0f), Vector3f::UnitY());

  viewer.create();
  viewer.makeCurrent();

  renderer->setSMPLData("smpl_neutral_lbs_10_207_0.h5", "vertex_segm24_col24_14.h5");

  using Image32FC1 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Image8UC1 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Image32FC1 gt_segm_image(H, W);
  Image32FC1 rendered_segm_image(H, W);

  Eigen::VectorXf gt_shape_param(10);
  Eigen::VectorXf gt_pose_param(69);
  gt_shape_param.setRandom();
  gt_pose_param.setRandom();

  {
    renderer->smplDrawer().shape() = gt_shape_param;
    renderer->smplDrawer().pose() = gt_pose_param;
    renderer->smplDrawer().updateShapeAndPose();
    viewer.render();
    viewer.readLabelBuffer(gt_segm_image.data());
  }

  {
    const std::vector<cv::Vec3b> smpl24_cmap = { { 55, 55, 55 }, { 47, 148, 84 }, { 231, 114, 177 }, { 89, 70,
        0 }, { 187, 43, 143 }, { 7, 70, 70 }, { 251, 92, 0 }, { 63, 252, 211 }, { 53, 144, 229 }, { 248, 150,
        144 }, { 52, 82, 125 }, { 189, 73, 1 }, { 42, 210, 52 }, { 253, 192, 40 }, { 231, 233, 157 }, { 109,
        131, 203 }, { 190, 195, 243 }, { 97, 70, 171 }, { 32, 137, 233 }, { 68, 43, 29 }, { 142, 35, 220 }, {
        243, 169, 53 }, { 119, 8, 153 }, { 217, 181, 152 }, { 32, 91, 213 } };
    Image8UC1 image = gt_segm_image.cast<unsigned char>();
    cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    cv::flip(cv_image, cv_image, 0);  // Can be done with Eigen tensor reverse also
    cv::imshow("gt_segm_image", RaC::getColoredImageFromLabels(cv_image, smpl24_cmap));
    cv::waitKey(1);
  }

  const Eigen::Index half_pixels = 1000;
  const float half_range = 1.0f;
  const float step_size = half_range / half_pixels;

  {
    Image32FC1 cost_map(10, 2 * half_pixels + 1);

    std::cout << "step_size = " <<  step_size << std::endl;
    std::cout << "Range = [" << -half_range << ", " << -half_range + (cost_map.cols() - 1) * step_size << "]" << std::endl;
    std::cout << "Rendering .. " << std::flush;
    for (Eigen::Index i = 0; i < cost_map.rows(); ++i) {
      for (Eigen::Index j = 0; j < cost_map.cols(); ++j) {
        renderer->smplDrawer().shape() = gt_shape_param;
        renderer->smplDrawer().pose() = gt_pose_param;

        renderer->smplDrawer().shape()[i] += -half_range + j * step_size;

        renderer->smplDrawer().updateShapeAndPose();
        viewer.render();
        viewer.readLabelBuffer(rendered_segm_image.data());
        cost_map(i, j) = 1.0f - RaC::computeIoU(gt_segm_image, rendered_segm_image);
      }
    }
    std::cout << "Done" << std::endl;

    {
       cv::Mat cv_image(cost_map.rows(), cost_map.cols(), CV_32FC1, cost_map.data());
       std::string name = "shape_cost_map_" + std::to_string(step_size);
       cv::imwrite(name + ".exr", cv_image);
       cv::imshow(name, cv_image);
       cv::waitKey(1);
    }
  }

  {
    Image32FC1 cost_map(69, 2 * half_pixels + 1);

    std::cout << "step_size = " <<  step_size << std::endl;
    std::cout << "Range = [" << -half_range << ", " << -half_range + (cost_map.cols() - 1) * step_size << "]" << std::endl;

    std::cout << "Rendering .. " << std::flush;
    for (Eigen::Index i = 0; i < cost_map.rows(); ++i) {
      for (Eigen::Index j = 0; j < cost_map.cols(); ++j) {
        renderer->smplDrawer().shape() = gt_shape_param;
        renderer->smplDrawer().pose() = gt_pose_param;

        renderer->smplDrawer().pose()[i] += -half_range + j * step_size;

        renderer->smplDrawer().updateShapeAndPose();
        viewer.render();
        viewer.readLabelBuffer(rendered_segm_image.data());
        cost_map(i, j) = 1.0f - RaC::computeIoU(gt_segm_image, rendered_segm_image);
      }
    }
    std::cout << "Done" << std::endl;

    {
       cv::Mat cv_image(cost_map.rows(), cost_map.cols(), CV_32FC1, cost_map.data());
       std::string name = "pose_cost_map_" + std::to_string(step_size);
       cv::imwrite(name + ".exr", cv_image);
       cv::imshow(name, cv_image);
       cv::waitKey(1);
    }
  }
  cv::waitKey(0);

  return EXIT_SUCCESS;
}



