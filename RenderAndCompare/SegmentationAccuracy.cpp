/**
 * @file SegmentationAccuracy.cpp
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#include "SegmentationAccuracy.h"

namespace RaC {

float computeIoU(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images, const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images) {
  if (gt_images.dimensions() != pred_images.dimensions())
    throw std::runtime_error("Dimension mismatch: gt_images.dimensions() ! = pred_images.dimensions()");
  const Eigen::Index images_per_blob = gt_images.dimension(0);

  float mean_iou = 0;
  for (Eigen::Index i = 0; i < images_per_blob; ++i) {
    using Image8UC1 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const Image8UC1> gt_image(&gt_images(i, 0, 0, 0), 240, 320);
    Eigen::Map<const Image8UC1> pred_image(&pred_images(i, 0, 0, 0), 240, 320);
    mean_iou += computeIoU(gt_image, pred_image);
  }
  mean_iou /= images_per_blob;
  return mean_iou;
}

}  // namespace RaC


