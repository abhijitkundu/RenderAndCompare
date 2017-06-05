/**
 * @file SegmentationAccuracy.h
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SEGMENTATION_ACCURACY_H_
#define RENDERANDCOMPARE_SEGMENTATION_ACCURACY_H_

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

namespace RaC {

float computeIoU(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images, const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images);

template<typename DG, typename DP >
float computeIoU(const Eigen::MatrixBase<DG>& gt_image, const Eigen::MatrixBase<DP>& pred_image) {
  const int num_of_labels = 25;
  Eigen::VectorXi total_pixels_class(num_of_labels);
  Eigen::VectorXi ok_pixels_class(num_of_labels);
  Eigen::VectorXi label_pixels(num_of_labels);

  total_pixels_class.setZero();
  ok_pixels_class.setZero();
  label_pixels.setZero();

  if (pred_image.size() != gt_image.size())
    throw std::runtime_error("pred_image.size() != gt_image.size()");

  for (Eigen::Index index = 0; index < gt_image.size(); ++index) {
    const int pred_label = static_cast<int>(pred_image(index));
    const int gt_label = static_cast<int>(gt_image(index));

    ++total_pixels_class[gt_label];
    ++label_pixels[pred_label];

    if (gt_label == pred_label) {
      ++ok_pixels_class[gt_label];
    }
  }

  float mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < num_of_labels; ++i) {
    int union_pixels = total_pixels_class[i] + label_pixels[i] - ok_pixels_class[i];
    if (union_pixels > 0)  {
      float class_iou = float(ok_pixels_class[i]) / union_pixels;
      mean_iou += class_iou;
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;
  return mean_iou;
}

}  // namespace RaC

#endif // end RENDERANDCOMPARE_SEGMENTATION_ACCURACY_H_
