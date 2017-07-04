/**
 * @file SegmentationAccuracy.h
 * @brief SegmentationAccuracy
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SEGMENTATIONACCURACY_H_
#define RENDERANDCOMPARE_SEGMENTATIONACCURACY_H_

#include "CuteGL/Utils/CudaUtils.h"
#include <Eigen/Core>

namespace RaC {

template<class ImageScalar, class CMScalar, class IoUScalar>
void cudaIoULoss(const int num_of_pixels,
                 const ImageScalar* const d_gt_image,
                 const ImageScalar* const d_pred_image,
                 const int num_of_labels,
                 CMScalar* d_conf_mat,
                 IoUScalar* iou_loss,
                 cudaStream_t stream = 0);

template <class DI>
typename DI::Scalar computeIoU(const Eigen::MatrixBase<DI>& gt_image, const Eigen::MatrixBase<DI>& rendered_image) {
  using Scalar = typename DI::Scalar;
  const int num_of_labels = 25;
  Eigen::VectorXi total_pixels_class(num_of_labels);
  Eigen::VectorXi ok_pixels_class(num_of_labels);
  Eigen::VectorXi label_pixels(num_of_labels);

  total_pixels_class.setZero();
  ok_pixels_class.setZero();
  label_pixels.setZero();

  assert(gt_image.maxCoeff() < 25);
  assert(gt_image.minCoeff() >= 0);
  assert(rendered_image.maxCoeff() < 25);
  assert(rendered_image.minCoeff() >= 0);

  for (Eigen::Index index = 0; index < rendered_image.size(); ++index) {
    const int pred_label = static_cast<int>(rendered_image(index));
    const int gt_label = static_cast<int>(gt_image(index));

    ++total_pixels_class[gt_label];
    ++label_pixels[pred_label];

    if (gt_label == pred_label) {
      ++ok_pixels_class[gt_label];
    }
  }

  Scalar mean_iou = 0;
  int valid_labels = 0;
  for (Eigen::Index i = 0; i < num_of_labels; ++i) {
    int union_pixels = total_pixels_class[i] + label_pixels[i] - ok_pixels_class[i];
    if (union_pixels > 0)  {
      Scalar class_iou = Scalar(ok_pixels_class[i]) / union_pixels;
      mean_iou += class_iou;
      ++valid_labels;
    }
  }
  mean_iou /= valid_labels;
  return mean_iou;
}

}  // namespace RaC

#endif // end RENDERANDCOMPARE_SEGMENTATIONACCURACY_H_
