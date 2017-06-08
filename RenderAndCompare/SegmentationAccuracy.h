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
#include <chrono>
#include <iostream>
#include "RenderAndCompare/CudaHelper.h"

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

template<typename DG, typename DP >
void computeSegHistsCPU(const Eigen::MatrixBase<DG>& gt_image, const Eigen::MatrixBase<DP>& pred_image) {
  const int num_of_labels = 25;
  Eigen::VectorXi total_pixels_class(num_of_labels);
  Eigen::VectorXi ok_pixels_class(num_of_labels);
  Eigen::VectorXi label_pixels(num_of_labels);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

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

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "CPU Time = " << elapsed_seconds.count() * 1000 << " ms.  ";

  std::cout << "mean_iou = " << mean_iou << "\n";
}

void compute_seg_histograms(const uint8_t* const gt_image,
                            const uint8_t* const pred_image,
                            int width, int height);

void compute_confusion_matrix(const uint8_t* const gt_image,
                              const uint8_t* const pred_image,
                              int width, int height);

void compute_confusion_tensor(const uint8_t* const gt_image,
                              const uint8_t* const pred_image,
                              int width, int height);

void compute_cmat_warped_iou(const uint8_t*
                             const gt_image,
                             const uint8_t* const pred_image,
                             int width, int height);

void compute_warped_cmat_warped_iou(const uint8_t* const gt_image,
                                    const uint8_t* const pred_image,
                                    int width,
                                    int height);


void computeHistogramWithAtomics(const uint8_t* const image, int width, int height, int *hist, int num_labels);
void computeHistogramWithSharedAtomics(const uint8_t* const image, int width, int height, int *hist, int num_labels);
void computeHistogramWithSharedBins(const uint8_t* const image, int width, int height, int *hist, int num_labels);
void computeHistogramWithPrivateBins(const uint8_t* const image, int width, int height, int *hist, int num_labels);
void computeHistogramWithThrust(const uint8_t* const image, int width, int height, int *hist, int num_labels);


template <class Derived>
Eigen::VectorXi computeHistogramWithCPU(const Eigen::MatrixBase<Derived>& image) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  Eigen::VectorXi hist(25);
  hist.setZero();

  for (int y = 0; y < image.rows(); ++y)
    for (int x = 0; x < image.cols(); ++x) {
      int label = static_cast<int>(image(y, x));
      ++hist[label];
    }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Time = " << elapsed_seconds.count() * 1000 << " ms\n";

  return hist;
}

void computeIoUwithCUDAseq(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images, const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images, const int trials = 3);
void computeIoUwithCUDApar(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images, const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images, const int trials = 3);
void computeIoUwithCUDAstreams(const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& gt_images, const Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>& pred_images, const int trials = 3);



}  // namespace RaC

#endif // end RENDERANDCOMPARE_SEGMENTATION_ACCURACY_H_
