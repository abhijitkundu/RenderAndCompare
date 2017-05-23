/**
 * @file ImageUtils.h
 * @brief ImageUtils
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_IMAGEUTILS_H_
#define RENDERANDCOMPARE_IMAGEUTILS_H_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace RaC {

template<int _Rows, int _Cols>
void saveImage(const Eigen::Matrix<unsigned char, _Rows, _Cols, Eigen::RowMajor>& image, const std::string& filename) {
  cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, (void*)image.data());
  cv::imwrite(filename, cv_image);
}

template<class Derived, class Scalar = double>
void saveImage(const Eigen::MatrixBase<Derived>& image, const std::string& filename, const Scalar scale = 255.0) {
  using Image8UC1 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Image8UC1 image_8UC1 = (image.array() * scale).template cast<unsigned char>();
  image_8UC1.colwise().reverseInPlace();
  saveImage(image_8UC1, filename);
}

}  // namespace RaC

#endif // end RENDERANDCOMPARE_IMAGEUTILS_H_
