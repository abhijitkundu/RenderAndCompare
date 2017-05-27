/**
 * @file ImageUtils.cpp
 * @brief ImageUtils
 *
 * @author Abhijit Kundu
 */

#include "ImageUtils.h"

namespace RaC {

cv::Mat getColoredImageFromLabels(const cv::Mat& label_image, const std::vector<cv::Vec3b>& colormap) {
  cv::Mat colored_image(label_image.rows, label_image.cols, CV_8UC3);
  for (int y = 0; y < label_image.rows; ++y)
    for (int x = 0; x < label_image.cols; ++x) {
      colored_image.at<cv::Vec3b>(y, x) = colormap.at(label_image.at<uchar>(y, x));
    }
  return colored_image;
}

}  // namespace RaC
