/**
 * @file ImageLoaders.cpp
 * @brief ImageLoaders
 *
 * @author Abhijit Kundu
 */

#include "ImageLoaders.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace RaC {

BatchImageLoader::BatchImageLoader(int width, int height)
    : width_(width),
      height_(height) {
}

void BatchImageLoader::setImageSize(int width, int height) {
  width_ = width;
  height_ = height;
}

void BatchImageLoader::preloadImages(const std::vector<std::string>& image_files) {
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Resize,Shuffle" << std::endl;
  std::vector<ImageType> new_images(image_files.size());

#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_COLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    cv::resize(cv_image, cv_image, cv::Size(width_, height_));

    const Eigen::array<ptrdiff_t, 3> shuffles({{1, 2, 0}});
    new_images[i] = Eigen::TensorMap<ImageType>(cv_image.data, height_, width_, 3).shuffle(shuffles);
  }

  images_.reserve(images_.size() + new_images.size());
  images_.insert(images_.end(), new_images.begin(), new_images.end());
  std::cout << "BatchImageLoader: Current number of images =  " << images_.size() << std::endl;
}

void BatchImageLoader::preloadImages(const std::vector<std::string>& image_files,
                                     const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings) {
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Crop,Resize,Shuffle" << std::endl;

  if (image_files.size() != croppings.size())
    throw std::runtime_error("BatchImageLoader::cropAndPreloadImages(): image_files.size() != croppings.size()");

  std::vector<ImageType> new_images(image_files.size());

#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_COLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    const Eigen::Vector4i& bbx = croppings[i];
    const cv::Rect roi(bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]);

    // Image is HWC
    cv::resize(cv_image(roi), cv_image, cv::Size(width_, height_));
    const Eigen::array<ptrdiff_t, 3> shuffles({{1, 2, 0}});


    new_images[i] = Eigen::TensorMap<ImageType>(cv_image.data, height_, width_, 3).shuffle(shuffles);
  }

  images_.reserve(images_.size() + new_images.size());
  images_.insert(images_.end(), new_images.begin(), new_images.end());
  std::cout << "BatchImageLoader: Current number of images =  " << images_.size() << std::endl;
}

}  // namespace RaC


