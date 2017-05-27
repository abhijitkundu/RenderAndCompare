/**
 * @file ImageLoaders.cpp
 * @brief ImageLoaders
 *
 * @author Abhijit Kundu
 */

#include "ImageLoaders.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

namespace RaC {

template<class Scalar_>
BatchImageLoader<Scalar_>::BatchImageLoader(int width, int height, int channels)
    : width_(width),
      height_(height),
      channels_(channels) {
}

template<class Scalar_>
void BatchImageLoader<Scalar_>::setImageSize(int width, int height, int channels) {
  width_ = width;
  height_ = height;
  channels_ = channels;
}

template<class Scalar_>
void BatchImageLoader<Scalar_>::preloadImages(const std::vector<std::string>& image_files) {
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Resize,Shuffle" << std::endl;
  assert(width_ > 0);
  assert(height_ > 0);
  assert(channels_ == 3 || channels_ == 1);

  const std::size_t prev_size = images_.size();
  images_.insert(images_.end(), image_files.size(), ImageType(channels_, height_, width_));

  boost::progress_display show_progress(image_files.size());
#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.channels() != channels_) {
      std::cout << "Got " << cv_image.channels() << " channels (Expected: " << channels_ << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.depth() != cv::DataType<Scalar>::depth) {
      std::cout << "Got " << cv_image.depth() << " depth (Expected: " << cv::DataType<Scalar>::depth << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    cv::resize(cv_image, cv_image, cv::Size(width_, height_));
    const Eigen::array<ptrdiff_t, 3> shuffles({{2, 0, 1}});
    images_[prev_size + i] = Eigen::TensorMap<ImageType>((Scalar*) cv_image.data, height_, width_, channels_).shuffle(shuffles);
    assert(images_[prev_size + i].dimension(0) == channels_);
    ++show_progress;
  }

  std::cout << "\nBatchImageLoader: Current number of images =  " << images_.size() << std::endl;
}

template<class Scalar_>
void BatchImageLoader<Scalar_>::preloadImages(const std::vector<std::string>& image_files,
                                     const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings) {
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Crop,Resize,Shuffle" << std::endl;
  assert(width_ > 0);
  assert(height_ > 0);
  assert(channels_ == 3);

  if (image_files.size() != croppings.size())
    throw std::runtime_error("BatchImageLoader::cropAndPreloadImages(): image_files.size() != croppings.size()");

  const std::size_t prev_size = images_.size();
  images_.insert(images_.end(), image_files.size(), ImageType(channels_, height_, width_));

  boost::progress_display show_progress(image_files.size());
#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.channels() != channels_) {
      std::cout << "Got " << cv_image.channels() << " channels (Expected: " << channels_ << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.depth() != cv::DataType<Scalar>::depth) {
      std::cout << "Got " << cv_image.depth() << " depth (Expected: " << cv::DataType<Scalar>::depth << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    const Eigen::Vector4i& bbx = croppings[i];
    const cv::Rect roi(bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]);

    // Image is HWC
    cv::resize(cv_image(roi), cv_image, cv::Size(width_, height_));
    const Eigen::array<ptrdiff_t, 3> shuffles({{2, 0, 1}});
    images_[prev_size + i] = Eigen::TensorMap<ImageType>((Scalar*) cv_image.data, height_, width_, channels_).shuffle(shuffles);
    assert(images_[prev_size + i].dimension(0) == channels_);
    ++show_progress;
  }

  std::cout << "\nBatchImageLoader: Current number of images =  " << images_.size() << std::endl;
}


// Instantiate BatchImageLoader
template class BatchImageLoader<uint8_t>;
template class BatchImageLoader<uint16_t>;

}  // namespace RaC


