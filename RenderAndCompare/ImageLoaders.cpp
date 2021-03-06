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

template<class S, int C>
BatchImageLoader<S, C>::BatchImageLoader(int width, int height)
    : width_(width),
      height_(height) {
}

template<class S, int C>
void BatchImageLoader<S, C>::setImageSize(int width, int height) {
  width_ = width;
  height_ = height;
}

template<class S, int C>
void BatchImageLoader<S, C>::preloadImages(const std::vector<std::string>& image_files, bool do_vertical_flip) {
  std::locale loc("");
  std::cout.imbue(loc);
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Resize,Shuffle" << std::endl;
  assert(width_ > 0);
  assert(height_ > 0);

  const std::size_t prev_size = images_.size();
  images_.insert(images_.end(), image_files.size(), ImageType(NumOfChannels, height_, width_));

  boost::progress_display show_progress(image_files.size());
#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.channels() != NumOfChannels) {
      std::cout << "Got " << cv_image.channels() << " channels (Expected: " << NumOfChannels << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.depth() != cv::DataType<Scalar>::depth) {
      std::cout << "Got " << cv_image.depth() << " depth (Expected: " << cv::DataType<Scalar>::depth << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    cv::resize(cv_image, cv_image, cv::Size(width_, height_));

    if (do_vertical_flip) {
      cv::flip(cv_image, cv_image, 0);
    }

    const Eigen::array<ptrdiff_t, 3> shuffles({{2, 0, 1}});
    images_[prev_size + i] = Eigen::TensorMap<ImageType>((Scalar*) cv_image.data, height_, width_, NumOfChannels).shuffle(shuffles);
    assert(images_[prev_size + i].dimension(0) == NumOfChannels);
    ++show_progress;
  }

  std::cout << "\nBatchImageLoader: Current number of images =  " << images_.size()
            << " (" << double(sizeInBytes())/1e9 << " GB)" << std::endl;
}

template<class S, int C>
void BatchImageLoader<S, C>::preloadImages(const std::vector<std::string>& image_files,
                                           const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings,
                                           const bool use_uniform_scaling) {
  std::locale loc("");
  std::cout.imbue(loc);
  std::cout << "BatchImageLoader: Preloading " << image_files.size() << " images. Ops: Crop,Resize,Shuffle" << std::endl;
  assert(width_ > 0);
  assert(height_ > 0);

  if (image_files.size() != croppings.size())
    throw std::runtime_error("BatchImageLoader::cropAndPreloadImages(): image_files.size() != croppings.size()");

  const std::size_t prev_size = images_.size();
  images_.insert(images_.end(), image_files.size(), ImageType(NumOfChannels, height_, width_));

  boost::progress_display show_progress(image_files.size());
#pragma omp parallel for
  for (std::size_t i = 0; i< image_files.size(); ++i) {
    cv::Mat cv_image = cv::imread(image_files[i], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if(cv_image.empty() )  {
      std::cout << "Failed to load image from " << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.channels() != NumOfChannels) {
      std::cout << "Got " << cv_image.channels() << " channels (Expected: " << NumOfChannels << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }
    if (cv_image.depth() != cv::DataType<Scalar>::depth) {
      std::cout << "Got " << cv_image.depth() << " depth (Expected: " << cv::DataType<Scalar>::depth << ") from "  << image_files[i] << std::endl;
      throw std::runtime_error("Image loading failed");
    }

    if (!use_uniform_scaling){
      const Eigen::Vector4i& bbx = croppings[i];
      const cv::Rect roi(bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]);
      cv::resize(cv_image(roi), cv_image, cv::Size(width_, height_));
    }
    else {
      const Eigen::Vector4i& tbbx = croppings[i];
      const int tbbx_height =  tbbx[3] - tbbx[1];
      const int tbbx_width =  tbbx[2] - tbbx[0];
      const int tbbx_hw_max = std::max(tbbx_height, tbbx_width);

      cv::Mat image(tbbx_hw_max, tbbx_hw_max, cv_image.type(), cv::Scalar::all(std::numeric_limits<Scalar>::max() / 2));

      cv::Rect roi_amodal, roi_src, roi_dst;

      roi_amodal.x = tbbx[0] - (tbbx_hw_max - tbbx_width) / 2;
      roi_amodal.y = tbbx[1] - (tbbx_hw_max - tbbx_height) / 2;
      roi_amodal.width = tbbx_hw_max;
      roi_amodal.height = tbbx_hw_max;

      roi_src.x = std::max(roi_amodal.x, 0);
      roi_src.y = std::max(roi_amodal.y, 0);
      roi_src.width = std::min(cv_image.cols, roi_amodal.x + tbbx_hw_max) - roi_src.x;
      roi_src.height = std::min(cv_image.rows, roi_amodal.y + tbbx_hw_max) - roi_src.y;

      roi_dst.x = roi_src.x - roi_amodal.x;
      roi_dst.y = roi_src.y - roi_amodal.y;
      roi_dst.width = roi_src.width;
      roi_dst.height = roi_src.height;

      cv_image(roi_src).copyTo(image(roi_dst));

      cv::resize(image, cv_image, cv::Size(width_, height_));
    }


    // Image is HWC
    const Eigen::array<ptrdiff_t, 3> shuffles({{2, 0, 1}});
    images_[prev_size + i] = Eigen::TensorMap<ImageType>((Scalar*) cv_image.data, height_, width_, NumOfChannels).shuffle(shuffles);
    assert(images_[prev_size + i].dimension(0) == NumOfChannels);
    ++show_progress;
  }

  std::cout << "\nBatchImageLoader: Current number of images =  " << images_.size()
            << " (" << double(sizeInBytes())/1e9 << " GB)" << std::endl;
}


// Instantiate BatchImageLoader
template class BatchImageLoader<uint8_t, 3>;
template class BatchImageLoader<uint8_t, 1>;
template class BatchImageLoader<uint16_t, 1>;

}  // namespace RaC


