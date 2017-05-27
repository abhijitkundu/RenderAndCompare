/**
 * @file ImageLoaders.h
 * @brief ImageLoaders
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_IMAGELOADERS_H_
#define RENDERANDCOMPARE_IMAGELOADERS_H_

#include <Eigen/CXX11/Tensor>

#include "RenderAndCompare/EigenTypedefs.h"
#include <vector>

namespace RaC {

template <class Scalar_ = unsigned char>
class BatchImageLoader {
 public:
  using Scalar = Scalar_;
  using ImageType = Eigen::Tensor<Scalar, 3, Eigen::RowMajor>;

  BatchImageLoader(int width = -1, int height =-1, int channels = 3);
  void setImageSize(int width, int height, int channels = 3);

  void preloadImages(const std::vector<std::string>& image_files);
  void preloadImages(const std::vector<std::string>& image_files, const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings);

  std::vector<ImageType>& images() {return images_;}
  const std::vector<ImageType>& images() const {return images_;}

  inline int width() const {return width_;}
  inline int height() const {return height_;}
  inline int channels() const {return channels_;}

 private:
  std::vector<ImageType> images_;
  int width_;
  int height_;
  int channels_;
};

}  // namespace RaC

#endif // end RENDERANDCOMPARE_IMAGELOADERS_H_
