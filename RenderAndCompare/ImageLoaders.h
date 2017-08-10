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

template <class Scalar_ = unsigned char, int NumOfChannels_ = 3>
class BatchImageLoader {
 public:
  using Scalar = Scalar_;
  static const int NumOfChannels = NumOfChannels_;

  using ImageType = Eigen::Tensor<Scalar, 3, Eigen::RowMajor>;

  BatchImageLoader(int width = -1, int height =-1);
  void setImageSize(int width, int height);

  void preloadImages(const std::vector<std::string>& image_files, bool do_vertical_flip = false);
  void preloadImages(const std::vector<std::string>& image_files,
                     const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings,
                     const bool use_uniform_scaling = true);

  std::vector<ImageType>& images() {return images_;}
  const std::vector<ImageType>& images() const {return images_;}

  inline int width() const {return width_;}
  inline int height() const {return height_;}

  inline std::size_t sizeInBytes() const {return images_.size() * NumOfChannels * height_ * width_ * sizeof(Scalar);}

 private:
  std::vector<ImageType> images_;
  int width_;
  int height_;
};

}  // namespace RaC

#endif // end RENDERANDCOMPARE_IMAGELOADERS_H_
