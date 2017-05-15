/**
 * @file ImageLoaders.h
 * @brief ImageLoaders
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_IMAGELOADERS_H_
#define RENDERANDCOMPARE_IMAGELOADERS_H_

#include <Eigen/CXX11/Tensor>
#include <opencv2/core/core.hpp>

#include "RenderAndCompare/EigenTypedefs.h"
#include <vector>

namespace RaC {

class BatchImageLoader {
 public:
  using ImageType = Eigen::Tensor<unsigned char, 3, Eigen::RowMajor>;

  BatchImageLoader(int width = -1, int height =-1);
  void setImageSize(int width, int height);

  void preloadImages(const std::vector<std::string>& image_files);
  void preloadImages(const std::vector<std::string>& image_files, const Eigen::AlignedStdVector<Eigen::Vector4i>& croppings);

  std::vector<ImageType>& images() {return images_;}
  const std::vector<ImageType>& images() const {return images_;}

  inline int width() const {return width_;}
  inline int height() const {return height_;}

 private:
  std::vector<ImageType> images_;
  int width_;
  int height_;
};

}  // namespace RaC



#endif // end RENDERANDCOMPARE_IMAGELOADERS_H_
