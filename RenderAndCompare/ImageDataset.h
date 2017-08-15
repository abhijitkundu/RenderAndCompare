/**
 * @file ImageDataset.h
 * @brief ImageDataset
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_IMAGEDATASET_H_
#define RENDERANDCOMPARE_IMAGEDATASET_H_

#include <array>
#include <vector>
#include <json.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

namespace RaC {

struct ImageObjectInfo {
  using Array2d = std::array<double, 2>;
  using Array3d = std::array<double, 3>;
  using Array4d = std::array<double, 4>;
  using VectorXd = std::vector<double>;

  // Object category
  boost::optional<std::string> category;

  // Shape params
  boost::optional<VectorXd> shape_param;

  // 3D length, width, height
  boost::optional<Array3d> dimension;

  // 2D bounding boxes
  boost::optional<Array4d> bbx_visible;
  boost::optional<Array4d> bbx_amodal;

  // model pose can be computed from origin_proj and viewpoint
  boost::optional<Array2d> origin_proj;
  boost::optional<Array4d> viewpoint;

  // Additional pose params (for articulated objects)
  boost::optional<VectorXd> pose_param;
};

struct ImageInfo {
  using Array2i = std::array<int, 2>;
  using Array9d = std::array<double, 9>;
  using ImageObjectInfos = std::vector<ImageObjectInfo>;

  boost::optional<std::string> image_file;
  boost::optional<std::string> segm_file;

  boost::optional<Array2i> image_size;
  boost::optional<Eigen::Matrix3d> image_intrinsic;

  boost::optional<ImageObjectInfos> objects;
};


struct ImageDataset {
  std::string name;
  boost::filesystem::path rootdir;
  std::vector<ImageInfo> annotations;
};

ImageDataset loadImageDatasetFromJson(const std::string& filepath);

void to_json(nlohmann::json& j, const ImageDataset& dataset);
void from_json(const nlohmann::json& j, ImageDataset& dataset);

void to_json(nlohmann::json& j, const ImageInfo& p);
void from_json(const nlohmann::json& j, ImageInfo& p);

void to_json(nlohmann::json& j, const ImageObjectInfo& p);
void from_json(const nlohmann::json& j, ImageObjectInfo& p);

}  // namespace RaC

#endif // end RENDERANDCOMPARE_IMAGEDATASET_H_
