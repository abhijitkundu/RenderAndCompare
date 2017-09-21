/**
 * @file ImageDataset.h
 * @brief ImageDataset
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_IMAGEDATASET_H_
#define RENDERANDCOMPARE_IMAGEDATASET_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <json.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

namespace RaC {

struct ImageObjectInfo {
  // object id
  boost::optional<int> id;

  // Object category
  boost::optional<std::string> category;

  // Shape params
  boost::optional<Eigen::VectorXd> shape_param;

  // 3D length, width, height
  boost::optional<Eigen::Vector3d> dimension;

  // 2D bounding boxes
  boost::optional<Eigen::Vector4d> bbx_visible;
  boost::optional<Eigen::Vector4d> bbx_amodal;

  // viewpoint (azimuth, elevation, tilt)
  boost::optional<Eigen::Vector3d> viewpoint;

  // projection of object center (origin) in image frame
  boost::optional<Eigen::Vector2d> center_proj;

  // distance of object center (origin)
  boost::optional<double> center_dist;

  // Additional pose params (for articulated objects)
  boost::optional<Eigen::VectorXd> pose_param;

 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ImageInfo {
  using ImageObjectInfos = std::vector<ImageObjectInfo, Eigen::aligned_allocator<ImageObjectInfo>>;

  boost::optional<std::string> image_file;
  boost::optional<std::string> segm_file;

  boost::optional<Eigen::Vector2i> image_size;
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
