/**
 * @file Dataset.h
 * @brief Dataset
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_DATASET_H_
#define RENDERANDCOMPARE_DATASET_H_

#include <array>
#include <vector>
#include <json.hpp>
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

namespace RaC {

// a simple struct to model a person
struct Annotation {
  using Array4d = std::array<double, 4>;
  using Array10d = std::array<double, 10>;
  using Array16d = std::array<double, 16>;
  using VectorXd = std::vector<double>;
  using MatrixRM4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

  boost::optional<std::string> image_file;
  boost::optional<Array4d> viewpoint;
  boost::optional<Array4d> bbx_amodal;
  boost::optional<Array4d> bbx_crop;
  boost::optional<Array4d> bbx_visible;
  boost::optional<Array10d> shape_param;
  boost::optional<VectorXd> pose_param;
  boost::optional<Array16d> camera_extrinsic;
  boost::optional<Array16d> model_pose;

  Eigen::VectorXd poseParam() const {
    return Eigen::Map<const Eigen::VectorXd>(pose_param.value().data(), pose_param.value().size());
  }

  Eigen::VectorXd shapeParam() const {
    return Eigen::Map<const Eigen::VectorXd>(shape_param.value().data(), shape_param.value().size());
  }

  MatrixRM4d cameraExtrinsic() const {
    return MatrixRM4d(camera_extrinsic.value().data());
  }

  MatrixRM4d modelPose() const {
    return MatrixRM4d(model_pose.value().data());
  }
};

struct Dataset {
  std::string name;
  boost::filesystem::path rootdir;
  std::vector<Annotation> annotations;
};

Dataset loadDatasetFromJson(const std::string& filepath);

void to_json(nlohmann::json& j, const Dataset& dataset);
void from_json(const nlohmann::json& j, Dataset& dataset);

void to_json(nlohmann::json& j, const Annotation& p);
void from_json(const nlohmann::json& j, Annotation& p);

std::ostream& operator<<(std::ostream& os, const Annotation& anno);

}  // end namespace RaC

#endif // end RENDERANDCOMPARE_DATASET_H_
