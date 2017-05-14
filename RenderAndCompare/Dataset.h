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

  boost::optional<std::string> image_file;
  boost::optional<Array4d> viewpoint;
  boost::optional<Array4d> bbx_amodal;
  boost::optional<Array4d> bbx_crop;
  boost::optional<Array4d> bbx_visible;
  boost::optional<Array10d> shape_param;
  boost::optional<Array10d> pose_param;
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
