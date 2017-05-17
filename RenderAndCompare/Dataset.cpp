/**
 * @file Dataset.cpp
 * @brief Dataset
 *
 * @author Abhijit Kundu
 */

#include "Dataset.h"
#include <fstream>
#include <boost/optional/optional_io.hpp>

namespace nlohmann {
template<typename T>
struct adl_serializer<boost::optional<T>> {
  static void to_json(json& j, const boost::optional<T>& opt) {
    if (opt == boost::none) {
      j = nullptr;
    } else {
      j = *opt;  // this will call adl_serializer<T>::to_json which will
                 // find the free function to_json in T's namespace!
    }
  }

  static void from_json(const json& j, boost::optional<T>& opt) {
    if (j.is_null()) {
      opt = boost::none;
    } else {
      opt = j.get<T>();  // same as above, but with
                         // adl_serializer<T>::from_json
    }
  }
};

} // end namespace nlohmann

namespace RaC {

Dataset loadDatasetFromJson(const std::string& filepath) {
  nlohmann::json dataset_json;
  {
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open File from " + filepath);
    }
    file >> dataset_json;
  }
  return dataset_json;
}

void to_json(nlohmann::json& j, const Dataset& dataset) {
  j = nlohmann::json {
    { "name", dataset.name },
    { "rootdir", dataset.rootdir.string() },
    { "annotations", dataset.annotations }
  };
}

void from_json(const nlohmann::json& j, Dataset& dataset) {
  dataset.name = j["name"].get<std::string>();
  dataset.rootdir = j["rootdir"].get<std::string>();
  dataset.annotations = j["annotations"].get<std::vector<Annotation>>();
}


void to_json(nlohmann::json& j, const Annotation& p) {
  j = nlohmann::json {
    { "image_file", p.image_file },
    { "viewpoint", p.viewpoint },
    { "bbx_amodal", p.bbx_amodal },
    { "bbx_crop", p.bbx_crop },
    { "bbx_visible", p.bbx_visible },
    { "shape_param", p.shape_param },
    { "pose_param", p.pose_param },
    { "camera_extrinsic", p.camera_extrinsic },
    { "model_pose", p.model_pose }
  };
}

template<typename KeyType, typename T, std::size_t N>
void from_json_if_present(const nlohmann::json& j, const KeyType& key, boost::optional<std::array<T, N>>& opt) {
  auto it = j.find(key);
  if (it != j.end()) {
    using VectorT = std::vector<T>;
    VectorT vec = it->template get<VectorT>();
    if (vec.size() == N) {
      opt.emplace();
      std::copy_n(vec.begin(), N, opt.value().begin());
      return;
    }
  }
  opt = boost::none;
}

template<typename KeyType, typename T>
void from_json_if_present(const nlohmann::json& j, const KeyType& key, boost::optional<T>& opt) {
  auto it = j.find(key);
  if (it != j.end()) {
    opt = it->template get<T>();
    return;
  }
  opt = boost::none;
}

void from_json(const nlohmann::json& j, Annotation& p) {
  from_json_if_present(j, "image_file", p.image_file);
  from_json_if_present(j, "viewpoint", p.viewpoint);
  from_json_if_present(j, "bbx_amodal", p.bbx_amodal);
  from_json_if_present(j, "bbx_crop", p.bbx_crop);
  from_json_if_present(j, "bbx_visible", p.bbx_visible);
  from_json_if_present(j, "shape_param", p.shape_param);
  from_json_if_present(j, "pose_param", p.pose_param);
  from_json_if_present(j, "camera_extrinsic", p.camera_extrinsic);
  from_json_if_present(j, "model_pose", p.model_pose);
}


template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& val) {
  for (const auto& s : val)
    os << s << ' ';
  return os;
}

template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const boost::optional<std::array<T, N>>& opt) {
  if (opt == boost::none) {
    os << "none";
  } else
    os << opt.value();
  return os;
}

std::ostream& operator<<(std::ostream& os, const Annotation& anno) {
  os << "image_file: " << anno.image_file;
  os << "\nviewpoint: " << anno.viewpoint;
  os << "\nbbx_amodal: " << anno.bbx_amodal;
  os << "\nbbx_crop: " << anno.bbx_crop;
  os << "\nbbx_visible: " << anno.bbx_visible;
  os << "\nshape_param: " << anno.shape_param;
  os << "\npose_param: " << anno.pose_param;
  os << "\ncamera_extrinsic: " << anno.camera_extrinsic;
  os << "\nmodel_pose: " << anno.model_pose;
  return os;
}

}  // end namespace RaC
