/**
 * @file DataLayer.cpp
 * @brief DataLayer
 *
 * @author Abhijit Kundu
 */

#include "ArticulatedObjectsDataLayer.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split

namespace caffe {

std::vector<std::string> tokenize(const std::string& input) {
  std::vector<std::string> tokens;
  boost::split(tokens, input, boost::is_any_of(", "), boost::token_compress_on);
  return tokens;
}

boost::program_options::variables_map parse_param_string(const std::string& param_str) {
  namespace po = boost::program_options;
  po::options_description config_options("Config");
    config_options.add_options()
        ("batch_size,b",  po::value<int>()->default_value(50), "Batch Size")
        ("width,w",  po::value<int>()->default_value(224), "Image Width")
        ("height,h",  po::value<int>()->default_value(224), "Image Height")
        ("mean_bgr,m",  po::value<std::vector<float>>()->multitoken()->default_value({103.0626238, 115.90288257, 123.15163084}, "103.0626238 115.90288257 123.15163084"), "Mean BGR color value")
        ("top_names,t", po::value<std::vector<std::string>>()->multitoken()->required(), "Top Names in Order")
        ("shape_param_size,s", po::value<int>()->default_value(10)->required(), "Shape param Size")
        ("pose_param_size,p", po::value<int>()->default_value(10)->required(), "Pose param Size")
        ;

  po::options_description options;
  options.add(config_options);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(tokenize(param_str)).options(options).run(), vm);
    po::notify(vm);
  } catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    std::cout << options << '\n';
    throw std::runtime_error("Invalid param_str. Cannot continue.");
  }

  return vm;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}


template<typename Dtype>
void ArticulatedObjectsDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "Setting up ArticulatedObjectsDataLayer";
  LOG(INFO) << "Done Setting up ArticulatedObjectsDataLayer";

  string param_str = this->layer_param_.string_param().param_str();
  LOG(INFO) << "Param = " << param_str;

  namespace po = boost::program_options;
  po::variables_map vm = parse_param_string(param_str);


  Eigen::Vector2i image_size(vm["width"].as<int>(), vm["height"].as<int>());
  batch_size_ = vm["batch_size"].as<int>();
  mean_bgr_ = Eigen::Vector3f(vm["mean_bgr"].as<vector<float>>().data()).cast<Dtype>();
  top_names_ = vm["top_names"].as<vector<string>>();

  shape_param_size_ = vm["shape_param_size"].as<int>();
  pose_param_size_ = vm["pose_param_size"].as<int>();

  const Eigen::IOFormat fmt(6, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
  LOG(INFO) << "----------Config---------------\n";
  LOG(INFO) << "batch_size = " << batch_size_;
  LOG(INFO) << "image_size = " << image_size.format(fmt);
  LOG(INFO) << "mean_bgr = " << mean_bgr_.format(fmt);
  LOG(INFO) << "top_names = " << top_names_;
  LOG(INFO) << "shape_param_size_ = " << shape_param_size_;
  LOG(INFO) << "pose_param_size = " << pose_param_size_;
  LOG(INFO) << "-------------------------------\n";

  CHECK_EQ(top.size(), top_names_.size()) << "Number of actual tops and top_names in param_str do not match";

  // Do the reshapes
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "input_image");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, 3, image_size.y(), image_size.x()}});
      input_image_loader_.setImageSize(image_size.x(), image_size.y());
    }
    else {
      LOG(WARNING) << "input_image blob not set";
    }
  }
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "segm_image");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, 1, 240, 320}});
      segm_image_loader_.setImageSize(320, 240);
    }
    else {
      LOG(WARNING) << "segm_image blob not set";
    }
  }
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "shape_param");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, shape_param_size_}});
    } else {
      LOG(WARNING) << "shape_param blob not set";
    }
  }
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "pose_param");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, pose_param_size_}});
    } else {
      LOG(WARNING) << "pose_param blob not set";
    }
  }
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "camera_extrinsic");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, 4, 4}});
    } else {
      LOG(WARNING) << "camera_extrinsic blob not set";
    }
  }
  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "model_pose");
    if (it != top_names_.end()) {
      top[std::distance(top_names_.begin(), it)]->Reshape( { {batch_size_, 4, 4}});
    } else {
      LOG(WARNING) << "model_pose blob not set";
    }
  }
}

template <typename Dtype>
void ArticulatedObjectsDataLayer<Dtype>::addDataset(const RaC::Dataset& dataset) {
  LOG(INFO) << "---- Adding data from " << dataset.name << "------";
  std::vector<std::string> image_files;
  std::vector<std::string> segm_files;
  Eigen::AlignedStdVector<Eigen::Vector4i> visible_boxes;

  image_files.reserve(dataset.annotations.size());
  segm_files.reserve(dataset.annotations.size());
  visible_boxes.reserve(dataset.annotations.size());

  for (std::size_t i = 0; i< dataset.annotations.size(); ++i) {
    const RaC::Annotation& anno = dataset.annotations[i];
    Eigen::Vector4d bbx(anno.bbx_visible.value().data());

    visible_boxes.push_back(bbx.cast<int>());
    image_files.push_back((dataset.rootdir / anno.image_file.value()).string());
    segm_files.push_back((dataset.rootdir / anno.segm_file.value()).string());

    VectorX shape_param = anno.shapeParam().cast<Dtype>();
    CHECK_EQ(shape_param.size(), shape_param_size_);
    shape_params_.push_back(shape_param);

    VectorX pose_param = anno.poseParam().cast<Dtype>();
    CHECK_EQ(pose_param.size(), pose_param_size_);
    pose_params_.push_back(pose_param);

    camera_extrinsics_.push_back(anno.cameraExtrinsic().cast<Dtype>());
    model_poses_.push_back(anno.modelPose().cast<Dtype>());
  }
  CHECK_EQ(image_files.size(), visible_boxes.size());
  input_image_loader_.preloadImages(image_files, visible_boxes);
  segm_image_loader_.preloadImages(segm_files, true);
}

template <typename Dtype>
void ArticulatedObjectsDataLayer<Dtype>::generateDatumIds() {
  LOG(INFO) << "Generating Data Ids";

  CHECK_GT(batch_size_, 0);
  CHECK_GE(input_image_loader_.images().size(), batch_size_);

  data_ids_.resize(input_image_loader_.images().size());
  std::iota(data_ids_.begin(), data_ids_.end(), 0);


  if (this->phase_ == caffe::TRAIN) {
    LOG(INFO) << "Shuffling Data Ids";
    std::shuffle(data_ids_.begin(), data_ids_.end(), rand_engine_);
  }
  LOG(INFO) << "Total number of data points = " << data_ids_.size();
  LOG(INFO) << "Approx " << data_ids_.size() / batch_size_ << " iterations required per epoch";
  curr_data_idx_ = 0;
}



template <typename Dtype>
void ArticulatedObjectsDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_GE(data_ids_.size(), batch_size_) << "batch size cannot be smaller than total number of data points";

  std::vector<std::size_t> batch_data_ids(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    if (curr_data_idx_ == data_ids_.size()) {
      // LOG(INFO) << "Shuffling Data Ids";
      // std::shuffle(data_ids_.begin(), data_ids_.end(), rand_engine_);
      curr_data_idx_ = 0;
    }
    batch_data_ids[i] = data_ids_[curr_data_idx_++];
    CHECK_LT(batch_data_ids[i], data_ids_.size());
  }

  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "input_image");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
      using ImageType = Eigen::Tensor<Dtype, 3, Eigen::RowMajor>;
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::TensorMap<ImageType> blob_image(top_data + top[index]->offset(i), 3, input_image_loader_.height(), input_image_loader_.width());
        blob_image = input_image_loader_.images()[batch_data_ids[i]].cast<Dtype>();

        blob_image.chip(0, 0) = blob_image.chip(0, 0) - mean_bgr_[0];
        blob_image.chip(1, 0) = blob_image.chip(1, 0) - mean_bgr_[1];
        blob_image.chip(2, 0) = blob_image.chip(2, 0) - mean_bgr_[2];
      }
    }
  }

  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "segm_image");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
      using ImageType = Eigen::Tensor<Dtype, 3, Eigen::RowMajor>;
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::TensorMap<ImageType> blob_image(top_data + top[index]->offset(i), 1, segm_image_loader_.height(), segm_image_loader_.width());
        blob_image = segm_image_loader_.images()[batch_data_ids[i]].cast<Dtype>();
      }
    }
  }

  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "shape_param");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::Map<VectorX>(top_data + top[index]->offset(i), shape_param_size_) = shape_params_[batch_data_ids[i]];
      }
    }
  }

  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "pose_param");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::Map<VectorX>(top_data + top[index]->offset(i), pose_param_size_) = pose_params_[batch_data_ids[i]];
      }
    }
  }


  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "camera_extrinsic");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::Map<Matrix4>(top_data + top[index]->offset(i), 4, 4) = camera_extrinsics_[batch_data_ids[i]];
      }
    }
  }


  {
    auto it = std::find(top_names_.begin(), top_names_.end(), "model_pose");
    if (it != top_names_.end()) {
      const auto index = std::distance(top_names_.begin(), it);
      Dtype* top_data = top[index]->mutable_cpu_data();
#pragma omp parallel for
      for (int i = 0; i< batch_size_; ++i) {
        Eigen::Map<Matrix4>(top_data + top[index]->offset(i), 4, 4) = model_poses_[batch_data_ids[i]];
      }
    }
  }
}

INSTANTIATE_CLASS(ArticulatedObjectsDataLayer);

}  // end namespace caffe
