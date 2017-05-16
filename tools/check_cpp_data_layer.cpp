/**
 * @file check_cpp_data_layer.cpp
 * @brief check_cpp_data_layer
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ArticulatedObjectsDataLayer.h"

#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {
REGISTER_LAYER_CLASS(ArticulatedObjectsData);
}  // namespace caffe

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;

int main(int argc, char **argv) {

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("dataset",  po::value<fs::path>(), "Path to dataset files (JSON)")
        ("network_model,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids")
        ("use_cpu",  po::value<bool>()->default_value(false), "Use CPU")
        ("pause_time,p",  po::value<int>()->default_value(0), "Pause time. Use 0 for pause")
        ;

  po::positional_options_description p;
  p.add("dataset", 1);

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
    po::notify(vm);
  } catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  if (vm.count("help")) {
    std::cout << cmdline_options << '\n';
    return EXIT_SUCCESS;
  }

  if (!vm.count("dataset")) {
    std::cout << "Please provide atleast one dataset file" << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  const fs::path dataset_file = vm["dataset"].as<fs::path>();
  const fs::path net_model_file = vm["network_model"].as<fs::path>();
  const int gpu_id = vm["gpu_id"].as<int>();
  const int pause_time = vm["pause_time"].as<int>();

  if (vm["use_cpu"].as<bool>()) {
    LOG(INFO)<< "Using CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    LOG(INFO) << "Using GPU with device ID " << gpu_id;

    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  }

  Dataset dataset = loadDatasetFromJson(dataset_file.string());
  const int num_of_images = dataset.annotations.size();

  LOG(INFO)<< "Instantiating network model from "<< net_model_file;
  Net<float> caffe_net(net_model_file.string(), caffe::TEST);
  LOG(INFO)<< "Network Instantiation successful";

  using DataLayerType = caffe::ArticulatedObjectsDataLayer<float>;
  auto data_layer_ptr = boost::dynamic_pointer_cast<DataLayerType>(caffe_net.layers()[0]);
  data_layer_ptr->addDataset(dataset);
  data_layer_ptr->generateDatumIds();

  auto data_blob_shape = caffe_net.blob_by_name("data")->shape();
  CHECK_EQ(data_blob_shape.size(), 4) << "Expects 4D data blob";
  CHECK_EQ(data_blob_shape[1], 3) << "Expects 2nd channel to be 3 for BGR image";
  const int batch_size = data_blob_shape[0];
  const int num_of_batches = int(std::ceil(num_of_images / float(batch_size)));

  const std::string windowname = "ImageViewer";
  cv::namedWindow(windowname, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);


  for (int b = 0; b < num_of_batches; ++b) {
    const int start_idx = batch_size * b;
    const int end_idx = std::min(batch_size * (b + 1), num_of_images);
    LOG(INFO) << "Working on batch: " << b  << "/" << num_of_batches
              << " (Images# " << start_idx  << "-" << end_idx << ")";

    caffe_net.Forward();

    using MatrixX10fRM = Eigen::Matrix<float, Eigen::Dynamic, 10, Eigen::RowMajor>;
    using RowVector10d = Eigen::Matrix<double, 1, 10>;

    auto gt_shape_param_blob_ptr =  caffe_net.blob_by_name("gt_shape_param");
    if (gt_shape_param_blob_ptr) {
      Eigen::Map<const MatrixX10fRM> gt_shape_param_blob(gt_shape_param_blob_ptr->cpu_data(), batch_size, 10);
      for (int i = start_idx; i < end_idx; ++i) {
        Annotation& anno = dataset.annotations[i];
//        std::cout  << RowVector10d(anno.shape_param.value().data()) << "\n";
//        std::cout << gt_shape_param_blob.row(i - start_idx) << "\n";
        CHECK(RowVector10d(anno.shape_param.value().data()).cast<float>().isApprox(gt_shape_param_blob.row(i - start_idx)));
      }
    } else {
      LOG(INFO) << "No shape Data";
    }

    auto gt_pose_param_blob_ptr =  caffe_net.blob_by_name("gt_pose_param");
    if (gt_pose_param_blob_ptr) {
      Eigen::Map<const MatrixX10fRM> gt_pose_param_blob(gt_pose_param_blob_ptr->cpu_data(), batch_size, 10);
      for (int i = start_idx; i < end_idx; ++i) {
        Annotation& anno = dataset.annotations[i];
        CHECK(RowVector10d(anno.pose_param.value().data()).cast<float>().isApprox(gt_pose_param_blob.row(i - start_idx)));
      }
    } else {
      LOG(INFO)<< "No pose Data";
    }

    bool exit_loop = false;
    auto image_data_blob_ptr =  caffe_net.blob_by_name("data");
    if (image_data_blob_ptr) {
      using Tensor4f = Eigen::Tensor<const float , 4, Eigen::RowMajor>;
      using Tensor3f = Eigen::Tensor<float , 3, Eigen::RowMajor>;
      using Tensor3u = Eigen::Tensor<unsigned char , 3, Eigen::RowMajor>;

      Eigen::TensorMap<Tensor4f>image_blob(image_data_blob_ptr->cpu_data(), batch_size, 3, data_blob_shape[2], data_blob_shape[3]);

      for (int i = 0; i < batch_size; ++i) {
        Tensor3f image = image_blob.chip(i, 0);

        const Eigen::Vector3f& mean_bgr = data_layer_ptr->mean_bgr();
        image.chip(0, 0) = image.chip(0, 0) + mean_bgr[0];
        image.chip(1, 0) = image.chip(1, 0) + mean_bgr[1];
        image.chip(2, 0) = image.chip(2, 0) + mean_bgr[2];

        const Eigen::array<ptrdiff_t, 3> shuffles({{1, 2, 0}});
        Tensor3u image_bgr = image.cast<unsigned char>().shuffle(shuffles);
        cv::Mat cv_image(image_bgr.dimension(0), image_bgr.dimension(1), CV_8UC3, image_bgr.data());
        cv::imshow(windowname, cv_image);
        const int key = cv::waitKey(pause_time) % 256;

        // Esc or Q or q
        if (key == 27 || key == 'q' || key == 'Q') {
          exit_loop = true;
          break;
        }
      }

      if (exit_loop)
        break;

    } else {
      LOG(INFO)<< "No Image Data";
    }

  }


  return EXIT_SUCCESS;
}


