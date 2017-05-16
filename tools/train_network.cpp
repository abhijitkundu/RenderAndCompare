/**
 * @file train_network.cpp
 * @brief train_network
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ArticulatedObjectsDataLayer.h"

#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>

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
        ("datasets,d",  po::value<std::vector<fs::path>>(), "Path to dataset files (JSON)")
        ("network_model,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids")
        ("use_cpu",  po::value<bool>()->default_value(false), "Use CPU")
        ;

  po::positional_options_description p;
  p.add("datasets", -1);

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

  if (!vm.count("datasets")) {
    std::cout << "Please provide atleast one dataset file" << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  const std::vector<fs::path> dataset_files(vm["datasets"].as<std::vector<fs::path>>());
  for (const auto& dataset_file : dataset_files) {
    if (!fs::exists(dataset_file)) {
      std::cout << "Error: " << dataset_file << " does not exist\n";
      return EXIT_FAILURE;
    }
  }

  std::vector<Dataset> datasets(dataset_files.size());
#pragma omp parallel for
  for (std::size_t i = 0; i< dataset_files.size(); ++i) {
    std::cout << "Loading dataset annotation from " <<  dataset_files[i] << std::endl;
    datasets[i] = loadDatasetFromJson(dataset_files[i].string());
  }
  std::cout << "Loaded " << datasets.size() << " datasets" << std::endl;

  const fs::path net_model_file = vm["network_model"].as<fs::path>();

  const int gpu_id = vm["gpu_id"].as<int>();

  if (vm["use_cpu"].as<bool>()) {
    LOG(INFO)<< "Using CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    LOG(INFO) << "Using GPU with device ID " << gpu_id;

    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  }

  LOG(INFO)<< "Instantiating network model from "<< net_model_file;
  Net<float> caffe_net(net_model_file.string(), caffe::TEST);
  LOG(INFO)<< "Network Instantiation successful";

  using DataLayerType = caffe::ArticulatedObjectsDataLayer<float>;
  auto data_layer_ptr = boost::dynamic_pointer_cast<DataLayerType>(caffe_net.layers()[0]);
  data_layer_ptr->addDataset(datasets[0]);
  data_layer_ptr->generateDatumIds();

  LOG(INFO) << "Doing forward";
  for (int i = 0; i < 1000; ++i) {
    caffe_net.Forward();

    auto image_data_blob =  caffe_net.blob_by_name("data");
    if (!image_data_blob) {
      LOG(INFO)<< "No Image Data";
    }

    auto gt_shape_param_blob =  caffe_net.blob_by_name("gt_shape_param");
    if (!gt_shape_param_blob) {
      LOG(INFO)<< "No shape Data";
    }

    auto gt_pose_param_blob =  caffe_net.blob_by_name("gt_pose_param");
    if (!gt_pose_param_blob) {
      LOG(INFO)<< "No pose Data";
    }
  }


  return EXIT_SUCCESS;
}



