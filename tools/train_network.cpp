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
using caffe::shared_ptr;

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace RaC;

// Train / Finetune a model.
void train(const fs::path& solver_proto,
           const std::vector<fs::path>& dataset_files,
           const fs::path& init_file, const int gpu_id) {
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(solver_proto.string(), &solver_param);

  solver_param.set_device_id(gpu_id);
  Caffe::SetDevice(gpu_id);
  Caffe::set_mode(Caffe::GPU);
//  Caffe::set_solver_count(gpus.size());

  LOG(INFO)<< "Using GPU with device ID " << gpu_id;

  shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  if (init_file.extension().string() == ".solverstate") {
    LOG(INFO)<< "Resuming from " << init_file;
    solver->Restore(init_file.c_str());
  }
  else if (init_file.extension().string() == ".caffemodel") {
    LOG(INFO) << "Copying weights from " << init_file;
    solver->net()->CopyTrainedLayersFrom(init_file.string());
  }
  else {
    LOG(WARNING) << "No initialization. Training from scratch";
  }


  using DataLayerType = caffe::ArticulatedObjectsDataLayer<float>;
  auto data_layer_ptr = boost::dynamic_pointer_cast<DataLayerType>(solver->net()->layers()[0]);
  for (const fs::path& dataset_file : dataset_files) {
    LOG(INFO) << "Loading dataset annotations from " <<  dataset_file;
    data_layer_ptr->addDataset(loadDatasetFromJson(dataset_file.string()));
  }
  data_layer_ptr->generateDatumIds();

  LOG(INFO)<< "Starting Optimization";
  solver->Solve();
  LOG(INFO)<< "Optimization Done.";
}

int main(int argc, char **argv) {
  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("datasets,d",  po::value<std::vector<fs::path>>(), "Path to dataset files (JSON)")
        ("network_prototxt,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("solver_prototxt,s",  po::value<fs::path>()->required(), "Path to Solver param file (prototxt)")
        ("initialization,i",  po::value<fs::path>()->required(), "Path to initialization weights/solverstate (prototxt)")
        ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids")
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

  const fs::path net_model_file = vm["network_prototxt"].as<fs::path>();
  const fs::path solver_params_file = vm["solver_prototxt"].as<fs::path>();
  const fs::path initialization_file = vm["initialization"].as<fs::path>();
  const int gpu_id = vm["gpu_id"].as<int>();

  train(solver_params_file, dataset_files, initialization_file, gpu_id);


  return EXIT_SUCCESS;
}



