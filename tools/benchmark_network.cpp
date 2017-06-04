/**
 * @file benchmark_network.cpp
 * @brief benchmark_network
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/SMPLRenderLayer.h"
#include "RenderAndCompare/SMPLRenderWithLossLayer.h"
#include "RenderAndCompare/ArticulatedObjectsDataLayer.h"
#include "RenderAndCompare/SegmAccuracyLayer.h"
#include "RenderAndCompare/Dataset.h"

#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <QGuiApplication>

namespace caffe {
REGISTER_LAYER_CLASS(SMPLRender);
REGISTER_LAYER_CLASS(SMPLRenderWithLoss);
REGISTER_LAYER_CLASS(ArticulatedObjectsData);
REGISTER_LAYER_CLASS(SegmAccuracy);
}  // namespace caffe


using caffe::Caffe;
using caffe::Net;
using caffe::Blob;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::Layer;
using caffe::vector;

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace RaC;

// Time: benchmark the execution time of a model.
void benchmark(const fs::path& net_proto_file,
               const std::vector<fs::path>& dataset_files,
               const int num_iters,
               const int gpu_id) {
  // Set device id and mode
  if (gpu_id >= 0) {
    LOG(INFO) << "Using GPU with device ID " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(net_proto_file.string(), caffe::TRAIN);

  LOG(INFO) << "Setting Data";
  using DataLayerType = caffe::ArticulatedObjectsDataLayer<float>;
  auto data_layer_ptr = boost::dynamic_pointer_cast<DataLayerType>(caffe_net.layers()[0]);
  for (const fs::path& dataset_file : dataset_files) {
    LOG(INFO) << "Loading dataset annotations from " <<  dataset_file;
    data_layer_ptr->addDataset(loadDatasetFromJson(dataset_file.string()));
  }
  data_layer_ptr->generateDatumIds();
  LOG(INFO) << "Done Setting Data";

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
  LOG(INFO) << "=============== Benchmark begins ===============";
  LOG(INFO) << "Testing for " << num_iters << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < num_iters; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (std::size_t i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 / num_iters << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 / num_iters << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 / num_iters << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 / num_iters << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() / num_iters << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "=============== Benchmark ends =================";
}

int main(int argc, char **argv) {
  QGuiApplication app(argc, argv);

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("datasets,d",  po::value<std::vector<fs::path>>(), "Path to dataset files (JSON)")
        ("network_prototxt,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("num_iters,i",  po::value<int>()->default_value(10), "Num of iterations")
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
  const int num_iters = vm["num_iters"].as<int>();
  const int gpu_id = vm["gpu_id"].as<int>();

  benchmark(net_model_file, dataset_files, num_iters, gpu_id);

  return EXIT_SUCCESS;
}


