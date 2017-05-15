/**
 * @file check_network_model.cpp
 * @brief check_network_model
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ImageUtils.h"
#include "RenderAndCompare/SMPLRenderLayer.h"
#include "RenderAndCompare/render_layer.hpp"


#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <QGuiApplication>

namespace caffe {

REGISTER_LAYER_CLASS(Render);
REGISTER_LAYER_CLASS(SMPLRender);

}  // namespace caffe

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;

int main(int argc, char **argv) {
  QGuiApplication app(argc, argv);

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("network_model,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids")
        ("use_cpu",  po::value<bool>()->default_value(false), "Use CPU")
        ;

  po::options_description cmdline_options;
  cmdline_options.add(generic_options).add(config_options);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
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
  // Instantiate the caffe net.
  Net<float> caffe_net(net_model_file.string(), caffe::TEST);

  LOG(INFO)<< "Network Instantiation successful";

  LOG(INFO)<< "Doing Forward pass";
  const std::vector<Blob<float>*> output = caffe_net.Forward();
  LOG(INFO)<< "Done Forward pass";

  LOG(INFO)<< "Output Shape = " << output[0]->shape_string();


  for (int i=0; i < output[0]->num(); ++i) {
    using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const Image> image(output[0]->cpu_data() + output[0]->offset(i, 0), output[0]->height(), output[0]->width());
    RaC::saveImage(image, "image_"+ std::to_string(i) + ".png");
  }

  return EXIT_SUCCESS;
}


