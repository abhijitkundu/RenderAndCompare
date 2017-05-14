/**
 * @file check_cpp_data_layer.cpp
 * @brief check_cpp_data_layer
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"

#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>

int main(int argc, char **argv) {

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("datasets,d",  po::value<std::vector<fs::path>>(), "Path to dataset files (JSON)")
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


  return EXIT_SUCCESS;
}


