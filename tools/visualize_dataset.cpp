/**
 * @file visualize_dataset.cpp
 * @brief visualize_dataset
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char **argv) {
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("dataset,d",  po::value<std::string>(), "Path to dataset file (JSON)")
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
    std::cout << "Please provide dataset file" << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  const fs::path dataset_file(vm["dataset"].as<std::string>());
  if (!fs::exists(dataset_file)) {
    std::cout << "Error:" << dataset_file << " does not exist\n";
    return EXIT_FAILURE;
  }

  RaC::Dataset dataset = RaC::loadDatasetFromJson(dataset_file.string());
  std::cout << "Loaded dataset \"" << dataset.name << "\" with " << dataset.annotations.size() << " annotations" << std::endl;

  const int num_of_annotations = dataset.annotations.size();

  const std::string windowname = "ImageViewer";
  cv::namedWindow(windowname, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

  std::cout << "Press \"P\" to PAUSE/UNPAUSE\n";
  std::cout << "Use Arrow keys or WASD keys for prev/next images\n";

  bool paused = true;
  int step = 1;
  for (int i = 0;;) {
    const RaC::Annotation& anno = dataset.annotations[i];
    const std::string image_path = (dataset.rootdir / anno.image_file.value()).string();
    cv::Mat cv_image = cv::imread(image_path, cv::IMREAD_COLOR);
    std::cout << anno << std::endl;

    cv::imshow(windowname, cv_image);
    const int key = cv::waitKey(!paused) % 256;

    if (key == 27 || key == 'q')  // Esc or Q or q
      break;
    else if (key == 123 || key == 125 || key == 50 || key == 52 || key == 81 || key == 84 || key == 'a' || key == 's')  // Down or Left Arrow key (including numpad) or 'a' and 's'
      step = -1;
    else if (key == 124 || key == 126 || key == 54 || key == 56 || key == 82 || key == 83 || key == 'w' || key == 'd')  // Up or Right Arrow or 'w' or 'd'
      step = 1;
    else if (key == 'p' || key == 'P')
      paused = !paused;

    i += step;
    i = std::max(0, i);
    i = std::min(i, num_of_annotations - 1);
  }

  return EXIT_SUCCESS;
}
