/**
 * @file visualize_image_dataset.cpp
 * @brief visualize_image_dataset
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/ImageDataset.h"
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


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

  RaC::ImageDataset dataset = RaC::loadImageDatasetFromJson(dataset_file.string());
  std::cout << "Loaded Image dataset \"" << dataset.name << "\" with " << dataset.annotations.size() << " images" << std::endl;

  const int num_of_annotations = dataset.annotations.size();

  const std::string windowname = "ImageViewer";
  cv::namedWindow(windowname, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

  std::cout << "Press \"P\" to PAUSE/UNPAUSE\n";
  std::cout << "Use Arrow keys or WASD keys for prev/next images\n";

  bool paused = true;
  int step = 1;
  for (int i = 0;;) {
    const RaC::ImageInfo& image_info = dataset.annotations[i];

    {
      std::cout << "---------------------------------------------------\n";
      nlohmann::json image_info_json(image_info);
      std::cout << image_info_json.dump(2) << std::endl;
    }

    const std::string image_path = (dataset.rootdir / image_info.image_file.value()).string();
    cv::Mat cv_image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image_info.objects) {
      // Loop over all objects
      for (const RaC::ImageObjectInfo& obj_info : image_info.objects.value()) {

        if(obj_info.bbx_visible) {
          auto bbx_visible = obj_info.bbx_visible.value();
          cv::rectangle(cv_image, cv::Point(bbx_visible[0], bbx_visible[1]), cv::Point(bbx_visible[2], bbx_visible[3]), cv::Scalar( 255, 0, 0));
        }

        if(obj_info.bbx_amodal) {
          auto bbx_amodal = obj_info.bbx_amodal.value();
          cv::rectangle(cv_image, cv::Point(bbx_amodal[0], bbx_amodal[1]), cv::Point(bbx_amodal[2], bbx_amodal[3]), cv::Scalar( 255, 255, 0));
        }

        if(obj_info.origin_proj) {
          auto origin_proj = obj_info.origin_proj.value();
          cv::circle(cv_image, cv::Point(origin_proj[0], origin_proj[1]), 5, cv::Scalar( 0, 0, 255 ), -1);
        }

      }
    }

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


