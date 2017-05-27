/**
 * @file demo_image_loaders.cpp
 * @brief demo_image_loaders
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ImageLoaders.h"
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

  using ImageLoader = BatchImageLoader<uint8_t>;
  ImageLoader image_loader(224, 224);

  for (const Dataset& dataset : datasets) {
    std::vector<std::string> image_files(dataset.annotations.size());
    Eigen::AlignedStdVector<Eigen::Vector4i> visible_boxes(dataset.annotations.size());
#pragma omp parallel for
    for (std::size_t i = 0; i< image_files.size(); ++i) {
      const Annotation& anno = dataset.annotations[i];
      image_files[i] = (dataset.rootdir / anno.image_file.value()).string();
      visible_boxes[i] = Eigen::Vector4d(anno.bbx_visible.value().data()).cast<int>();
    }
//    image_loader.preloadImages(image_files);
    image_loader.preloadImages(image_files, visible_boxes);
  }


  const std::string windowname = "ImageViewer";
  cv::namedWindow(windowname, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

  std::cout << "Press \"P\" to PAUSE/UNPAUSE\n";
  std::cout << "Use Arrow keys or WASD keys for prev/next images\n";

  const int num_of_images = image_loader.images().size();

  bool paused = true;
  int step = 1;
  for (int i = 0;;) {
    const Eigen::array<ptrdiff_t, 3> shuffles({{1, 2, 0}});
    ImageLoader::ImageType image = image_loader.images()[i].shuffle(shuffles);
    cv::Mat cv_image(image_loader.height(), image_loader.width(), CV_8UC3, image.data());

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
    i = std::min(i, num_of_images - 1);
  }

  return EXIT_SUCCESS;
}


