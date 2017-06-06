/**
 * @file demo_compute_iou.cpp
 * @brief demo_compute_iou
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/SegmentationAccuracy.h"
#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ImageLoaders.h"
#include <boost/program_options.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>


int main(int argc, char **argv) {
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("dataset,d",  po::value<fs::path>(), "Path to dataset file (JSON)")
        ("images_per_blob,i", po::value<int>()->default_value(400), "number of images per blob")
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
    std::cout << "Please provide one dataset file" << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  const fs::path dataset_file(vm["dataset"].as<fs::path>());
  if (!fs::exists(dataset_file)) {
    std::cout << "Error: " << dataset_file << " does not exist\n";
    return EXIT_FAILURE;
  }

  const int images_per_blob = vm["images_per_blob"].as<int>();

  std::cout << "Loading dataset annotation from " <<  dataset_file << std::endl;
  Dataset dataset = loadDatasetFromJson(dataset_file.string());
  std::cout << "Loaded datatset with " << dataset.annotations.size() << " annotations." << std::endl;

  if (int(dataset.annotations.size()) < 2 * images_per_blob) {
    std::cout << "Not enough images to create blobs\n";
    return EXIT_FAILURE;
  }

  // Create two blobs of images pred_images, and, gt_images
  using Images = Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>;
  Images gt_images(images_per_blob, 1, 240, 320);
  Images pred_images(images_per_blob, 1, 240, 320);

  {
    using ImageLoader = BatchImageLoader<uint8_t, 1>;
    ImageLoader image_loader(320, 240);

    {
      std::vector<std::string> segm_files(images_per_blob * 2);
  #pragma omp parallel for
      for (std::size_t i = 0; i < segm_files.size(); ++i) {
        const Annotation& anno = dataset.annotations[i];
        segm_files[i] = (dataset.rootdir / anno.segm_file.value()).string();
      }
      image_loader.preloadImages(segm_files);
    }
    for (int i = 0; i < images_per_blob; ++i) {
      gt_images.chip(i, 0) = image_loader.images()[2*i];
      pred_images.chip(i, 0) = image_loader.images()[2*i + 1];
    }
  }

  cudaCheckError(cudaSetDevice(0));

  {
    std::cout << "------------------------------------------------" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    float mean_iou = 0;
    int count = 0;
    for (int i = 0; i < images_per_blob; ++i) {
      using Image8UC1 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<Image8UC1> gt_image(&gt_images(i, 0, 0, 0), 240, 320);
      Eigen::Map<Image8UC1> pred_image(&pred_images(i, 0, 0, 0), 240, 320);
      mean_iou += computeIoU(gt_image, pred_image);
      ++ count;
    }
    mean_iou /= count;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Mean IoU= " << mean_iou << std::endl;
    std::cout << "Time = " << elapsed_seconds.count() * 1000 << " ms\n";
  }

  {
    std::cout << "------------------------------------------------" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    float mean_iou = computeIoU(gt_images, pred_images);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Mean IoU= " << mean_iou << std::endl;
    std::cout << "Time = " << elapsed_seconds.count() * 1000 << " ms\n";
  }

//  {
//    std::cout << "------------------------------------------------" << std::endl;
//    std::chrono::time_point<std::chrono::system_clock> start, end;
//    start = std::chrono::system_clock::now();
//    float mean_iou = computeIoUwithCUDA(gt_images, pred_images);
//    end = std::chrono::system_clock::now();
//    std::chrono::duration<double> elapsed_seconds = end-start;
//    std::cout << "Mean IoU= " << mean_iou << std::endl;
//    std::cout << "Time = " << elapsed_seconds.count() * 1000 << " ms\n";
//  }


  {
    std::cout << "---computeHistogramWithCPU---------------------------------------------" << std::endl;
    using Image8UC1 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Image8UC1> gt_image(&gt_images(0, 0, 0, 0), 240, 320);
    Eigen::VectorXi hist = computeHistogramWithCPU(gt_image);
    hist = computeHistogramWithCPU(gt_image);
    hist = computeHistogramWithCPU(gt_image);
    hist = computeHistogramWithCPU(gt_image);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << "hist = " << hist.format(fmt) << "\n";
  }

  {
    std::cout << "---computeHistogramWithAtomics---------------------------------------------" << std::endl;
    Eigen::VectorXi hist(25);
    computeHistogramWithAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << "hist = " << hist.format(fmt) << "\n";
  }

  {
    std::cout << "---computeHistogramWithSharedAtomics---------------------------------------------" << std::endl;
    Eigen::VectorXi hist(25);
    computeHistogramWithSharedAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedAtomics(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << "hist = " << hist.format(fmt) << "\n";
  }

  {
    std::cout << "---computeHistogramWithSharedBins---------------------------------------------" << std::endl;
    Eigen::VectorXi hist(25);
    computeHistogramWithSharedBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithSharedBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << "hist = " << hist.format(fmt) << "\n";
  }

  {
    std::cout << "---computeHistogramWithPrivateBins---------------------------------------------" << std::endl;
    Eigen::VectorXi hist(25);
    computeHistogramWithPrivateBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithPrivateBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithPrivateBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    computeHistogramWithPrivateBins(&gt_images(0, 0, 0, 0), 320, 240, hist.data(), 25);
    const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << "hist = " << hist.format(fmt) << "\n";
  }

  {
    std::cout << "------------------------------------------------" << std::endl;
    gt_images = gt_images * uint8_t(10);
    pred_images = pred_images * uint8_t(10);
    for (int i = 0; i < images_per_blob; ++i) {
      using Image8UC1 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<Image8UC1> gt_image(&gt_images(i, 0, 0, 0), 240, 320);
      Eigen::Map<Image8UC1> pred_image(&pred_images(i, 0, 0, 0), 240, 320);

      {
        cv::Mat cv_image(240, 320, CV_8UC1, gt_image.data());
        cv::imshow("gt_image", cv_image);
      }
      {
        cv::Mat cv_image(240, 320, CV_8UC1, pred_image.data());
        cv::imshow("pred_image", cv_image);
      }
      cv::waitKey(1);
    }
  }

}

