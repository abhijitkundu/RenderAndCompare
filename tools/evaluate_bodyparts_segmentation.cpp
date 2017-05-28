/**
 * @file evaluate_bodyparts_segmentation.cpp
 * @brief evaluate_bodyparts_segmentation
 *
 * @author Abhijit Kundu
 */

#include <CuteGL/IO/H5EigenDense.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

Eigen::MatrixXi readSegmH5File(const std::string& filepath) {
  H5::H5File file(filepath, H5F_ACC_RDONLY);
  Eigen::MatrixXi segmm;
  H5Eigen::load(file, "segmm", segmm);
  assert(segmm.maxCoeff() < 25);
  return segmm;
}

void merge24LabelsTo14(Eigen::MatrixXi& smpl_p24_label_img, const std::map<int, int>& label_map) {
  for (Eigen::Index y = 0; y < smpl_p24_label_img.rows(); ++y)
    for (Eigen::Index x = 0; x < smpl_p24_label_img.cols(); ++x) {
      smpl_p24_label_img(y, x) = label_map.at(smpl_p24_label_img(y, x));
    }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("split_file,s", po::value<fs::path>()->required(), "Path to list of images")
        ("gt_folder,g", po::value<fs::path>()->required(), "GT Directory")
        ("result_folder,r", po::value<fs::path>()->required(), "Output dir")
        ("use_24_parts", po::value<bool>()->default_value(false), "Use 24 parts instead of 14 parts")
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

  const fs::path split_file(vm["split_file"].as<fs::path>());
  if (!fs::exists(split_file)) {
    std::cout << "Error:" << split_file << " does not exist\n";
    return EXIT_FAILURE;
  }

  const fs::path gt_folder(vm["gt_folder"].as<fs::path>());
  const fs::path result_folder(vm["result_folder"].as<fs::path>());

  const bool use_24_parts = vm["use_24_parts"].as<bool>();

  // number of labels
  const int M = use_24_parts ? 25 : 15;
  std::cout << "Number of labels = " << M << "\n";

  if (!fs::exists(gt_folder) || !fs::is_directory(gt_folder)) {
    std::cout << "Provided GTFolder is invalid: " << gt_folder << "\n";
    return EXIT_FAILURE;
  }

  if (!fs::exists(result_folder) || !fs::is_directory(result_folder)) {
    std::cout << "Provided ResultFolder is invalid: " << result_folder << "\n";
    return EXIT_FAILURE;
  }

  std::vector<std::string> image_names;
  {
    std::ifstream file(split_file.c_str());
    if (!file) {
      std::cout << "Error opening output file: " << split_file << std::endl;
      return EXIT_FAILURE;
    }

    for (std::string line; std::getline( file, line ); /**/ )
      image_names.push_back( line );

    file.close();
  }


  std::array<std::string, 25> parts_24 = { "background", "hips", "leftUpLeg", "rightUpLeg", "spine", "leftLeg", "rightLeg",
      "spine1", "leftFoot", "rightFoot", "spine2", "leftToeBase", "rightToeBase", "neck", "leftShoulder",
      "rightShoulder", "head", "leftArm", "rightArm", "leftForeArm", "rightForeArm", "leftHand", "rightHand",
      "leftHandIndex1", "rightHandIndex1" };

  std::array<std::string, 15> parts_14 = {  "background",
                                            "head",
                                            "torso",
                                            "leftUpLeg", "rightUpLeg",
                                            "leftLowLeg", "rightLowLeg",
                                            "leftUpArm", "rightUpArm",
                                            "leftLowArm", "rightLowArm",
                                            "leftHand", "rightHand",
                                            "leftFoot", "rightFoot"};

  using PartsMapType = std::map<std::string, std::vector<std::string> >;

  PartsMapType partsmap_14_to_24 ={ {"background", {"background"}},
                                    {"head", {"head"}},
                                    {"torso", {"hips", "spine", "spine1", "spine2", "neck", "leftShoulder", "rightShoulder"}},
                                    {"leftUpLeg", {"leftUpLeg"}},
                                    {"rightUpLeg", {"rightUpLeg"}},
                                    {"leftLowLeg", {"leftLeg"}},
                                    {"rightLowLeg", {"rightLeg"}},
                                    {"leftUpArm", {"leftArm"}},
                                    {"rightUpArm", {"rightArm"}},
                                    {"leftLowArm", {"leftForeArm"}},
                                    {"rightLowArm", {"rightForeArm"}},
                                    {"leftHand", {"leftHand", "leftHandIndex1"}},
                                    {"rightHand", {"rightHand", "rightHandIndex1"}},
                                    {"leftFoot", {"leftFoot", "leftToeBase"}},
                                    {"rightFoot", {"rightFoot", "rightToeBase"}} };

  assert (partsmap_14_to_24.size() == 15);

  using LabelMapType = std::map<int, int>;
  LabelMapType lalbelmap_24_to_14;
  for (const auto &key_vals : partsmap_14_to_24) {
    auto p14_id = std::distance(parts_14.begin(), std::find(parts_14.begin(), parts_14.end(), key_vals.first));
    for (const auto& p24 : key_vals.second) {
      auto p24_id = std::distance(parts_24.begin(), std::find(parts_24.begin(), parts_24.end(), p24));
      assert(p24_id <  25);
      lalbelmap_24_to_14[p24_id] = p14_id;
    }
  }

  if (lalbelmap_24_to_14.size() != 25) {
    std::cout << "lalbelmap_24_to_14.size() != 25\n";
    return EXIT_FAILURE;
  }

  std::cout << "[";
  for (const auto& key_and_val : lalbelmap_24_to_14) {
    std::cout << key_and_val.second << ", ";
  }
  std::cout << "\b\b]";


  std::cout << "Evaluating on " << image_names.size() << " Images" << std::endl;


  int total_pixels = 0;
  int ok_pixels = 0;

  Eigen::VectorXi total_pixels_class = Eigen::VectorXi::Zero(M);
  Eigen::VectorXi ok_pixels_class = Eigen::VectorXi::Zero(M);
  Eigen::VectorXi label_pixels = Eigen::VectorXi::Zero(M);

  boost::progress_display show_progress(image_names.size());
  for (const std::string& image_name : image_names) {
    fs::path gt_image_fp = gt_folder / fs::path(image_name);
    if (!fs::exists(gt_image_fp) || !fs::is_regular_file(gt_image_fp)) {
      std::cout << "Invalid Image path: " << gt_image_fp << "\n";
      return EXIT_FAILURE;
    }

    fs::path result_image_fp = result_folder / fs::path(image_name);
    if (!fs::exists(result_image_fp) || !fs::is_regular_file(result_image_fp)) {
      std::cout << "Invalid Image path: " << result_image_fp << "\n";
      return EXIT_FAILURE;
    }

    Eigen::MatrixXi gt_image = readSegmH5File(gt_image_fp.string());
    Eigen::MatrixXi result_image = readSegmH5File(result_image_fp.string());



    if((gt_image.rows() != result_image.rows()) || (gt_image.cols() != result_image.cols())) {
      std::cout << "ERROR: gt_image.size != result_image.size\n";
      return EXIT_FAILURE;
    }

    if (!use_24_parts) {
      merge24LabelsTo14(gt_image, lalbelmap_24_to_14);
      merge24LabelsTo14(result_image, lalbelmap_24_to_14);
    }


    {
      for (Eigen::Index y = 0; y < gt_image.rows(); ++y)
        for (Eigen::Index x = 0; x < gt_image.cols(); ++x) {
          int gt_label = gt_image(y, x);

          ++total_pixels;
          ++total_pixels_class[gt_label];

          int result_label = result_image(y, x);
          ++label_pixels[result_label];

          if (gt_label == result_label) {
            ++ok_pixels;
            ++ok_pixels_class[gt_label];
          }
        }
    }

    ++show_progress;
  }

  Eigen::VectorXd iou_scores = 100.0 * ok_pixels_class.cast<double>().array()
      / (total_pixels_class + label_pixels - ok_pixels_class).cast<double>().array();

  Eigen::VectorXd class_pixel_acc = 100.0 * ok_pixels_class.cast<double>().array() / label_pixels.cast<double>().array();

  std::cout << "ClassIOU= " << iou_scores.transpose() << std::endl;
  std::cout << "MeanIOU= " << iou_scores.mean() << std::endl;
  std::cout << "ClassPixelAccuracy= " << class_pixel_acc.transpose() << std::endl;
  std::cout << "MeanClassPixelAccuracy= " << class_pixel_acc.mean() << std::endl;
  std::cout << "GlobalPixelAccuracy= " << (ok_pixels * 100.0) / double (total_pixels) << std::endl;

  return EXIT_SUCCESS;
}



