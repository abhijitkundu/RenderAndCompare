/**
 * @file check_smpl_render_layer.cpp
 * @brief check_smpl_render_layer
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/SMPLRenderLayer.h"
#include "RenderAndCompare/SMPLRenderWithLossLayer.h"
#include "RenderAndCompare/ArticulatedObjectsDataLayer.h"
#include "RenderAndCompare/SegmAccuracyLayer.h"
#include "RenderAndCompare/Dataset.h"
#include "RenderAndCompare/ImageUtils.h"

#include "caffe/caffe.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QApplication>

namespace caffe {
REGISTER_LAYER_CLASS(SMPLRender);
REGISTER_LAYER_CLASS(SMPLRenderWithLoss);
REGISTER_LAYER_CLASS(ArticulatedObjectsData);
REGISTER_LAYER_CLASS(SegmAccuracy);
}  // namespace caffe

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;

int main(int argc, char **argv) {
  QApplication app(argc, argv);

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("dataset",  po::value<fs::path>(), "Path to dataset files (JSON)")
        ("network_model,n",  po::value<fs::path>()->required(), "Path to network model file (prototxt)")
        ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids (Use -ve value to force CPU)")
        ("pause_time,p",  po::value<int>()->default_value(0), "Pause time. Use 0 for pause")
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
    std::cout << "Please provide atleast one dataset file" << '\n';
    std::cout << cmdline_options << '\n';
    return EXIT_FAILURE;
  }

  const fs::path dataset_file = vm["dataset"].as<fs::path>();
  const fs::path net_model_file = vm["network_model"].as<fs::path>();
  const int gpu_id = vm["gpu_id"].as<int>();
  const int pause_time = vm["pause_time"].as<int>();

  if (gpu_id >= 0) {
    LOG(INFO) << "Using GPU with device ID " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Dataset dataset = loadDatasetFromJson(dataset_file.string());
  const int num_of_images = dataset.annotations.size();

  LOG(INFO)<< "Instantiating network model from "<< net_model_file;
  Net<float> caffe_net(net_model_file.string(), caffe::TEST);
  LOG(INFO)<< "Network Instantiation successful";

  using DataLayerType = caffe::ArticulatedObjectsDataLayer<float>;
  auto data_layer_ptr = boost::dynamic_pointer_cast<DataLayerType>(caffe_net.layers()[0]);
  CHECK_NOTNULL(data_layer_ptr.get());
  data_layer_ptr->addDataset(dataset);
  data_layer_ptr->generateDatumIds();


  auto image_data_blob_ptr =  caffe_net.blob_by_name("data");
  CHECK_NOTNULL(image_data_blob_ptr.get());
  auto data_blob_shape = image_data_blob_ptr->shape();
  CHECK_EQ(data_blob_shape.size(), 4) << "Expects 4D data blob";
  CHECK_EQ(data_blob_shape[1], 3) << "Expects 2nd channel to be 3 for BGR image";
  const int batch_size = data_blob_shape[0];
  const int num_of_batches = int(std::ceil(num_of_images / float(batch_size)));

  auto segm_image_blob_shape = caffe_net.blob_by_name("gt_segm_image")->shape();
  CHECK_EQ(segm_image_blob_shape.size(), 4) << "Expects 4D data blob";
  CHECK_EQ(segm_image_blob_shape[1], 1) << "Expects 2nd channel to be 1 for Segm image";
  CHECK_EQ(segm_image_blob_shape[0], batch_size) << "Expects 1st dime to be batch size";

  cv::namedWindow("InputImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::namedWindow("SegmImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::namedWindow("BlobInputImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::namedWindow("BlobSegmImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::namedWindow("RenderedImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);


  const std::vector<cv::Vec3b> smpl24_cmap = {{55, 55, 55},
                                              {47, 148, 84},
                                              {231, 114, 177},
                                              {89, 70, 0},
                                              {187, 43, 143},
                                              {7, 70, 70},
                                              {251, 92, 0},
                                              {63, 252, 211},
                                              {53, 144, 229},
                                              {248, 150, 144},
                                              {52, 82, 125},
                                              {189, 73, 1},
                                              {42, 210, 52},
                                              {253, 192, 40},
                                              {231, 233, 157},
                                              {109, 131, 203},
                                              {190, 195, 243},
                                              {97, 70, 171},
                                              {32, 137, 233},
                                              {68, 43, 29},
                                              {142, 35, 220},
                                              {243, 169, 53},
                                              {119, 8, 153},
                                              {217, 181, 152},
                                              {32, 91, 213}};


  for (int b = 0; b < num_of_batches; ++b) {
    const int start_idx = batch_size * b;
    const int end_idx = std::min(batch_size * (b + 1), num_of_images);
    LOG(INFO) << "Working on batch: " << b  << "/" << num_of_batches
              << " (Images# " << start_idx  << "-" << end_idx << ")";

    caffe_net.Forward();

    {
      auto segm_class_iou_loss_blob_ptr =  caffe_net.blob_by_name("segm_class_iou_loss");
      if (segm_class_iou_loss_blob_ptr)
        LOG(INFO) << "class_iou_loss = " << segm_class_iou_loss_blob_ptr->cpu_data()[0];
    }

    {
      auto segm_pixel_acc_blob_ptr =  caffe_net.blob_by_name("segm_pixel_acc");
      auto segm_class_acc_blob_ptr =  caffe_net.blob_by_name("segm_class_acc");
      auto segm_class_iou_blob_ptr =  caffe_net.blob_by_name("segm_class_iou");

      if (segm_pixel_acc_blob_ptr && segm_class_acc_blob_ptr && segm_class_iou_blob_ptr) {

        float mean_pixel_acc = segm_pixel_acc_blob_ptr->cpu_data()[0];
        float mean_class_acc = segm_class_acc_blob_ptr->cpu_data()[0];
        float mean_class_iou = segm_class_iou_blob_ptr->cpu_data()[0];

        LOG(INFO) << "class_iou= " << mean_class_iou << " class_acc= " << mean_class_acc << " pixel_acc= " << mean_pixel_acc;
      }
      else {
        LOG(INFO) << "No Segmentation Accuracy Data";
      }
    }

    auto gt_shape_param_blob_ptr =  caffe_net.blob_by_name("gt_shape_param");
    auto gt_pose_param_blob_ptr =  caffe_net.blob_by_name("gt_pose_param");

    auto input_image_blob_ptr =  caffe_net.blob_by_name("data");
    auto segm_image_blob_ptr =  caffe_net.blob_by_name("gt_segm_image");
    auto rendered_image_blob_ptr =  caffe_net.blob_by_name("rendered_image");

    CHECK_NOTNULL(input_image_blob_ptr.get());
    CHECK_NOTNULL(segm_image_blob_ptr.get());

    if (!gt_shape_param_blob_ptr) LOG(INFO)<< "No shape Data";
    if (!gt_pose_param_blob_ptr)  LOG(INFO)<< "No pose Data";
    if (!rendered_image_blob_ptr) LOG(INFO)<< "No RenderedImage Data";

    using Tensor4f = Eigen::Tensor<const float , 4, Eigen::RowMajor>;
    Eigen::TensorMap<Tensor4f>input_image_blob(input_image_blob_ptr->cpu_data(), batch_size, 3, data_blob_shape[2], data_blob_shape[3]);
    Eigen::TensorMap<Tensor4f>segm_image_blob(segm_image_blob_ptr->cpu_data(), batch_size, 1, segm_image_blob_shape[2], segm_image_blob_shape[3]);

    bool exit_loop = false;
    for (int idx = start_idx; idx < end_idx; ++idx) {
      int rel_idx = idx - start_idx;
      const Annotation& anno = dataset.annotations[idx];

      {
        const std::string image_path = (dataset.rootdir / anno.image_file.value()).string();
        cv::Mat cv_image = cv::imread(image_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        cv::imshow("InputImage", cv_image);
      }

      {
        const std::string image_path = (dataset.rootdir / anno.segm_file.value()).string();
        cv::Mat cv_image = cv::imread(image_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        cv::imshow("SegmImage", getColoredImageFromLabels(cv_image, smpl24_cmap));
      }

      if (gt_shape_param_blob_ptr) {
        auto shape_param_blob_shape = gt_shape_param_blob_ptr->shape();
        CHECK_EQ(shape_param_blob_shape.size(), 2) << "Expects 2D data blob";
        CHECK_EQ(shape_param_blob_shape[0], batch_size);
        const auto shape_param_size = shape_param_blob_shape[1];

        Eigen::VectorXd anno_shape_param = anno.shapeParam();
        CHECK_EQ(anno_shape_param.size(), shape_param_size);

        Eigen::Map<const Eigen::VectorXf> blob_shape_param(gt_shape_param_blob_ptr->cpu_data() + rel_idx * shape_param_size, shape_param_size);
        CHECK(anno_shape_param.cast<float>().isApprox(blob_shape_param));
      }

      if (gt_pose_param_blob_ptr) {
        auto pose_param_blob_shape = gt_pose_param_blob_ptr->shape();
        CHECK_EQ(pose_param_blob_shape.size(), 2) << "Expects 2D data blob";
        CHECK_EQ(pose_param_blob_shape[0], batch_size);
        const auto pose_param_size = pose_param_blob_shape[1];

        Eigen::VectorXd anno_pose_param = anno.poseParam();
        CHECK_EQ(anno_pose_param.size(), pose_param_size);

        Eigen::Map<const Eigen::VectorXf> blob_pose_param(gt_pose_param_blob_ptr->cpu_data() + rel_idx * pose_param_size, pose_param_size);
        CHECK(anno_pose_param.cast<float>().isApprox(blob_pose_param));
      }

      {
        using Tensor3u = Eigen::Tensor<unsigned char , 3, Eigen::RowMajor>;
        Tensor3u image = segm_image_blob.chip(rel_idx, 0).cast<unsigned char>();
        CHECK_EQ(image.dimension(0), 1);
        cv::Mat cv_image(image.dimension(1), image.dimension(2), CV_8UC1, image.data());
        cv::flip(cv_image, cv_image, 0); // Can be done with Eigen tensor reverse also
        cv::imshow("BlobSegmImage", getColoredImageFromLabels(cv_image, smpl24_cmap));
      }

      if (rendered_image_blob_ptr) {
        const int H = rendered_image_blob_ptr->height();
        const int W = rendered_image_blob_ptr->width();

        using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const Image> image(rendered_image_blob_ptr->cpu_data() + rendered_image_blob_ptr->offset(rel_idx, 0), H, W);

        using Image8UC1 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Image8UC1 image_8UC1 = image.template cast<unsigned char>();
        image_8UC1.colwise().reverseInPlace();

        cv::Mat cv_image(image_8UC1.rows(), image_8UC1.cols(), CV_8UC1, (void*)image_8UC1.data());
        cv::imshow("RenderedImage", getColoredImageFromLabels(cv_image, smpl24_cmap));
      }

      {
        using Tensor3f = Eigen::Tensor<float , 3, Eigen::RowMajor>;
        using Tensor3u = Eigen::Tensor<unsigned char , 3, Eigen::RowMajor>;

        Tensor3f image = input_image_blob.chip(rel_idx, 0);

        const Eigen::Vector3f& mean_bgr = data_layer_ptr->mean_bgr();
        image.chip(0, 0) = image.chip(0, 0) + mean_bgr[0];
        image.chip(1, 0) = image.chip(1, 0) + mean_bgr[1];
        image.chip(2, 0) = image.chip(2, 0) + mean_bgr[2];

        const Eigen::array<ptrdiff_t, 3> shuffles({{1, 2, 0}});
        Tensor3u image_bgr = image.cast<unsigned char>().shuffle(shuffles);
        cv::Mat cv_image(image_bgr.dimension(0), image_bgr.dimension(1), CV_8UC3, image_bgr.data());
        cv::imshow("BlobInputImage", cv_image);
        const int key = cv::waitKey(pause_time) % 256;

        // Esc or Q or q
        if (key == 27 || key == 'q' || key == 'Q') {
          exit_loop = true;
          break;
        }
      }
    } // end loop idx \in [start_idx --- end_idx)

    if (exit_loop)
      break;

  } // end loop b \in [0 --- num_of_batches)


  return EXIT_SUCCESS;
}


