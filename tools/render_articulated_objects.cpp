/**
 * @file render_articulated_objects.cpp
 * @brief render_articulated_objects
 *
 * @author Abhijit Kundu
 */

#include "CuteGL/Renderer/SMPLRenderer.h"
#include "CuteGL/Surface/OffScreenRenderViewer.h"
#include <CuteGL/Core/PoseUtils.h>
#include "RenderAndCompare/Dataset.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QGuiApplication>
#include <memory>
#include <iostream>

struct PosePCADecoder {

  PosePCADecoder(const std::string pose_pca_file) {
    H5::H5File file(pose_pca_file, H5F_ACC_RDONLY);
    H5Eigen::load(file, "pose_mean", pose_mean);
    H5Eigen::load(file, "pose_basis", pose_basis);
    H5Eigen::load(file, "encoded_training_data", encoded_training_data);
    assert(pose_basis.rows() == 69);
    assert(pose_mean.size() == 69);
  }


  template <class Derived>
  Eigen::VectorXf operator()(const Eigen::MatrixBase<Derived>& encoded_pose) const {
    Eigen::Matrix<float, 72, 1> pose_param;
    if (encoded_pose.size() == pose_basis.cols())
      pose_param.tail<69>() = pose_mean + pose_basis * encoded_pose;
    else if (encoded_pose.size() == 69)
      pose_param.tail<69>() = encoded_pose;
    else
      throw std::runtime_error("Invalid pose size");
    pose_param.head<3>().setZero();
    return pose_param;
  }

  Eigen::Matrix<float, 69, 1> pose_mean;
  Eigen::Matrix<float, 69, Eigen::Dynamic> pose_basis;
  Eigen::Matrix<float, 10, Eigen::Dynamic> encoded_training_data;
};

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  namespace fs = boost::filesystem;
  using namespace RaC;
  using namespace CuteGL;
  using Eigen::Vector3f;

  po::options_description generic_options("Generic Options");
    generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
    config_options.add_options()
        ("dataset,d",  po::value<fs::path>(), "Path to dataset file (JSON)")
        ("smpl_model_file,m", po::value<fs::path>()->default_value("smpl_neutral_lbs_10_207_0.h5"), "Path to SMPL model file")
        ("smpl_segmm_file,s", po::value<fs::path>()->default_value("vertex_segm24_col24_14.h5"), "Path to SMPL segmentation file")
        ("smpl_pose_pca_file,p", po::value<fs::path>()->default_value("smpl_pose_pca36_cmu_h36m.h5"), "Path to SMPL pca pose file")
        ("output_dir,o", po::value<fs::path>()->required(), "Output dir")
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

  const fs::path dataset_file(vm["dataset"].as<fs::path>());
  if (!fs::exists(dataset_file)) {
    std::cout << "Error:" << dataset_file << " does not exist\n";
    return EXIT_FAILURE;
  }

  const fs::path out_dir(vm["output_dir"].as<fs::path>());

  Dataset dataset = loadDatasetFromJson(dataset_file.string());
  std::cout << "Loaded dataset \"" << dataset.name << "\" with " << dataset.annotations.size() << " annotations" << std::endl;

  QGuiApplication app(argc, argv);

  // Create the Renderer
  std::unique_ptr<SMPLRenderer> renderer(new SMPLRenderer);
  renderer->setDisplayGrid(false);
  renderer->setDisplayAxis(false);

  const Eigen::Vector2i image_size(320, 240);
  const Eigen::Matrix3f K((Eigen::Matrix3f() << 600.0f, 0.0f, 160.0f,
                                                0.0f, 600.0f, 120.0f,
                                                0.0f, 0.0f, 1.0f).finished());

  OffScreenRenderViewer viewer(renderer.get());
  // viewer.setBackgroundColor(115, 139, 163);
  viewer.setBackgroundColor(0, 0, 0, 0);

  viewer.resize(image_size.x(), image_size.y());

  viewer.camera().intrinsics() = getGLPerspectiveProjection(K,
                                                            image_size.x(), image_size.y(),
                                                            0.01f, 100.0f);

  viewer.create();
  viewer.makeCurrent();

  const fs::path smpl_model_file = vm["smpl_model_file"].as<fs::path>();
  const fs::path smpl_segmm_file = vm["smpl_segmm_file"].as<fs::path>();
  const fs::path smpl_pose_pca_file = vm["smpl_pose_pca_file"].as<fs::path>();


  const PosePCADecoder pca_decoder(smpl_pose_pca_file.string());

  // renderer->setSMPLData(smpl_model_file.string(), smpl_segmm_file.string());
  renderer->setSMPLData(smpl_model_file.string());




  for (const Annotation& anno : dataset.annotations) {
    using Matrix4dRM = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

    Matrix4dRM model_pose_mat(anno.model_pose.value().data());
    Matrix4dRM camera_extrinsic_mat(anno.camera_extrinsic.value().data());

    viewer.camera().extrinsics() = camera_extrinsic_mat.cast<float>();
    renderer->modelPose() = model_pose_mat.cast<float>();

    renderer->smplDrawer().pose() = pca_decoder(anno.poseParam().cast<float>());
    renderer->smplDrawer().shape() = anno.shapeParam().cast<float>();

    renderer->smplDrawer().updateShapeAndPose();

    viewer.render();


    std::string frame_name = (out_dir / fs::path(anno.image_file.value())).string();
    boost::replace_first(frame_name, "image", "segm");

    std::cout << "Working on " << frame_name << "\n";

    fs::path parant_dir = fs::path(frame_name).parent_path();
    if (!fs::exists(parant_dir)){
      fs::create_directories(parant_dir);
    }


    {
      QImage image = viewer.grabColorBuffer();
      image.save(QString::fromStdString(fs::path(frame_name).replace_extension(".png").string()));
    }

    // {
    //   using Image32FC1 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    //   Image32FC1 buffer(image_size.y(), image_size.x());
    //   viewer.grabDepthBuffer(buffer.data());
    //   buffer.colwise().reverseInPlace();
    //   cv::Mat cv_image(buffer.rows(), buffer.cols(), CV_32FC1, buffer.data());
    //   cv::imwrite(fs::path(frame_name).replace_extension(".exr").string(), cv_image);
    // }


//    {
//      using Image32FC1 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//      using Image8UC1 = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//      Image32FC1 buffer(image_size.y(), image_size.x());
//      viewer.grabLabelBuffer(buffer.data());
//      buffer.colwise().reverseInPlace();
//      Image8UC1 label_image = buffer.cast<unsigned char>();
//      if (label_image.maxCoeff() > 24) {
//        throw std::runtime_error("Bad label value");
//      }
//      {
//        H5::H5File file(fs::path(frame_name).replace_extension(".h5").string(), H5F_ACC_TRUNC);
//        H5Eigen::save(file, "segmm", label_image);
//      }
//    }
  }

  return EXIT_SUCCESS;
}
