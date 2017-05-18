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
    pose_param.tail<69>() = pose_mean + pose_basis * encoded_pose;
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
        ("dataset,d",  po::value<std::string>(), "Path to dataset file (JSON)")
        ("smpl_model_file,m", po::value<std::string>()->default_value("smpl_neutral_lbs_10_207_0.h5"), "Path to SMPL model file")
        ("smpl_segmm_file,s", po::value<std::string>()->default_value("vertex_segm24_col24_14.h5"), "Path to SMPL segmentation file")
        ("smpl_pose_pca_file,p", po::value<std::string>()->default_value("smpl_pose_pca36_cmu_h36m.h5"), "Path to SMPL pca pose file")
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
  viewer.setBackgroundColor(0, 0, 0);

  viewer.resize(image_size.x(), image_size.y());

  viewer.camera().intrinsics() = getGLPerspectiveProjection(K,
                                                            image_size.x(), image_size.y(),
                                                            0.01f, 100.0f);

  viewer.create();
  viewer.makeCurrent();

  const std::string smpl_model_file = vm["smpl_model_file"].as<std::string>();
  const std::string smpl_segmm_file = vm["smpl_segmm_file"].as<std::string>();
  const std::string smpl_pose_pca_file = vm["smpl_pose_pca_file"].as<std::string>();


  const PosePCADecoder pca_decoder(smpl_pose_pca_file);

  renderer->setSMPLData(smpl_model_file, smpl_segmm_file);

  using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


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


    const fs::path frame_name = fs::path("output") / fs::path(anno.image_file.value());
    std::cout << "Ouput: " << frame_name << "\n";
    {
      QImage image = viewer.grabColorBuffer();
      image.save(QString::fromStdString(frame_name.string()));
    }

//    {
//      Image buffer(H, W);
//      viewer.grabLabelBuffer(buffer.data());
//      buffer.colwise().reverseInPlace();
//
//      std::cout << buffer.minCoeff() << "  " << buffer.maxCoeff() << "\n";
//
//      QImage image(image_size.x(), image_size.y(), QImage::Format_RGB888);
//      for (int x = 0; x < image_size.x(); ++x) {
//        for (int y = 0; y < image_size.y(); ++y) {
//          int cval = buffer(y, x) * 1.0;
//          image.setPixel(x, y, qRgb(cval, cval, cval));
//        }
//      }
//      image.save(QString("label_" + .png"));
//    }
  }



  return EXIT_SUCCESS;
}



;
