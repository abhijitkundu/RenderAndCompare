/**
 * @file visualize_image_dataset.cpp
 * @brief visualize_image_dataset
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/ImageDataset.h"
#include "CuteGL/Core/PoseUtils.h"
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template <class DerivedV, class DerivedO>
Eigen::Transform<typename DerivedV::Scalar, 3, Eigen::Isometry>
computeObjectPose(const Eigen::MatrixBase<DerivedV>& viewpoint,
                  const Eigen::MatrixBase<DerivedO>& object_center) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedV, 3);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedO, 3);

  using Scalar = typename DerivedV::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

  Isometry3 pose = CuteGL::getExtrinsicsFromViewPoint(viewpoint.x(), viewpoint.y(), viewpoint.z(), object_center.norm());
  pose = Eigen::Quaternion<Scalar>::FromTwoVectors(Vector3::UnitZ(), object_center) * pose;
  return pose;
}

template <class Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 8>
createCorners(const Eigen::MatrixBase<Derived>& bbx_size) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

  using Scalar = typename Derived::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using AlignedBox3 = Eigen::AlignedBox<Scalar, 3>;

  const Vector3 bbx_half_size = bbx_size / 2;
  const AlignedBox3 bbx(-bbx_half_size , bbx_half_size);

  Eigen::Matrix<Scalar, 3, 8> obj_corners;
  obj_corners.col(0) = bbx.corner(AlignedBox3::BottomLeftFloor);
  obj_corners.col(1) = bbx.corner(AlignedBox3::BottomRightFloor);
  obj_corners.col(2) = bbx.corner(AlignedBox3::TopLeftFloor);
  obj_corners.col(3) = bbx.corner(AlignedBox3::TopRightFloor);
  obj_corners.col(4) = bbx.corner(AlignedBox3::BottomLeftCeil);
  obj_corners.col(5) = bbx.corner(AlignedBox3::BottomRightCeil);
  obj_corners.col(6) = bbx.corner(AlignedBox3::TopLeftCeil);
  obj_corners.col(7) = bbx.corner(AlignedBox3::TopRightCeil);

  return obj_corners;
}

template <class Derived>
cv::Point vec2point(const Eigen::DenseBase<Derived>& vec) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 2);
 return cv::Point(vec.x(), vec.y());
}

template <class Derived>
void draw3DBoxOnImage(cv::Mat& image, const Eigen::MatrixBase<Derived>& img_corners) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 8);
  using Scalar = typename Derived::Scalar;
  using AlignedBox3 = Eigen::AlignedBox<Scalar, 3>;

  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomLeftFloor)), vec2point(img_corners.col(AlignedBox3::BottomRightFloor)), CV_RGB(255, 0, 0), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::TopLeftFloor)), vec2point(img_corners.col(AlignedBox3::TopRightFloor)), CV_RGB(255, 0, 0), 1);

  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomLeftCeil)), vec2point(img_corners.col(AlignedBox3::BottomRightCeil)), CV_RGB(255, 0, 0), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::TopLeftCeil)), vec2point(img_corners.col(AlignedBox3::TopRightCeil)), CV_RGB(255, 0, 0), 1);

  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomLeftFloor)), vec2point(img_corners.col(AlignedBox3::BottomLeftCeil)), CV_RGB(0, 0, 255), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomRightFloor)), vec2point(img_corners.col(AlignedBox3::BottomRightCeil)), CV_RGB(0, 0, 255), 1);

  cv::line(image, vec2point(img_corners.col(AlignedBox3::TopLeftFloor)), vec2point(img_corners.col(AlignedBox3::TopLeftCeil)), CV_RGB(0, 0, 255), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::TopRightFloor)), vec2point(img_corners.col(AlignedBox3::TopRightCeil)), CV_RGB(0, 0, 255), 1);

  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomLeftFloor)), vec2point(img_corners.col(AlignedBox3::TopLeftFloor)), CV_RGB(0, 255, 0), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomRightFloor)), vec2point(img_corners.col(AlignedBox3::TopRightFloor)), CV_RGB(0, 255, 0), 1);

  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomLeftCeil)), vec2point(img_corners.col(AlignedBox3::TopLeftCeil)), CV_RGB(0, 255, 0), 1);
  cv::line(image, vec2point(img_corners.col(AlignedBox3::BottomRightCeil)), vec2point(img_corners.col(AlignedBox3::TopRightCeil)), CV_RGB(0, 255, 0), 1);
}


cv::Mat visualizeObjects(const cv::Mat& cv_image, const RaC::ImageInfo& image_info) {
  cv::Mat image = cv_image.clone();
  if (image_info.objects) {
    // Loop over all objects
    for (const RaC::ImageObjectInfo& obj_info : image_info.objects.value()) {

      if (obj_info.bbx_visible) {
        auto bbx_visible = obj_info.bbx_visible.value();
        cv::rectangle(image, cv::Point(bbx_visible[0], bbx_visible[1]), cv::Point(bbx_visible[2], bbx_visible[3]), cv::Scalar(255, 0, 0));
      }

      if (obj_info.bbx_amodal) {
        auto bbx_amodal = obj_info.bbx_amodal.value();
        cv::rectangle(image, cv::Point(bbx_amodal[0], bbx_amodal[1]), cv::Point(bbx_amodal[2], bbx_amodal[3]),
                      cv::Scalar(255, 255, 0));
      }

      if (obj_info.center_proj) {
        // project object_center to image
        const Eigen::Vector2d object_center_image = obj_info.center_proj.value();
        cv::circle(image, cv::Point(object_center_image.x(), object_center_image.y()), 5, cv::Scalar(0, 0, 255), -1);

        if (image_info.image_intrinsic && obj_info.center_dist && obj_info.viewpoint && obj_info.dimension) {
          const Eigen::Matrix3d K = image_info.image_intrinsic.value();
          const Eigen::Vector3d object_center = obj_info.center_dist.value()
              * (K.inverse() * object_center_image.homogeneous()).normalized();

          auto viewpoint = obj_info.viewpoint.value();
          auto obj_pose = computeObjectPose(viewpoint, object_center);
          const Eigen::Matrix<double, 2, 8> img_corners = (K * obj_pose
              * createCorners(obj_info.dimension.value())).colwise().hnormalized();
          draw3DBoxOnImage(image, img_corners);
        }
      }
    }
  }
  return image;
}

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

    const fs::path image_path = dataset.rootdir / image_info.image_file.value();
    cv::Mat cv_image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    cv::Mat image = visualizeObjects(cv_image, image_info);

    cv::imshow(windowname, image);
    cv::displayOverlay(windowname, image_path.stem().string());
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


