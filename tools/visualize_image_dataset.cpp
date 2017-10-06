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

RaC::ImageInfo flip(const RaC::ImageInfo& image_info_) {
  RaC::ImageInfo image_info = image_info_;

  const int W = image_info.image_size.value().x();

  if (image_info.image_intrinsic) {
    image_info.image_intrinsic.value()(0, 2) = W - image_info.image_intrinsic.value()(0, 2);
  }

  if (image_info.objects) {
    for (RaC::ImageObjectInfo& obj_info : image_info.objects.value()) {
      if (obj_info.bbx_visible) {
        Eigen::Vector4d& bbx_visible = obj_info.bbx_visible.value();
        bbx_visible[0] = W - bbx_visible[0];
        bbx_visible[2] = W - bbx_visible[2];
        std::swap(bbx_visible[0], bbx_visible[2]);
      }

      if (obj_info.bbx_amodal) {
        Eigen::Vector4d& bbx_amodal = obj_info.bbx_amodal.value();
        bbx_amodal[0] = W - bbx_amodal[0];
        bbx_amodal[2] = W - bbx_amodal[2];
        std::swap(bbx_amodal[0], bbx_amodal[2]);
      }

      if (obj_info.center_proj) {
        Eigen::Vector2d& center_proj = obj_info.center_proj.value();
        center_proj[0] = W - center_proj[0];
      }

      if (obj_info.viewpoint) {
        Eigen::Vector3d& viewpoint = obj_info.viewpoint.value();
        viewpoint[0] = -viewpoint[0]; // azimuth = -azimuth
        viewpoint[2] = -viewpoint[2]; // tilt = -tilt
      }

    }
  }

  return image_info;
}


cv::Mat visualizeObjects(const cv::Mat& cv_image_, const RaC::ImageInfo& image_info_, bool do_flipping = false) {
  cv::Mat cv_image;
  RaC::ImageInfo image_info;

  if (do_flipping) {
    cv::flip(cv_image_, cv_image, 1);
    image_info = flip(image_info_);
  } else {
    cv_image_.copyTo(cv_image);
    image_info = image_info_;
  }


  if (image_info.objects) {
    // Loop over all objects
    for (const RaC::ImageObjectInfo& obj_info : image_info.objects.value()) {

      if (obj_info.bbx_visible) {
        auto bbx_visible = obj_info.bbx_visible.value();
        cv::rectangle(cv_image, cv::Point(bbx_visible[0], bbx_visible[1]), cv::Point(bbx_visible[2], bbx_visible[3]), cv::Scalar(255, 0, 0));
        cv::line(cv_image, cv::Point(bbx_visible[0], bbx_visible[1]), cv::Point(bbx_visible[2], bbx_visible[3]), cv::Scalar(255, 0, 0));
      }

      if (obj_info.bbx_amodal) {
        auto bbx_amodal = obj_info.bbx_amodal.value();
        cv::rectangle(cv_image, cv::Point(bbx_amodal[0], bbx_amodal[1]), cv::Point(bbx_amodal[2], bbx_amodal[3]),
                      cv::Scalar(255, 255, 0));
      }

      if (obj_info.center_proj) {
        // project object_center to image
        const Eigen::Vector2d object_center_image = obj_info.center_proj.value();
        cv::circle(cv_image, cv::Point(object_center_image.x(), object_center_image.y()), 5, cv::Scalar(0, 0, 255), -1);

        if (image_info.image_intrinsic && obj_info.center_dist && obj_info.viewpoint && obj_info.dimension) {
          const Eigen::Matrix3d K = image_info.image_intrinsic.value();
          const Eigen::Vector3d object_center = obj_info.center_dist.value()
              * (K.inverse() * object_center_image.homogeneous()).normalized();

          auto viewpoint = obj_info.viewpoint.value();
          auto obj_pose = computeObjectPose(viewpoint, object_center);
          const Eigen::Matrix<double, 2, 8> img_corners = (K * obj_pose
              * createCorners(obj_info.dimension.value())).colwise().hnormalized();
          draw3DBoxOnImage(cv_image, img_corners);
        }
      }
    }
  }
  return cv_image;
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

  if (!fs::exists(dataset.rootdir)) {
    std::cout << "Error: Dataset rootdir " << dataset.rootdir << " does not exist\n";
    return EXIT_FAILURE;
  }

  if (!fs::is_directory(dataset.rootdir)) {
    std::cout << "Error: Dataset rootdir " << dataset.rootdir << " is not a valid directory\n";
    return EXIT_FAILURE;
  }

  const int num_of_annotations = dataset.annotations.size();

  const std::string img_window_original = "Original Image";
  const std::string img_window_flipped = "Flipped Image";

  cv::namedWindow(img_window_original, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::namedWindow(img_window_flipped, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

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

    if (image_info.image_size.value() != Eigen::Vector2i(cv_image.cols, cv_image.rows)) {
      std::cout << "Bad image_size: " << cv_image.size << "\n";
      return EXIT_FAILURE;
    }

    cv::Mat original_view = visualizeObjects(cv_image, image_info, false);
    cv::Mat flipped_view = visualizeObjects(cv_image, image_info, true);

    cv::imshow(img_window_original, original_view);
    cv::imshow(img_window_flipped, flipped_view);

    cv::displayOverlay(img_window_original, image_path.stem().string());
    cv::displayOverlay(img_window_flipped, image_path.stem().string());

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


