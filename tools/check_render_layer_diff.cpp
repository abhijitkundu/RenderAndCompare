/**
 * @file check_render_layer_diff.cpp
 * @brief check_render_layer_diff
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/SMPLRenderWithLossLayer.h"
#include "CuteGL/Renderer/SMPLRenderer.h"
#include "CuteGL/Core/PoseUtils.h"
#include "RenderAndCompare/ImageUtils.h"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QApplication>

namespace caffe {

// The gradient checker adds a L2 normalization loss function on top of the
// top blobs, and checks the gradient.
template <typename Dtype>
class GradientChecker {
 public:
  // kink and kink_range specify an ignored nonsmooth region of the form
  // kink - kink_range <= |feature value| <= kink + kink_range,
  // which accounts for all nonsmoothness in use by caffe
  GradientChecker(const Dtype stepsize, const Dtype threshold)
      : stepsize_(stepsize), threshold_(threshold){}
  // Checks the gradient of a layer, with provided bottom layers and top
  // layers.
  // Note that after the gradient check, we do not guarantee that the data
  // stored in the layer parameters and the blobs are unchanged.
  void CheckGradient(Layer<Dtype>* layer, const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, int check_bottom = -1) {
      layer->SetUp(bottom, top);
      CheckGradientSingle(layer, bottom, top, check_bottom, -1, -1);
  }
  void CheckGradientExhaustive(Layer<Dtype>* layer,
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
      int check_bottom = -1, int top_id = 0);

  // Checks the gradient of a single output with respect to particular input
  // blob(s).  If check_bottom = i >= 0, check only the ith bottom Blob.
  // If check_bottom == -1, check everything -- all bottom Blobs and all
  // param Blobs.  Otherwise (if check_bottom < -1), check only param Blobs.
  void CheckGradientSingle(Layer<Dtype>* layer,
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
      int check_bottom, int top_id, int top_data_id);

 protected:
  Dtype GetObjAndGradient(const Layer<Dtype>& layer,
      const vector<Blob<Dtype>*>& top, int top_id = -1, int top_data_id = -1);
  Dtype stepsize_;
  Dtype threshold_;
};


template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(Layer<Dtype>* layer,
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int check_bottom, int top_id, int top_data_id) {
  // First, figure out what blobs we need to check against, and zero init
  // parameter blobs.
  vector<Blob<Dtype>*> blobs_to_check;
  vector<bool> propagate_down(bottom.size(), check_bottom == -1);
  for (size_t i = 0; i < layer->blobs().size(); ++i) {
    Blob<Dtype>* blob = layer->blobs()[i].get();
    caffe_set(blob->count(), static_cast<Dtype>(0), blob->mutable_cpu_diff());
    blobs_to_check.push_back(blob);
  }
  if (check_bottom == -1) {
    for (size_t i = 0; i < bottom.size(); ++i) {
      blobs_to_check.push_back(bottom[i]);
    }
  } else if (check_bottom >= 0) {
    CHECK_LT(check_bottom, bottom.size());
    blobs_to_check.push_back(bottom[check_bottom]);
    propagate_down[check_bottom] = true;
  }
  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
  // Compute the gradient analytically using Backward
  // Ignore the loss from the layer (it's just the weighted sum of the losses
  // from the top blobs, whose gradients we may want to test individually).
  layer->Forward(bottom, top);
  // Get additional loss from the objective
  GetObjAndGradient(*layer, top, top_id, top_data_id);
  layer->Backward(top, propagate_down, bottom);
  // Store computed gradients for all checked blobs
  vector<shared_ptr<Blob<Dtype> > >
      computed_gradient_blobs(blobs_to_check.size());
  for (size_t blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    Blob<Dtype>* current_blob = blobs_to_check[blob_id];
    computed_gradient_blobs[blob_id].reset(new Blob<Dtype>());
    computed_gradient_blobs[blob_id]->ReshapeLike(*current_blob);
    const int count = blobs_to_check[blob_id]->count();
    const Dtype* diff = blobs_to_check[blob_id]->cpu_diff();
    Dtype* computed_gradients =
        computed_gradient_blobs[blob_id]->mutable_cpu_data();
    caffe_copy(count, diff, computed_gradients);
  }
  // Compute derivative of top w.r.t. each bottom and parameter input using
  // finite differencing.
  // LOG(ERROR) << "Checking " << blobs_to_check.size() << " blobs.";
  for (size_t blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    Blob<Dtype>* current_blob = blobs_to_check[blob_id];
    const Dtype* computed_gradients =
        computed_gradient_blobs[blob_id]->cpu_data();
    // LOG(ERROR) << "Blob " << blob_id << ": checking "
    //     << current_blob->count() << " parameters.";
    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
      // For an element-wise layer, we only need to do finite differencing to
      // compute the derivative of top[top_id][top_data_id] w.r.t.
      // bottom[blob_id][i] only for i == top_data_id.  For any other
      // i != top_data_id, we know the derivative is 0 by definition, and simply
      // check that that's true.
      Dtype estimated_gradient = 0;
      Dtype positive_objective = 0;
      Dtype negative_objective = 0;
      {
        // Do finite differencing.
        // Compute loss with stepsize_ added to input.
        current_blob->mutable_cpu_data()[feat_id] += stepsize_;
        layer->Forward(bottom, top);
        positive_objective =
            GetObjAndGradient(*layer, top, top_id, top_data_id);
        // Compute loss with stepsize_ subtracted from input.
        current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
        layer->Forward(bottom, top);
        negative_objective =
            GetObjAndGradient(*layer, top, top_id, top_data_id);
        // Recover original input value.
        current_blob->mutable_cpu_data()[feat_id] += stepsize_;
        estimated_gradient = (positive_objective - negative_objective) /
            stepsize_ / 2.;
      }
      Dtype computed_gradient = computed_gradients[feat_id];
//      LOG(INFO) << estimated_gradient << " " << computed_gradient;
//      LOG(ERROR) << "debug: " << current_blob->cpu_data()[feat_id] << " " << current_blob->cpu_diff()[feat_id];
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        Dtype scale = std::max<Dtype>(
            std::max(fabs(computed_gradient), fabs(estimated_gradient)),
            Dtype(1.));
        CHECK_NEAR(computed_gradient, estimated_gradient, threshold_ * scale);
//          << "debug: (top_id, top_data_id, blob_id, feat_id)="
//          << top_id << "," << top_data_id << "," << blob_id << "," << feat_id
//          << "; feat = " << feature
//          << "; objective+ = " << positive_objective
//          << "; objective- = " << negative_objective;
      // LOG(ERROR) << "Feature: " << current_blob->cpu_data()[feat_id];
      // LOG(ERROR) << "computed gradient: " << computed_gradient
      //    << " estimated_gradient: " << estimated_gradient;
    }
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(Layer<Dtype>* layer,
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int check_bottom, int top_id) {
  layer->SetUp(bottom, top);
  CHECK_GE(top_id, 0) << "top_id should be >=0";
  CHECK_LT(top_id, top.size()) << "top_id should be less than number of tops";
  LOG(INFO) << "Exhaustive Mode.";
  LOG(INFO)<< "Exhaustive: Top Blob " << top_id << " size " << top[top_id]->count();
  for (int j = 0; j < top[top_id]->count(); ++j) {
    // LOG(ERROR) << "Exhaustive: blob " << i << " data " << j;
    CheckGradientSingle(layer, bottom, top, check_bottom, top_id, j);
  }
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(const Layer<Dtype>& layer,
    const vector<Blob<Dtype>*>& top, int top_id, int top_data_id) {
  Dtype loss = 0;
  if (top_id < 0) {
    // the loss will be half of the sum of squares of all outputs
    for (size_t i = 0; i < top.size(); ++i) {
      Blob<Dtype>* top_blob = top[i];
      const Dtype* top_blob_data = top_blob->cpu_data();
      Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
      int count = top_blob->count();
      for (int j = 0; j < count; ++j) {
        loss += top_blob_data[j] * top_blob_data[j];
      }
      // set the diff: simply the data.
      caffe_copy(top_blob->count(), top_blob_data, top_blob_diff);
    }
    loss /= 2.;
  } else {
    // the loss will be the top_data_id-th element in the top_id-th blob.
    for (size_t i = 0; i < top.size(); ++i) {
      Blob<Dtype>* top_blob = top[i];
      Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
      caffe_set(top_blob->count(), Dtype(0), top_blob_diff);
    }
    const Dtype loss_weight = 2;
    loss = top[top_id]->cpu_data()[top_data_id] * loss_weight;
    top[top_id]->mutable_cpu_diff()[top_data_id] = loss_weight;
  }
  return loss;
}

}  // namespace caffe

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  using namespace caffe;
  using Dtype = float;
  using BlobType = Blob<Dtype>;

  QApplication app(argc, argv);

  po::options_description generic_options("Generic Options");
  generic_options.add_options()("help,h", "Help screen");

  po::options_description config_options("Config");
  config_options.add_options()
      ("gpu_id,g",  po::value<int>()->default_value(0), "GPU Decice Ids (Use -ve value to force CPU)")
      ("pause_time,p",  po::value<int>()->default_value(0), "Pause time. Use 0 for pause")
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


  vector<BlobType*> blob_bottom_vec;
  vector<BlobType*> blob_top_vec;

  const int num_frames = 10;

  std::unique_ptr<BlobType> shape_params(new BlobType({num_frames, 10}));
  std::unique_ptr<BlobType> pose_params(new BlobType({num_frames, 69}));
  std::unique_ptr<BlobType> camera_extrinsics(new BlobType({num_frames, 4, 4}));
  std::unique_ptr<BlobType> model_poses(new BlobType({num_frames, 4, 4}));
  std::unique_ptr<BlobType> gt_segm_images(new BlobType(num_frames, 1, 240, 320));

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

  {
    using namespace CuteGL;
    using Eigen::Vector3f;

    // Create the Renderer
    std::unique_ptr<SMPLRenderer> renderer(new SMPLRenderer);
    renderer->setDisplayGrid(false);
    renderer->setDisplayAxis(false);

    OffScreenRenderViewer viewer(renderer.get());
    viewer.setBackgroundColor(0, 0, 0);

    const int W = 320;
    const int H = 240;
    viewer.resize(W, H);

    const Eigen::Matrix3f K((Eigen::Matrix3f() << 600.f, 0.f, 160.f, 0.f, 600.f, 120.f, 0.f, 0.f, 1.f).finished());
    viewer.camera().intrinsics() = getGLPerspectiveProjection(K, W, H, 0.01f, 100.0f);

    // Set camera pose via lookAt
    viewer.setCameraToLookAt(Vector3f(0.0f, 0.85f, 5.0f),
                             Vector3f(0.0f, 0.85f, 0.0f),
                             Vector3f::UnitY());

    viewer.create();
    viewer.makeCurrent();

    renderer->setSMPLData("smpl_neutral_lbs_10_207_0.h5", "vertex_segm24_col24_14.h5");

    using MatrixX = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<MatrixX>(shape_params->mutable_cpu_data(), 10, num_frames).setRandom();
    Eigen::Map<MatrixX>(pose_params->mutable_cpu_data(), 69, num_frames).setRandom();
    for (int i = 0; i < num_frames; ++i) {
      using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;
      using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
      Eigen::Map<Matrix4>(camera_extrinsics->mutable_cpu_data() + i * 16, 4, 4) = viewer.camera().extrinsics().matrix();
      Eigen::Map<Matrix4>(model_poses->mutable_cpu_data() + i * 16, 4, 4) = renderer->modelPose().matrix();

      renderer->smplDrawer().shape() = Eigen::Map<const VectorX>(shape_params->cpu_data() + i * 10, 10);
      renderer->smplDrawer().pose().head(3).setZero();
      renderer->smplDrawer().pose().tail(69) = Eigen::Map<const VectorX>(pose_params->cpu_data() + i * 69, 69);
      renderer->smplDrawer().updateShapeAndPose();
      viewer.render();
      viewer.readFrameBuffer(GL_COLOR_ATTACHMENT3, GL_RED, gt_segm_images->mutable_cpu_data() + gt_segm_images->offset(i, 0));
    }
  }

  blob_bottom_vec.push_back(shape_params.get());
  blob_bottom_vec.push_back(pose_params.get());
  blob_bottom_vec.push_back(camera_extrinsics.get());
  blob_bottom_vec.push_back(model_poses.get());
  blob_bottom_vec.push_back(gt_segm_images.get());

  std::unique_ptr<BlobType> loss(new BlobType());
  std::unique_ptr<BlobType> rendered_images(new BlobType());
  blob_top_vec.push_back(loss.get());
  blob_top_vec.push_back(rendered_images.get());

  LayerParameter layer_param;
  layer_param.add_loss_weight(Dtype(3.7));

  SMPLRenderWithLossLayer<Dtype> layer(layer_param);
  layer.SetUp(blob_bottom_vec, blob_top_vec);

  CHECK_EQ(loss->count(), 1);
  CHECK_EQ(rendered_images->count(), num_frames * 320 * 240);

  {
    // Check Fwd pass
    layer.Forward(blob_bottom_vec, blob_top_vec);

    {
      // Visualize the gt segm blob
      using Tensor4f = Eigen::Tensor<const float , 4, Eigen::RowMajor>;
      Eigen::TensorMap<Tensor4f>gt_segm_images_tensor(gt_segm_images->cpu_data(), num_frames, 1, 240, 320);
      Eigen::TensorMap<Tensor4f>rendered_images_tensor(rendered_images->cpu_data(), num_frames, 1, 240, 320);
      cv::namedWindow("GTSegmImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
      cv::moveWindow("GTSegmImage", 100, 100);
      cv::namedWindow("RenderedSegmImage", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
      cv::moveWindow("RenderedSegmImage", 420, 100);
      for (int i = 0; i < num_frames; ++i) {
        using Tensor3u = Eigen::Tensor<unsigned char , 3, Eigen::RowMajor>;
        {
          Tensor3u image = gt_segm_images_tensor.chip(i, 0).cast<unsigned char>();
          cv::Mat cv_image(image.dimension(1), image.dimension(2), CV_8UC1, image.data());
          cv::flip(cv_image, cv_image, 0);  // Can be done with Eigen tensor reverse also
          cv::imshow("GTSegmImage", RaC::getColoredImageFromLabels(cv_image, smpl24_cmap));
        }
        {
          Tensor3u image = rendered_images_tensor.chip(i, 0).cast<unsigned char>();
          cv::Mat cv_image(image.dimension(1), image.dimension(2), CV_8UC1, image.data());
          cv::flip(cv_image, cv_image, 0);  // Can be done with Eigen tensor reverse also
          cv::imshow("RenderedSegmImage", RaC::getColoredImageFromLabels(cv_image, smpl24_cmap));
        }
        cv::waitKey(pause_time);
      }
    }

    CHECK_EQ(loss->cpu_data()[0], 0.0f) << "Expects loss to be close to zero";
    const Dtype* gt_segm_images_data = gt_segm_images->cpu_data();
    const Dtype* rendered_images_data = rendered_images->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < gt_segm_images->count(); ++i) {
      Dtype expected_value = gt_segm_images_data[i];
      Dtype precision = std::max(Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
      CHECK_NEAR(expected_value, rendered_images_data[i], precision);
    }
  }

  {
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec, blob_top_vec, 0);
  }

  return EXIT_SUCCESS;
}



