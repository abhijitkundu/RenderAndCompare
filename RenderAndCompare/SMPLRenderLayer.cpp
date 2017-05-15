/**
 * @file SMPLRenderLayer.cpp
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#include "SMPLRenderLayer.h"
#include "ImageUtils.h"
#include "H5EigenDense.h"
#include "H5EigenTensor.h"

#include "CuteGL/Geometry/ComputeAlignedBox.h"
#include "CuteGL/Core/MeshUtils.h"
#include "CuteGL/Core/Config.h"
#include "CuteGL/IO/ImportViaAssimp.h"
#include <CuteGL/Geometry/ComputeNormals.h>
#include <CuteGL/Core/PoseUtils.h>

namespace caffe {

template<typename Dtype>
void SMPLRenderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<< "Setting up SMPLRenderLayer";
  using namespace CuteGL;

  CHECK_EQ(top.size(), 1);
  CHECK_GE(bottom.size(), 4);

  const int num_frames = bottom[0]->shape(0);
  for (const auto& blob : bottom) {
    CHECK_EQ(blob->shape(0), num_frames);
  }

  const Eigen::Array2i image_size(320, 240);
  const Eigen::Matrix3f K ((Eigen::Matrix3f() << 600.0, 0.0, 160.0, 0.0, 600.0, 120.0, 0.0, 0.0, 1.0).finished());

  std::vector<int> shape = {num_frames, 1, image_size.y(), image_size.x()};
  top[0]->Reshape(shape);

  renderer_.reset(new SMPLMeshRenderer());
  renderer_->setDisplayGrid(true);
  renderer_->setDisplayAxis(true);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(image_size.x(), image_size.y());

  viewer_->camera().intrinsics() = getGLPerspectiveProjection(K, image_size.x(), image_size.y(), 0.01f, 100.0f);

  LOG(INFO)<< "image_size= " << image_size;
  LOG(INFO)<< "K=\n" << K;

  LOG(INFO)<< "Loading data for SMPL";
  smpl_.setDataFromHDF5("smpl_neutral_lbs_10_207_0.h5");

  using MeshType = SMPLMeshRenderer::MeshType;
  MeshType mesh;

  mesh.positions = smpl_.template_vertices;
  mesh.colors = MeshType::ColorType(180, 180, 180, 255).replicate(mesh.positions.rows(), 1);
  mesh.faces = smpl_.faces;
  computeNormals(mesh);

  viewer_->create();
  viewer_->makeCurrent();

  renderer_->initMeshDrawer(mesh);
  renderer_->initLineDrawer();

  LOG(INFO)<< "Done Setting up SMPLRenderLayer";
}

template <typename Dtype>
void SMPLRenderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SMPLRenderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
//   Dtype* top_data = top[0]->mutable_cpu_data();
   viewer_->makeCurrent();
//   for (int i = 0; i < 100; ++i) {
//     using Image = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//     Eigen::Map<Image> image(top_data + top[0]->offset(i, 0), 480, 640);
//     viewer_->render();
//     viewer_->grabZBuffer((float*)image.data());
////     RaC::saveImage(image, "image_"+ std::to_string(i) + ".png");
//    for (auto& pose : renderer_->modelDrawers().poses()) {
//      pose = pose * Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitY());
//    }
//   }
   viewer_->doneCurrent();
}

INSTANTIATE_CLASS(SMPLRenderLayer);

}  // end namespace caffe
