#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "render_layer.hpp"

#include "CuteGL/Geometry/ComputeAlignedBox.h"
#include "CuteGL/Core/MeshUtils.h"
#include "CuteGL/Core/Config.h"
#include "CuteGL/IO/ImportViaAssimp.h"

#include <QGuiApplication>
#include <QTime>
#include <iostream>

namespace caffe {

template<typename Dtype>
void RenderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
//  CHECK_EQ(bottom.size(), 2);

  LOG(INFO)<< "Setting up RenderLayer";
  using namespace CuteGL;

  renderer_.reset(new MultiObjectRenderer());
  renderer_->setDisplayGrid(true);
  renderer_->setDisplayAxis(true);

  viewer_.reset(new OffScreenRenderViewer(renderer_.get()));
  viewer_->setBackgroundColor(0, 0, 0);
  viewer_->resize(640, 480);

  viewer_->setCameraToLookAt(Eigen::Vector3f(0.5f, -0.2f, 1.0f),
                             Eigen::Vector3f::Zero(),
                             Eigen::Vector3f::UnitY());

  viewer_->create();

  viewer_->makeCurrent();
  {
    std::string asset_file = CUTEGL_ASSETS_FOLDER "/Sphere.nff";
    std::vector<MeshData> meshes = loadMeshesViaAssimp(asset_file);
    Eigen::AlignedBox3f bbx = computeAlignedBox(meshes);
    Eigen::Affine3f model_pose = Eigen::UniformScaling<float>(0.5f / bbx.sizes().maxCoeff())
        * Eigen::Translation3f(-bbx.center());
    renderer_->modelDrawers().addItem(model_pose, meshes);
  }

  {
    std::string asset_file = CUTEGL_ASSETS_FOLDER "/cow.off";
    std::vector<MeshData> meshes = loadMeshesViaAssimp(asset_file);
    Eigen::AlignedBox3f bbx = computeAlignedBox(meshes);
    Eigen::Affine3f model_pose = Eigen::UniformScaling<float>(
          1.0f / bbx.sizes().maxCoeff()) * Eigen::Translation3f(-bbx.center());

    renderer_->modelDrawers().addItem(model_pose, meshes);
  }


  LOG(INFO)<< "Done Setting up RenderLayer";
}

template <typename Dtype>
void RenderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape = {100, 1, 480, 640};
  top[0]->Reshape(shape);
}

template <typename Dtype>
void RenderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  viewer_->makeCurrent();
  for (int i = 0; i < 100; ++i) {
    using Image = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Image> image(top_data + top[0]->offset(i, 0), 480, 640);
    viewer_->render();
    viewer_->readZBuffer(image.data());
//     RaC::saveImage(image, "image_"+ std::to_string(i) + ".png");
   for (auto& pose : renderer_->modelDrawers().poses()) {
     pose = pose * Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitY());
   }
  }
  viewer_->doneCurrent();
}

INSTANTIATE_CLASS(RenderLayer);
//REGISTER_LAYER_CLASS(Render);

}  // namespace caffe
