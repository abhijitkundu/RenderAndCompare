/**
 * @file SMPLMeshRenderer.h
 * @brief SMPLMeshRenderer
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SMPLMESHRENDERER_H_
#define RENDERANDCOMPARE_SMPLMESHRENDERER_H_

//#include <CuteGL/Renderer/SMPLRenderer.h> // Should be included 1st
#include <CuteGL/Renderer/BasicLightRenderer.h>
#include <CuteGL/Drawers/MeshDrawer.h>
#include <CuteGL/Drawers/LineDrawer.h>

namespace CuteGL {

class SMPLMeshRenderer : public BasicLightRenderer {
 public:

  using MeshType = Mesh<float, float, unsigned char>;

  SMPLMeshRenderer()
   :line_drawer_(),
    mesh_drawer_(),
    model_mat_(Eigen::Affine3f::Identity()) {
  }

  void initMeshDrawer(const MeshType& mesh);
  void initLineDrawer();

  const Eigen::Affine3f& modelMat() const {return model_mat_;}
  Eigen::Affine3f& modelMat() {return model_mat_;}

  MeshDrawer& meshDrawer() {return mesh_drawer_;}
  const MeshDrawer& meshDrawer() const {return mesh_drawer_;}

  LineDrawer& lineDrawer() {return line_drawer_;}
  const LineDrawer& lineDrawer() const {return line_drawer_;}

 protected:
  virtual void draw(PhongShader& shader);
  virtual void draw(BasicShader& shader);

 private:
  LineDrawer line_drawer_;
  MeshDrawer mesh_drawer_;
  Eigen::Affine3f model_mat_;

 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // end namespace CuteGL

#endif // end RENDERANDCOMPARE_SMPLMESHRENDERER_H_
