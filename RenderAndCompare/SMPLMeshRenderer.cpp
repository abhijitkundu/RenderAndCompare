/**
 * @file SMPLMeshRenderer.cpp
 * @brief SMPLMeshRenderer
 *
 * @author Abhijit Kundu
 */

#include "SMPLMeshRenderer.h"

namespace CuteGL {

void SMPLMeshRenderer::initMeshDrawer(const MeshType& mesh) {
  mesh_drawer_.init(phongShader().program, mesh);
  phongShader().program.bind();
  glfuncs_->glUniform3f(
      phongShader().program.uniformLocation("light_position_world"),
      10.0f, 0.0f, 0.0f);
  phongShader().program.release();
}

void SMPLMeshRenderer::initLineDrawer() {
  line_drawer_.init(basicShader().program);
}

void SMPLMeshRenderer::draw(PhongShader& shader) {
  shader.setModelPose(model_mat_);
  mesh_drawer_.draw();
}

void SMPLMeshRenderer::draw(BasicShader& shader) {
  shader.setModelPose(model_mat_);
  glFuncs()->glLineWidth(1.0);
  line_drawer_.draw();
  glFuncs()->glLineWidth(1.0);
}

}  // end namespace CuteGL
