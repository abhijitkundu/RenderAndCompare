/**
 * @file SMPLmodel.h
 * @brief SMPLmodel
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SMPLMODEL_H_
#define RENDERANDCOMPARE_SMPLMODEL_H_

#include "CuteGL/Core/SMPLData.h"

namespace RaC {

template <class PositionScalar_ = float, class IndexScalar_ = unsigned int>
class SMPLmodel : public CuteGL::SMPLData<PositionScalar_, IndexScalar_> {
 public:
  SMPLmodel() {};
  SMPLmodel(const std::string& smpl_model_h5_file);


  using SMPLDataType = CuteGL::SMPLData<PositionScalar_, IndexScalar_>;

  using PositionScalar = typename SMPLDataType::PositionScalar;
  using IndexScalar = typename SMPLDataType::IndexScalar;
  using MatrixX1u = typename SMPLDataType::MatrixX1u;
  using MatrixX3u = typename SMPLDataType::MatrixX3u;
  using MatrixX3 = typename SMPLDataType::MatrixX3;
  using MatrixXX = typename SMPLDataType::MatrixXX;
  using Tensor3 = typename SMPLDataType::Tensor3;
  using Tensor2 = typename SMPLDataType::Tensor2;
  using Tensor1 = typename SMPLDataType::Tensor1;

  void setDataFromHDF5(const std::string& smpl_model_h5_file);

  template<class DerivedPose, class DerivedShape>
  MatrixX3 computeVertices(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                           const Eigen::DenseBase<DerivedShape>& shape_coeffs) const;

  template<class DerivedPose, class DerivedShape>
  void computeVertices(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                       const Eigen::DenseBase<DerivedShape>& shape_coeffs,
                       const MatrixX3& posed_vertices, const MatrixX3& joint_positions) const;

  template<class DerivedPose, class DerivedShape>
  MatrixX3 computePosedJoints(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                              const Eigen::DenseBase<DerivedShape>& shape_coeffs) const;

  using SMPLDataType::parents;
  using SMPLDataType::faces;
  using SMPLDataType::template_vertices;

  using SMPLDataType::blend_weights;
  using SMPLDataType::joint_regressor;
  using SMPLDataType::pose_displacements;
  using SMPLDataType::shape_displacements;

};

template <class SMPLDataType>
SMPLDataType loadSMPLDataFromHDF5(const std::string& smpl_model_h5_file);

template <class SMPLDataType>
void loadSMPLDataFromHDF5(SMPLDataType& smpl_data, const std::string& smpl_model_h5_file);

}  // end namespace RaC

#include "RenderAndCompare/SMPLmodel-impl.hpp"

#endif // end RENDERANDCOMPARE_SMPLMODEL_H_
