/**
 * @file SMPLmodel-impl.hpp
 * @brief SMPLmodel-impl
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_SMPLMODEL_IMPL_HPP_
#define RENDERANDCOMPARE_SMPLMODEL_IMPL_HPP_

#include "RenderAndCompare/EigenTypedefs.h"
#include "RenderAndCompare/H5EigenTensor.h"

#include <boost/version.hpp>
#if BOOST_VERSION>=105600
#include <boost/core/ignore_unused.hpp>
#endif

namespace RaC {

namespace detail {

template <class DerivedPose>
Eigen::AlignedStdVector<Eigen::Matrix<typename DerivedPose::Scalar, 3, 3, Eigen::RowMajor> >
makeRotationMatrices(const Eigen::DenseBase<DerivedPose>& pose_coeffs) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedPose);
  using Scalar = typename DerivedPose::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  assert(pose_coeffs.size() % 3 == 0);
  Eigen::Index num_of_rotations = pose_coeffs.size() / 3;

  Eigen::AlignedStdVector<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> > rotations(num_of_rotations);
  for (Eigen::Index i = 0; i<num_of_rotations; ++i) {
    Vector3 aa = pose_coeffs.template segment<3>(3*i);
    rotations[i] = Eigen::AngleAxis<Scalar>(aa.norm(), aa.stableNormalized()).toRotationMatrix();
  }

  return rotations;
}


template<class DerivedV, class DerivedP, class RotMat, class A>
Eigen::AlignedStdVector<Eigen::Transform<typename DerivedV::Scalar, 3, Eigen::Isometry> >
computeBonePoses(const Eigen::DenseBase<DerivedV>& joint_positions,
                 const Eigen::DenseBase<DerivedP>& parents,
                 const std::vector<RotMat, A>& rotations) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedP);

  using Scalar = typename DerivedV::Scalar;
  using Isometry3 = Eigen::Transform<Scalar,3,Eigen::Isometry>;

  const Eigen::Index K = parents.size()  - 1;

  assert(Eigen::Index(rotations.size()) == (K + 1));
  assert(parents.size() == (K + 1));

  Eigen::AlignedStdVector<Isometry3> world_tfms_rest(K + 1, Isometry3::Identity());
  {
    for (Eigen::Index k = 0; k < (K + 1); ++k) {
      world_tfms_rest[k].translation() = joint_positions.row(k);
    }
  }

  Eigen::AlignedStdVector<Isometry3> relative_tfms_rest(K + 1, Isometry3::Identity());
  {
    relative_tfms_rest[0] = world_tfms_rest[0];
    for (Eigen::Index k = 1; k < (K + 1); ++k) {
      relative_tfms_rest[k] = world_tfms_rest[parents[k]].inverse() * world_tfms_rest[k];
    }
  }

  Eigen::AlignedStdVector<Isometry3> world_tfms_posed(K + 1, Isometry3::Identity());
  {
    world_tfms_posed[0] = relative_tfms_rest[0] * Isometry3(rotations[0]);
    for (Eigen::Index k = 1; k < (K + 1); ++k) {
      Isometry3 RT = relative_tfms_rest[k] * Isometry3(rotations[k]);
      world_tfms_posed[k] = world_tfms_posed[parents[k]] * RT;
    }
  }

  Eigen::AlignedStdVector<Isometry3> rel_world_tfms(K + 1);
  for (std::size_t i = 0; i < rel_world_tfms.size(); ++i)
    rel_world_tfms[i] = world_tfms_posed[i] * world_tfms_rest[i].inverse();

  return rel_world_tfms;
}

template<class DerivedV, class DerivedW, class A>
Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> linearBlendSkinning(
    const Eigen::DenseBase<DerivedV>& vertices,
    const Eigen::DenseBase<DerivedW>& blend_weights,
    const std::vector<Eigen::Transform<typename DerivedV::Scalar,3,Eigen::Isometry>, A>& bone_poses) {

  using Scalar = typename DerivedV::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using MatrixX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;

  const Eigen::Index N = vertices.rows();
  const Eigen::Index K = blend_weights.cols() - 1;

  assert(Eigen::Index(bone_poses.size()) == (K + 1));

  MatrixX3 posed_vertices(N, 3);
  posed_vertices.setZero();

  for (Eigen::Index i = 0; i < N; ++i) {
    Vector3 posed_vertex = Vector3::Zero();
    const Vector3 vertex = vertices.row(i);
    for (Eigen::Index k = 0; k < (K + 1); ++k) {
      const Vector3 tfm_vertex = bone_poses[k] * vertex;
      posed_vertex += blend_weights(i, k) * tfm_vertex;
    }

    posed_vertices.row(i) = posed_vertex;
  }

  return posed_vertices;
}

}  // end namespace detail

template<class PS, class IS>
template<class DerivedPose, class DerivedShape>
typename SMPLmodel<PS, IS>::MatrixX3
SMPLmodel<PS, IS>::computeVertices(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                                    const Eigen::DenseBase<DerivedShape>& shape_coeffs) const {

  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedPose);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedShape);

  static_assert(std::is_same<typename DerivedPose::Scalar, PositionScalar>::value, "pose_coeffs has diff type");
  static_assert(std::is_same<typename DerivedShape::Scalar, PositionScalar>::value, "shape_coeffs has diff type");

  const Eigen::Index N = template_vertices.rows();
  const Eigen::Index K = blend_weights.cols() - 1;
  const Eigen::Index num_shape_params = shape_displacements.dimension(2);

  assert(shape_coeffs.size() == num_shape_params);
  assert(pose_coeffs.size() == 3 * (K + 1));

  Tensor1 betas(num_shape_params);
  Eigen::Map<Eigen::Matrix<PositionScalar, Eigen::Dynamic, 1>> betas_map(betas.data(), num_shape_params, 1);
  betas_map = shape_coeffs;

  using DimPair = typename Tensor1::DimensionPair;
  Eigen::array<DimPair, 1> product_dims = { { DimPair(2, 0) } };

  Tensor2 SD = shape_displacements.contract(betas, product_dims);
  assert(SD.dimension(0) == N);
  assert(SD.dimension(1) == 3);

  // T' = T + BS
  MatrixX3 vertices = template_vertices + Eigen::Map<MatrixX3>(SD.data(), N, 3);

  // Joint positions
  MatrixX3 joint_positions = joint_regressor * vertices;
  assert(joint_positions.rows() == K + 1);
  assert(joint_positions.cols() == 3);

  const auto rotations = detail::makeRotationMatrices(pose_coeffs);
  assert(Eigen::Index(rotations.size()) == (K + 1));

  // Relative joint locations
  Tensor1 relative_rotations(9 * K);
  for (Eigen::Index i = 0; i < K; ++i) {
    using Matrix33 = Eigen::Matrix<PositionScalar, 3, 3, Eigen::RowMajor>;
    Matrix33 rel_rot = rotations[i + 1] - Matrix33::Identity();

    const std::array<Eigen::Index, 1> offsets = { 9 * i };
    const std::array<Eigen::Index, 1> extents = { 9 };
    relative_rotations.slice(offsets, extents) = Eigen::TensorMap<Tensor1>(rel_rot.data(), 9);
  }

  Tensor2 PD = pose_displacements.contract(relative_rotations, product_dims);
  assert(PD.dimension(0) == N);
  assert(PD.dimension(1) == 3);

  vertices += Eigen::Map<MatrixX3>(PD.data(), N, 3);

  using BonePoses = Eigen::AlignedStdVector<Eigen::Transform<PositionScalar, 3, Eigen::Isometry> >;
  BonePoses bone_poses = detail::computeBonePoses(joint_positions, parents, rotations);
  MatrixX3 posed_vertices = detail::linearBlendSkinning(vertices, blend_weights, bone_poses);

  return posed_vertices;
}

template<class PS, class IS>
template<class DerivedPose, class DerivedShape>
void SMPLmodel<PS, IS>::computeVertices(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                                        const Eigen::DenseBase<DerivedShape>& shape_coeffs,
                                        const MatrixX3& posed_vertices_, const MatrixX3& joint_positions_) const {

  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedPose);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedShape);

  static_assert(std::is_same<typename DerivedPose::Scalar, PositionScalar>::value, "pose_coeffs has diff type");
  static_assert(std::is_same<typename DerivedShape::Scalar, PositionScalar>::value, "shape_coeffs has diff type");

  const Eigen::Index N = template_vertices.rows();
  const Eigen::Index K = blend_weights.cols() - 1;
  const Eigen::Index num_shape_params = shape_displacements.dimension(2);

  assert(shape_coeffs.size() == num_shape_params);
  assert(pose_coeffs.size() == 3 * (K + 1));

  Tensor1 betas(num_shape_params);
  Eigen::Map<Eigen::Matrix<PositionScalar, Eigen::Dynamic, 1>> betas_map(betas.data(), num_shape_params, 1);
  betas_map = shape_coeffs;

  using DimPair = typename Tensor1::DimensionPair;
  Eigen::array<DimPair, 1> product_dims = { { DimPair(2, 0) } };

  Tensor2 SD = shape_displacements.contract(betas, product_dims);
  assert(SD.dimension(0) == N);
  assert(SD.dimension(1) == 3);

  // T' = T + BS
  MatrixX3 vertices = template_vertices + Eigen::Map<MatrixX3>(SD.data(), N, 3);

  // Joint positions
  MatrixX3& joint_positions = const_cast<MatrixX3&>(joint_positions_);
  joint_positions = joint_regressor * vertices;
  assert(joint_positions.rows() == K + 1);
  assert(joint_positions.cols() == 3);

  const auto rotations = detail::makeRotationMatrices(pose_coeffs);
  assert(Eigen::Index(rotations.size()) == (K + 1));

  // Relative joint locations
  Tensor1 relative_rotations(9 * K);
  for (Eigen::Index i = 0; i < K; ++i) {
    using Matrix33 = Eigen::Matrix<PositionScalar, 3, 3, Eigen::RowMajor>;
    Matrix33 rel_rot = rotations[i + 1] - Matrix33::Identity();

    const std::array<Eigen::Index, 1> offsets = { 9 * i };
    const std::array<Eigen::Index, 1> extents = { 9 };
    relative_rotations.slice(offsets, extents) = Eigen::TensorMap<Tensor1>(rel_rot.data(), 9);
  }

  Tensor2 PD = pose_displacements.contract(relative_rotations, product_dims);
  assert(PD.dimension(0) == N);
  assert(PD.dimension(1) == 3);

  vertices += Eigen::Map<MatrixX3>(PD.data(), N, 3);

  MatrixX3& posed_vertices = const_cast<MatrixX3&>(posed_vertices_);

  using BonePoses = Eigen::AlignedStdVector<Eigen::Transform<PositionScalar, 3, Eigen::Isometry> >;
  BonePoses bone_poses = detail::computeBonePoses(joint_positions, parents, rotations);
  posed_vertices = detail::linearBlendSkinning(vertices, blend_weights, bone_poses);
}

template<class PS, class IS>
template<class DerivedPose, class DerivedShape>
typename SMPLmodel<PS, IS>::MatrixX3
SMPLmodel<PS, IS>::computePosedJoints(const Eigen::DenseBase<DerivedPose>& pose_coeffs,
                                      const Eigen::DenseBase<DerivedShape>& shape_coeffs) const {

  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedPose);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY (DerivedShape);

  static_assert(std::is_same<typename DerivedPose::Scalar, PositionScalar>::value, "pose_coeffs has diff type");
  static_assert(std::is_same<typename DerivedShape::Scalar, PositionScalar>::value, "shape_coeffs has diff type");

  const Eigen::Index N = template_vertices.rows();
  const Eigen::Index K = blend_weights.cols() - 1;
  const Eigen::Index num_shape_params = shape_displacements.dimension(2);

  assert(shape_coeffs.size() == num_shape_params);
  assert(pose_coeffs.size() == 3 * (K + 1));

  Tensor1 betas(num_shape_params);
  Eigen::Map<Eigen::Matrix<PositionScalar, Eigen::Dynamic, 1>> betas_map(betas.data(), num_shape_params, 1);
  betas_map = shape_coeffs;

  using DimPair = typename Tensor1::DimensionPair;
  Eigen::array<DimPair, 1> product_dims = { { DimPair(2, 0) } };

  Tensor2 SD = shape_displacements.contract(betas, product_dims);
  assert(SD.dimension(0) == N);
  assert(SD.dimension(1) == 3);

  // T' = T + BS
  MatrixX3 vertices = template_vertices + Eigen::Map<MatrixX3>(SD.data(), N, 3);

  // Joint positions
  MatrixX3 joint_positions = joint_regressor * vertices;
  assert(joint_positions.rows() == K + 1);
  assert(joint_positions.cols() == 3);

  const auto rotations = detail::makeRotationMatrices(pose_coeffs);
  assert(Eigen::Index(rotations.size()) == (K + 1));

  using BonePoses = Eigen::AlignedStdVector<Eigen::Transform<PositionScalar, 3, Eigen::Isometry> >;
  BonePoses bone_poses = detail::computeBonePoses(joint_positions, parents, rotations);


  MatrixX3 posed_joints(K + 1, 3);
  for (Eigen::Index k = 0; k < (K + 1); ++k) {
    using Vector3 = Eigen::Matrix<PositionScalar, 3, 1>;
    const Vector3 rest_joint_location = joint_positions.row(k);
    const Vector3 posed_joint_location = bone_poses[k] * rest_joint_location;
    posed_joints.row(k) = posed_joint_location;
  }

  return posed_joints;
}

template<class PS, class IS>
SMPLmodel<PS, IS>::SMPLmodel(const std::string& smpl_model_h5_file) {
  loadSMPLDataFromHDF5(*this, smpl_model_h5_file);
}

template<class PS, class IS>
void SMPLmodel<PS, IS>::setDataFromHDF5(const std::string& smpl_model_h5_file) {
  loadSMPLDataFromHDF5(*this, smpl_model_h5_file);
}


template<class SMPLDataType>
SMPLDataType loadSMPLDataFromHDF5(const std::string& smpl_model_h5_file) {
  SMPLDataType smpl_data;
  loadSMPLDataFromHDF5(smpl_data, smpl_model_h5_file);
  return smpl_data;
}

template<class SMPLDataType>
void loadSMPLDataFromHDF5(SMPLDataType& smpl_data, const std::string& smpl_model_h5_file) {
  H5::H5File file(smpl_model_h5_file, H5F_ACC_RDONLY);

  H5Eigen::load(file, "parents", smpl_data.parents);
  H5Eigen::load(file, "faces", smpl_data.faces);

  H5Eigen::load(file, "template_vertices", smpl_data.template_vertices);
  H5Eigen::load(file, "blend_weights", smpl_data.blend_weights);
  H5Eigen::load(file, "joint_regressor", smpl_data.joint_regressor);

  H5Eigen::load(file, "pose_displacements", smpl_data.pose_displacements);
  H5Eigen::load(file, "shape_displacements", smpl_data.shape_displacements);


  // Check dimensions
  const Eigen::Index N = smpl_data.template_vertices.rows();
  const Eigen::Index K = smpl_data.blend_weights.cols() - 1;

  // parents K+1 vector
  assert(smpl_data.parents.size() == K+1);

  // blend_weights N x K+1
  assert(smpl_data.blend_weights.rows() == N);
  assert(smpl_data.blend_weights.cols() == K+1);

  // joint_regressor K+1 x N
  assert(smpl_data.joint_regressor.rows() == K+1);
  assert(smpl_data.joint_regressor.cols() == N);

  // shape_displacements N x 3 x 10
  assert(smpl_data.shape_displacements.dimension(0) == N);
  assert(smpl_data.shape_displacements.dimension(1) == 3);

  // pose_displacements N x 3 x 9K
  assert(smpl_data.pose_displacements.dimension(0) == N);
  assert(smpl_data.pose_displacements.dimension(1) == 3);
  assert(smpl_data.pose_displacements.dimension(2) == 9*K);

#if BOOST_VERSION>=105600
  // ignore unused warning
  boost::ignore_unused(N, K);
#endif
}

}  // namespace RaC

#endif // end RENDERANDCOMPARE_SMPLMODEL_IMPL_HPP_
