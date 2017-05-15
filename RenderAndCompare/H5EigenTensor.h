/**
 * @file H5EigenTensor.h
 * @brief H5EigenTensor
 *
 * @author Abhijit Kundu
 */


#ifndef _H5EIGEN_TENSOR_H_
#define _H5EIGEN_TENSOR_H_

#include "RenderAndCompare/H5EigenDense.h"
#include <Eigen/CXX11/Tensor>
#include <iostream>

namespace H5Eigen {

namespace internal {


// For RowMajor tensors
template<typename DataSet, typename Scalar, int NumIndices, int Options, typename IndexType>
void loadAsTensor(const DataSet& dataset,
                  Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor,
                  std::true_type) {
  static_assert((Options & Eigen::RowMajor), "Expects RowMajor only here");
  const H5::DataSpace dataspace = dataset.getSpace();
  const std::size_t ndims = dataspace.getSimpleExtentNdims();
  assert(ndims > 0);

  if (ndims != NumIndices) {
    throw std::runtime_error("HDF5 array has different dimension than tensor");
  }

  std::array<IndexType, NumIndices> dimensions;
  {
    std::array<hsize_t, NumIndices> dimensions_;
    dataspace.getSimpleExtentDims(dimensions_.data());
    std::copy(begin(dimensions_), end(dimensions_), begin(dimensions));
  }

  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  tensor.resize(dimensions);
  read_data(dataset, tensor.data(), *datatype);
}

// For ColMajor tensors
template<typename DataSet, typename Scalar, int NumIndices, int Options, typename IndexType>
void loadAsTensor(const DataSet& dataset,
                  Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor,
                  std::false_type) {
  static_assert(!(Options & Eigen::RowMajor), "Expects ColMajor only here");
  const H5::DataSpace dataspace = dataset.getSpace();
  const std::size_t ndims = dataspace.getSimpleExtentNdims();
  assert(ndims > 0);

  if (ndims != NumIndices) {
    throw std::runtime_error("HDF5 array has different dimension than tensor");
  }

  std::array<IndexType, NumIndices> dimensions;
  {
    std::array<hsize_t, NumIndices> dimensions_;
    dataspace.getSimpleExtentDims(dimensions_.data());
    std::copy(begin(dimensions_), end(dimensions_), begin(dimensions));
  }

  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();

  const static int RowMajorOtions = Options | Eigen::RowMajor;
  Eigen::Tensor<Scalar, NumIndices, RowMajorOtions, IndexType> row_major;
  row_major.resize(dimensions);
  read_data(dataset, row_major.data(), *datatype);


  // Swap the layout and preserve the order of the dimensions
  std::array<IndexType, NumIndices> shuffle;
  std::iota(shuffle.begin(), shuffle.end(), 0);
  std::reverse(shuffle.begin(), shuffle.end());

  tensor = row_major.swap_layout().shuffle(shuffle);
}

template<typename DataSet, typename Scalar, int NumIndices, int Options, typename IndexType>
void loadAsTensor(const DataSet& dataset,
                  Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor) {
  loadAsTensor(dataset, tensor, std::integral_constant<bool, Options & Eigen::RowMajor>());
}


template<typename Scalar, int NumIndices, int Options, typename IndexType>
H5::DataSpace create_dataspace(const Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor) {
  std::array<hsize_t, NumIndices> dimensions;
  for(int i = 0; i < NumIndices; ++i)
    dimensions[i] = static_cast<hsize_t>(tensor.dimension(i));

  return H5::DataSpace(NumIndices, dimensions.data());
}

template<typename PlainObjectType, int Options_, template <class> class MakePointer_>
H5::DataSpace create_dataspace(const Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>& tensor_map) {
  using TensorType = Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>;

  std::array<hsize_t, TensorType::NumIndices> dimensions;
  for(int i = 0; i < TensorType::NumIndices; ++i)
    dimensions[i] = static_cast<hsize_t>(tensor_map.dimension(i));

  return H5::DataSpace(TensorType::NumIndices, dimensions.data());
}

// For RowMajor tensors
template<typename TensorType>
void saveAsTensor(const H5::DataSet& dataset,
                  const H5::DataType& datatype,
                  const TensorType& row_major_tensor,
                  std::true_type) {
  static_assert((TensorType::Layout & Eigen::RowMajor), "Expects RowMajor only here");
  dataset.write(row_major_tensor.data(), datatype);
}

// For ColMajor tensors
template<typename TensorType>
void saveAsTensor(const H5::DataSet& dataset,
                  const H5::DataType& datatype,
                  const TensorType& col_major_tensor,
                  std::false_type) {
  static_assert(!(TensorType::Layout & Eigen::RowMajor), "Expects ColMajor only here");


  // Swap the layout and preserve the order of the dimensions
  std::array<typename TensorType::Index, TensorType::NumIndices> shuffle;
  std::iota(shuffle.begin(), shuffle.end(), 0);
  std::reverse(shuffle.begin(), shuffle.end());

  using RowMajorTensor = Eigen::Tensor<typename TensorType::Scalar, TensorType::NumIndices, Eigen::RowMajor, typename TensorType::Index>;

  const RowMajorTensor row_major = col_major_tensor.swap_layout().shuffle(shuffle);

  dataset.write(row_major.data(), datatype);
}

}  // namespace internal


template<typename Scalar, int NumIndices, int Options, typename IndexType>
void load(const H5::CommonFG& h5group,
          const std::string& name,
          Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor) {
  const H5::DataSet dataset = h5group.openDataSet(name);
  internal::loadAsTensor(dataset, tensor);
}

template<typename Scalar, int NumIndices, int Options, typename IndexType>
void load_attribute(const H5::H5Location &h5obj, const std::string &name,
                    Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor) {
  const H5::Attribute dataset = h5obj.openAttribute(name);
  internal::loadAsTensor(dataset, tensor);
}

template<typename Scalar, int NumIndices, int Options, typename IndexType>
void save(const H5::CommonFG& h5group,
          const std::string& name,
          const Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor,
          const H5::DSetCreatPropList &plist = H5::DSetCreatPropList::DEFAULT) {

  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  const H5::DataSpace dataspace = internal::create_dataspace(tensor);

  H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);

  internal::saveAsTensor(dataset, *datatype, tensor, std::integral_constant<bool, Options & Eigen::RowMajor>());
}

template<typename Scalar, int NumIndices, int Options, typename IndexType>
void replace(const H5::CommonFG& h5group,
             const std::string& name,
             const Eigen::Tensor<Scalar, NumIndices, Options, IndexType>& tensor,
             const H5::DSetCreatPropList &plist = H5::DSetCreatPropList::DEFAULT) {

  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  const H5::DataSpace dataspace = internal::create_dataspace(tensor);

  H5::DataSet dataset = h5group.openDataSet(name);

  internal::saveAsTensor(dataset, *datatype, tensor, std::integral_constant<bool, Options & Eigen::RowMajor>());
}

template<typename PlainObjectType, int Options_, template <class> class MakePointer_>
void save(const H5::CommonFG& h5group,
          const std::string& name,
          const Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>& tensor_map,
          const H5::DSetCreatPropList &plist = H5::DSetCreatPropList::DEFAULT) {

  using TensorType = Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>;
  using Scalar = typename TensorType::Scalar;

  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  const H5::DataSpace dataspace = internal::create_dataspace(tensor_map);

  H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);

  internal::saveAsTensor(dataset, *datatype, tensor_map, std::integral_constant<bool, PlainObjectType::Options & Eigen::RowMajor>());
}

}  // namespace H5Eigen


#endif // end _H5EIGEN_TENSOR_H_
