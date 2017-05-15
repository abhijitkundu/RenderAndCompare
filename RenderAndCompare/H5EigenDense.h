/**
 * @file H5EigenDense.h
 * @brief H5Eigen for Dense eigen objects
 *
 * @author Abhijit Kundu
 */


#ifndef _H5EIGEN_DENSE_H_
#define _H5EIGEN_DENSE_H_

#include "RenderAndCompare/H5EigenBase.h"

namespace H5Eigen {

/**@brief loads a Eigen dense object
 *
 * @param h5group[in]  some H5::H5File file
 * @param name[in]     dataset name
 * @param mat[out]     output matrix
 */
template<typename Derived>
void load(const H5::CommonFG &h5group, const std::string &name,
          const Eigen::DenseBase<Derived> &mat) {
  const H5::DataSet dataset = h5group.openDataSet(name);
  internal::loadAsDense(dataset, mat);
}

template<typename Derived>
void load_attribute(const H5::H5Location &h5obj, const std::string &name,
                    const Eigen::DenseBase<Derived> &mat) {
  const H5::Attribute dataset = h5obj.openAttribute(name);
  internal::loadAsDense(dataset, mat);
}


template<typename T>
void save_scalar_attribute(const H5::H5Location &h5obj, const std::string &name,
                           const T &value) {
  const H5::DataType * const datatype = DatatypeSpecialization<T>::get();
  H5::DataSpace dataspace(H5S_SCALAR);
  H5::Attribute att = h5obj.createAttribute(name, *datatype, dataspace);
  att.write(*datatype, &value);
}

template<>
inline void save_scalar_attribute(const H5::H5Location &h5obj,
                                  const std::string &name,
                                  const std::string &value) {
  save_scalar_attribute(h5obj, name, value.c_str());
}

template<typename Derived>
void save(H5::CommonFG &h5group, const std::string &name,
          const Eigen::EigenBase<Derived> &mat,
          const H5::DSetCreatPropList &plist = H5::DSetCreatPropList::DEFAULT) {
  typedef typename Derived::Scalar Scalar;
  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  const H5::DataSpace dataspace = internal::create_dataspace(mat);
  H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace,
                                              plist);

  bool written = false;  // flag will be true when the data has been written
  if (mat.derived().Flags & Eigen::RowMajor) {
    written = internal::write_rowmat(mat, datatype, &dataset, &dataspace);
  } else {
    written = internal::write_colmat(mat, datatype, &dataset, &dataspace);
  }

  if (!written) {
    // data has not yet been written, so there is nothing else to try but copy the input
    // matrix to a row major matrix and write it.
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
    dataset.write(row_major_mat.data(), *datatype);
  }
}

template<typename Derived>
void save_attribute(const H5::H5Location &h5obj, const std::string &name,
                    const Eigen::EigenBase<Derived> &mat) {
  typedef typename Derived::Scalar Scalar;
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
  const H5::DataSpace dataspace = internal::create_dataspace(mat);
  const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
  H5::Attribute dataset = h5obj.createAttribute(name, *datatype, dataspace);
  dataset.write(*datatype, row_major_mat.data());
}

}  // namespace H5Eigen


#endif // end _H5EIGEN_DENSE_H_
