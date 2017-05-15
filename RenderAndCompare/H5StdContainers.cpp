/**
 * @file H5StdContainers.cpp
 * @brief H5StdContainers
 *
 * @author Abhijit Kundu
 */

#include "RenderAndCompare/H5StdContainers.h"
#include <stdexcept>
#include <cassert>

namespace H5StdContainers {

void save(const H5::CommonFG& h5fg, const std::string& dsname, const std::vector<std::string>& strings) {
  H5::Exception::dontPrint();

  try {
    // HDF5 only understands vector of char* :-(
    std::vector<const char*> arr_c_str;
    for (size_t ii = 0; ii < strings.size(); ++ii) {
      arr_c_str.push_back(strings[ii].c_str());
    }

    //
    //  one dimension
    //
    hsize_t str_dimsf[1] { arr_c_str.size() };
    H5::DataSpace dataspace(1, str_dimsf);

    // Variable length string
    H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE );
    H5::DataSet str_dataset = h5fg.createDataSet(dsname, datatype, dataspace);

    str_dataset.write(arr_c_str.data(), datatype);
  } catch (H5::Exception& err) {
    throw std::runtime_error(std::string("HDF5 Error in ") + err.getFuncName() + ": " + err.getDetailMsg());

  }
}

void save(const H5::CommonFG& h5fg, const std::string& dsname, const std::string& string) {
  H5::StrType datatype(0, H5T_VARIABLE);
  H5::DataSpace dataspace(H5S_SCALAR);
  H5::DataSet dataset = h5fg.openDataSet(dsname);
  dataset.write(string, datatype, dataspace);
}

void load(const H5::CommonFG& h5fg, const std::string& dsname, std::string& string) {
  H5::StrType datatype(0, H5T_VARIABLE);
  H5::DataSpace dataspace(H5S_SCALAR);
  H5::DataSet dataset = h5fg.openDataSet(dsname);
  dataset.read(string, datatype, dataspace);
}

void load(const H5::CommonFG& h5fg, const std::string& dsname, std::vector<std::string>& strings) {
  strings = read_vector_of_strings(h5fg, dsname);
}

std::vector<std::string> read_vector_of_strings(const H5::CommonFG& h5fg, const std::string& dsname) {
  H5::DataSet cdataset = h5fg.openDataSet(dsname);

  H5::DataSpace space = cdataset.getSpace();
  assert(space.getSimpleExtentNdims() == 1);

  hsize_t dims_out[1];
  space.getSimpleExtentDims(dims_out, NULL);

  size_t length = dims_out[0];

  std::vector<const char*> tmpvect(length, NULL);

  std::vector<std::string> strs(length);
  H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE );
  cdataset.read(tmpvect.data(), datatype);

  for (size_t x = 0; x < tmpvect.size(); ++x) {
    strs[x] = tmpvect[x];
  }

  return strs;
}

}  // end namespace H5StdContainers
