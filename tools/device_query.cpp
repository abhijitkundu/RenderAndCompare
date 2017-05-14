/**
 * @file device_query.cpp
 * @brief device_query
 *
 * @author Abhijit Kundu
 */

#include "caffe/caffe.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>


int device_query() {
  LOG(INFO) << "Querying GPUs ";
  std::vector<int> gpus = {0, 1, 2};
  for (std::size_t i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}

int main(int argc, char** argv) {
  // caffe::GlobalInit(&argc, &argv);
  device_query();
  return 0;
}
