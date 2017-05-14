#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::vector;

int device_query() {
  LOG(INFO) << "Querying GPUs ";
  vector<int> gpus = {0, 1, 2};
  for (int i = 0; i < gpus.size(); ++i) {
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
