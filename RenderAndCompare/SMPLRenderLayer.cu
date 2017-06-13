/**
 * @file SMPLRenderLayer.cu
 * @brief SMPLRenderLayer
 *
 * @author Abhijit Kundu
 */

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "SMPLRenderLayer.h"

texture<float, 2, cudaReadModeElementType> texRef;

namespace caffe {

template <typename Scalar>
__global__
void CopyTextureKernel(Scalar* dst, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Early-out if we are beyond the texture coordinates for our texture.
  if (x > W || y > H)
    return;

  dst[y * W + x] = tex2D(texRef, x, y);
}

template <typename Scalar>
void copyTexture(Scalar* dest, cudaArray *in_array, int W, int H) {
  cudaCheckError(cudaBindTextureToArray( texRef, in_array));

  dim3 block(16, 16, 1);
  dim3 grid((W + block.x -1) / block.x, (H + block.y -1) / block.y, 1);
  CopyTextureKernel<<< grid, block >>>(dest, W, H);

  // Unbind the texture reference
  cudaCheckError(cudaUnbindTexture(texRef));
}

template<typename Dtype>
void SMPLRenderLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const int num_frames = top[0]->shape(0);
  const int height = top[0]->height();
  const int width = top[0]->width();

  const Dtype* shape_param_data = bottom[0]->cpu_data();
  const Dtype* pose_param_data = bottom[1]->cpu_data();
  const Dtype* camera_extrinsic_data = bottom[2]->cpu_data();
  const Dtype* model_pose_data = bottom[3]->cpu_data();

  Dtype* image_top_data = top[0]->mutable_gpu_data();


  viewer_->makeCurrent();
  for (int i = 0; i < num_frames; ++i) {
    using Matrix4 = Eigen::Matrix<Dtype, 4, 4, Eigen::RowMajor>;
    using VectorX = Eigen::Matrix<Dtype, Eigen::Dynamic, 1>;

    viewer_->camera().extrinsics() = Eigen::Map<const Matrix4>(camera_extrinsic_data + bottom[2]->offset(i), 4, 4).template cast<float>();
    renderer_->modelPose() = Eigen::Map<const Matrix4>(model_pose_data + bottom[3]->offset(i), 4, 4).template cast<float>();

    renderer_->smplDrawer().shape() = Eigen::Map<const VectorX>(shape_param_data + bottom[0]->offset(i), bottom[0]->shape(1)).template cast<float>();

    VectorX full_pose(72);
    full_pose.setZero();
    full_pose.tail(69) = Eigen::Map<const VectorX>(pose_param_data + bottom[1]->offset(i), bottom[1]->shape(1));
    renderer_->smplDrawer().pose() = full_pose.template cast<float>();

    renderer_->smplDrawer().updateShapeAndPose();

    viewer_->render();

    cudaArray *in_array;
    cudaCheckError(cudaGraphicsMapResources(1, &cuda_gl_resource_, 0));
    cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_gl_resource_, 0, 0));
    copyTexture(image_top_data + top[0]->offset(i, 0), in_array, width, height);
    cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_gl_resource_, 0));
  }
  viewer_->doneCurrent();
}

INSTANTIATE_LAYER_GPU_FUNCS(SMPLRenderLayer);


}  // namespace caffe
