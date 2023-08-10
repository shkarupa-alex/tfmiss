#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "euclidean_dist.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow
{
namespace miss
{

template <typename IT, typename OT>
__global__ void EuclideanDistanceColumnGPUKernel(
    const IT *__restrict__ input, const int batch, const int height, const int width, const int channel,
    OT *__restrict__ output)
{
  float *f = new float[height];
  float *d = new float[height];
  int *v = new int[height];
  float *z = new float[height + 1];

  const int num_kernels = batch * width * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / width;
    const int column_id = index / channel % width;
    const int channel_id = index % channel;

    euclidean_distance_column<IT, OT>(
        input, batch_id, column_id, channel_id, height, width, channel, output, f, d, v, z);
  }

  delete[] f;
  delete[] d;
  delete[] v;
  delete[] z;
}

template <typename OT>
__global__ void EuclideanDistanceRowGPUKernel(
    const int batch, const int height, const int width, const int channel, OT *__restrict__ output)
{
  float *f = new float[width];
  float *d = new float[width];
  int *v = new int[width];
  float *z = new float[width + 1];

  const int num_kernels = batch * height * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / height;
    const int row_id = index / channel % height;
    const int channel_id = index % channel;

    euclidean_distance_row<OT>(batch_id, row_id, channel_id, height, width, channel, output, f, d, v, z);
  }

  delete[] f;
  delete[] d;
  delete[] v;
  delete[] z;
}

template <typename IT, typename OT>
struct EuclideanDistanceFunctor<GPUDevice, IT, OT>
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      OT *output) const
  {
    const int num_kernels_column = batch * width * channel, num_kernels_row = batch * height * channel;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();

    GpuLaunchConfig config_column = GetGpuLaunchConfig(num_kernels_column, eigen_gpu);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceColumnGPUKernel<IT, OT>, config_column.block_count, config_column.thread_per_block, 0,
        eigen_gpu.stream(), input, batch, height, width, channel, output));

    GpuLaunchConfig config_row = GetGpuLaunchConfig(num_kernels_row, eigen_gpu);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceRowGPUKernel<OT>, config_row.block_count, config_row.thread_per_block, 0, eigen_gpu.stream(),
        batch, height, width, channel, output));
  }
};

#define DECLARE(T)                                                         \
  template struct EuclideanDistanceFunctor<GPUDevice, T, Eigen::half>;     \
  template struct EuclideanDistanceFunctor<GPUDevice, T, Eigen::bfloat16>; \
  template struct EuclideanDistanceFunctor<GPUDevice, T, float>;           \
  template struct EuclideanDistanceFunctor<GPUDevice, T, double>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE);
TF_CALL_bool(DECLARE);
#undef DECLARE

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA