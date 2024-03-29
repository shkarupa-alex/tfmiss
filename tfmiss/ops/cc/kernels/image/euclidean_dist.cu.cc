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

template <typename T>
__global__ void EuclideanDistanceColumnGPUKernel(
    const T *__restrict__ input, const int batch, const int height, const int width, const int channel,
    float *__restrict__ fd, int *__restrict__ v, float *__restrict__ z, float *__restrict__ output)
{
  const int num_kernels = batch * width * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / width;
    const int column_id = index / channel % width;
    const int channel_id = index % channel;

    euclidean_distance_column<T>(input, batch_id, column_id, channel_id, height, width, channel, fd, v, z, output);
  }
}

__global__ void EuclideanDistanceRowGPUKernel(
    const int batch, const int height, const int width, const int channel, float *__restrict__ fd, int *__restrict__ v,
    float *__restrict__ z, float *__restrict__ output)
{
  const int num_kernels = batch * height * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / height;
    const int row_id = index / channel % height;
    const int channel_id = index % channel;

    euclidean_distance_row(batch_id, row_id, channel_id, height, width, channel, fd, v, z, output);
  }
}

template <typename T>
struct EuclideanDistanceFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,
      float *fd, int *v, float *z, float *output) const
  {
    const int num_kernels_column = batch * width * channel, num_kernels_row = batch * height * channel;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();

    GpuLaunchConfig config_column = GetGpuLaunchConfig(num_kernels_column, eigen_gpu);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceColumnGPUKernel<T>, config_column.block_count, config_column.thread_per_block, 0,
        eigen_gpu.stream(), input, batch, height, width, channel, fd, v, z, output));

    GpuLaunchConfig config_row = GetGpuLaunchConfig(num_kernels_row, eigen_gpu);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceRowGPUKernel, config_row.block_count, config_row.thread_per_block, 0, eigen_gpu.stream(),
        batch, height, width, channel, fd, v, z, output));
  }
};

#define DECLARE(T) template struct EuclideanDistanceFunctor<GPUDevice, T>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE);
TF_CALL_bool(DECLARE);
#undef DECLARE

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA