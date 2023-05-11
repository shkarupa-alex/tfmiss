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

template <typename IT, OT>
__global__ void EuclideanDistanceColumnGPUKernel(
    const IT *__restrict__ input, const int batch, const int height, const int width, const int channel,
    OT *__restrict__ output)
{
  const int num_kernels = batch * width * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / width;
    const int column_id = index / channel % width;
    const int channel_id = index % channel;

    euclidean_distance_column<IT, OT>(input, batch_id, column_id, channel_id, height, width, channel, output);
  }
}

template <typename IT, OT>
__global__ void EuclideanDistanceRowGPUKernel(
    const IT *__restrict__ input, const int batch, const int height, const int width, const int channel,
    OT *__restrict__ output)
{
  const int num_kernels = batch * height * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    const int batch_id = index / channel / height;
    const int row_id = index / channel % height;
    const int channel_id = index % channel;

    euclidean_distance_row<OT>(batch_id, row_id, channel_id, height, width, channel, output);
  }
}

template <typename IT, typename OT>
struct EuclideanDistanceFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      OT *output) const
  {
    const int num_kernels_column = batch * width * channel, num_kernels_row = batch * height * channel;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();

    GpuLaunchConfig config_column =
        GetGpuLaunchConfig(num_kernels_column, eigen_gpu, EuclideanDistanceColumnGPUKernel<IT, OT>);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceColumnGPUKernel<IT, OT>, config_column.block_count, config_column.thread_per_block, 0,
        eigen_gpu.stream(), input, batch, height, width, channel, output));

    GpuLaunchConfig config_row = GetGpuLaunchConfig(num_kernels_row, eigen_gpu, EuclideanDistanceRowGPUKernel<OT>);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceRowGPUKernel<OT>, config_row.block_count, config_row.thread_per_block, 0, eigen_gpu.stream(),
        batch, height, width, channel, output));
  }
};

#define DECLARE_FUNCTOR(TYPE)                                                 \
  template struct EuclideanDistanceFunctor<GPUDevice, TYPE, Eigen::half>;     \
  template struct EuclideanDistanceFunctor<GPUDevice, TYPE, Eigen::bfloat16>; \
  template struct EuclideanDistanceFunctor<GPUDevice, TYPE, float>;           \
  template struct EuclideanDistanceFunctor<GPUDevice, TYPE, double>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE_FUNCTOR);
TF_CALL_bool(DECLARE_FUNCTOR);

#undef DECLARE_FUNCTOR

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA