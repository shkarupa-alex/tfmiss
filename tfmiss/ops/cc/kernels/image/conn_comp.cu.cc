#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "conn_comp.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow
{
namespace miss
{

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T atomic_min(T *address, T val)
{
  return GpuAtomicMin<T, T>(address, val);
}

template <typename T>
__global__ void ConnectedComponentsInitGPUKernel(
    const T *__restrict__ input, const int batch, const int height, const int width, const int channel,
    int64 *__restrict__ output)
{
  const int num_kernels = batch * height * width * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    init_labels<T>(input, index, height, width, channel, output);
  }
}

__global__ void ConnectedComponentsResolveGPUKernel(
    const int batch, const int height, const int width, const int channel, int64 *__restrict__ output)
{
  const int num_kernels = batch * height * width * channel;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    resolve_labels(index, height, width, channel, output);
  }
}

template <typename T>
__global__ void ConnectedComponentsReduceGPUKernel(
    const T *__restrict__ input, const int batch, const int height, const int width, const int channel,
    int64 *__restrict__ output)
{
  const int num_kernels = batch * height * width * channel;

  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    reduce_labels<T>(input, index, height, width, channel, output);
  }
}

__global__ void ConnectedComponentsNormalizeGPUKernel(
    const int batch, const int height, const int width, const int channel, int64 *__restrict__ output)
{
  const int num_kernels = batch * channel;

  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    normalize_labels(index, height, width, channel, output);
  }
}

__global__ void ConnectedComponentsMinimizeGPUKernel(
    const int batch, const int height, const int width, const int channel, int64 *__restrict__ output)
{
  const int num_kernels = batch * height * width * channel;

  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    minimize_labels(index, height, width, channel, output);
  }
}

template <typename T>
struct ConnectedComponentsFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const bool norm, const int batch, const int height, const int width,
      const int channel, int64 *output) const
  {
    const int num_kernels_full = batch * height * width * channel, num_kernels_safe = batch * channel;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();

    GpuLaunchConfig config_full = GetGpuLaunchConfig(num_kernels_full, eigen_gpu);

    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsInitGPUKernel<T>, config_full.block_count, config_full.thread_per_block, 0,
        eigen_gpu.stream(), input, batch, height, width, channel, output));

    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsResolveGPUKernel, config_full.block_count, config_full.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsReduceGPUKernel<T>, config_full.block_count, config_full.thread_per_block, 0,
        eigen_gpu.stream(), input, batch, height, width, channel, output));

    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsResolveGPUKernel, config_full.block_count, config_full.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    if (norm)
    {
      GpuLaunchConfig config_safe = GetGpuLaunchConfig(num_kernels_safe, eigen_gpu);
      TF_CHECK_OK(GpuLaunchKernel(
          ConnectedComponentsNormalizeGPUKernel, config_safe.block_count, config_safe.thread_per_block, 0,
          eigen_gpu.stream(), batch, height, width, channel, output));
    }
    else if (batch > 1 || channel > 1)
    {
      TF_CHECK_OK(GpuLaunchKernel(
          ConnectedComponentsMinimizeGPUKernel, config_full.block_count, config_full.thread_per_block, 0,
          eigen_gpu.stream(), batch, height, width, channel, output));
    }
  }
};

#define DECLARE(T) template struct ConnectedComponentsFunctor<GPUDevice, T>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE);
TF_CALL_bool(DECLARE);
#undef DECLARE

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA