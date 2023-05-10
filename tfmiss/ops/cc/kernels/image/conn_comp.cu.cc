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

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 atomic_min(int64 *address, int64 val) { return atomicMin(address, val); }

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

template <typename T>
struct ConnectedComponentsFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      OT *output) const
  {
    const int num_kernels_full = batch * height * width * channel, num_kernels_safe = batch * channel;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();

    GpuLaunchConfig config_init = GetGpuLaunchConfig(num_kernels_full, eigen_gpu, ConnectedComponentsInitGPUKernel<T>);
    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsInitGPUKernel<T>, config_init.block_count, config_init.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    GpuLaunchConfig config_resolve =
        GetGpuLaunchConfig(num_kernels_full, eigen_gpu, ConnectedComponentsResolveGPUKernel);
    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsResolveGPUKernel, config_resolve.block_count, config_resolve.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    GpuLaunchConfig config_reduce =
        GetGpuLaunchConfig(num_kernels_full, eigen_gpu, ConnectedComponentsReduceGPUKernel<T>);
    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsReduceGPUKernel<T>, config_reduce.block_count, config_reduce.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    TF_CHECK_OK(GpuLaunchKernel(
        ConnectedComponentsResolveGPUKernel, config_resolve.block_count, config_resolve.thread_per_block, 0,
        eigen_gpu.stream(), batch, height, width, channel, output));

    if (norm)
    {
      GpuLaunchConfig config_norm =
          GetGpuLaunchConfig(num_kernels_safe, eigen_gpu, ConnectedComponentsNormalizeGPUKernel);
      TF_CHECK_OK(GpuLaunchKernel(
          ConnectedComponentsNormalizeGPUKernel, config_norm.block_count, config_norm.thread_per_block, 0,
          eigen_gpu.stream(), batch, height, width, channel, output));
    }
  }
};

#define DECLARE_FUNCTOR(TYPE) template struct ConnectedComponentsFunctor<GPUDevice, TYPE>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE_FUNCTOR);
TF_CALL_bool(DECLARE_FUNCTOR);

#undef DECLARE_FUNCTOR

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA