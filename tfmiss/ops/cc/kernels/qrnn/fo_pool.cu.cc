#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "fo_pool.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow
{
namespace miss
{
template <typename T>
__global__ void FoPoolForwardGPUKernel(
    const T *__restrict__ input, const T *__restrict__ forget, const T *__restrict__ init, const int batch_size,
    const int time_size, const int channel_size, T *__restrict__ output)
{
  const int num_kernels = batch_size * channel_size;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    fo_pool_forward_body<T>(index, input, forget, init, batch_size, time_size, channel_size, output);
  }
}

template <typename T>
struct FoPoolForwardFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *init, const int batch_size, const int time_size,
      const int channel_size, T *output) const
  {
    const int num_kernels = batch_size * channel_size;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, eigen_gpu);

    TF_CHECK_OK(GpuLaunchKernel(
        FoPoolForwardGPUKernel<T>, config.block_count, config.thread_per_block, 0, eigen_gpu.stream(), input, forget,
        init, batch_size, time_size, channel_size, output));
  }
};

template struct FoPoolForwardFunctor<GPUDevice, bfloat16>;
template struct FoPoolForwardFunctor<GPUDevice, Eigen::half>;
template struct FoPoolForwardFunctor<GPUDevice, float>;
template struct FoPoolForwardFunctor<GPUDevice, double>;

template <typename T>
__global__ void FoPoolBackwardGPUKernel(
    const T *__restrict__ input, const T *__restrict__ forget, const T *__restrict__ hidden, const T *__restrict__ grad,
    const int batch_size, const int time_size, const int channel_size, T *__restrict__ grad_input,
    T *__restrict__ grad_forget, T *__restrict__ grad_init)
{
  const int num_kernels = batch_size * channel_size;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    fo_pool_backward_body<T>(
        index, input, forget, hidden, grad, batch_size, time_size, channel_size, grad_input, grad_forget, grad_init);
  }
}

template <typename T>
struct FoPoolBackwardFunctor<GPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *hidden, const T *grad, const int batch_size,
      const int time_size, const int channel_size, T *grad_input, T *grad_forget, T *grad_init) const
  {
    const int num_kernels = batch_size * channel_size;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, eigen_gpu);

    TF_CHECK_OK(GpuLaunchKernel(
        FoPoolBackwardGPUKernel<T>, config.block_count, config.thread_per_block, 0, eigen_gpu.stream(), input, forget,
        hidden, grad, batch_size, time_size, channel_size, grad_input, grad_forget, grad_init));
  }
};

template struct FoPoolBackwardFunctor<GPUDevice, bfloat16>;
template struct FoPoolBackwardFunctor<GPUDevice, Eigen::half>;
template struct FoPoolBackwardFunctor<GPUDevice, float>;
template struct FoPoolBackwardFunctor<GPUDevice, double>;

}  // namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA