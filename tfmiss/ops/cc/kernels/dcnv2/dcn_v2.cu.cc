#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "dcn_v2.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow
{
namespace miss
{
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void atomic_add(T *ptr, const T value)
{
  GpuAtomicAdd(ptr, value);
}

template <typename T, typename PT>
__global__ void ModulatedDeformableColumnForwardGPUKernel(
    const T *__restrict__ input, const T *__restrict__ offset, const T *__restrict__ mask, const int batch_size,
    const int height_in, const int width_in, const int channel_in, const int height_out, const int width_out,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group, T *__restrict__ column)
{
  const int num_kernels = batch_size * channel_in * height_out * width_out;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    modulated_deformable_im2col_body<T, PT>(
        index, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out, kernel_h,
        kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group, column);
  }
}

template <typename T, typename PT>
struct ModulatedDeformableColumnForwardFunctor<GPUDevice, T, PT>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in,
      const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const
  {
    const int num_kernels = batch_size * channel_in * height_out * width_out;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, eigen_gpu);

    TF_CHECK_OK(GpuLaunchKernel(
        ModulatedDeformableColumnForwardGPUKernel<T, PT>, config.block_count, config.thread_per_block, 0,
        eigen_gpu.stream(), input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group, column));
  }
};

template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, Eigen::half, float>;
template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, Eigen::bfloat16, float>;
template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, float, float>;
template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, double, double>;

template <typename T, typename PT>
__global__ void ModulatedDeformableColumnBackwardGPUKernel(
    const T *__restrict__ input, const T *__restrict__ offset, const T *__restrict__ mask, const T *__restrict__ column,
    const T *__restrict__ grad, const int batch_size, const int height_in, const int width_in, const int channel_in,
    const int height_out, const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group,
    PT *__restrict__ grad_input, PT *__restrict__ grad_offset, PT *__restrict__ grad_mask)
{
  const int num_kernels = batch_size * channel_in * height_out * width_out;
  for (int index : GpuGridRangeX<int>(num_kernels))
  {
    modulated_deformable_col2im_body<T, PT>(
        index, input, offset, mask, column, grad, batch_size, height_in, width_in, channel_in, height_out, width_out,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group, grad_input,
        grad_offset, grad_mask);
  }
}

template <typename T, typename PT>
struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, T, PT>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *column, const T *grad,
      const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,
      const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
      const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input,
      PT *grad_offset, PT *grad_mask) const
  {
    const int num_kernels = batch_size * channel_in * height_out * width_out;
    auto eigen_gpu = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, eigen_gpu);

    TF_CHECK_OK(GpuLaunchKernel(
        ModulatedDeformableColumnBackwardGPUKernel<T, PT>, config.block_count, config.thread_per_block, 0,
        eigen_gpu.stream(), input, offset, mask, column, grad, batch_size, height_in, width_in, channel_in, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group,
        grad_input, grad_offset, grad_mask));
  }
};

template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, Eigen::half, float>;
template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, Eigen::bfloat16, float>;
template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, float, float>;
template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, double, double>;

template <typename T>
struct SetZeroFunctor<GPUDevice, T>
{
  void operator()(OpKernelContext *ctx, typename TTypes<T>::Flat output)
  {
    const auto &device = ctx->eigen_device<GPUDevice>();
    output.device(device) = output.constant(T(0));
  }
};
template struct SetZeroFunctor<GPUDevice, Eigen::half>;
template struct SetZeroFunctor<GPUDevice, Eigen::bfloat16>;
template struct SetZeroFunctor<GPUDevice, float>;
template struct SetZeroFunctor<GPUDevice, double>;

template <typename T, typename PT>
struct CastToFunctor<GPUDevice, T, PT>
{
  void operator()(OpKernelContext *ctx, typename TTypes<T>::Flat output, typename TTypes<PT>::Flat input)
  {
    const auto &device = ctx->eigen_device<GPUDevice>();
    output.device(device) = input.template cast<T>();
  }
};

template struct CastToFunctor<GPUDevice, Eigen::half, float>;

}  // end namespace miss
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA