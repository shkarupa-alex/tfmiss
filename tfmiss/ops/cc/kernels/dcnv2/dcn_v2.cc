#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "dcn_v2.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow
{
namespace miss
{

template <typename T, typename PT>
struct ModulatedDeformableColumnForwardFunctor<CPUDevice, T, PT>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in,
      const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const
  {
    const int num_kernels = batch_size * channel_in * height_out * width_out;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    // TODO: Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. thread_pool->ParallelFor assumes each cost unit is 1ns, minimum
    // cost per shard
    // being 10us.
    //    thread_pool->ParallelFor(num_kernels, height_out * width_out * 1000,
    //        [&](int64 start_index, int64 end_index) {
    //            for (int index = start_index; index < end_index; index++)
    //            {
    //            modulated_deformable_im2col_body<T, PT>(
    //              index, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out,
    //              kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group,
    //              column);
    //            }
    //        });

    for (int index = 0; index < num_kernels; index++)
    {
      modulated_deformable_im2col_body<T, PT>(
          index, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out, kernel_h,
          kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group, column);
    }
  }
};

template struct ModulatedDeformableColumnForwardFunctor<CPUDevice, Eigen::bfloat16, float>;
template struct ModulatedDeformableColumnForwardFunctor<CPUDevice, Eigen::half, float>;
template struct ModulatedDeformableColumnForwardFunctor<CPUDevice, float, float>;
template struct ModulatedDeformableColumnForwardFunctor<CPUDevice, double, double>;

template <typename Device, typename T>
class ModulatedDeformableColumnOp : public OpKernel
{
 private:
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_hb;
  int pad_ha;
  int pad_h;
  int pad_wb;
  int pad_wa;
  int pad_w;
  int dilation_h;
  int dilation_w;
  int deformable_group;

 public:
  explicit ModulatedDeformableColumnOp(OpKernelConstruction *ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_h", &kernel_h));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_w", &kernel_w));
    OP_REQUIRES(ctx, kernel_h > 0 && kernel_h > 0, errors::InvalidArgument("Kernel sizes should be larger than 0"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_h", &stride_h));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_w", &stride_w));
    OP_REQUIRES(ctx, stride_h > 0 && stride_w > 0, errors::InvalidArgument("Strides should be larger than 0"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_hb", &pad_hb));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_ha", &pad_ha));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_wb", &pad_wb));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_wa", &pad_wa));
    OP_REQUIRES(
        ctx, pad_hb >= 0 && pad_ha >= 0 && pad_wb >= 0 && pad_wa >= 0,
        errors::InvalidArgument("Paddings should be larger or equal to 0"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilation_h", &dilation_h));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilation_w", &dilation_w));
    OP_REQUIRES(
        ctx, dilation_w > 0 && dilation_h > 0, errors::InvalidArgument("Dilated rates should be larger than 0"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("deformable_group", &deformable_group));
    OP_REQUIRES(
        ctx, deformable_group > 0, errors::InvalidArgument("Number of deformable groups should be larger than 0"));
  }

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 4, errors::InvalidArgument("Input tensor must have rank 4"));
    OP_REQUIRES(
        ctx, deformable_group <= input_tensor->shape().dim_size(1),
        errors::InvalidArgument(
            "Number of deformable groups should be less or equals to input channel dimension size"));
    const int batch_size = input_tensor->shape().dim_size(0);
    const int height_in = input_tensor->shape().dim_size(1);
    const int width_in = input_tensor->shape().dim_size(2);
    const int channel_in = input_tensor->shape().dim_size(3);

    const int height_out = floor((height_in + pad_hb + pad_ha - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    const int width_out = floor((width_in + pad_wb + pad_wa - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;
    OP_REQUIRES(ctx, height_out > 0 && width_out > 0, errors::InvalidArgument("Output height and width can't be 0"));
    const int channel_out = channel_in * kernel_h * kernel_w;

    const Tensor *offset_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("offset", &offset_tensor));
    OP_REQUIRES(ctx, offset_tensor->shape().dims() == 4, errors::InvalidArgument("Offset tensor must have rank 4"));
    OP_REQUIRES(
        ctx, batch_size == offset_tensor->shape().dim_size(0),
        errors::InvalidArgument("Offset batch size is different from the input one"));
    OP_REQUIRES(
        ctx, height_out == offset_tensor->shape().dim_size(1),
        errors::InvalidArgument("Offset height is different from the output one"));
    OP_REQUIRES(
        ctx, width_out == offset_tensor->shape().dim_size(2),
        errors::InvalidArgument("Offset width is different from the output one"));
    OP_REQUIRES(
        ctx, deformable_group * kernel_h * kernel_w * 2 == offset_tensor->shape().dim_size(3),
        errors::InvalidArgument("Wrong offset channel size"));

    const Tensor *mask_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("mask", &mask_tensor));
    OP_REQUIRES(ctx, mask_tensor->shape().dims() == 4, errors::InvalidArgument("Mask tensor must have rank 4"));
    OP_REQUIRES(
        ctx, batch_size == mask_tensor->shape().dim_size(0),
        errors::InvalidArgument("Mask batch size is different from the input one"));
    OP_REQUIRES(
        ctx, height_out == mask_tensor->shape().dim_size(1),
        errors::InvalidArgument("Mask height is different from the output one"));
    OP_REQUIRES(
        ctx, width_out == mask_tensor->shape().dim_size(2),
        errors::InvalidArgument("Mask width is different from the output one"));
    OP_REQUIRES(
        ctx, deformable_group * kernel_h * kernel_w == mask_tensor->shape().dim_size(3),
        errors::InvalidArgument("Wrong mask channel size"));

    // Prepare output
    Tensor *output_tensor;
    TensorShape output_shape({batch_size, height_out * width_out, channel_out});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    output_tensor->flat<T>().setZero();

    // Do calculations
    const T *input = input_tensor->flat<T>().data();
    const T *offset = offset_tensor->flat<T>().data();
    const T *mask = mask_tensor->flat<T>().data();
    T *output = output_tensor->flat<T>().data();

    if (!std::is_same<T, Eigen::half>::value && !std::is_same<T, Eigen::bfloat16>::value)
    {
      ModulatedDeformableColumnForwardFunctor<Device, T, T> im2col_functor;

      im2col_functor(
          ctx, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out, kernel_h,
          kernel_w, pad_hb, pad_wb, stride_h, stride_w, dilation_h, dilation_w, deformable_group, output);
    }
    else
    {
      ModulatedDeformableColumnForwardFunctor<Device, T, float> im2col_functor;

      im2col_functor(
          ctx, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out, kernel_h,
          kernel_w, pad_hb, pad_wb, stride_h, stride_w, dilation_h, dilation_w, deformable_group, output);
    }
  }
};

#define REGISTER(TYPE)                                                                      \
  REGISTER_KERNEL_BUILDER(                                                                  \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_CPU).TypeConstraint<TYPE>("FT"), \
      ModulatedDeformableColumnOp<CPUDevice, TYPE>)

TF_CALL_bfloat16(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define DECLARE_FUNCTOR(TYPE)                                                                                          \
  template <>                                                                                                          \
  void ModulatedDeformableColumnForwardFunctor<GPUDevice, TYPE>::operator()(                                           \
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in, \
      const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,         \
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,                    \
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const;                        \
  extern template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, TYPE>

TF_CALL_bfloat16(DECLARE_FUNCTOR);
TF_CALL_half(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

#define REGISTER(TYPE)                                                                      \
  REGISTER_KERNEL_BUILDER(                                                                  \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_GPU).TypeConstraint<TYPE>("FT"), \
      ModulatedDeformableColumnOp<GPUDevice, TYPE>)

TF_CALL_bfloat16(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // namespace miss
}  // end namespace tensorflow