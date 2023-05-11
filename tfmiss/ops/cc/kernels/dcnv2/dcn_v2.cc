#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "dcn_v2.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cast_op_impl.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow
{
namespace miss
{
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void atomic_add(T *ptr, const T value)
{
  *ptr += value;
}

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
    thread_pool->ParallelFor(
        num_kernels, kernel_h * kernel_w * 15,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            modulated_deformable_im2col_body<T, PT>(
                index, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out,
                kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group, column);
          }
        });
  }
};

template <typename T, typename PT>
struct ModulatedDeformableColumnBackwardFunctor<CPUDevice, T, PT>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *column, const T *grad,
      const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,
      const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
      const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input,
      PT *grad_offset, PT *grad_mask) const
  {
    const int num_kernels = batch_size;
    const int batch_kernels = channel_in * height_out * width_out;

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        num_kernels, batch_kernels * kernel_h * kernel_w * 25,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            const int batch_shift = index * batch_kernels;

            for (int chw = 0; chw < batch_kernels; chw++)
            {
              modulated_deformable_col2im_body<T, PT>(
                  batch_shift + chw, input, offset, mask, column, grad, batch_size, height_in, width_in, channel_in,
                  height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                  deformable_group, grad_input, grad_offset, grad_mask);
            }
          }
        });
  }
};

template <typename T>
struct SetZeroFunctor<CPUDevice, T>
{
  void operator()(OpKernelContext *ctx, typename TTypes<T>::Flat output)
  {
    const auto &device = ctx->eigen_device<CPUDevice>();
    output.device(device) = output.constant(T(0));
  }
};

template <typename T, typename PT>
struct CastToFunctor<CPUDevice, T, PT>
{
  void operator()(OpKernelContext *ctx, typename TTypes<T>::Flat output, typename TTypes<PT>::Flat input)
  {
    const auto &device = ctx->eigen_device<CPUDevice>();
    output.device(device) = input.template cast<T>();
  }
};

template <typename Device, typename T, typename PT>
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
  SetZeroFunctor<Device, T> zero_functor;
  ModulatedDeformableColumnForwardFunctor<Device, T, PT> im2col_functor;

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
        ctx, deformable_group <= input_tensor->shape().dim_size(3),
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

    // Do calculations
    const T *input = input_tensor->flat<T>().data();
    const T *offset = offset_tensor->flat<T>().data();
    const T *mask = mask_tensor->flat<T>().data();
    T *output = output_tensor->flat<T>().data();

    zero_functor(ctx, output_tensor->flat<T>());

    im2col_functor(
        ctx, input, offset, mask, batch_size, height_in, width_in, channel_in, height_out, width_out, kernel_h,
        kernel_w, pad_hb, pad_wb, stride_h, stride_w, dilation_h, dilation_w, deformable_group, output);
  }
};

template <typename Device, typename T, typename PT>
class ModulatedDeformableColumnBackwardOp : public OpKernel
{
 private:
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_hb;
  int pad_ha;
  int pad_wb;
  int pad_wa;
  int dilation_h;
  int dilation_w;
  int deformable_group;
  SetZeroFunctor<Device, PT> zero_functor;
  ModulatedDeformableColumnBackwardFunctor<Device, T, PT> col2im_functor;

 public:
  explicit ModulatedDeformableColumnBackwardOp(OpKernelConstruction *ctx) : OpKernel(ctx)
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
        ctx, deformable_group <= input_tensor->shape().dim_size(3),
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

    const Tensor *column_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("column", &column_tensor));
    OP_REQUIRES(ctx, column_tensor->shape().dims() == 3, errors::InvalidArgument("Column tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == column_tensor->shape().dim_size(0),
        errors::InvalidArgument("Column batch size is different from the input one"));
    OP_REQUIRES(
        ctx, height_out * width_out == column_tensor->shape().dim_size(1),
        errors::InvalidArgument("Column height*width is different from the expected one"));
    OP_REQUIRES(
        ctx, channel_out == column_tensor->shape().dim_size(2),
        errors::InvalidArgument("Column channel size is different from the expected one"));

    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    OP_REQUIRES(ctx, grad_tensor->shape().dims() == 3, errors::InvalidArgument("Grad tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == grad_tensor->shape().dim_size(0),
        errors::InvalidArgument("Grad batch size is different from the input one"));
    OP_REQUIRES(
        ctx, height_out * width_out == grad_tensor->shape().dim_size(1),
        errors::InvalidArgument("Grad height*width is different from the expected one"));
    OP_REQUIRES(
        ctx, channel_out == grad_tensor->shape().dim_size(2),
        errors::InvalidArgument("Grad channel size is different from the expected one"));

    // Prepare output
    Tensor *grad_input_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor->shape(), &grad_input_tensor));

    Tensor *grad_offset_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, offset_tensor->shape(), &grad_offset_tensor));

    Tensor *grad_mask_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, mask_tensor->shape(), &grad_mask_tensor));

    // Do calculations
    const T *input = input_tensor->flat<T>().data();
    const T *offset = offset_tensor->flat<T>().data();
    const T *mask = mask_tensor->flat<T>().data();
    const T *column = column_tensor->flat<T>().data();
    const T *grad = grad_tensor->flat<T>().data();

    if (!std::is_same<T, Eigen::half>::value && !std::is_same<T, Eigen::bfloat16>::value)  // T == PT == float/double
    {
      zero_functor(ctx, grad_input_tensor->flat<PT>());
      zero_functor(ctx, grad_offset_tensor->flat<PT>());
      zero_functor(ctx, grad_mask_tensor->flat<PT>());

      PT *grad_input = grad_input_tensor->flat<PT>().data();
      PT *grad_offset = grad_offset_tensor->flat<PT>().data();
      PT *grad_mask = grad_mask_tensor->flat<PT>().data();

      col2im_functor(
          ctx, input, offset, mask, column, grad, batch_size, height_in, width_in, channel_in, height_out, width_out,
          kernel_h, kernel_w, pad_hb, pad_wb, stride_h, stride_w, dilation_h, dilation_w, deformable_group, grad_input,
          grad_offset, grad_mask);
    }
    else  // T == half or bfloat16, PT = float
    {
      // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/resize_bilinear_op.cc#L361
      // Accumulate output to float instead of half/bfloat16 tensor, since float accumulation is more numerically
      // stable and GPU half implementation is slow.
      Tensor temp_grad_input_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, input_tensor->shape(), &temp_grad_input_tensor));

      Tensor temp_grad_offset_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, offset_tensor->shape(), &temp_grad_offset_tensor));

      Tensor temp_grad_mask_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, mask_tensor->shape(), &temp_grad_mask_tensor));

      zero_functor(ctx, temp_grad_input_tensor.flat<PT>());
      zero_functor(ctx, temp_grad_offset_tensor.flat<PT>());
      zero_functor(ctx, temp_grad_mask_tensor.flat<PT>());

      PT *grad_input = temp_grad_input_tensor.flat<PT>().data();
      PT *grad_offset = temp_grad_offset_tensor.flat<PT>().data();
      PT *grad_mask = temp_grad_mask_tensor.flat<PT>().data();

      col2im_functor(
          ctx, input, offset, mask, column, grad, batch_size, height_in, width_in, channel_in, height_out, width_out,
          kernel_h, kernel_w, pad_hb, pad_wb, stride_h, stride_w, dilation_h, dilation_w, deformable_group, grad_input,
          grad_offset, grad_mask);

      CastToFunctor<Device, T, PT> cast_functor;
      cast_functor(ctx, grad_input_tensor->flat<T>(), temp_grad_input_tensor.flat<PT>());
      cast_functor(ctx, grad_offset_tensor->flat<T>(), temp_grad_offset_tensor.flat<PT>());
      cast_functor(ctx, grad_mask_tensor->flat<T>(), temp_grad_mask_tensor.flat<PT>());
    }
  }
};

#define REGISTER_FLOAT(T, PT)                                                            \
  REGISTER_KERNEL_BUILDER(                                                               \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_CPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnOp<CPUDevice, T, float>)

TF_CALL_half(REGISTER_FLOAT);
TF_CALL_bfloat16(REGISTER_FLOAT);
#undef REGISTER_FLOAT

#define REGISTER_SAME(T)                                                                 \
  REGISTER_KERNEL_BUILDER(                                                               \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_CPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnOp<CPUDevice, T, T>)

TF_CALL_float(REGISTER_SAME);
TF_CALL_double(REGISTER_SAME);
#undef REGISTER_SAME

#define REGISTER_FLOAT(T)                                                                        \
  REGISTER_KERNEL_BUILDER(                                                                       \
      Name("Miss>ModulatedDeformableColumnBackward").Device(DEVICE_CPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnBackwardOp<CPUDevice, T, float>)

TF_CALL_half(REGISTER_FLOAT);
TF_CALL_bfloat16(REGISTER_FLOAT);
#undef REGISTER_FLOAT

#define REGISTER_SAME(T)                                                                         \
  REGISTER_KERNEL_BUILDER(                                                                       \
      Name("Miss>ModulatedDeformableColumnBackward").Device(DEVICE_CPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnBackwardOp<CPUDevice, T, T>)

TF_CALL_float(REGISTER_SAME);
TF_CALL_double(REGISTER_SAME);
#undef REGISTER_SAME

#if GOOGLE_CUDA

#define DECLARE_FLOAT(T)                                                                                               \
  template <>                                                                                                          \
  void ModulatedDeformableColumnForwardFunctor<GPUDevice, T, float>::operator()(                                       \
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in, \
      const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,         \
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,                    \
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const;                        \
  extern template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, T, float>

TF_CALL_half(DECLARE_FLOAT);
TF_CALL_bfloat16(DECLARE_FLOAT);
#undef DECLARE_FLOAT

#define DECLARE_SAME(T)                                                                                                \
  template <>                                                                                                          \
  void ModulatedDeformableColumnForwardFunctor<GPUDevice, T, T>::operator()(                                           \
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in, \
      const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,         \
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,                    \
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const;                        \
  extern template struct ModulatedDeformableColumnForwardFunctor<GPUDevice, T, T>

TF_CALL_float(DECLARE_SAME);
TF_CALL_double(DECLARE_SAME);
#undef DECLARE_SAME

#define REGISTER_FLOAT(T)                                                                \
  REGISTER_KERNEL_BUILDER(                                                               \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_GPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnOp<GPUDevice, T, float>)

TF_CALL_half(REGISTER_FLOAT);
TF_CALL_bfloat16(REGISTER_FLOAT);
#undef REGISTER_FLOAT

#define REGISTER_SAME(T)                                                                 \
  REGISTER_KERNEL_BUILDER(                                                               \
      Name("Miss>ModulatedDeformableColumn").Device(DEVICE_GPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnOp<GPUDevice, T, T>)

TF_CALL_float(REGISTER_SAME);
TF_CALL_double(REGISTER_SAME);
#undef REGISTER_SAME

#define DECLARE_FLOAT(T)                                                                                              \
  template <>                                                                                                         \
  void ModulatedDeformableColumnBackwardFunctor<GPUDevice, T, float>::operator()(                                     \
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *column, const T *grad,           \
      const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,      \
      const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,                  \
      const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, \
      float *grad_input, float *grad_offset, float *grad_mask) const;                                                 \
  extern template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, T, float>

TF_CALL_half(DECLARE_FLOAT);
TF_CALL_bfloat16(DECLARE_FLOAT);
#undef DECLARE_FLOAT

#define DECLARE_SAME(T)                                                                                               \
  template <>                                                                                                         \
  void ModulatedDeformableColumnBackwardFunctor<GPUDevice, T, PT>::operator()(                                        \
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *column, const T *grad,           \
      const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,      \
      const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,                  \
      const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, \
      T *grad_input, T *grad_offset, T *grad_mask) const;                                                             \
  extern template struct ModulatedDeformableColumnBackwardFunctor<GPUDevice, T, T>

TF_CALL_float(DECLARE_SAME);
TF_CALL_double(DECLARE_SAME);
#undef DECLARE_SAME

#define DECLARE(T)                                                                                      \
  template <>                                                                                           \
  void SetZeroFunctor<GPUDevice, T>::operator()(OpKernelContext *ctx, typename TTypes<T>::Flat output); \
  extern template struct SetZeroFunctor<GPUDevice, T>

TF_CALL_half(DECLARE);
TF_CALL_bfloat16(DECLARE);
TF_CALL_float(DECLARE);
TF_CALL_double(DECLARE);
#undef DECLARE

#define DECLARE(T)                                                                                \
  template <>                                                                                     \
  void CastToFunctor<GPUDevice, T, float>::operator()(                                            \
      OpKernelContext *ctx, typename TTypes<T>::Flat output, typename TTypes<float>::Flat input); \
  extern template struct CastToFunctor<GPUDevice, T, float>

TF_CALL_half(DECLARE);
TF_CALL_bfloat16(DECLARE);
#undef DECLARE

#define REGISTER_FLOAT(T)                                                                        \
  REGISTER_KERNEL_BUILDER(                                                                       \
      Name("Miss>ModulatedDeformableColumnBackward").Device(DEVICE_GPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnBackwardOp<GPUDevice, T, float>)

TF_CALL_half(REGISTER_FLOAT);
TF_CALL_bfloat16(REGISTER_FLOAT);
#undef REGISTER_FLOAT

#define REGISTER_SAME(T)                                                                         \
  REGISTER_KERNEL_BUILDER(                                                                       \
      Name("Miss>ModulatedDeformableColumnBackward").Device(DEVICE_GPU).TypeConstraint<T>("FT"), \
      ModulatedDeformableColumnBackwardOp<GPUDevice, T, T>)
TF_CALL_float(REGISTER_SAME);
TF_CALL_double(REGISTER_SAME);
#undef REGISTER_SAME

#endif  // GOOGLE_CUDA

}  // end namespace miss
}  // end namespace tensorflow