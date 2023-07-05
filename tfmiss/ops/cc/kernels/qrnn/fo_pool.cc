#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "fo_pool.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow
{
namespace miss
{
template <typename T>
struct FoPoolForwardFunctor<CPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *init, const int batch_size, const int time_size,
      const int channel_size, T *output) const
  {
    const int num_kernels = batch_size * channel_size;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        num_kernels, time_size * 10,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            fo_pool_forward_body<T>(index, input, forget, init, batch_size, time_size, channel_size, output);
          }
        });
  }
};

template <typename T>
struct FoPoolBackwardFunctor<CPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *hidden, const T *grad, const int batch_size,
      const int time_size, const int channel_size, T *grad_input, T *grad_forget, T *grad_init) const
  {
    const int num_kernels = batch_size * channel_size;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        num_kernels, time_size * 15,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            fo_pool_backward_body<T>(
                index, input, forget, hidden, grad, batch_size, time_size, channel_size, grad_input, grad_forget,
                grad_init);
          }
        });
  }
};

template <typename Device, typename T>
class FoPoolOp : public OpKernel
{
 private:
  FoPoolForwardFunctor<Device, T> fo_pool_functor;

 public:
  explicit FoPoolOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 3, errors::InvalidArgument("Input tensor must have rank 3"));
    const int batch_size = input_tensor->shape().dim_size(0);
    const int time_size = input_tensor->shape().dim_size(1);
    const int channel_size = input_tensor->shape().dim_size(2);

    const Tensor *forget_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("forget", &forget_tensor));
    OP_REQUIRES(ctx, forget_tensor->shape().dims() == 3, errors::InvalidArgument("Forget tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == forget_tensor->shape().dim_size(0),
        errors::InvalidArgument("Forget batch size is different from the input one"));
    OP_REQUIRES(
        ctx, time_size == forget_tensor->shape().dim_size(1),
        errors::InvalidArgument("Forget time size is different from the input one"));
    OP_REQUIRES(
        ctx, channel_size == forget_tensor->shape().dim_size(2),
        errors::InvalidArgument("Forget channel size is different from the input one"));

    const Tensor *init_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("init", &init_tensor));
    OP_REQUIRES(
        ctx, init_tensor->shape().dims() == 2, errors::InvalidArgument("Initial state tensor must have rank 2"));
    OP_REQUIRES(
        ctx, batch_size == init_tensor->shape().dim_size(0),
        errors::InvalidArgument("Initial state batch size is different from the input one"));
    OP_REQUIRES(
        ctx, channel_size == init_tensor->shape().dim_size(1),
        errors::InvalidArgument("Initial state channel size is different from the input one"));

    // Prepare output
    Tensor *output_tensor;
    TensorShape output_shape = input_tensor->shape();
    output_shape.set_dim(1, output_shape.dim_size(1) + 1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    // Do calculations
    const T *input = input_tensor->flat<T>().data();
    const T *forget = forget_tensor->flat<T>().data();
    const T *init = init_tensor->flat<T>().data();
    T *output = output_tensor->flat<T>().data();

    fo_pool_functor(ctx, input, forget, init, batch_size, time_size, channel_size, output);
  }
};

template <typename Device, typename T>
class FoPoolBackwardOp : public OpKernel
{
 private:
  FoPoolBackwardFunctor<Device, T> fo_pool_backward_functor;

 public:
  explicit FoPoolBackwardOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 3, errors::InvalidArgument("Input tensor must have rank 3"));
    const int batch_size = input_tensor->shape().dim_size(0);
    const int time_size = input_tensor->shape().dim_size(1);
    const int channel_size = input_tensor->shape().dim_size(2);

    const Tensor *forget_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("forget", &forget_tensor));
    OP_REQUIRES(ctx, forget_tensor->shape().dims() == 3, errors::InvalidArgument("Forget tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == forget_tensor->shape().dim_size(0),
        errors::InvalidArgument("Forget batch size is different from the input one"));
    OP_REQUIRES(
        ctx, time_size == forget_tensor->shape().dim_size(1),
        errors::InvalidArgument("Forget time size is different from the input one"));
    OP_REQUIRES(
        ctx, channel_size == forget_tensor->shape().dim_size(2),
        errors::InvalidArgument("Forget channel size is different from the input one"));

    const Tensor *hidden_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("hidden", &hidden_tensor));
    OP_REQUIRES(
        ctx, hidden_tensor->shape().dims() == 3, errors::InvalidArgument("Hidden state tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == hidden_tensor->shape().dim_size(0),
        errors::InvalidArgument("Hidden state batch size is different from the input one"));
    OP_REQUIRES(
        ctx, time_size + 1 == hidden_tensor->shape().dim_size(1),
        errors::InvalidArgument("Hidden state time size is different from the input + initial one"));
    OP_REQUIRES(
        ctx, channel_size == hidden_tensor->shape().dim_size(2),
        errors::InvalidArgument("Hidden state channel size is different from the input one"));

    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    OP_REQUIRES(ctx, grad_tensor->shape().dims() == 3, errors::InvalidArgument("Gradient tensor must have rank 3"));
    OP_REQUIRES(
        ctx, batch_size == grad_tensor->shape().dim_size(0),
        errors::InvalidArgument("Gradient batch size is different from the input one"));
    OP_REQUIRES(
        ctx, time_size + 1 == grad_tensor->shape().dim_size(1),
        errors::InvalidArgument("Gradient time size is different from the hidden one"));
    OP_REQUIRES(
        ctx, channel_size == grad_tensor->shape().dim_size(2),
        errors::InvalidArgument("Gradient channel size is different from the input one"));

    // Prepare output
    Tensor *grad_input_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor->shape(), &grad_input_tensor));

    Tensor *grad_forget_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, forget_tensor->shape(), &grad_forget_tensor));

    Tensor *grad_init_tensor;
    TensorShape grad_init_shape({batch_size, channel_size});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, grad_init_shape, &grad_init_tensor));

    // Do calculations
    const T *input = input_tensor->flat<T>().data();
    const T *forget = forget_tensor->flat<T>().data();
    const T *hidden = hidden_tensor->flat<T>().data();
    const T *grad = grad_tensor->flat<T>().data();
    T *grad_input = grad_input_tensor->flat<T>().data();
    T *grad_forget = grad_forget_tensor->flat<T>().data();
    T *grad_init = grad_init_tensor->flat<T>().data();

    fo_pool_backward_functor(
        ctx, input, forget, hidden, grad, batch_size, time_size, channel_size, grad_input, grad_forget, grad_init);
  }
};

#define REGISTER(T)                                                                                                \
  REGISTER_KERNEL_BUILDER(Name("Miss>FoPool").Device(DEVICE_CPU).TypeConstraint<T>("FT"), FoPoolOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                                                                         \
      Name("Miss>FoPoolBackward").Device(DEVICE_CPU).TypeConstraint<T>("FT"), FoPoolBackwardOp<CPUDevice, T>)

TF_CALL_FLOAT_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

#define REGISTER(T)                                                                                                    \
  template <>                                                                                                          \
  void FoPoolForwardFunctor<GPUDevice, T>::operator()(                                                                 \
      OpKernelContext *ctx, const T *input, const T *forget, const T *init, const int batch_size, const int time_size, \
      const int channel_size, T *output) const;                                                                        \
  extern template struct FoPoolForwardFunctor<GPUDevice, T>;                                                           \
  REGISTER_KERNEL_BUILDER(Name("Miss>FoPool").Device(DEVICE_GPU).TypeConstraint<T>("FT"), FoPoolOp<GPUDevice, T>);     \
  template <>                                                                                                          \
  void FoPoolBackwardFunctor<GPUDevice, T>::operator()(                                                                \
      OpKernelContext *ctx, const T *input, const T *forget, const T *hidden, const T *grad, const int batch_size,     \
      const int time_size, const int channel_size, T *grad_input, T *grad_forget, T *grad_init) const;                 \
  extern template struct FoPoolBackwardFunctor<GPUDevice, T>;                                                          \
  REGISTER_KERNEL_BUILDER(                                                                                             \
      Name("Miss>FoPoolBackward").Device(DEVICE_GPU).TypeConstraint<T>("FT"), FoPoolBackwardOp<GPUDevice, T>)

TF_CALL_FLOAT_TYPES(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA
}  // end namespace miss
}  // namespace tensorflow
