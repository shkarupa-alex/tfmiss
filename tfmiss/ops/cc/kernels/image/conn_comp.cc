#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "conn_comp.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow
{
namespace miss
{

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 atomic_min(int64 *address, int64 val)
{
  const int64 old = *address;

  *address = Eigen::numext::mini(old, val);

  return old;
}

template <typename T>
struct ConnectedComponentsFunctor<CPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const bool norm, const int batch, const int height, const int width,
      const int channel, int64 *output) const
  {
    const int num_kernels_full = batch * height * width * channel, num_kernels_safe = batch * channel;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;

    // thread_pool->ParallelFor(
    //     num_kernels_full, 12 * 2,
    //     [&](int64 start_index, int64 end_index)
    //     {
    //       for (int index = start_index; index < end_index; index++)
    //       {
    //         init_labels<T>(input, index, height, width, channel, output);
    //       }
    //     });

          for (int index = 0; index < num_kernels_full; index++)
          {
            init_labels<T>(input, index, height, width, channel, output);
          }

    // thread_pool->ParallelFor(
    //     num_kernels_full, 8 * 2,
    //     [&](int64 start_index, int64 end_index)
    //     {
    //       for (int index = start_index; index < end_index; index++)
    //       {
    //         resolve_labels(index, height, width, channel, output);
    //       }
    //     });

    // thread_pool->ParallelFor(
    //     num_kernels_safe, 9 * 2 * height * width,
    //     [&](int64 start_index, int64 end_index)
    //     {
    //       for (int index = start_index; index < end_index; index++)
    //       {
    //         const int batch_id = index / channel;
    //         const int channel_id = index % channel;
    //         const int offset0 = batch_id * height * width * channel + channel_id;

    //         for (int row_id = 0; row_id < height; row_id++)
    //         {
    //           const int offset1 = row_id * width * channel;

    //           for (int column_id = 0; column_id < width; column_id++)
    //           {
    //             reduce_labels<T>(input, offset0 + offset1 + column_id * channel, height, width, channel, output);
    //           }
    //         }
    //       }
    //     });

    // thread_pool->ParallelFor(
    //     num_kernels_full, 7 * 2,
    //     [&](int64 start_index, int64 end_index)
    //     {
    //       for (int index = start_index; index < end_index; index++)
    //       {
    //         resolve_labels(index, height, width, channel, output);
    //       }
    //     });

    // if (norm)
    // {
    //   thread_pool->ParallelFor(
    //       num_kernels_safe, 50 * 2 * height * width,
    //       [&](int64 start_index, int64 end_index)
    //       {
    //         for (int index = start_index; index < end_index; index++)
    //         {
    //           normalize_labels(index, height, width, channel, output);
    //         }
    //       });
    // }
  }
};

template <typename Device, typename T> 
class ConnectedComponentsOp : public OpKernel
{
 private:
  bool normalize;
  ConnectedComponentsFunctor<Device, T> cc_functor;

 public:
  explicit ConnectedComponentsOp(OpKernelConstruction *ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize", &normalize));
  }

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 4, errors::InvalidArgument("Input tensor must have rank 4"));
    OP_REQUIRES(
        ctx, input_tensor->NumElements() <= Eigen::NumTraits<int64>::highest(),
        errors::InvalidArgument("Input images' size exceeds 2^32-1"));

    const int batch = input_tensor->shape().dim_size(0);
    const int height = input_tensor->shape().dim_size(1);
    const int width = input_tensor->shape().dim_size(2);
    const int channel = input_tensor->shape().dim_size(3);

    // Prepare output
    Tensor *output_tensor;
    TensorShape output_shape(input_tensor->shape());
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    // Do calculations
    const T *input = input_tensor->tensor<T, 4>().data();
    int64 *output = output_tensor->tensor<int64, 4>().data();

    cc_functor(ctx, input, normalize, batch, height, width, channel, output);
  }
};

#define REGISTER(TYPE)                                                                \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name("Miss>ConnectedComponents").Device(DEVICE_CPU).TypeConstraint<TYPE>("DT"), \
      ConnectedComponentsOp<CPUDevice, TYPE>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

#define DECLARE_FUNCTOR(TYPE)                                                                                       \
  template <>                                                                                                       \
  void ConnectedComponentsFunctor<GPUDevice, TYPE>::operator()(                                                     \
      OpKernelContext *ctx, const TYPE *input, const bool norm, const int batch, const int height, const int width, \
      const int channel, int64 *output) const;                                                                      \
  extern template struct ConnectedComponentsFunctor<GPUDevice, TYPE>;

TF_CALL_REAL_NUMBER_TYPES(DECLARE_FUNCTOR);
TF_CALL_bool(DECLARE_FUNCTOR);

#undef DECLARE_FUNCTOR

#define REGISTER(TYPE)                                                                \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name("Miss>ConnectedComponents").Device(DEVICE_GPU).TypeConstraint<TYPE>("DT"), \
      ConnectedComponentsOp<GPUDevice, TYPE>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);

#endif  // GOOGLE_CUDA

}  // end namespace miss
}  // end namespace tensorflow