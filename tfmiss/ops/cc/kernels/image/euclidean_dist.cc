#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "euclidean_dist.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow
{
namespace miss
{

template <typename T>
struct EuclideanDistanceFunctor<CPUDevice, T>
{
  void operator()(
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,
      float *fd, int *v, float *z, float *output) const
  {
    const int num_kernels_column = batch * width * channel, num_kernels_row = batch * height * channel;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;

    thread_pool->ParallelFor(
        num_kernels_column, 17 * 2 * height,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            const int batch_id = index / channel / width;
            const int column_id = index / channel % width;
            const int channel_id = index % channel;

            euclidean_distance_column<T>(
                input, batch_id, column_id, channel_id, height, width, channel, fd, v, z, output);
          }
        });

    thread_pool->ParallelFor(
        num_kernels_row, 8 * 2 * width,
        [&](int64 start_index, int64 end_index)
        {
          for (int index = start_index; index < end_index; index++)
          {
            const int batch_id = index / channel / height;
            const int row_id = index / channel % height;
            const int channel_id = index % channel;

            euclidean_distance_row(batch_id, row_id, channel_id, height, width, channel, fd, v, z, output);
          }
        });
  }
};

template <typename Device, typename T>
class EuclideanDistanceOp : public OpKernel
{
 private:
  EuclideanDistanceFunctor<Device, T> ed_functor;

 public:
  explicit EuclideanDistanceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 4, errors::InvalidArgument("Input tensor must have rank 4"));

    const int batch = input_tensor->shape().dim_size(0);
    const int height = input_tensor->shape().dim_size(1);
    const int width = input_tensor->shape().dim_size(2);
    const int channel = input_tensor->shape().dim_size(3);
    OP_REQUIRES(
        ctx, height * width <= Eigen::NumTraits<int>::highest(),
        errors::InvalidArgument("Input images' size exceeds 2^32-1"));

    // Prepare temp
    Tensor fd_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, input_tensor->shape(), &fd_tensor));

    Tensor v_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, input_tensor->shape(), &v_tensor));

    Tensor z_tenzor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({batch, height + 1, width + 1, channel}), &z_tenzor));

    // Prepare output
    Tensor *output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    // Do calculations
    const T *input = input_tensor->tensor<T, 4>().data();

    float *fd = fd_tensor.tensor<float, 4>().data();
    int *v = v_tensor.tensor<int, 4>().data();
    float *z = z_tenzor.tensor<float, 4>().data();

    float *output = output_tensor->tensor<float, 4>().data();

    ed_functor(ctx, input, batch, height, width, channel, fd, v, z, output);
  }
};

#define REGISTER(T)        \
  REGISTER_KERNEL_BUILDER( \
      Name("Miss>EuclideanDistance").Device(DEVICE_CPU).TypeConstraint<T>("DT"), EuclideanDistanceOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

#define REGISTER(T)                                                                                                \
  template <>                                                                                                      \
  void EuclideanDistanceFunctor<GPUDevice, T>::operator()(                                                         \
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel, \
      float *fd, int *v, float *z, float *output) const;                                                           \
  extern template struct EuclideanDistanceFunctor<GPUDevice, T>;                                                   \
  REGISTER_KERNEL_BUILDER(                                                                                         \
      Name("Miss>EuclideanDistance").Device(DEVICE_GPU).TypeConstraint<T>("DT"), EuclideanDistanceOp<GPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace miss
}  // end namespace tensorflow