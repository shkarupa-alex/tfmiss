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

template <typename IT, typename OT>
struct EuclideanDistanceFunctor<CPUDevice, IT, OT>
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      OT *output) const
  {
    const int num_kernels_column = batch * width * channel, num_kernels_row = batch * height * channel;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;

    thread_pool->ParallelFor(
        num_kernels_column, 17 * 2 * height,
        [&](int64 start_index, int64 end_index)
        {
          float *f = new float[height];
          float *d = new float[height];
          int *v = new int[height];
          float *z = new float[height + 1];

          for (int index = start_index; index < end_index; index++)
          {
            const int batch_id = index / channel / width;
            const int column_id = index / channel % width;
            const int channel_id = index % channel;

            euclidean_distance_column<IT, OT>(
                input, batch_id, column_id, channel_id, height, width, channel, output, f, d, v, z);
          }

          delete[] f;
          delete[] d;
          delete[] v;
          delete[] z;
        });

    thread_pool->ParallelFor(
        num_kernels_row, 8 * 2 * width,
        [&](int64 start_index, int64 end_index)
        {
          float *f = new float[width];
          float *d = new float[width];
          int *v = new int[width];
          float *z = new float[width + 1];

          for (int index = start_index; index < end_index; index++)
          {
            const int batch_id = index / channel / height;
            const int row_id = index / channel % height;
            const int channel_id = index % channel;

            euclidean_distance_row<OT>(batch_id, row_id, channel_id, height, width, channel, output, f, d, v, z);
          }

          delete[] f;
          delete[] d;
          delete[] v;
          delete[] z;
        });
  }
};

template <typename Device, typename IT, typename OT>
class EuclideanDistanceOp : public OpKernel
{
 private:
  EuclideanDistanceFunctor<Device, IT, OT> ed_functor;

 public:
  explicit EuclideanDistanceOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override
  {
    // Prepare input
    const Tensor *input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, input_tensor->shape().dims() == 4, errors::InvalidArgument("Input tensor must have rank 4"));
    OP_REQUIRES(
        ctx, input_tensor->NumElements() <= Eigen::NumTraits<int>::highest(),
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
    const IT *input = input_tensor->tensor<IT, 4>().data();
    OT *output = output_tensor->tensor<OT, 4>().data();

    ed_functor(ctx, input, batch, height, width, channel, output);
  }
};

#define REGISTER(T)                                                                                                   \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_CPU).TypeConstraint<T>("DT").TypeConstraint<Eigen::half>("dtype"), \
      EuclideanDistanceOp<CPUDevice, T, Eigen::half>);                                                                \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance")                                                                                  \
          .Device(DEVICE_CPU)                                                                                         \
          .TypeConstraint<T>("DT")                                                                                    \
          .TypeConstraint<Eigen::bfloat16>("dtype"),                                                                  \
      EuclideanDistanceOp<CPUDevice, T, Eigen::bfloat16>);                                                            \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_CPU).TypeConstraint<T>("DT").TypeConstraint<float>("dtype"),       \
      EuclideanDistanceOp<CPUDevice, T, float>);                                                                      \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_CPU).TypeConstraint<T>("DT").TypeConstraint<double>("dtype"),      \
      EuclideanDistanceOp<CPUDevice, T, double>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA

#define REGISTER(T)                                                                                                   \
  template <>                                                                                                         \
  void EuclideanDistanceFunctor<GPUDevice, T, Eigen::half>::operator()(                                               \
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,    \
      Eigen::half *output) const;                                                                                     \
  extern template struct EuclideanDistanceFunctor<GPUDevice, T, Eigen::half>;                                         \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_GPU).TypeConstraint<T>("DT").TypeConstraint<Eigen::half>("dtype"), \
      EuclideanDistanceOp<GPUDevice, T, Eigen::half>);                                                                \
  template <>                                                                                                         \
  void EuclideanDistanceFunctor<GPUDevice, T, Eigen::bfloat16>::operator()(                                           \
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,    \
      Eigen::bfloat16 *output) const;                                                                                 \
  extern template struct EuclideanDistanceFunctor<GPUDevice, T, Eigen::bfloat16>;                                     \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance")                                                                                  \
          .Device(DEVICE_GPU)                                                                                         \
          .TypeConstraint<T>("DT")                                                                                    \
          .TypeConstraint<Eigen::bfloat16>("dtype"),                                                                  \
      EuclideanDistanceOp<GPUDevice, T, Eigen::bfloat16>);                                                            \
  template <>                                                                                                         \
  void EuclideanDistanceFunctor<GPUDevice, T, float>::operator()(                                                     \
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,    \
      float *output) const;                                                                                           \
  extern template struct EuclideanDistanceFunctor<GPUDevice, T, float>;                                               \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_GPU).TypeConstraint<T>("DT").TypeConstraint<float>("dtype"),       \
      EuclideanDistanceOp<GPUDevice, T, float>);                                                                      \
  template <>                                                                                                         \
  void EuclideanDistanceFunctor<GPUDevice, T, double>::operator()(                                                    \
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,    \
      double *output) const;                                                                                          \
  extern template struct EuclideanDistanceFunctor<GPUDevice, T, double>;                                              \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Miss>EuclideanDistance").Device(DEVICE_GPU).TypeConstraint<T>("DT").TypeConstraint<double>("dtype"),      \
      EuclideanDistanceOp<GPUDevice, T, double>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
TF_CALL_bool(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace miss
}  // end namespace tensorflow