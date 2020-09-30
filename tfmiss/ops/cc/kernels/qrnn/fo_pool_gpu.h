#pragma once

#if GOOGLE_CUDA

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "fo_pool.h"

/* TIME MAJOR */
void TimeMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void TimeMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void TimeMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void TimeMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);

/* BATCH MAJOR */
void BatchMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BatchMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BatchMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BatchMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);

namespace tensorflow {
namespace miss {

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

/* TIME MAJOR */

// Specialise the FoPool op for GPUs
template <typename FT, bool time_major>
class FoPool<GPUDevice, FT, time_major> : public tensorflow::OpKernel
{
public:
  explicit FoPool(tensorflow::OpKernelConstruction *context) : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override
  {
    namespace tf = tensorflow;

    // Create variables for input tensors
    const auto &in_x = context->input(0);
    const auto &in_forget = context->input(1);
    const auto &in_hinit = context->input(2);

    // Allocate output tensors
    // Allocate space for output tensor 'output'
    tf::Tensor *output_ptr = nullptr;
    auto in_x_shape = in_x.shape();
    tf::TensorShape output_shape = in_x_shape;
    if (time_major)
    {
      output_shape.set_dim(0, output_shape.dim_size(0) + 1);
    }
    else
    {
      output_shape.set_dim(1, output_shape.dim_size(1) + 1);
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_ptr));

    // Get pointers to flattened tensor data buffers
    const auto fin_x = in_x.flat<FT>().data();
    const auto fin_forget = in_forget.flat<FT>().data();
    const auto fin_hinit = in_hinit.flat<FT>().data();
    auto fout_output = output_ptr->flat<FT>().data();

    // Get the GPU device
    const auto &device = context->eigen_device<GPUDevice>();

    // Call the qrnn_fo_pool CUDA kernel
    if (time_major)
    {
      TimeMajorFoPoolLauncher(fout_output, fin_x, fin_forget, fin_hinit, in_x_shape.dim_size(0),
        output_shape.dim_size(1), output_shape.dim_size(2), device.stream());
    }
    else
    {
      BatchMajorFoPoolLauncher(fout_output, fin_x, fin_forget, fin_hinit, in_x_shape.dim_size(1),
        output_shape.dim_size(0), output_shape.dim_size(2), device.stream());
    }
  }
};

template <typename FT, bool time_major>
class BwdFoPool<GPUDevice, FT, time_major> : public tensorflow::OpKernel
{
public:
  explicit BwdFoPool(tensorflow::OpKernelConstruction *context) : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override
  {
    namespace tf = tensorflow;

    const auto &in_h = context->input(0);
    const auto &in_x = context->input(1);
    const auto &in_forget = context->input(2);
    const auto &in_gh = context->input(3);

    // Extract Eigen tensors
    auto h = in_h.flat<FT>().data();
    auto x = in_x.flat<FT>().data();
    auto forget = in_forget.flat<FT>().data();
    auto gh = in_gh.flat<FT>().data();

    // Allocate output tensors
    // Allocate space for output tensor 'output'
    tf::Tensor *out_gf = nullptr;
    tf::Tensor *out_gx = nullptr;
    tf::Tensor *out_ginitial_state = nullptr;

    auto in_x_shape = in_x.shape();
    tf::TensorShape grad_shape = in_x_shape;
    int batch_size = time_major ? in_x_shape.dim_size(1) : in_x_shape.dim_size(0);
    tf::TensorShape ginitial_state_shape({batch_size, in_x_shape.dim_size(2)});

    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &out_gx));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_shape, &out_gf));
    OP_REQUIRES_OK(context, context->allocate_output(2, ginitial_state_shape, &out_ginitial_state));
    auto gf = out_gf->flat<FT>().data();
    auto gx = out_gx->flat<FT>().data();
    auto ginitial_state = out_ginitial_state->flat<FT>().data();

    // Get the GPU device
    const auto &device = context->eigen_device<GPUDevice>();

    if (time_major)
    {
      TimeMajorBwdFoPoolLauncher(h, x, forget, gh, gx, gf, ginitial_state, grad_shape.dim_size(0),
        grad_shape.dim_size(1), grad_shape.dim_size(2), device.stream());
    }
    else
    {
      BatchMajorBwdFoPoolLauncher(h, x, forget, gh, gx, gf, ginitial_state, grad_shape.dim_size(1),
        grad_shape.dim_size(0), grad_shape.dim_size(2), device.stream());
    }
  }
};

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorFoPool")
  .TypeConstraint<Eigen::half>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, Eigen::half, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorBwdFoPool")
  .TypeConstraint<Eigen::half>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, Eigen::half, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorFoPool")
  .TypeConstraint<float>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorBwdFoPool")
  .TypeConstraint<float>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorFoPool")
  .TypeConstraint<double>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>TimeMajorBwdFoPool")
  .TypeConstraint<double>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorFoPool")
  .TypeConstraint<Eigen::half>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, Eigen::half, false>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorBwdFoPool")
  .TypeConstraint<Eigen::half>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, Eigen::half, false>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorFoPool")
  .TypeConstraint<float>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorBwdFoPool")
  .TypeConstraint<float>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorFoPool")
  .TypeConstraint<double>("FT")
  .Device(tensorflow::DEVICE_GPU),
  FoPool<GPUDevice, double, false>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>BatchMajorBwdFoPool")
  .TypeConstraint<double>("FT")
  .Device(tensorflow::DEVICE_GPU),
  BwdFoPool<GPUDevice, double, false>);

} // end namespace miss
} // namespace tensorflow

#endif // #if GOOGLE_CUDA
