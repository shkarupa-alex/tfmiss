#if GOOGLE_CUDA

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "fo_pool.h"

struct KernelParams
{
  dim3 grid;
  dim3 blocks;
  KernelParams(int HIDDEN, int batch_size) : grid(std::ceil(double(HIDDEN / double(min(HIDDEN, 512)))), batch_size, 1), blocks(min(HIDDEN, 512), 1, 1){};
};

/* TIME MAJOR */

template <typename FT>
__global__ void time_major_fo_pool(FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (hid >= HIDDEN || batch_id >= batch_size)
    return;
  //
  dst[batch_id * HIDDEN + hid] = initial_state[batch_id * HIDDEN + hid];
  for (int ts = 0 + 1; ts < SEQ + 1; ts++)
  {
    // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
    // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
    // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
    // To move timesteps, we step HIDDEN * batch_size
    // To move batches, we move HIDDEN
    // To move neurons, we move +- 1
    // Note: dst[dst_i] = ts * 100 + batch_id * 10 + hid; is useful for debugging
    int i = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
    int dst_i = (ts - 0) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
    int dst_iminus1 = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
    dst[dst_i] = f[i] * x[i];
    dst[dst_i] += ((FT)1 - f[i]) * dst[dst_iminus1];
  }
}

template <typename FT>
__global__ void time_major_bwd_fo_pool(const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (hid >= HIDDEN || batch_id >= batch_size)
    return;
  //
  FT running_f = (FT)0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--)
  {
    int i = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
    int dst_iminus1 = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
    //
    running_f += gh[dst_iminus1];
    // Gradient of X
    gx[i] = f[i] * running_f;
    // Gradient of F
    gf[i] = (x[i] - h[dst_iminus1]) * running_f;
    //
    // The line below is likely more numerically stable than (1 - f[i]) * running_f;
    running_f = running_f - f[i] * running_f;
  }
  ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN + hid];
}

void TimeMajorFoPoolLauncher(Eigen::half *dst, const Eigen::half *x, const Eigen::half *f, const Eigen::half *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorBwdFoPoolLauncher(const Eigen::half *h, const Eigen::half *x, const Eigen::half *f, const Eigen::half *gh, Eigen::half *gx, Eigen::half *gf, Eigen::half *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  time_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}

/* BATCH MAJOR */

template <typename FT>
__global__ void batch_major_fo_pool(FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (hid >= HIDDEN || batch_id >= batch_size)
    return;
  //
  dst[batch_id * HIDDEN * (SEQ + 1) + hid] = initial_state[batch_id * HIDDEN + hid];
  for (int ts = 0 + 1; ts < SEQ + 1; ts++)
  {
    // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
    // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
    // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
    // To move timesteps, we step HIDDEN * batch_size
    // To move batches, we move HIDDEN
    // To move neurons, we move +- 1
    // Note: dst[dst_i] = ts * 100 + batch_id * 10 + hid; is useful for debugging
    int i = (ts - 1) * HIDDEN + batch_id * HIDDEN * SEQ + hid;
    int dst_i = (ts - 0) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
    int dst_iminus1 = (ts - 1) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
    dst[dst_i] = f[i] * x[i];
    dst[dst_i] += ((FT)1 - f[i]) * dst[dst_iminus1];
  }
}

template <typename FT>
__global__ void batch_major_bwd_fo_pool(const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (hid >= HIDDEN || batch_id >= batch_size)
    return;
  //
  FT running_f = (FT)0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--)
  {
    int i = (ts - 1) * HIDDEN + batch_id * HIDDEN * SEQ + hid;
    int dst_iminus1 = (ts - 1) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
    //
    running_f += gh[dst_iminus1];
    // Gradient of X
    gx[i] = f[i] * running_f;
    // Gradient of F
    gf[i] = (x[i] - h[dst_iminus1]) * running_f;
    //
    // The line below is likely more numerically stable than (1 - f[i]) * running_f;
    running_f = running_f - f[i] * running_f;
  }
  ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN * (SEQ + 1) + hid];
}

void BatchMajorFoPoolLauncher(Eigen::half *dst, const Eigen::half *x, const Eigen::half *f, const Eigen::half *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorBwdFoPoolLauncher(const Eigen::half *h, const Eigen::half *x, const Eigen::half *f, const Eigen::half *gh, Eigen::half *gx, Eigen::half *gf, Eigen::half *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream)
{
  KernelParams l(HIDDEN, batch_size);
  batch_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}

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
