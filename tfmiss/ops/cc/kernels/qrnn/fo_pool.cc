#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/work_sharder.h" // TODO
#include "tensorflow/core/framework/types.h"
#include "fo_pool.h"
#include "thread_pool.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

namespace tensorflow {
namespace miss {

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename FT>
void time_major_fo_pool(OpKernelContext *context, FT *dst, const FT *x, const FT *f,
  const FT *initial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  const int64 cost = SEQ * HIDDEN * 1000;
  Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
    [&batch_size, x, f, initial_state, dst, &HIDDEN, &SEQ](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id)
      {
        for (int hid = 0; hid < HIDDEN; hid++)
        {
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
      }
    });
}

template <typename FT>
void batch_major_fo_pool(OpKernelContext *context, FT *dst, const FT *x, const FT *f,
  const FT *initial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  const int64 cost = SEQ * HIDDEN * 1000;
  Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
    [x, f, initial_state, dst, &HIDDEN, &SEQ](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id)
      {
        for (int hid = 0; hid < HIDDEN; hid++)
        {
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
      }
    });
}

template <typename FT>
void time_major_bwd_fo_pool(OpKernelContext *context, const FT *h, const FT *x, const FT *f, const FT *gh,
  FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  const int64 cost = SEQ * HIDDEN * 1000;
  Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
    [&batch_size, h, f, x, gh, gf, gx, ginitial_state, &HIDDEN, &SEQ](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id)
      {
        for (int hid = 0; hid < HIDDEN; hid++)
        {
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
            // The line below is likely more numerically stable than (1 - f[i]) * running_f;
            running_f = running_f - f[i] * running_f;
          }
          ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN + hid];
        }
      }
    });
}

template <typename FT>
void batch_major_bwd_fo_pool(OpKernelContext *context, const FT *h, const FT *x, const FT *f, const FT *gh,
  FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN)
{
  /*
Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
This means dst array has a separate index than that of f or x
*/
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  const int64 cost = SEQ * HIDDEN * 1000;
  Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
    [h, f, x, gh, gf, gx, ginitial_state, &HIDDEN, &SEQ](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id)
      {
        for (int hid = 0; hid < HIDDEN; hid++)
        {
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
            // The line below is likely more numerically stable than (1 - f[i]) * running_f;
            running_f = running_f - f[i] * running_f;
          }
          ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN * (SEQ + 1) + hid];
        }
      }
    });
}

// Specialise the FoPool op for CPUs
template <typename FT, bool time_major>
class FoPool<CPUDevice, FT, time_major> : public OpKernel
{
public:
  explicit FoPool(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    // Create reference to input Tensorflow tensors
    const auto &in_x = context->input(0);
    const auto &in_forget = context->input(1);
    const auto &in_initial_state = context->input(2);

    // Extract Eigen tensors
    auto x = in_x.flat<FT>().data();
    auto forget = in_forget.flat<FT>().data();
    auto initial_state = in_initial_state.flat<FT>().data();

    // Allocate output tensors
    // Allocate space for output tensor 'output'
    Tensor *output_ptr = nullptr;
    auto in_x_shape = in_x.shape();
    TensorShape output_shape = in_x_shape;
    if (time_major)
    {
      output_shape.set_dim(0, output_shape.dim_size(0) + 1);
    }
    else
    {
      output_shape.set_dim(1, output_shape.dim_size(1) + 1);
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_ptr));

    auto out = output_ptr->flat<FT>().data();
    if (time_major)
    {
      time_major_fo_pool(context, out, x, forget, initial_state, in_x_shape.dim_size(0), output_shape.dim_size(1),
        output_shape.dim_size(2));
    }
    else
    {
      batch_major_fo_pool(context, out, x, forget, initial_state, in_x_shape.dim_size(1), output_shape.dim_size(0),
        output_shape.dim_size(2));
    }
  }
};

template <typename FT, bool time_major>
class BwdFoPool<CPUDevice, FT, time_major> : public OpKernel
{
public:
  explicit BwdFoPool(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
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
    Tensor *out_gx = nullptr;
    Tensor *out_gf = nullptr;
    Tensor *out_ginitial_state = nullptr;

    auto in_x_shape = in_x.shape();
    TensorShape grad_shape = in_x_shape;
    int batch_size = time_major ? in_x_shape.dim_size(1) : in_x_shape.dim_size(0);
    TensorShape ginitial_state_shape({batch_size, in_x_shape.dim_size(2)});

    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &out_gx));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_shape, &out_gf));
    OP_REQUIRES_OK(context, context->allocate_output(2, ginitial_state_shape, &out_ginitial_state));
    auto gx = out_gx->flat<FT>().data();
    auto gf = out_gf->flat<FT>().data();
    auto ginitial_state = out_ginitial_state->flat<FT>().data();

    if (time_major)
    {
      time_major_bwd_fo_pool(context, h, x, forget, gh, gx, gf, ginitial_state, grad_shape.dim_size(0),
        grad_shape.dim_size(1), grad_shape.dim_size(2));
    }
    else
    {
      batch_major_bwd_fo_pool(context, h, x, forget, gh, gx, gf, ginitial_state, grad_shape.dim_size(1),
        grad_shape.dim_size(0), grad_shape.dim_size(2));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorFoPool")
    .TypeConstraint<Eigen::half>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, Eigen::half, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorFoPool")
    .TypeConstraint<Eigen::half>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, Eigen::half, false>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(DEVICE_CPU),
    FoPool<CPUDevice, double, false>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorBwdFoPool")
    .TypeConstraint<Eigen::half>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, Eigen::half, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>TimeMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorBwdFoPool")
    .TypeConstraint<Eigen::half>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, Eigen::half, false>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("Miss>BatchMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(DEVICE_CPU),
    BwdFoPool<CPUDevice, double, false>);

} // end namespace miss
} // namespace tensorflow
