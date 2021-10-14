#pragma once

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{
namespace miss
{
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void fo_pool_forward_body(
    const int index, const T *input, const T *forget, const T *init, const int batch_size, const int time_size,
    const int channel_size, T *output)
{
  const int b = index / channel_size;
  const int c = index % channel_size;

  output[b * (time_size + 1) * channel_size + c] = init[b * channel_size + c];

  for (int t = 0; t < time_size; t++)
  {
    const int inp_idx = (b * time_size + t) * channel_size + c;
    const int out_idx = (b * (time_size + 1) + t + 1) * channel_size + c;

    output[out_idx] = forget[inp_idx] * input[inp_idx];
    output[out_idx] += (static_cast<T>(1.) - forget[inp_idx]) * output[out_idx - channel_size];
  }
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void fo_pool_backward_body(
    const int index, const T *input, const T *forget, const T *hidden, const T *grad, const int batch_size,
    const int time_size, const int channel_size, T *grad_input, T *grad_forget, T *grad_init)
{
  const int b = index / channel_size;
  const int c = index % channel_size;

  T running_grad = static_cast<T>(0.);

  for (int t = time_size - 1; t >= 0; t--)
  {
    const int inp_idx = (b * time_size + t) * channel_size + c;
    const int grd_idx = (b * (time_size + 1) + t + 1) * channel_size + c;

    running_grad += grad[grd_idx];
    grad_input[inp_idx] = forget[inp_idx] * running_grad;
    grad_forget[inp_idx] = (input[inp_idx] - hidden[grd_idx - channel_size]) * running_grad;
    running_grad -= forget[inp_idx] * running_grad;
  }

  grad_init[b * channel_size + c] = running_grad + grad[b * (time_size + 1) * channel_size + c];
}

template <typename Device, typename T>
struct FoPoolForwardFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *init, const int batch_size, const int time_size,
      const int channel_size, T *output) const;
};

template <typename Device, typename T>
struct FoPoolBackwardFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *forget, const T *hidden, const T *grad, const int batch_size,
      const int time_size, const int channel_size, T *grad_input, T *grad_forget, T *grad_init) const;
};

}  // end namespace miss
}  // namespace tensorflow
