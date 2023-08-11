#pragma once

#define EIGEN_USE_THREADS

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{
namespace miss
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void init_labels(
    const T *input, const int index, const int height, const int width, const int channel, int64 *output)
{
  const T pyx = input[index];

  if (!pyx)
  {
    output[index] = 0;
  }
  else
  {
    const int row_id = index / channel / width % height;
    const int column_id = index / channel % width;

    const int nym1x = index - width * channel;
    const int nyxm1 = index - channel;

    if (row_id && input[nym1x])
    {
      output[index] = nym1x + 1;
    }
    else if (column_id && input[nyxm1])
    {
      output[index] = nyxm1 + 1;
    }
    else
    {
      output[index] = index + 1;
    }
  }
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void resolve_labels(
    const int index, const int height, const int width, const int channel, int64 *output)
{
  int64 label = output[index];

  if (label)
  {
    int64 next = output[label - 1];

    while (label != next)
    {
      label = next;
      next = output[label - 1];
    }

    output[index] = label;
  }
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T atomic_min(T *address, T val);

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void reduce_labels(
    const T *input, const int index, const int height, const int width, const int channel, int64 *output)
{
  const int row_id = index / channel / width % height;
  const int column_id = index / channel % width;

  const int nyxm1 = index - channel;
  const int nym1x = index - width * channel;
  const int nym1xm1 = nym1x - channel;

  const T pyx = input[index];

  if (column_id && row_id && pyx && input[nyxm1] && input[nym1x] && !input[nym1xm1])
  {
    int64 label1 = output[index];
    int64 label2 = output[nyxm1];

    int64 next1 = (label1 != label2) ? output[label1 - 1] : 0;
    int64 next2 = (label1 != label2) ? output[label2 - 1] : 0;

    while (label1 != label2 && label1 != next1)
    {
      label1 = next1;
      next1 = output[label1 - 1];
    }

    while (label1 != label2 && label2 != next2)
    {
      label2 = next2;
      next2 = output[label2 - 1];
    }

    int64 label3;
    while (label1 != label2)
    {
      if (label1 < label2)
      {
        label3 = label1;
        label1 = label2;
        label2 = label3;
      }

      label3 = atomic_min<int64>(&output[label1 - 1], label2);
      label1 = (label1 == label3) ? label2 : label3;
    }
  }
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void normalize_labels(
    const int index, const int height, const int width, const int channel, int64 *output)
{
  const int batch_id = index / channel;
  const int channel_id = index % channel;
  const int offset0 = batch_id * height * width * channel + channel_id;
  int counter = 1;

  for (int row_id = 0; row_id < height; row_id++)
  {
    const int offset1 = offset0 + row_id * width * channel;

    for (int column_id = 0; column_id < width; column_id++)
    {
      const int nyx = offset1 + column_id * channel;
      const int64 label = output[nyx];

      if (label)
      {
        if (label - 1 == nyx)
        {
          output[nyx] = counter++;
        }
        else
        {
          output[nyx] = output[label - 1];
        }
      }
    }
  }
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void minimize_labels(
    const int index, const int height, const int width, const int channel, int64 *output)
{
  const int batch_id = index / channel / width / height;
  const int row_id = index / channel / width % height;
  const int column_id = index / channel % width;
  const int channel_id = index % channel;

  if (output[index])
  {
    output[index] -= batch_id * height * width * channel + channel_id;
  }
}

template <typename Device, typename T>
struct ConnectedComponentsFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const bool norm, const int batch, const int height, const int width,
      const int channel, int64 *output) const;
};

}  // end namespace miss
}  // end namespace tensorflow
