#pragma once

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{
namespace miss
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_1d(
    const float *f, float *d, int *v, float *z, const int n, const int fdvs, const int zs)
{
  int k = 0;
  v[fdvs * 0] = 0;
  z[zs * 0] = z[zs * 1] = Eigen::NumTraits<float>::lowest();
  float s, s_, d_;

  for (int q = 1; q < n; q++)
  {
    s_ = f[fdvs * q] + static_cast<float>(Eigen::numext::pow(q, 2));

    k++;
    do
    {
      k--;
      s = s_ - f[fdvs * v[fdvs * k]] - static_cast<float>(Eigen::numext::pow(v[fdvs * k], 2));
      s /= static_cast<float>((q - v[fdvs * k]) * 2);
    } while (s <= z[zs * k]);

    k++;
    v[fdvs * k] = q;
    z[zs * k] = s;
    z[zs * (k + 1)] = Eigen::NumTraits<float>::highest();
  }

  k = 0;
  for (int q = 0; q < n; q++)
  {
    while (z[zs * (k + 1)] < static_cast<float>(q))
    {
      k++;
    }

    d_ = static_cast<float>(Eigen::numext::pow(q - v[fdvs * k], 2)) + f[fdvs * v[fdvs * k]];
    d_ = Eigen::numext::isinf(d_) ? Eigen::NumTraits<float>::highest() : d_;
    d[fdvs * q] = d_;
  }
}

template <typename IT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_column(
    const IT *input, const int batch_id, const int column_id, const int channel_id, const int height, const int width,
    const int channels, float *fd, int *v, float *z, float *output)
{
  const int fdvo = (batch_id * height * width + column_id) * channels + channel_id;
  const int fdvs = width * channels;
  const int zo = (batch_id * (height + 1) * (width + 1) + column_id) * channels + channel_id;
  const int zs = (width + 1) * channels;
  int i;

  for (int y = 0; y < height; y++)
  {
    i = fdvo + fdvs * y;
    fd[i] = input[i] ? Eigen::NumTraits<float>::highest() : 0;
  }

  euclidean_distance_1d(fd + fdvo, output + fdvo, v + fdvo, z + zo, height, fdvs, zs);
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_row(
    const int batch_id, const int row_id, const int channel_id, const int height, const int width, const int channels,
    float *fd, int *v, float *z, float *output)
{
  const int fdvo = (batch_id * height + row_id) * width * channels + channel_id;
  const int zo = (batch_id * (height + 1) + row_id) * (width + 1) * channels + channel_id;
  const int fdvzs = channels;
  int i;

  euclidean_distance_1d(output + fdvo, fd + fdvo, v + fdvo, z + zo, width, fdvzs, fdvzs);

  for (int x = 0; x < width; x++)
  {
    i = fdvo + fdvzs * x;
    output[i] =
        fd[i] == Eigen::NumTraits<float>::highest() ? Eigen::NumTraits<float>::highest() : Eigen::numext::sqrt(fd[i]);
  }
}

template <typename Device, typename IT>
struct EuclideanDistanceFunctor
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      float *fd, int *v, float *z, float *output) const;
};

}  // end namespace miss
}  // end namespace tensorflow
