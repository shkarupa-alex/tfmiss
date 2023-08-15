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

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_column(
    const T *input, const int batch_id, const int column_id, const int channel_id, const int height, const int width,
    const int channels, float *f, int *v, float *z, float *output)
{
  const int fv_offset = (batch_id * height * width + column_id) * channels + channel_id;
  const int fv_shift = width * channels;
  const int z_offset = (batch_id * (height + 1) * (width + 1) + column_id) * channels + channel_id;
  const int z_shift = (width + 1) * channels;
  int h_index;

  for (int y = 0; y < height; y++)
  {
    h_index = y * fv_shift + fv_offset;
    f[h_index] = input[h_index] ? Eigen::NumTraits<float>::highest() : 0;
  }

  euclidean_distance_1d(f + fv_offset, output + fv_offset, v + fv_offset, z + z_offset, height, fv_shift, z_shift);
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_row(
    const int batch_id, const int row_id, const int channel_id, const int height, const int width, const int channels,
    float *d, int *v, float *z, float *output)
{
  const int dv_offset = (batch_id * height + row_id) * width * channels + channel_id;
  const int z_offset = (batch_id * (height + 1) + row_id) * (width + 1) * channels + channel_id;
  const int dvz_shift = channels;
  int w_index;

  euclidean_distance_1d(output + dv_offset, d + dv_offset, v + dv_offset, z + z_offset, width, dvz_shift, dvz_shift);

  for (int x = 0; x < width; x++)
  {
    w_index = x * dvz_shift + dv_offset;
    output[w_index] = d[w_index] == Eigen::NumTraits<float>::highest() ? Eigen::NumTraits<float>::highest()
                                                                       : Eigen::numext::sqrt(d[w_index]);
  }
}

template <typename Device, typename T>
struct EuclideanDistanceFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const int batch, const int height, const int width, const int channel,
      float *fd, int *v, float *z, float *output) const;
};

}  // end namespace miss
}  // end namespace tensorflow
