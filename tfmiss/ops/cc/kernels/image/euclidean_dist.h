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
    const float *f, float *d, int *v, float *z, const int n)
{
  int k = 0;
  v[0] = 0;
  z[0] = z[1] = Eigen::NumTraits<float>::lowest();
  float s, s_;

  for (int q = 1; q < n; q++)
  {
    s_ = f[q] + static_cast<float>(Eigen::numext::pow(q, 2));

    k++;
    do
    {
      k--;
      s = s_ - f[v[k]] - static_cast<float>(Eigen::numext::pow(v[k], 2));
      s /= static_cast<float>((q - v[k]) * 2);
    } while (s <= z[k]);

    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Eigen::NumTraits<float>::highest();
  }

  k = 0;
  for (int q = 0; q < n; q++)
  {
    while (z[k + 1] < static_cast<float>(q))
    {
      k++;
    }
    d[q] = static_cast<float>(Eigen::numext::pow(q - v[k], 2)) + f[v[k]];
  }
}

template <typename IT, typename OT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_column(
    const IT *input, const int batch_id, const int column_id, const int channel_id, const int height, const int width,
    const int channels, OT *output)
{
  const int o = (batch_id * height * width + column_id) * channels + channel_id;
  const int s = width * channels;
  float *f = new float[height];
  float *d = new float[height];
  int *v = new int[height];
  float *z = new float[height + 1];

  for (int y = 0; y < height; y++)
  {
    f[y] = input[o + s * y] ? Eigen::NumTraits<float>::highest() : 0;
  }

  euclidean_distance_1d(f, d, v, z, height);

  for (int y = 0; y < height; y++)
  {
    output[o + s * y] = Eigen::numext::isinf(d[y]) || d[y] >= static_cast<float>(Eigen::NumTraits<OT>::highest())
                            ? Eigen::NumTraits<OT>::highest()
                            : static_cast<OT>(d[y]);
  }

  delete[] f;
  delete[] d;
  delete[] v;
  delete[] z;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_row(
    const int batch_id, const int row_id, const int channel_id, const int height, const int width, const int channels,
    T *output)
{
  const int o = (batch_id * height + row_id) * width * channels + channel_id;
  const int s = channels;
  float *f = new float[width];
  float *d = new float[width];
  int *v = new int[width];
  float *z = new float[width + 1];

  for (int x = 0; x < width; x++)
  {
    f[x] = static_cast<float>(output[o + s * x]);
  }

  euclidean_distance_1d(f, d, v, z, width);

  for (int x = 0; x < width; x++)
  {
    float v = Eigen::numext::sqrt(d[x]);
    if (std::is_same<T, double>::value)
    {
      output[o + s * x] = Eigen::numext::isinf(d[x]) || d[x] == Eigen::NumTraits<float>::highest()
                              ? Eigen::NumTraits<T>::highest()
                              : static_cast<T>(v);
    }
    else
    {
      output[o + s * x] = Eigen::numext::isinf(d[x]) || d[x] == static_cast<float>(Eigen::NumTraits<T>::highest()) ||
                                  v > static_cast<float>(Eigen::NumTraits<T>::highest())
                              ? Eigen::NumTraits<T>::highest()
                              : static_cast<T>(v);
    }
  }

  delete[] f;
  delete[] d;
  delete[] v;
  delete[] z;
}

template <typename Device, typename IT, typename OT>
struct EuclideanDistanceFunctor
{
  void operator()(
      OpKernelContext *ctx, const IT *input, const int batch, const int height, const int width, const int channel,
      OT *output) const;
};

}  // end namespace miss
}  // end namespace tensorflow
