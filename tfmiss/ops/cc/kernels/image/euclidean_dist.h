#pragma once

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{
namespace miss
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_1d(const T *f, T *d, int *v, T *z, const int n)
{
  int k = 0;
  v[0] = 0;
  z[0] = Eigen::NumTraits<T>::lowest();
  z[1] = Eigen::NumTraits<T>::highest();
  T s, s_;

  for (int q = 1; q < n; q++)
  {
    s_ = f[q] + static_cast<T>(Eigen::numext::pow(q, 2));

    k++;
    do
    {
      k--;
      s = T(0.5) * (s_ - f[v[k]] - static_cast<T>(Eigen::numext::pow(v[k], 2))) / static_cast<T>(q - v[k]);
    } while (s <= z[k]);

    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Eigen::NumTraits<T>::highest();
  }

  k = 0;
  for (int q = 0; q < n; q++)
  {
    while (z[k + 1] < static_cast<T>(q))
    {
      k++;
    }
    d[q] = static_cast<T>(Eigen::numext::pow(q - v[k], 2)) + f[v[k]];
  }
}

template <typename IT, typename OT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void euclidean_distance_column(
    const IT *input, const int batch_id, const int column_id, const int channel_id, const int height, const int width,
    const int channels, OT *output)
{
  const int o = (batch_id * height * width + column_id) * channels + channel_id;
  const int s = width * channels;
  OT *f = new OT[height];
  OT *d = new OT[height];
  int *v = new int[height];
  OT *z = new OT[height + 1];

  for (int y = 0; y < height; y++)
  {
    int index = o + s * y;
    if (input[index] == 0)
    {
      f[y] = static_cast<OT>(0);
    }
    else
    {
      f[y] = Eigen::NumTraits<OT>::highest();
    }
  }

  euclidean_distance_1d<OT>(f, d, v, z, height);

  for (int y = 0; y < height; y++)
  {
    int index = o + s * y;
    if (Eigen::numext::isinf(d[y]))
    {
      output[index] = Eigen::NumTraits<OT>::highest();
    }
    else
    {
      output[index] = d[y];
    }
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
  const T r(Eigen::numext::sqrt(Eigen::NumTraits<T>::highest()));
  T *f = new T[width];
  T *d = new T[width];
  int *v = new int[width];
  T *z = new T[width + 1];

  for (int x = 0; x < width; x++)
  {
    int index = o + s * x;
    f[x] = output[index];
  }

  euclidean_distance_1d<T>(f, d, v, z, width);

  for (int x = 0; x < width; x++)
  {
    int index = o + s * x;
    if (Eigen::numext::isinf(d[x]))
    {
      output[index] = r;
    }
    else
    {
      output[index] = Eigen::numext::sqrt(d[x]);
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
