#pragma once

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace miss {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T im2col_bilinear(const T *data, const int height, const int width, const int channels, const T h, const T w)
{
  const T zero = static_cast<T>(0.);
  const T one = static_cast<T>(1.);

  const int h_low = floor(static_cast<float>(h));
  const int w_low = floor(static_cast<float>(w));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const bool h_low_within = 0 <= h_low && h_low < height;
  const bool w_low_within = 0 <= w_low && w_low < width;
  const bool h_high_within = 0 <= h_high && h_high < height;
  const bool w_high_within = 0 <= w_high && w_high < width;

  const T lh = h - static_cast<T>(h_low);
  const T lw = w - static_cast<T>(w_low);
  const T hh = one - lh;
  const T hw = one - lw;

  T value = zero;

  if (h_low_within && w_low_within)
  {
    value += data[(h_low * width + w_low) * channels] * hh * hw;
  }

  if (h_low_within && w_high_within)
  {
    value += data[(h_low * width + w_high) * channels] * hh * lw;
  }

  if (h_high_within && w_low_within)
  {
    value += data[(h_high * width + w_low) * channels] * lh * hw;
  }

  if (h_high_within && w_high_within)
  {
    value += data[(h_high * width + w_high) * channels] * lh * lw;
  }

  return value;
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T coordinate_weight(
    const T *data, const int height, const int width, const int channels, const T h, const T w, const bool horizontal)
{
  const T zero = static_cast<T>(0.);
  const T one = static_cast<T>(1.);
  const T two = static_cast<T>(2.);
//  const T epsilon = static_cast<T>(1e-13);
  const T epsilon = Eigen::NumTraits<T>::dummy_precision();

  const int h_low = floor(h);
  const int w_low = floor(w);
  const T lh = h - static_cast<T>(h_low);
  const T lw = w - static_cast<T>(w_low);

  T weight = zero;

  if (horizontal && (zero == lh || one == lh))
  {
    weight += coordinate_weight(data, height, width, channels, h - epsilon, w, horizontal);
    weight += coordinate_weight(data, height, width, channels, h + epsilon, w, horizontal);
    weight /= two;

    return weight;
  }

  if (!horizontal && (zero == lw || one == lw))
  {
    weight += coordinate_weight(data, height, width, channels, h, w - epsilon, horizontal);
    weight += coordinate_weight(data, height, width, channels, h, w + epsilon, horizontal);
    weight /= two;

    return weight;
  }


  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const T hh = one - lh;
  const T hw = one - lw;

  const bool h_low_within = 0 <= h_low && h_low < height;
  const bool w_low_within = 0 <= w_low && w_low < width;
  const bool h_high_within = 0 <= h_high && h_high < height;
  const bool w_high_within = 0 <= w_high && w_high < width;

  if (h_low_within && w_low_within)
  {
    const T value = data[(h_low * width + w_low) * channels];
    weight += value * (horizontal ? -hw : -hh);
  }
  if (h_low_within && w_high_within)
  {
    const T value = data[(h_low * width + w_high) * channels];
    weight += value * (horizontal ? -lw : hh);
  }
  if (h_high_within && w_low_within)
  {
    const T value = data[(h_high * width + w_low) * channels];
    weight += value * (horizontal ? hw : -lh);
  }
  if (h_high_within && w_high_within)
  {
    const T value = data[(h_high * width + w_high) * channels];
    weight += value * (horizontal ? lw : lh);
  }

  return weight;
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool input_weight(
  const T h, const T w, const int height, const int width, const int dir, int *h_in, int *w_in, T *weight)
{
  const T zero = static_cast<T>(0.);
  const T one = static_cast<T>(1.);

  const int h_low = floor(static_cast<float>(h));
  const int w_low = floor(static_cast<float>(w));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const T lh = h - static_cast<T>(h_low);
  const T lw = w - static_cast<T>(w_low);
  const T hh = one - lh;
  const T hw = one - lw;

  const bool h_low_within = 0 <= h_low && h_low < height;
  const bool w_low_within = 0 <= w_low && w_low < width;
  const bool h_high_within = 0 <= h_high && h_high < height;
  const bool w_high_within = 0 <= w_high && w_high < width;


  if (0 == dir && h_low_within && w_low_within)
  {
    *h_in = h_low;
    *w_in = w_low;
    *weight = hh * hw;

    return true;
  }
  else if (1 == dir && h_low_within && w_high_within)
  {
    *h_in = h_low;
    *w_in = w_high;
    *weight = hh * lw;

    return true;
  }
  else if (2 == dir && h_high_within && w_low_within)
  {
    *h_in = h_high;
    *w_in = w_low;
    *weight = lh * hw;

    return true;
  }
  else if (3 == dir && h_high_within && w_high_within)
  {
    *h_in = h_high;
    *w_in = w_high;
    *weight = lh * lw;

    return true;
  }

  *h_in = 0;
  *w_in = 0;
  *weight = zero;

  return false;
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void modulated_deformable_im2col_body(
    const int index, const T *input, const T *offset, const T *mask, const int batch_size, const int height_in,
    const int width_in, const int channel_in, const int height_out, const int width_out, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, T *output)
{
  const int b = index / channel_in / width_out / height_in;
  const int h = index / channel_in / width_out % height_in;
  const int w = index / channel_in % width_out;
  const int c = index % channel_in;
  const int g = c % deformable_group;

  const int h_in = h * stride_h - pad_h;
  const int w_in = w * stride_w - pad_w;

  const T *input_slice = input +
      b * height_in * width_in * channel_in +
      //          h * width_in * channel_in +
      //                     w * channel_in +
                                          c;

  const T *offset_slice = offset +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                   h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                               w * deformable_group * kernel_h * kernel_w * 2 +
                                                  g * kernel_h * kernel_w * 2;
      //                                                     i * kernel_w * 2 +
      //                                                                j * 2 +
      //                                                                    0;
  const T *mask_slice = mask +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                   h * width_out * deformable_group * kernel_h * kernel_w +
                               w * deformable_group * kernel_h * kernel_w +
                                                  g * kernel_h * kernel_w;
      //                                                     i * kernel_w +
      //                                                                j;

  T *output_slice = output +
      b * height_out * width_out * channel_in * kernel_h * kernel_w +
                   h * width_out * channel_in * kernel_h * kernel_w +
                               w * channel_in * kernel_h * kernel_w +
                                            c * kernel_h * kernel_w;
      //                                               i * kernel_w +
      //                                                          j;

  for (int i = 0; i < kernel_h; ++i) {
    for (int j = 0; j < kernel_w; ++j) {
      const T offset_h_ = offset_slice[i * kernel_w * 2 + j * 2];
      const T offset_w_ = offset_slice[i * kernel_w * 2 + j * 2 + 1];
      const T mask_ = mask_slice[i * kernel_w + j];

      const T h_im = static_cast<T>(h_in + i * dilation_h) + offset_h_;
      const T w_im = static_cast<T>(w_in + j * dilation_w) + offset_w_;

      const T value = im2col_bilinear(input_slice, height_in, width_in, channel_in, h_im, w_im);

      output_slice[i * kernel_w + j] = mask_ * value;
    }
  }
}


template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
void modulated_deformable_col2im_body(
    const int index, const T *input, const T *offset, const T *mask, const T *grad, const int batch_size,
    const int height_in, const int width_in, const int channel_in, const int height_out, const int width_out,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group, T *grad_input, T *grad_offset, T *grad_mask)
{
  const int b = index / channel_in / width_out / height_in;
  const int h = index / channel_in / width_out % height_in;
  const int w = index / channel_in % width_out;
  const int c = index % channel_in;
  const int g = c % deformable_group;

  const int h_in = h * stride_h - pad_h;
  const int w_in = w * stride_w - pad_w;

  const T *input_slice = input +
      b * height_in * width_in * channel_in +
      //          h * width_in * channel_in +
      //                     w * channel_in +
                                          c;

  const T *offset_slice = offset +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                   h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                               w * deformable_group * kernel_h * kernel_w * 2 +
                                                  g * kernel_h * kernel_w * 2;
      //                                                     i * kernel_w * 2 +
      //                                                                j * 2 +
      //                                                                    0;
  const T *mask_slice = mask +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                   h * width_out * deformable_group * kernel_h * kernel_w +
                               w * deformable_group * kernel_h * kernel_w +
                                                  g * kernel_h * kernel_w;
      //                                                     i * kernel_w +
      //                                                                j;

  const T *grad_slice = grad +
      b * height_out * width_out * channel_in * kernel_h * kernel_w +
                   h * width_out * channel_in * kernel_h * kernel_w +
                               w * channel_in * kernel_h * kernel_w +
                                            c * kernel_h * kernel_w;
      //                                               i * kernel_w +
      //                                                          j;

  T *grad_input_slice = grad_input +
      b * height_out * width_out * channel_in +
//                   h * width_out * channel_in +
//                               w * channel_in +
                                            c;

  T *grad_offset_slice = grad_offset +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                   h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                               w * deformable_group * kernel_h * kernel_w * 2 +
                                                  g * kernel_h * kernel_w * 2;
      //                                                     i * kernel_w * 2 +
      //                                                                j * 2 +
      //                                                                    0;

  T *grad_mask_slice = grad_mask +
      b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                   h * width_out * deformable_group * kernel_h * kernel_w +
                               w * deformable_group * kernel_h * kernel_w +
                                                  g * kernel_h * kernel_w;
      //                                                     i * kernel_w +
      //                                                                j;

  for (int i = 0; i < kernel_h; ++i) {
    for (int j = 0; j < kernel_w; ++j) {
      const T offset_h_ = offset_slice[i * kernel_w * 2 + j * 2];
      const T offset_w_ = offset_slice[i * kernel_w * 2 + j * 2 + 1];
      const T mask_ = mask_slice[i * kernel_w + j];
      const T grad_ = grad_slice[i * kernel_w + j];

      const T h_im = static_cast<T>(h_in + i * dilation_h) + offset_h_;
      const T w_im = static_cast<T>(w_in + j * dilation_w) + offset_w_;

      const T top_grad = grad_ * mask_;

      // Mask gradient
      const T m_grad = grad_ * im2col_bilinear(input_slice, height_in, width_in, channel_in, h_im, w_im);
      grad_mask_slice[i * kernel_w + j] += m_grad;


      // Offset gradient
      const T o_h_grad = top_grad * coordinate_weight(input_slice, height_in, width_in, channel_in, h_im, w_im, true);
      const T o_w_grad = top_grad * coordinate_weight(input_slice, height_in, width_in, channel_in, h_im, w_im, false);
      grad_offset_slice[i * kernel_w * 2 + j * 2] += o_h_grad;
      grad_offset_slice[i * kernel_w * 2 + j * 2 + 1] += o_w_grad;

      // Input gradient
      for (int z=0; z < 4; z++) {
          int i_height, i_width;
          T i_grad;

          const bool update = input_weight(h_im, w_im, height_in, width_in, z, &i_height, &i_width, &i_grad);
          if (update)
          {
            i_grad *= top_grad;
            grad_input_slice[(i_height * width_out + i_width) * channel_in] += i_grad;
          }
      }
    }
  }
}

template <typename Device, typename T>
struct ModulatedDeformableColumnForwardFunctor {
  void operator()(OpKernelContext *ctx, const T *input, const T *offset,
                  const T *mask, const int batch_size, const int channel_in,
                  const int height_in, const int width_in, const int height_out,
                  const int width_out, const int kernel_h, const int kernel_w,
                  const int pad_h, const int pad_w, const int stride_h,
                  const int stride_w, const int dilation_h,
                  const int dilation_w, const int deformable_group,
                  T *column) const;
};

template <typename Device, typename T>
struct ModulatedDeformableColumnBackwardFunctor {
  void operator()(OpKernelContext *ctx, const T *input, const T *offset,
                  const T *mask, const T *grad, const int batch_size,
                  const int height_in, const int width_in, const int channel_in,
                  const int height_out, const int width_out, const int kernel_h,
                  const int kernel_w, const int pad_h, const int pad_w,
                  const int stride_h, const int stride_w, const int dilation_h,
                  const int dilation_w, const int deformable_group,
                  T *grad_offset, T *grad_mask) const;
};

}  // end namespace miss
}  // namespace tensorflow
