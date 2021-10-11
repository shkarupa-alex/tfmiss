#pragma once

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow
{
namespace miss
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PT
im2col_bilinear(const T *data, const int height, const int width, const int channels, const PT h, const PT w)
{
  const PT zero = static_cast<PT>(0.);
  const PT one = static_cast<PT>(1.);

  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const bool h_low_within = 0 <= h_low && h_low < height;
  const bool w_low_within = 0 <= w_low && w_low < width;
  const bool h_high_within = 0 <= h_high && h_high < height;
  const bool w_high_within = 0 <= w_high && w_high < width;

  const PT lh = h - static_cast<PT>(h_low);
  const PT lw = w - static_cast<PT>(w_low);
  const PT hh = one - lh;
  const PT hw = one - lw;

  PT value = zero;

  if (h_low_within && w_low_within)
  {
    const PT h_low_w_low = data[(h_low * width + w_low) * channels];
    value += h_low_w_low * hh * hw;
  }

  if (h_low_within && w_high_within)
  {
    const PT h_low_w_high = data[(h_low * width + w_high) * channels];
    value += h_low_w_high * hh * lw;
  }

  if (h_high_within && w_low_within)
  {
    const PT h_high_w_low = data[(h_high * width + w_low) * channels];
    value += h_high_w_low * lh * hw;
  }

  if (h_high_within && w_high_within)
  {
    const PT h_high_w_high = data[(h_high * width + w_high) * channels];
    value += h_high_w_high * lh * lw;
  }

  return value;
}

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PT coordinate_weight(
    const T *data, const int height, const int width, const int channels, const PT h, const PT w, const bool horizontal,
    const bool recursive)
{
  const PT zero = static_cast<PT>(0.);
  const PT one = static_cast<PT>(1.);
  const PT two = static_cast<PT>(2.);
  const PT epsilon = Eigen::NumTraits<PT>::dummy_precision();

  const int h_low = floor(h);
  const int w_low = floor(w);
  const PT lh = h - static_cast<PT>(h_low);
  const PT lw = w - static_cast<PT>(w_low);

  PT weight = zero;

  if (!recursive && horizontal && (zero == lh || one == lh))
  {
    weight += coordinate_weight(data, height, width, channels, h - epsilon, w, horizontal, true);
    weight += coordinate_weight(data, height, width, channels, h + epsilon, w, horizontal, true);
    weight /= two;

    return weight;
  }

  if (!recursive && !horizontal && (zero == lw || one == lw))
  {
    weight += coordinate_weight(data, height, width, channels, h, w - epsilon, horizontal, true);
    weight += coordinate_weight(data, height, width, channels, h, w + epsilon, horizontal, true);
    weight /= two;

    return weight;
  }

  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const PT hh = one - lh;
  const PT hw = one - lw;

  const bool h_low_within = 0 <= h_low && h_low < height;
  const bool w_low_within = 0 <= w_low && w_low < width;
  const bool h_high_within = 0 <= h_high && h_high < height;
  const bool w_high_within = 0 <= w_high && w_high < width;

  if (h_low_within && w_low_within)
  {
    const PT value = data[(h_low * width + w_low) * channels];
    weight += value * (horizontal ? -hw : -hh);
  }
  if (h_low_within && w_high_within)
  {
    const PT value = data[(h_low * width + w_high) * channels];
    weight += value * (horizontal ? -lw : hh);
  }
  if (h_high_within && w_low_within)
  {
    const PT value = data[(h_high * width + w_low) * channels];
    weight += value * (horizontal ? hw : -lh);
  }
  if (h_high_within && w_high_within)
  {
    const PT value = data[(h_high * width + w_high) * channels];
    weight += value * (horizontal ? lw : lh);
  }

  return weight;
}

template <typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool input_weight(
    const PT h, const PT w, const int height, const int width, const int dir, int *h_in, int *w_in, PT *weight)
{
  const PT zero = static_cast<PT>(0.);
  const PT one = static_cast<PT>(1.);

  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const PT lh = h - static_cast<PT>(h_low);
  const PT lw = w - static_cast<PT>(w_low);
  const PT hh = one - lh;
  const PT hw = one - lw;

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

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void modulated_deformable_im2col_body(
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

  const T *input_slice = input + b * height_in * width_in * channel_in +
                         //          h * width_in * channel_in +
                         //                     w * channel_in +
                         c;

  const T *offset_slice = offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //                                                     i * kernel_w * 2 +
  //                                                                j * 2 +
  //                                                                    0;
  const T *mask_slice = mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                        h * width_out * deformable_group * kernel_h * kernel_w +
                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //                                                     i * kernel_w +
  //                                                                j;

  T *output_slice = output + b * height_out * width_out * channel_in * kernel_h * kernel_w +
                    h * width_out * channel_in * kernel_h * kernel_w + w * channel_in * kernel_h * kernel_w +
                    c * kernel_h * kernel_w;
  //                                               i * kernel_w +
  //                                                          j;

  for (int i = 0; i < kernel_h; ++i)
  {
    for (int j = 0; j < kernel_w; ++j)
    {
      const PT offset_h_ = static_cast<PT>(offset_slice[i * kernel_w * 2 + j * 2]);
      const PT offset_w_ = static_cast<PT>(offset_slice[i * kernel_w * 2 + j * 2 + 1]);
      const PT mask_ = static_cast<PT>(mask_slice[i * kernel_w + j]);

      const PT h_im = static_cast<PT>(h_in + i * dilation_h) + offset_h_;
      const PT w_im = static_cast<PT>(w_in + j * dilation_w) + offset_w_;

      const PT value = mask_ * im2col_bilinear<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im);

      output_slice[i * kernel_w + j] = static_cast<T>(value);
    }
  }
}

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void modulated_deformable_col2im_body(
    const int index, const T *input, const T *offset, const T *mask, const T *grad, const int batch_size,
    const int height_in, const int width_in, const int channel_in, const int height_out, const int width_out,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input, PT *grad_offset,
    PT *grad_mask)
{
  const int b = index / channel_in / width_out / height_in;
  const int h = index / channel_in / width_out % height_in;
  const int w = index / channel_in % width_out;
  const int c = index % channel_in;
  const int g = c % deformable_group;

  const int h_in = h * stride_h - pad_h;
  const int w_in = w * stride_w - pad_w;

  const T *input_slice = input + b * height_in * width_in * channel_in +
                         //          h * width_in * channel_in +
                         //                     w * channel_in +
                         c;

  const T *offset_slice = offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //                                                     i * kernel_w * 2 +
  //                                                                j * 2 +
  //                                                                    0;
  const T *mask_slice = mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                        h * width_out * deformable_group * kernel_h * kernel_w +
                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //                                                     i * kernel_w +
  //                                                                j;

  const T *grad_slice = grad + b * height_out * width_out * channel_in * kernel_h * kernel_w +
                        h * width_out * channel_in * kernel_h * kernel_w + w * channel_in * kernel_h * kernel_w +
                        c * kernel_h * kernel_w;
  //                                               i * kernel_w +
  //                                                          j;

  PT *grad_input_slice = grad_input + b * height_out * width_out * channel_in +
                         //                   h * width_out * channel_in +
                         //                               w * channel_in +
                         c;

  PT *grad_offset_slice = grad_offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //                                                     i * kernel_w * 2 +
  //                                                                j * 2 +
  //                                                                    0;

  PT *grad_mask_slice = grad_mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
                        h * width_out * deformable_group * kernel_h * kernel_w +
                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //                                                     i * kernel_w +
  //                                                                j;

  for (int i = 0; i < kernel_h; ++i)
  {
    for (int j = 0; j < kernel_w; ++j)
    {
      const PT offset_h_ = offset_slice[i * kernel_w * 2 + j * 2];
      const PT offset_w_ = offset_slice[i * kernel_w * 2 + j * 2 + 1];
      const PT mask_ = mask_slice[i * kernel_w + j];
      const PT grad_ = grad_slice[i * kernel_w + j];

      const PT h_im = static_cast<PT>(h_in + i * dilation_h) + offset_h_;
      const PT w_im = static_cast<PT>(w_in + j * dilation_w) + offset_w_;

      const PT top_grad = grad_ * mask_;

      // TODO: atomic add

      // Mask gradient
      const PT m_grad = grad_ * im2col_bilinear<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im);
      grad_mask_slice[i * kernel_w + j] += m_grad;

      // Offset gradient
      const PT o_h_grad =
          top_grad * coordinate_weight<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im, true, false);
      const PT o_w_grad =
          top_grad * coordinate_weight<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im, false, false);
      grad_offset_slice[i * kernel_w * 2 + j * 2] += o_h_grad;
      grad_offset_slice[i * kernel_w * 2 + j * 2 + 1] += o_w_grad;

      // Input gradient
      int i_height, i_width;
      PT i_grad;
      for (int z = 0; z < 4; z++)
      {
        const bool update = input_weight<PT>(h_im, w_im, height_in, width_in, z, &i_height, &i_width, &i_grad);
        if (update)
        {
          i_grad *= top_grad;
          grad_input_slice[(i_height * width_out + i_width) * channel_in] += i_grad;
        }
      }
    }
  }
}

template <typename Device, typename T, typename PT>
struct ModulatedDeformableColumnForwardFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const int batch_size, const int channel_in,
      const int height_in, const int width_in, const int height_out, const int width_out, const int kernel_h,
      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w, const int deformable_group, T *column) const;
};

template <typename Device, typename T, typename PT>
struct ModulatedDeformableColumnBackwardFunctor
{
  void operator()(
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *grad, const int batch_size,
      const int height_in, const int width_in, const int channel_in, const int height_out, const int width_out,
      const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input, PT *grad_offset,
      PT *grad_mask) const;
};

template <typename Device, typename T>
struct CastFloatFunctor
{
  void operator()(OpKernelContext *ctx, typename TTypes<float>::ConstFlat input, typename TTypes<T>::Flat output);
};

}  // end namespace miss
}  // namespace tensorflow
