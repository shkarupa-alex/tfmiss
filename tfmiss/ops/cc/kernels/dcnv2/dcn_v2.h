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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void atomic_add(T *ptr, const T value);

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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PT coordinate_weight_default(
    const T *data, const int height, const int width, const int channels, const PT h, const PT w, const bool horizontal)
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

  PT weight = zero;

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

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PT coordinate_weight(
    const T *data, const int height, const int width, const int channels, const PT h, const PT w, const bool horizontal)
{
  const PT zero = static_cast<PT>(0.);
  const PT one = static_cast<PT>(1.);
  const PT two = static_cast<PT>(2.);
  const PT epsilon = Eigen::NumTraits<PT>::dummy_precision();

  const int h_low = floor(h);
  const PT lh = h - static_cast<PT>(h_low);

  PT weight = zero;

  if (horizontal && (zero == lh || one == lh))
  {
    weight += coordinate_weight_default<T, PT>(data, height, width, channels, h - epsilon, w, horizontal);
    weight += coordinate_weight_default<T, PT>(data, height, width, channels, h + epsilon, w, horizontal);
    weight /= two;

    return weight;
  }

  const int w_low = floor(w);
  const PT lw = w - static_cast<PT>(w_low);

  if (!horizontal && (zero == lw || one == lw))
  {
    weight += coordinate_weight_default<T, PT>(data, height, width, channels, h, w - epsilon, horizontal);
    weight += coordinate_weight_default<T, PT>(data, height, width, channels, h, w + epsilon, horizontal);
    weight /= two;

    return weight;
  }

  return coordinate_weight_default<T, PT>(data, height, width, channels, h, w, horizontal);
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

  //  const T *input_slice = input + b * height_in * width_in * channel_in +
  //                         //                  h * width_in * channel_in +
  //                         //                             w * channel_in +
  //                         c;
  const T *input_slice = input + b * height_in * width_in * channel_in + c;

  //  const T *offset_slice = offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //  //                                                                                  i * kernel_w * 2 +
  //  //                                                                                             j * 2 +
  //  //                                                                                                 0;
  const T *offset_slice =
      offset + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w * 2;

  //  const T *mask_slice = mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
  //                        h * width_out * deformable_group * kernel_h * kernel_w +
  //                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //  //                                                                            i * kernel_w +
  //  //                                                                                       j;
  const T *mask_slice = mask + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w;

  //  T *output_slice = output + b * height_out * width_out * channel_in * kernel_h * kernel_w +
  //                    h * width_out * channel_in * kernel_h * kernel_w + w * channel_in * kernel_h * kernel_w +
  //                    c * kernel_h * kernel_w;
  //  //                                                                                           i * kernel_w +
  //  //                                                                                                      j;
  T *output_slice = output + (((b * height_out + h) * width_out + w) * channel_in + c) * kernel_h * kernel_w;

  for (int i = 0; i < kernel_h; ++i)
  {
    const int kernel_h_shift = i * kernel_w;

    for (int j = 0; j < kernel_w; ++j)
    {
      const int kernel_hw_shift = kernel_h_shift + j;
      const int kernel_hw_shift_x2 = kernel_hw_shift * 2;

      const PT offset_h_ = static_cast<PT>(offset_slice[kernel_hw_shift_x2]);
      const PT offset_w_ = static_cast<PT>(offset_slice[kernel_hw_shift_x2 + 1]);
      const PT mask_ = static_cast<PT>(mask_slice[kernel_hw_shift]);

      const PT h_im = static_cast<PT>(h_in + i * dilation_h) + offset_h_;
      const PT w_im = static_cast<PT>(w_in + j * dilation_w) + offset_w_;

      const PT value = im2col_bilinear<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im);

      output_slice[i * kernel_w + j] = static_cast<T>(value * mask_);
    }
  }
}

template <typename T, typename PT>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void modulated_deformable_col2im_body(
    const int index, const T *input, const T *offset, const T *mask, const T *column, const T *grad,
    const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,
    const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input,
    PT *grad_offset, PT *grad_mask)
{
  const int b = index / channel_in / width_out / height_in;
  const int h = index / channel_in / width_out % height_in;
  const int w = index / channel_in % width_out;
  const int c = index % channel_in;
  const int g = c % deformable_group;

  const int h_in = h * stride_h - pad_h;
  const int w_in = w * stride_w - pad_w;

  //  const T *input_slice = input + b * height_in * width_in * channel_in +
  //                         //                  h * width_in * channel_in +
  //                         //                             w * channel_in +
  //                         c;
  const T *input_slice = input + b * height_in * width_in * channel_in + c;

  //  const T *offset_slice = offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //  //                                                                                  i * kernel_w * 2 +
  //  //                                                                                             j * 2 +
  //  //                                                                                                 0;
  const T *offset_slice =
      offset + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w * 2;

  //  const T *mask_slice = mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
  //                        h * width_out * deformable_group * kernel_h * kernel_w +
  //                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //  //                                                                            i * kernel_w +
  //  //                                                                                       j;
  const T *mask_slice = mask + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w;

  //  const T *column_slice = column + b * height_out * width_out * channel_in * kernel_h * kernel_w +
  //                        h * width_out * channel_in * kernel_h * kernel_w + w * channel_in * kernel_h * kernel_w +
  //                        c * kernel_h * kernel_w;
  //  //                                                                                               i * kernel_w +
  //  //                                                                                                          j;
  const T *column_slice = column + (((b * height_out + h) * width_out + w) * channel_in + c) * kernel_h * kernel_w;

  //  const T *grad_slice = grad + b * height_out * width_out * channel_in * kernel_h * kernel_w +
  //                        h * width_out * channel_in * kernel_h * kernel_w + w * channel_in * kernel_h * kernel_w +
  //                        c * kernel_h * kernel_w;
  //  //                                                                                               i * kernel_w +
  //  //                                                                                                          j;
  const T *grad_slice = grad + (((b * height_out + h) * width_out + w) * channel_in + c) * kernel_h * kernel_w;

  //  PT *grad_input_slice = grad_input + b * height_out * width_out * channel_in +
  //                         //                        h * width_out * channel_in +
  //                         //                                    w * channel_in +
  //                         c;
  PT *grad_input_slice = grad_input + b * height_out * width_out * channel_in + c;

  //  PT *grad_offset_slice = grad_offset + b * height_out * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          h * width_out * deformable_group * kernel_h * kernel_w * 2 +
  //                          w * deformable_group * kernel_h * kernel_w * 2 + g * kernel_h * kernel_w * 2;
  //  //                                                                                  i * kernel_w * 2 +
  //  //                                                                                             j * 2 +
  //  //                                                                                                 0;
  PT *grad_offset_slice =
      grad_offset + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w * 2;

  //  PT *grad_mask_slice = grad_mask + b * height_out * width_out * deformable_group * kernel_h * kernel_w +
  //                        h * width_out * deformable_group * kernel_h * kernel_w +
  //                        w * deformable_group * kernel_h * kernel_w + g * kernel_h * kernel_w;
  //  //                                                                            i * kernel_w +
  //  //                                                                                       j;
  PT *grad_mask_slice =
      grad_mask + (((b * height_out + h) * width_out + w) * deformable_group + g) * kernel_h * kernel_w;

  for (int i = 0; i < kernel_h; ++i)
  {
    const int kernel_h_shift = i * kernel_w;

    for (int j = 0; j < kernel_w; ++j)
    {
      const int kernel_hw_shift = kernel_h_shift + j;
      const int kernel_hw_shift_x2 = kernel_hw_shift * 2;

      const PT offset_h_ = offset_slice[kernel_hw_shift_x2];
      const PT offset_w_ = offset_slice[kernel_hw_shift_x2 + 1];
      const PT mask_ = mask_slice[kernel_hw_shift];
      const PT column_ = column_slice[kernel_hw_shift];
      const PT grad_ = grad_slice[kernel_hw_shift];

      const PT h_im = static_cast<PT>(h_in + i * dilation_h) + offset_h_;
      const PT w_im = static_cast<PT>(w_in + j * dilation_w) + offset_w_;

      const PT top_grad = grad_ * mask_;

      // Mask gradient
      const PT value = (0. == mask_) ? im2col_bilinear<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im)
                                     : column_ / mask_;
      atomic_add<PT>(grad_mask_slice + kernel_hw_shift, grad_ * value);

      // Offset gradient
      const PT o_h_weight = coordinate_weight<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im, true);
      const PT o_w_weight = coordinate_weight<T, PT>(input_slice, height_in, width_in, channel_in, h_im, w_im, false);
      atomic_add<PT>(grad_offset_slice + kernel_hw_shift_x2, top_grad * o_h_weight);
      atomic_add<PT>(grad_offset_slice + kernel_hw_shift_x2 + 1, top_grad * o_w_weight);

      // Input gradient
      for (int z = 0; z < 4; z++)
      {
        int i_height;
        int i_width;
        PT i_weight;
        const bool update = input_weight<PT>(h_im, w_im, height_in, width_in, z, &i_height, &i_width, &i_weight);
        if (update)
        {
          atomic_add<PT>(grad_input_slice + (i_height * width_out + i_width) * channel_in, top_grad * i_weight);
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
      OpKernelContext *ctx, const T *input, const T *offset, const T *mask, const T *column, const T *grad,
      const int batch_size, const int height_in, const int width_in, const int channel_in, const int height_out,
      const int width_out, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
      const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group, PT *grad_input,
      PT *grad_offset, PT *grad_mask) const;
};

}  // end namespace miss
}  // namespace tensorflow
