#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow
{
namespace miss
{
REGISTER_OP("Miss>ModulatedDeformableColumn")
    .Input("input: FT")
    .Input("offset: FT")
    .Input("mask: FT")
    .Attr("kernel_h: int")
    .Attr("kernel_w: int")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("pad_hb: int")
    .Attr("pad_ha: int")
    .Attr("pad_wb: int")
    .Attr("pad_wa: int")
    .Attr("dilation_h: int")
    .Attr("dilation_w: int")
    .Attr("deformable_group: int")
    .Attr("FT: {bfloat16, half, float32, float64} = DT_FLOAT")
    .Output("output: FT")
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
          shape_inference::ShapeHandle input;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(0), 4, &input),
              "input rank must equals 4, but provided shape is " + c->DebugString(input));

          shape_inference::ShapeHandle offset;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(1), 4, &offset),
              "offset rank must equals 4, but provided shape is " + c->DebugString(offset));

          shape_inference::ShapeHandle mask;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(2), 4, &mask),
              "mask rank must equals 4, but provided shape is " + c->DebugString(mask));

          shape_inference::DimensionHandle batch_dim = c->Dim(input, 0);
          const int height_in = c->Value(c->Dim(input, 1));
          const int width_in = c->Value(c->Dim(input, 2));
          const int channel_in = c->Value(c->Dim(input, 3));

          int kernel_h, kernel_w;
          TF_RETURN_IF_ERROR(c->GetAttr("kernel_h", &kernel_h));
          TF_RETURN_IF_ERROR(c->GetAttr("kernel_w", &kernel_w));
          if (kernel_h <= 0 || kernel_w <= 0)
          {
            return errors::InvalidArgument("Kernel sizes should be larger than 0.");
          }

          int stride_h, stride_w;
          TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
          TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));
          if (stride_h <= 0 || stride_w <= 0)
          {
            return errors::InvalidArgument("Strides should be larger than 0.");
          }

          int pad_hb, pad_ha, pad_wb, pad_wa;
          TF_RETURN_IF_ERROR(c->GetAttr("pad_hb", &pad_hb));
          TF_RETURN_IF_ERROR(c->GetAttr("pad_ha", &pad_ha));
          TF_RETURN_IF_ERROR(c->GetAttr("pad_wb", &pad_wb));
          TF_RETURN_IF_ERROR(c->GetAttr("pad_wa", &pad_wa));
          if (pad_hb < 0 || pad_ha < 0 || pad_wb < 0 || pad_wa < 0)
          {
            return errors::InvalidArgument("Paddings should be larger or equal to 0.");
          }

          int dilation_h, dilation_w;
          TF_RETURN_IF_ERROR(c->GetAttr("dilation_h", &dilation_h));
          TF_RETURN_IF_ERROR(c->GetAttr("dilation_w", &dilation_w));
          if (dilation_h <= 0 || dilation_w <= 0)
          {
            return errors::InvalidArgument("Dilated rates should be larger than 0.");
          }

          const int height_out = floor((height_in + pad_hb + pad_ha - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
          const int width_out = floor((width_in + pad_wb + pad_wa - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;
          const int channel_out = channel_in * kernel_h * kernel_w;

          shape_inference::ShapeHandle output_shape = c->MakeShape({batch_dim, height_out * width_out, channel_out});
          c->set_output(0, output_shape);

          return Status::OK();
        });

REGISTER_OP("Miss>ModulatedDeformableColumnBackward")
    .Input("input: FT")
    .Input("offset: FT")
    .Input("mask: FT")
    .Input("column: FT")
    .Input("grad: FT")
    .Attr("kernel_h: int")
    .Attr("kernel_w: int")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("pad_hb: int")
    .Attr("pad_ha: int")
    .Attr("pad_wb: int")
    .Attr("pad_wa: int")
    .Attr("dilation_h: int")
    .Attr("dilation_w: int")
    .Attr("deformable_group: int")
    .Attr("FT: {bfloat16, half, float32, float64} = DT_FLOAT")
    .Output("grad_input: FT")
    .Output("grad_offset: FT")
    .Output("grad_mask: FT")
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
          shape_inference::ShapeHandle input;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(0), 4, &input),
              "input rank must equals 4, but provided shape is " + c->DebugString(input));

          shape_inference::ShapeHandle offset;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(1), 4, &offset),
              "offset rank must equals 4, but provided shape is " + c->DebugString(offset));

          shape_inference::ShapeHandle mask;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(2), 4, &mask),
              "mask rank must equals 4, but provided shape is " + c->DebugString(mask));

          shape_inference::ShapeHandle grad;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(3), 3, &grad),
              "grad rank must equals 3, but provided shape is " + c->DebugString(grad));

          c->set_output(0, input);
          c->set_output(1, offset);
          c->set_output(2, mask);

          return Status::OK();
        });

}  // end namespace miss
}  // namespace tensorflow
