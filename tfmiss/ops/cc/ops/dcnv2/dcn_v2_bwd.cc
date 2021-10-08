#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>ModulatedDeformableColumnBackward")
  .Input("input: FT")
  .Input("offset: FT")
  .Input("mask: FT")
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
  .Attr("FT: {half, float32, float64, bfloat16} = DT_FLOAT")
  .Output("grad_input: FT")
  .Output("grad_offset: FT")
  .Output("grad_mask: FT")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle input;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 4, &input),
                                      "input rank must equals 4, but provided shape is " + c->DebugString(input));

      shape_inference::ShapeHandle offset;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 4, &offset),
                                      "offset rank must equals 4, but provided shape is " + c->DebugString(offset));

      shape_inference::ShapeHandle mask;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 4, &mask),
                                      "mask rank must equals 4, but provided shape is " + c->DebugString(mask));

      shape_inference::ShapeHandle grad;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(3), 3, &grad),
                                      "grad rank must equals 3, but provided shape is " + c->DebugString(grad));

      c->set_output(0, input);
      c->set_output(1, offset);
      c->set_output(2, mask);

      return Status::OK();
  });

} // end namespace miss
} // namespace tensorflow
