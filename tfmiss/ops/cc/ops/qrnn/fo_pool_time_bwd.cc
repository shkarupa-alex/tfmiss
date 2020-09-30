#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>TimeMajorBwdFoPool")
  .Input("h: FT")
  .Input("x: FT")
  .Input("forget: FT")
  .Input("gh: FT")
  .Attr("FT: {float, double} = DT_FLOAT")
  .Output("gx: FT")
  .Output("gf: FT")
  .Output("ginitial_state: FT")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input;
    shape_inference::DimensionHandle d;

    shape_inference::ShapeHandle in_h = c->input(0);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_h, 3, &input),
      "h must have shape [None, None, None] but is " + c->DebugString(in_h));

    shape_inference::ShapeHandle in_x = c->input(1);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
      "x must have shape [None, None, None] but is " + c->DebugString(in_h));

    shape_inference::ShapeHandle in_forget = c->input(2);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
      "forget must have shape [None, None, None] but is " + c->DebugString(in_forget));

    shape_inference::ShapeHandle in_gh = c->input(3);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_gh, 3, &input),
      "gh must have shape [None, None, None] but is " + c->DebugString(in_gh));

    std::vector<shape_inference::DimensionHandle> dims({c->Dim(in_gh, 1), c->Dim(in_gh, 2)});

    c->set_output(0, in_x);
    c->set_output(1, in_forget);
    c->set_output(2, c->MakeShape(dims));

    return Status::OK();
  });

} // end namespace miss
} // namespace tensorflow
