#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>TimeMajorFoPool")
  .Input("x: FT")
  .Input("forget: FT")
  .Input("initial_state: FT")
  .Attr("FT: {float, double} = DT_FLOAT")
  .Output("output: FT")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input;
    shape_inference::DimensionHandle d;

    shape_inference::ShapeHandle in_x = c->input(0);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
      "x must have shape [None, None, None] but is " + c->DebugString(in_x));

    shape_inference::ShapeHandle in_forget = c->input(1);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
      "forget must have shape [None, None, None] but is " + c->DebugString(in_forget));

    shape_inference::ShapeHandle in_hinit = c->input(2);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_hinit, 2, &input),
      "hinit must have shape [None, None] but is " + c->DebugString(in_hinit));

    std::vector<shape_inference::DimensionHandle> dims(3);
    for (int i = 1; i < 3; i++)
    {
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(in_x, i), c->Dim(in_hinit, i - 1), &dims[i]));
    }
    for (int i = 0; i < 3; i++)
    {
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(in_x, i), c->Dim(in_forget, i), &dims[i]));
    }
    TF_RETURN_IF_ERROR(c->Add(c->Dim(in_x, 0), static_cast<tensorflow::int64>(1), &dims[0]));

    c->set_output(0, c->MakeShape(dims));

    return Status::OK();
  });

} // end namespace miss
} // namespace tensorflow
