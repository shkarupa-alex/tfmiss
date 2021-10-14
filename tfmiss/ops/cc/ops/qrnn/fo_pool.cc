#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
namespace miss
{
REGISTER_OP("Miss>FoPool")
    .Input("input: FT")
    .Input("forget: FT")
    .Input("init: FT")
    .Attr("FT: {half, float32, float64} = DT_FLOAT")
    .Output("output: FT")
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
          shape_inference::ShapeHandle input;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(0), 3, &input), "input rank must equals 3, but provided shape is " + c->DebugString(input));

          shape_inference::ShapeHandle forget;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(1), 3, &forget),
              "forget rank must equals 3, but provided shape is " + c->DebugString(forget));

          shape_inference::ShapeHandle init;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(2), 2, &init),
              "initial state rank must equals 2, but provided shape is " + c->DebugString(init));

          std::vector<shape_inference::DimensionHandle> dims(3);
          TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, 0), c->Dim(init, 0), &dims[0]));
          TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, 2), c->Dim(init, 1), &dims[2]));

          for (int i = 0; i < 3; i++)
          {
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, i), c->Dim(forget, i), &dims[i]));
          }
          TF_RETURN_IF_ERROR(c->Add(c->Dim(input, 1), static_cast<tensorflow::int64>(1), &dims[1]));

          c->set_output(0, c->MakeShape(dims));

          return Status::OK();
        });

REGISTER_OP("Miss>FoPoolBackward")
    .Input("input: FT")
    .Input("forget: FT")
    .Input("hidden: FT")
    .Input("grad: FT")
    .Attr("FT: {half, float32, float64} = DT_FLOAT")
    .Output("grad_input: FT")
    .Output("grad_forget: FT")
    .Output("grad_init: FT")
    .SetShapeFn(
        [](shape_inference::InferenceContext *c)
        {
          shape_inference::ShapeHandle input;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(0), 3, &input), "input rank must equals 3, but provided shape is " + c->DebugString(input));

          shape_inference::ShapeHandle forget;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(1), 3, &forget),
              "forget rank must equals 3, but provided shape is " + c->DebugString(forget));

          shape_inference::ShapeHandle hidden;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(2), 3, &hidden), "hidden rank must equals 3, but provided shape is " + c->DebugString(hidden));

          shape_inference::ShapeHandle grad;
          TF_RETURN_WITH_CONTEXT_IF_ERROR(
              c->WithRank(c->input(3), 3, &grad), "grad rank must equals 3, but provided shape is " + c->DebugString(grad));

          std::vector<shape_inference::DimensionHandle> init_dims({c->Dim(input, 0), c->Dim(input, 2)});

          c->set_output(0, input);
          c->set_output(1, forget);
          c->set_output(2, c->MakeShape(init_dims));

          return Status::OK();
        });

}  // end namespace miss
}  // namespace tensorflow
