#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>ContBow")
    .Input("source_values: string")
    .Input("source_splits: T")
    .Attr("window: int")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: {int32, int64} = DT_INT64")
    .Output("target: string")
    .Output("context_values: string")
    .Output("context_splits: T")
    .Output("context_positions: int32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused)); // source_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused)); // source_splits

      c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(3, c->Vector(shape_inference::InferenceContext::kUnknownDim));

      return OkStatus();
    });

} // end namespace miss
} // namespace tensorflow
