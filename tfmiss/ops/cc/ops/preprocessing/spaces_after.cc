#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>SpacesAfter")
    .Input("source_values: string")
    .Input("source_splits: T")
    .Attr("T: {int32, int64} = DT_INT64")
    .Output("token_values: string")
    .Output("space_values: string")
    .Output("common_splits: T")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused)); // source_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused)); // source_splits

      c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(shape_inference::InferenceContext::kUnknownDim));

      return OkStatus();
    });

} // end namespace miss
} // namespace tensorflow
