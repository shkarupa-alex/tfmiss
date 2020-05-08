#include "tensorflow/core/framework/op.h"
#include "expand_shape.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>SplitWords")
    .Input("source: string")
    .Attr("extended: bool = false")
    .Attr("skip: list(string)")
    .Attr("T: {int32, int64} = DT_INT64")
    .Output("result_values: string")
    .Output("result_splits: T")
    .SetShapeFn(ExpandShape);

} // end namespace miss
} // namespace tensorflow
