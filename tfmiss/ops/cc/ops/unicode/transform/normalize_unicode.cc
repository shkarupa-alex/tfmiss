#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>NormalizeUnicode")
    .Input("source: string")
    .Attr("form: {'NFC', 'NFD', 'NFKC', 'NFKD'}")
    .Attr("skip: list(string)")
    .Output("result: string")
    .SetShapeFn(shape_inference::UnchangedShape);

} // end namespace miss
} // namespace tensorflow
