#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("ReplaceString")
    .Input("source: string")
    .Attr("needle: list(string) >= 1")
    .Attr("haystack: list(string) >= 1")
    .Output("result: string")
    .SetShapeFn(shape_inference::UnchangedShape);
