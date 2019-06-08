#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("WrapWith")
    .Input("source: string")
    .Attr("left: string")
    .Attr("right: string")
    .Output("result: string")
    .SetShapeFn(shape_inference::UnchangedShape);
