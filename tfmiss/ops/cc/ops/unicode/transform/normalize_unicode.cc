#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("NormalizeUnicode")
    .Input("source: string")
    .Attr("form: {'NFC', 'NFD', 'NFKC', 'NFKD'}")
    .Output("result: string")
    .SetShapeFn(shape_inference::UnchangedShape);
