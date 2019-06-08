#include "tensorflow/core/framework/op.h"
#include "expand_shape.h"

REGISTER_OP("CharNgrams")
    .Input("source: string")
    .Attr("minn: int")
    .Attr("maxn: int")
    .Attr("itself: {'ASIS', 'NEVER', 'ALWAYS', 'ALONE'}")
    .Attr("T: {int32, int64} = DT_INT64")
    .Output("result_values: string")
    .Output("result_splits: T")
    .SetShapeFn(ExpandShape);
