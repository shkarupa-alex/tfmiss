#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>SampleMask")
    .Input("source: string")
    .Attr("keys: list(string) >= 1")
    .Attr("freqs: list(int) >= 1")
    .Attr("threshold: float")
    .Attr("min_freq: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Output("mask: bool")
    .SetShapeFn(shape_inference::UnchangedShape);

} // end namespace miss
} // namespace tensorflow
