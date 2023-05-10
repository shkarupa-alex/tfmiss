#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
namespace miss
{

REGISTER_OP("Miss>EuclideanDistance")
    .Input("input: DT")
    .Attr("DT: {bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, half, bfloat16, float, double}")
    .Attr("dtype: {half, bfloat16, float, double} = DT_FLOAT")
    .Output("output: dtype")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // end namespace miss
}  // namespace tensorflow
