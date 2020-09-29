#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow
{
namespace miss
{

// Register the FoPool operator.
REGISTER_OP("TimeMajorFoPool")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("initial_state: FT")
    .Output("output: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool operation.)doc")
    .SetShapeFn(time_major_fo_pool_shape_function);

REGISTER_OP("BatchMajorFoPool")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("initial_state: FT")
    .Output("output: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool operation.)doc")
    .SetShapeFn(batch_major_fo_pool_shape_function);

} // end namespace miss
} // namespace tensorflow
