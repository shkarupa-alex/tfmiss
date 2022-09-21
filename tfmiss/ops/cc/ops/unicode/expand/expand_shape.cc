#include "expand_shape.h"

namespace tensorflow
{
namespace miss
{

Status ExpandShape(shape_inference::InferenceContext *c)
{
    c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim)); // result_values
    c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim)); // result_splits

    return OkStatus();
}

} // end namespace miss
} // namespace tensorflow
