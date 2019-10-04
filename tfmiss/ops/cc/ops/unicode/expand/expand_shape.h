#pragma once

#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
namespace miss
{

Status ExpandShape(shape_inference::InferenceContext *c);

} // end namespace miss
} // namespace tensorflow
