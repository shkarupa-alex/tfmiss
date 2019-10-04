#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

void UnicodeTransformOp::Compute(OpKernelContext *ctx)
{
  // Prepare source
  const Tensor *source_tensor;
  OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
  const auto source_values = source_tensor->flat<string>();

  // Allocate result
  Tensor *result_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("result", TensorShape({source_tensor->shape()}), &result_tensor));
  auto result_values = result_tensor->flat<string>();

  // Transform and write to output
  for (int64 i = 0; i < source_values.size(); i++)
  {
    string binary_string = source_values(i);
    bool transform_ok = transform_raw(binary_string);

    OP_REQUIRES(ctx, transform_ok,
                errors::InvalidArgument("Unicode transformation failed"));

    result_values(i) = binary_string;
  }
}

bool UnicodeTransformOp::transform_raw(string &binary_string)
{
  UnicodeString unicode_string = UnicodeString::fromUTF8(binary_string);
  bool transform_ok = transform_unicode(unicode_string);

  if (transform_ok)
  {
    binary_string.clear();
    unicode_string.toUTF8String(binary_string);
  }

  return transform_ok;
}

}  // end namespace miss
}  // namespace tensorflow
