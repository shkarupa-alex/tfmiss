#include "unicode_expand.h"

namespace tensorflow {
namespace miss {

UnicodeExpandOp::UnicodeExpandOp(OpKernelConstruction *ctx) : OpKernel(ctx) {

  // Load skip list
  std::vector<string> skip;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("skip", &skip));
  for (string key : skip)
  {
    _skip.insert(key);
  }
}

void UnicodeExpandOp::Compute(OpKernelContext *ctx)
{
  // Prepare source
  const Tensor *source_tensor;
  OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
  const auto source_values = source_tensor->flat<tstring>();

  // Prepare intermediate result storage
  uint64 expand_max = expand_rate();

  std::vector<string> intermediate_values;
  intermediate_values.reserve(source_values.size() * expand_max);

  std::vector<uint64> intermediate_splits;
  intermediate_splits.reserve(source_values.size());

  std::vector<UnicodeString> expand_storage;
  expand_storage.reserve(expand_max);

  // Expand source values
  for (int64 i = 0; i < source_values.size(); i++)
  {
    intermediate_splits.push_back(intermediate_values.size());

    string binary_string = source_values(i);

    if (_skip.count(binary_string))
    {
      intermediate_values.push_back(binary_string);
    }
    else
    {
      UnicodeString unicode_string = UnicodeString::fromUTF8(binary_string);
      expand_storage.clear();

      bool expand_ok = expand_unicode(unicode_string, expand_storage);
      OP_REQUIRES(ctx, expand_ok,
                  errors::InvalidArgument("Unicode expansion failed"));

      for (uint64 j = 0; j < expand_storage.size(); j++)
      {
        binary_string.clear();
        expand_storage[j].toUTF8String(binary_string);
        intermediate_values.push_back(binary_string);
      }
    }
  }
  intermediate_splits.push_back(intermediate_values.size());

  // Allocate result
  Tensor *result_values_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("result_values", TensorShape({(int64)intermediate_values.size()}), &result_values_tensor));
  auto result_values = result_values_tensor->flat<tstring>();

  Tensor *result_splits_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("result_splits", TensorShape({(int64)intermediate_splits.size()}), &result_splits_tensor));
  auto result_splits = result_splits_tensor->flat<int64>();

  // Fill result
  for (uint64 i = 0; i < intermediate_values.size(); i++)
  {
    result_values(i) = intermediate_values[i];
  }
  for (uint64 i = 0; i < intermediate_splits.size(); i++)
  {
    result_splits(i) = intermediate_splits[i];
  }
}

uint64 UnicodeExpandOp::expand_rate() { return (uint64)1; }

}  // end namespace miss
}  // namespace tensorflow
