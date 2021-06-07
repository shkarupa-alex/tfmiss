#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace icu;

namespace tensorflow {
namespace miss {

template <typename T>
class SpacesAfterOp : public OpKernel
{
public:
  explicit SpacesAfterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override
  {

    // Load source values & splits
    const Tensor *source_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_values", &source_values_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(source_values_tensor->shape()),
                errors::InvalidArgument("Values must be a vector, got shape: ", source_values_tensor->shape().DebugString()));
    const auto source_values = source_values_tensor->flat<tstring>();

    const Tensor *source_splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_splits", &source_splits_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(source_splits_tensor->shape()),
                errors::InvalidArgument("Splits must be a vector, got shape: ", source_splits_tensor->shape().DebugString()));
    const auto source_splits = source_splits_tensor->flat<T>();

    // Prepare vectors to store output
    std::vector<string> result_values;
    std::vector<T> result_splits;
    result_splits.reserve(source_splits.size() / 3);

    // Separate tokens and spaces
    result_splits.push_back(0);
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_start = source_splits(row);
      int64 row_stop = source_splits(row + 1);
      string curr_space;
      bool first_token = true;

      for (int64 i = row_start; i < row_stop; i++)
      {
        string curr_binary = source_values(i);
        UnicodeString curr_unicode = UnicodeString::fromUTF8(curr_binary);
        bool space = is_space(curr_unicode);

        if (space && first_token) {
            continue;
        }

        if (!space && first_token) {
            first_token = false;
            result_values.push_back(curr_binary);
            continue;
        }

        if (space) {
            curr_space += curr_binary;
            continue;
        }

        result_values.push_back(curr_space);
        curr_space.clear();

        result_values.push_back(curr_binary);
      }

      if (!first_token) {
        result_values.push_back(curr_space);
      }

      result_splits.push_back(result_values.size() / 2);
    }

    // Create output tensors
    Tensor *token_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("token_values", TensorShape({(int64)result_values.size() / 2}), &token_values_tensor));
    auto token_values = token_values_tensor->vec<tstring>();

    Tensor *space_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("space_values", TensorShape({(int64)result_values.size() / 2}), &space_values_tensor));
    auto space_values = space_values_tensor->vec<tstring>();

    Tensor *common_splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("common_splits", TensorShape({(int64)result_splits.size()}), &common_splits_tensor));
    auto common_splits = common_splits_tensor->vec<T>();

    // Fill outputs
    for (uint64 i = 0; i < result_values.size() / 2; i++)
    {
      token_values(i) = result_values[i * 2];
      space_values(i) = result_values[i * 2 + 1];
    }

    for (uint64 i = 0; i < result_splits.size(); i++)
    {
      common_splits(i) = result_splits[i];
    }
  }

private:
  std::set<UChar> spaces = {
    9, // \t
    10, // \n
    11, // \x0b
    12, // \x0c
    13, // \x1c
    28, // \x1c
    29, // \x1d
    30, // \x1e
    31, // \x1f
    32, // \s
    133, // \x85
    160, // \xa0
    5760, // \u1680
    8192, // \u2000
    8193, // \u2001
    8194, // \u2002
    8195, // \u2003
    8196, // \u2004
    8197, // \u2005
    8198, // \u2006
    8199, // \u2007
    8200, // \u2008
    8201, // \u2009
    8202, // \u200a
    8232, // \u2028
    8233, // \u2029
    8203, // \u200b
    8239, // \u202f
    8287, // \u205f
    8288, // \u2060
    8289, // \u2061
    10240, // \u2800
    12288, // \u3000
    65279, // \ufeff
  };

protected:
  bool is_space(const UnicodeString &haystack)
  {
    for (int32_t pos = 0; pos < haystack.length(); pos++)
    {
      if (spaces.count(haystack.char32At(pos))) {
        return true;
      }
    }

    return false;
  }
};

REGISTER_KERNEL_BUILDER(
  Name("Miss>SpacesAfter")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  SpacesAfterOp<int32>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>SpacesAfter")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("T"),
  SpacesAfterOp<int64>);

}  // end namespace miss
}  // namespace tensorflow
