#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace miss {

template <typename T>
class CbowContextOp : public OpKernel
{
public:
  explicit CbowContextOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    // Load window
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window", &window_size));
    OP_REQUIRES(ctx, window_size > 0,
                errors::InvalidArgument("Window must be greater than zero, got ", window_size));

    // Load value for empty context
    OP_REQUIRES_OK(ctx, ctx->GetAttr("empty", &empty_context));
  }

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

    // Prepare vectors to store context_*
    std::vector<string> cbow_context_values;
    std::vector<int32> cbow_context_positions;
    uint64 reserve_context_values = 0;
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_length = source_splits(row + 1) - source_splits(row);
      int64 row_reserve = (row_length - window_size) * (window_size + 1);
      reserve_context_values += row_reserve > 0 ? row_reserve : 0;
    }
    cbow_context_values.reserve(reserve_context_values);
    cbow_context_positions.reserve(reserve_context_values);

    std::vector<uint64> cbow_context_splits;
    cbow_context_splits.reserve(source_values.size());

    // Estimate cont-bow pairs
    cbow_context_splits.push_back(0);
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_start = source_splits(row);
      int64 row_stop = source_splits(row + 1);

      if (row_start == row_stop)
        continue; // Empty row

      for (int64 target = row_start; target < row_stop; target++)
      {
        int64 window_start = std::max(row_start, target - window_size);
        int64 window_stop = std::max((int64)0, std::min(row_stop - 1, target + window_size));
        bool empty_row = true;

        for (int64 context = window_start; context <= window_stop; context++)
        {
          if (context == target)
            continue; // words not equals by index
          if (!source_values(context).size())
            continue; // context word empty

          // Not found such filtration in FastText or Word2Vec sources
          // https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc#L402
          // https://github.com/dav/word2vec/blob/master/src/word2vec.c#L447
          // if (source_values(context) == source_values(target)) // words not equals by value
          //  continue;

          cbow_context_values.push_back(source_values(context));
          cbow_context_positions.push_back(context - target);
          empty_row = false;
        }

        if (empty_row) {
          cbow_context_values.push_back(empty_context);
          cbow_context_values.push_back(empty_context);
          cbow_context_positions.push_back(-1);
          cbow_context_positions.push_back(1);
        }

        cbow_context_splits.push_back(cbow_context_values.size());
      }
    }

    // Create context_* outputs
    Tensor *context_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context_values", TensorShape({(int64)cbow_context_values.size()}), &context_values_tensor));
    auto context_values = context_values_tensor->vec<tstring>();

    Tensor *context_splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context_splits", TensorShape({(int64)cbow_context_splits.size()}), &context_splits_tensor));
    auto context_splits = context_splits_tensor->vec<T>();

    Tensor *context_positions_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context_positions", TensorShape({(int64)cbow_context_positions.size()}), &context_positions_tensor));
    auto context_positions = context_positions_tensor->vec<int32>();

    // Fill outputs
    for (uint64 i = 0; i < cbow_context_values.size(); i++)
    {
      context_values(i) = cbow_context_values[i];
    }

    for (uint64 i = 0; i < cbow_context_splits.size(); i++)
    {
      context_splits(i) = cbow_context_splits[i];
    }

    for (uint64 i = 0; i < cbow_context_positions.size(); i++)
    {
      context_positions(i) = cbow_context_positions[i];
    }
  }

private:
  int64 window_size;
  string empty_context;
};

REGISTER_KERNEL_BUILDER(
  Name("Miss>CbowContext")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  CbowContextOp<int32>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>CbowContext")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("T"),
  CbowContextOp<int64>);

}  // end namespace miss
}  // namespace tensorflow
