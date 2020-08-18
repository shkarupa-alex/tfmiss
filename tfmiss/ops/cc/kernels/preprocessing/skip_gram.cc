#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace miss {

template <typename T>
class SkipGramOp : public OpKernel
{
public:
  explicit SkipGramOp(OpKernelConstruction *ctx) : OpKernel(ctx)
  {
    // Load window
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window", &window_size));
    OP_REQUIRES(ctx, window_size > 0,
                errors::InvalidArgument("Window must be greater than zero, got ", window_size));

    // Load random seeds
    OP_REQUIRES_OK(ctx, _random_generator.Init(ctx));
  }

  void Compute(OpKernelContext *ctx) override
  {

    // Load source values & splits
    const Tensor *values_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_values", &values_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values_tensor->shape()),
                errors::InvalidArgument("Values must be a vector, got shape: ", values_tensor->shape().DebugString()));
    const auto source_values = values_tensor->flat<tstring>();

    const Tensor *splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_splits", &splits_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(splits_tensor->shape()),
                errors::InvalidArgument("Splits must be a vector, got shape: ", splits_tensor->shape().DebugString()));
    const auto source_splits = splits_tensor->flat<T>();

    // Prepare random generator
    random::PhiloxRandom philox_rng = _random_generator.ReserveSamples128(source_values.size());
    random::SimplePhilox simple_philox(&philox_rng);

    // Prepare skip-grams storage
    std::vector<string> skip_grams;
    uint64 reserve_pairs = 0;
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_length = source_splits(row + 1) - source_splits(row);
      int64 row_reserve = (row_length - window_size) * (window_size + 1);
      reserve_pairs += row_reserve > 0 ? row_reserve : 0;
    }
    skip_grams.reserve(reserve_pairs * 2);

    // Estimate skip-gram pairs
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_start = source_splits(row);
      int64 row_stop = source_splits(row + 1);

      if (row_start == row_stop)
        continue; // Empty row

      for (int64 target = row_start; target < row_stop; target++)
      {
        if (!source_values(target).size())
          continue; // target word empty

        int64 window_reduce = simple_philox.Uniform64(window_size);
        int64 window_start = std::max(row_start, target - window_size + window_reduce);
        int64 window_stop = std::max((int64)0, std::min(row_stop - 1, target + window_size - window_reduce));

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

          skip_grams.push_back(source_values(target));
          skip_grams.push_back(source_values(context));
        }
      }
    }

    // Create target & context outputs
    Tensor *target_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("target", TensorShape({(int64)skip_grams.size() / 2}), &target_tensor));
    auto target = target_tensor->vec<tstring>();

    Tensor *context_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context", TensorShape({(int64)skip_grams.size() / 2}), &context_tensor));
    auto context = context_tensor->vec<tstring>();

    // Fill outputs
    for (uint64 i = 0; i < skip_grams.size() / 2; i++)
    {
      target(i) = skip_grams[i * 2];
      context(i) = skip_grams[i * 2 + 1];
    }
  }

private:
  GuardedPhiloxRandom _random_generator;
  int64 window_size;
};

REGISTER_KERNEL_BUILDER(
  Name("Miss>SkipGram")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  SkipGramOp<int32>);

REGISTER_KERNEL_BUILDER(
  Name("Miss>SkipGram")
  .Device(DEVICE_CPU)
  .TypeConstraint<int64>("T"),
  SkipGramOp<int64>);

}  // end namespace miss
}  // namespace tensorflow
