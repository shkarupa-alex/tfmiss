#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

using namespace tensorflow;

class ContBowOp : public OpKernel
{
public:
  explicit ContBowOp(OpKernelConstruction *ctx) : OpKernel(ctx)
  {

    // Load random seeds
    OP_REQUIRES_OK(ctx, _random_generator.Init(ctx));
  }

  void Compute(OpKernelContext *ctx) override
  {

    // Load source values & splits
    const Tensor *source_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_values", &source_values_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(source_values_tensor->shape()),
                errors::InvalidArgument("Values must be a vector, got shape: ", source_values_tensor->shape().DebugString()));
    const auto source_values = source_values_tensor->flat<string>();

    const Tensor *source_splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source_splits", &source_splits_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(source_splits_tensor->shape()),
                errors::InvalidArgument("Splits must be a vector, got shape: ", source_splits_tensor->shape().DebugString()));
    const auto source_splits = source_splits_tensor->flat<int64>();

    // Load window
    const Tensor *window_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("window", &window_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(window_tensor->shape()),
                errors::InvalidArgument("Window should be a scalar, got shape: ", window_tensor->shape().DebugString()));
    int64 window_size = window_tensor->flat<int64>()(0);
    OP_REQUIRES(ctx, window_size > 0,
                errors::InvalidArgument("Window must be greater than zero, got ", window_size));

    // Prepare random generator
    random::PhiloxRandom philox_rng = _random_generator.ReserveSamples128(source_values.size());
    random::SimplePhilox simple_philox(&philox_rng);

    // Prepare vectors to store target and context_*
    std::vector<string> cbow_target;
    cbow_target.reserve(source_values.size());

    std::vector<string> cbow_context_values;
    uint64 reserve_context_values = 0;
    for (int64 row = 0; row < source_splits.size() - 1; row++)
    {
      int64 row_length = source_splits(row + 1) - source_splits(row);
      int64 row_reserve = (row_length - window_size) * (window_size + 1);
      reserve_context_values += row_reserve > 0 ? row_reserve : 0;
    }
    cbow_context_values.reserve(reserve_context_values);

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
        if (!source_values(target).size())
          continue; // target word empty

        int64 window_reduce = simple_philox.Uniform64(window_size);
        int64 window_start = std::max(row_start, target - window_size + window_reduce);
        int64 window_stop = std::max((int64)0, std::min(row_stop - 1, target + window_size - window_reduce));
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
          empty_row = false;
        }

        if (!empty_row)
        {
          cbow_target.push_back(source_values(target));
          cbow_context_splits.push_back(cbow_context_values.size());
        }
      }
    }

    // Create target & context_* outputs
    Tensor *target_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("target", TensorShape({(int64)cbow_target.size()}), &target_tensor));
    auto target = target_tensor->vec<string>();

    Tensor *context_source_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context_values", TensorShape({(int64)cbow_context_values.size()}), &context_source_values_tensor));
    auto context_values = context_source_values_tensor->vec<string>();

    Tensor *context_source_splits_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("context_splits", TensorShape({(int64)cbow_context_splits.size()}), &context_source_splits_tensor));
    auto context_splits = context_source_splits_tensor->vec<int64>();

    // Fill outputs
    for (uint64 i = 0; i < cbow_target.size(); i++)
    {
      target(i) = cbow_target[i];
    }

    for (uint64 i = 0; i < cbow_context_values.size(); i++)
    {
      context_values(i) = cbow_context_values[i];
    }

    for (uint64 i = 0; i < cbow_context_splits.size(); i++)
    {
      context_splits(i) = cbow_context_splits[i];
    }
  }

private:
  GuardedPhiloxRandom _random_generator;
};

REGISTER_KERNEL_BUILDER(Name("ContBow").Device(DEVICE_CPU), ContBowOp);
