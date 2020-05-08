#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace miss {

class SampleMaskOp : public OpKernel
{
public:
  explicit SampleMaskOp(OpKernelConstruction *ctx) : OpKernel(ctx)
  {

    // Load vocabulary
    std::vector<string> keys;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keys", &keys));
    for (string key : keys)
    {
      OP_REQUIRES(ctx, std::count(keys.begin(), keys.end(), key) == 1,
                  errors::InvalidArgument("Keys must be unique", key));
    }

    std::vector<int64> freqs;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("freqs", &freqs));
    for (int64 freq : freqs)
    {
      OP_REQUIRES(ctx, freq > 0,
                  errors::InvalidArgument("Frequencies must be greater then zero", freq));
    }
    OP_REQUIRES(ctx, std::is_sorted(freqs.rbegin(), freqs.rend()),
                errors::InvalidArgument("Frequencies must be sorted in descending order"));

    OP_REQUIRES(ctx, keys.size() > 0,
                errors::InvalidArgument("Keys must not be empty"));
    OP_REQUIRES(ctx, keys.size() == freqs.size(),
                errors::InvalidArgument("Sizes of keys and frequencies must be equal"));

    // Load sampling parameters
    int64 min_freq = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_freq", &min_freq));
    OP_REQUIRES(ctx, min_freq >= 0,
                errors::InvalidArgument("Minimum frequency must be greater or equal zero", min_freq));

    float threshold;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold));
    OP_REQUIRES(ctx, threshold > 0.0,
                errors::InvalidArgument("Threshold must be greater then zero"));

    // Load random seeds
    OP_REQUIRES_OK(ctx, _random_generator.Init(ctx));

    // Estimate sample rates
    // "Keep probabilities" below estimated with word2vec equation from source code (not from paper)
    // See https://github.com/maxoodf/word2vec#subsampling-down-sampling
    uint64 total_freq = std::accumulate(freqs.begin(), freqs.end(), 0);
    float total_thold = threshold * total_freq;

    // Count non-tracked (unique) samples and their total frequency
    uint64 uniq_freq = 0;
    uint64 uniq_count = 0;
    for (int64 freq : freqs)
    {
      if (freq < min_freq)
      {
        uniq_freq += freq;
        uniq_count++;
      }
    }

    // Estimate unique samples keep probability
    if (uniq_count > 0)
    {
      _sample_uniq = true;
      _uniq_prob = (std::sqrt(uniq_freq / total_thold) + 1) * total_thold / uniq_freq;
      _uniq_prob = (_uniq_prob > 1.0) ? 1.0 : _uniq_prob;
    }

    // Estimate other samples keep probability
    float keep_count = uniq_count * _uniq_prob;
    for (uint64 i = 0; i < freqs.size() - uniq_count; i++)
    {
      string key = keys[i];
      uint64 val = freqs[i];

      float prob = (std::sqrt(val / total_thold) + 1) * total_thold / val;

      if (!_sample_uniq && prob >= 1.0)
        break; // Store only sampling keys if there is no unique ones

      prob = (prob >= 1.0) ? 1.0 : prob;
      keep_count += val * prob;
      _keep_probs[key] = prob;
    }

    if (round(keep_count) == 0)
    {
      LOG(WARNING) << "SampleMask with minimum frequency " << min_freq
                   << " and threshold " << threshold << " will discard all samples. "
                   << "Consider removing this operation or choose smaller threshold.";
    }
    else if (round(keep_count) == total_freq)
    {
      LOG(WARNING) << "SampleMask with minimum frequency " << min_freq
                   << " and threshold " << threshold << " will not discard any samples. "
                   << "Consider removing this operation or choose bigger threshold";
    }
    else
    {
      LOG(INFO) << "SampleMask with minimum frequency " << min_freq
                << " and threshold " << threshold << " will discard ~"
                << round(10000 * (total_freq - keep_count) / total_freq) / 100 << "% of samples";
    }
  }

  void Compute(OpKernelContext *ctx) override
  {

    // Prepare input
    const Tensor *source_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
    const auto source_values = source_tensor->flat<tstring>();
    const uint64 num_elements = source_tensor->shape().num_elements();

    // Allocate output
    Tensor *mask_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("mask", TensorShape({source_tensor->shape()}), &mask_tensor));
    auto mask_values = mask_tensor->flat<bool>();

    // Fill defaults and estimate required randoms count
    std::vector<float> source_probs;
    source_probs.resize(num_elements);

    uint64 randoms_count = 0;
    for (uint64 i = 0; i < num_elements; i++)
    {
      string key = source_values(i);
      mask_values(i) = true;

      try {
        source_probs[i] = _keep_probs.at(key);
      }
      catch (const std::out_of_range&) {
        source_probs[i] = _sample_uniq ? _uniq_prob : 1.0;
      }

      if (source_probs[i] >= 1.0)
        continue;

      randoms_count += 1;
    }

    // Prepare random generator
    random::PhiloxRandom philox_rng = _random_generator.ReserveSamples128(randoms_count);
    random::SimplePhilox simple_philox(&philox_rng);

    // Evaluate "keep" flag
    for (uint64 i = 0; i < num_elements; i++)
    {
      if (source_probs[i] >= 1.0)
        continue;

      float random_prob = simple_philox.RandFloat();
      mask_values(i) = random_prob < source_probs[i];
    }
  }

private:
  bool _sample_uniq = false;
  float _uniq_prob = 1.0;
  std::map<string, float> _keep_probs;
  GuardedPhiloxRandom _random_generator;
};

REGISTER_KERNEL_BUILDER(Name("Miss>SampleMask").Device(DEVICE_CPU), SampleMaskOp);

}  // end namespace miss
}  // namespace tensorflow
