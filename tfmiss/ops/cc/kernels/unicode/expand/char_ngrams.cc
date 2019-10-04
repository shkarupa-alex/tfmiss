#include "unicode_expand.h"

namespace tensorflow {
namespace miss {

class CharNgramsOp : public UnicodeExpandOp
{
public:
  explicit CharNgramsOp(OpKernelConstruction *ctx) : UnicodeExpandOp(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("minn", &_minn));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxn", &_maxn));
    OP_REQUIRES(ctx, _minn > 0, errors::InvalidArgument("minn should be above 0"));
    OP_REQUIRES(ctx, _maxn >= _minn, errors::InvalidArgument("maxn should be above or equal minn"));

    string itself;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("itself", &itself));
    std::transform(itself.begin(), itself.end(), itself.begin(), ::toupper);
    if ("NEVER" == itself)
    {
      _itself = NgramItself::NEVER;
    }
    else if ("ALWAYS" == itself)
    {
      _itself = NgramItself::ALWAYS;
    }
    else if ("ALONE" == itself)
    {
      _itself = NgramItself::ALONE;
    }
    else
    {
      _itself = NgramItself::ASIS;
    }
  }

private:
  enum class NgramItself
  {
    ASIS,
    NEVER,
    ALWAYS,
    ALONE
  };

  int _minn;
  int _maxn;
  NgramItself _itself;

protected:
  uint64 expand_rate() override
  {
    const int mean_len = 5; // Mean word length
    uint64 total_rate = 0;  // As is

    for (int i = _minn; i <= _maxn; i++)
    {
      int current_rate = mean_len - i + 1;

      current_rate = current_rate < 0 ? 0 : current_rate;
      total_rate += current_rate;
    }

    return total_rate;
  }

  bool expand_unicode(const UnicodeString &unicode_string, std::vector<UnicodeString> &expanded_strings) override
  {
    int64 length = unicode_string.length(); // Convert length to signed int. Required to allow negative values.

    // Split ngrams
    for (int64 n = _minn; n <= _maxn; n++)
    {
      if ((NgramItself::NEVER == _itself || NgramItself::ALONE == _itself) && length == n)
        continue;

      for (int64 pos = 0; pos <= length - n; pos++)
      {
        UnicodeString ngram = UnicodeString(unicode_string, pos, n);

        expanded_strings.push_back(ngram);
      }
    }

    if (NgramItself::ALWAYS == _itself && (length < _minn || length > _maxn))
    {
      expanded_strings.push_back(unicode_string);
    }

    if (NgramItself::ALONE == _itself && 0 == expanded_strings.size())
    {
      expanded_strings.push_back(unicode_string);
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>CharNgrams").Device(DEVICE_CPU), CharNgramsOp);

}  // end namespace miss
}  // namespace tensorflow
