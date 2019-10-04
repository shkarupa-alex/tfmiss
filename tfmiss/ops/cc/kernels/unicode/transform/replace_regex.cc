#include "re2/re2.h"
#include "tensorflow/core/util/ptr_util.h"
#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class ReplaceRegexOp : public UnicodeTransformOp
{
public:
  explicit ReplaceRegexOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Prepare attrs
    std::vector<string> pattern;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern));
    _pattern.resize(pattern.size());
    for (uint64 i = 0; i < pattern.size(); i++)
    {
      OP_REQUIRES(ctx, pattern[i].size() > 0,
                  errors::InvalidArgument("Items of \"pattern\" could not be empty"));

      _pattern[i] = MakeUnique<RE2>(pattern[i]);
      OP_REQUIRES(ctx, _pattern[i]->ok(),
                  errors::InvalidArgument("Invalid pattern: ", pattern[i], ", error: ", _pattern[i]->error()));
    }

    OP_REQUIRES_OK(ctx, ctx->GetAttr("rewrite", &_rewrite));
    OP_REQUIRES(ctx, _pattern.size() == _rewrite.size(),
                errors::InvalidArgument("Sizes are different for \"pattern\" and \"rewrite\""));
  }

private:
  std::vector<std::unique_ptr<RE2>> _pattern;
  std::vector<string> _rewrite;

protected:
  bool transform_raw(string &binary_string) override
  {
    for (uint64 i = 0; i < _pattern.size(); i++)
    {
      RE2::GlobalReplace(&binary_string, *_pattern[i], _rewrite[i]);
    }

    return true;
  }

  bool transform_unicode(UnicodeString &unicode_string) override { return true; }
};

REGISTER_KERNEL_BUILDER(Name("Miss>ReplaceRegex").Device(DEVICE_CPU), ReplaceRegexOp);

}  // end namespace miss
}  // namespace tensorflow
