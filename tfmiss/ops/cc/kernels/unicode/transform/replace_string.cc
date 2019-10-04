#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class ReplaceStringOp : public UnicodeTransformOp
{
public:
  explicit ReplaceStringOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Prepare attrs
    std::vector<string> needle;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("needle", &needle));
    _needle.resize(needle.size());
    for (uint64 i = 0; i < needle.size(); i++)
    {
      _needle[i] = UnicodeString::fromUTF8(needle[i]);
      OP_REQUIRES(ctx, _needle[i].length() > 0,
                  errors::InvalidArgument("Items of \"needle\" could not be empty"));
    }

    std::vector<string> haystack;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("haystack", &haystack));
    _haystack.resize(haystack.size());
    for (uint64 i = 0; i < haystack.size(); i++)
    {
      _haystack[i] = UnicodeString::fromUTF8(haystack[i]);
    }

    OP_REQUIRES(ctx, _needle.size() == _haystack.size(),
                errors::InvalidArgument("Sizes are different for \"needle\" and \"haystack\""));
  }

private:
  std::vector<UnicodeString> _needle;
  std::vector<UnicodeString> _haystack;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    for (uint64 i = 0; i < _needle.size(); i++)
    {
      unicode_string.findAndReplace(_needle[i], _haystack[i]);
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>ReplaceString").Device(DEVICE_CPU), ReplaceStringOp);

}  // end namespace miss
}  // namespace tensorflow
