#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class SubStringOp : public UnicodeTransformOp
{
public:
  explicit SubStringOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Prepare attrs
    OP_REQUIRES_OK(ctx, ctx->GetAttr("start", &_start));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("limit", &_limit));
  }

private:
  int _start;
  int _limit;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    int start = _start >= 0 ? _start : unicode_string.length() + _start;
    start = std::max(0, std::min(unicode_string.length(), start));

    int limit = _limit >= 0 ? _limit : unicode_string.length();
    limit = std::min(unicode_string.length() - start, limit);

    unicode_string.extract(start, limit, unicode_string);

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>SubString").Device(DEVICE_CPU), SubStringOp);

}  // end namespace miss
}  // namespace tensorflow
