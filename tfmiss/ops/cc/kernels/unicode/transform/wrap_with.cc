#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class WrapWithOp : public UnicodeTransformOp
{
public:
  explicit WrapWithOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Prepare attrs
    string left;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left", &left));
    _left = UnicodeString::fromUTF8(left);

    string right;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right", &right));
    _right = UnicodeString::fromUTF8(right);
  }

private:
  UnicodeString _left;
  UnicodeString _right;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    unicode_string = _left + unicode_string + _right;

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>WrapWith").Device(DEVICE_CPU), WrapWithOp);

}  // end namespace miss
}  // namespace tensorflow
