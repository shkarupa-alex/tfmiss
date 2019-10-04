#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class UpperCaseOp : public UnicodeTransformOp
{
public:
  explicit UpperCaseOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx) {}

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    unicode_string.toUpper();

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>UpperCase").Device(DEVICE_CPU), UpperCaseOp);

}  // end namespace miss
}  // namespace tensorflow
