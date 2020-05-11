#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class LowerCaseOp : public UnicodeTransformOp
{
public:
  explicit LowerCaseOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx) {}

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    unicode_string.toLower();

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>LowerCase").Device(DEVICE_CPU), LowerCaseOp);

}  // end namespace miss
}  // namespace tensorflow
