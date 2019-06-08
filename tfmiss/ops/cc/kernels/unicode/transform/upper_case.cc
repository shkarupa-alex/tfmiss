#include "unicode_transform.h"

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

REGISTER_KERNEL_BUILDER(Name("UpperCase").Device(DEVICE_CPU), UpperCaseOp);
