#include <unicode/unistr.h>
#include "unicode_transform.h"

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

REGISTER_KERNEL_BUILDER(Name("LowerCase").Device(DEVICE_CPU), LowerCaseOp);
