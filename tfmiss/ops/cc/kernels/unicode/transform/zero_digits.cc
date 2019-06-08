#include <unicode/uchar.h>
#include "unicode_transform.h"

class ZeroDigitsOp : public UnicodeTransformOp
{
public:
  explicit ZeroDigitsOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx) {}

private:
  const UChar _zero = 48; // Unicode 0

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    for (int32_t pos = 0; pos < unicode_string.length(); pos++)
    {
      if (u_isdigit(unicode_string.charAt(pos)))
      {
        unicode_string.replace(pos, 1, _zero);
      }
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroDigits").Device(DEVICE_CPU), ZeroDigitsOp);
