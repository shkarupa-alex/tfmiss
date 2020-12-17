#include "unicode_transform.h"
#include <unicode/uchar.h>

namespace tensorflow {
namespace miss {

class CharCategoryOp : public UnicodeTransformOp
{
public:
  explicit CharCategoryOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx) {
    // Prepare attrs
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first", &_first));
  }

private:
  bool _first;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    uint8 category = 0;
    if (unicode_string.length()) {
      if (_first) {
        category = u_charType(unicode_string[0]);
      } else {
        category = u_charType(unicode_string[unicode_string.length() - 1]);
      }
    }

    unicode_string = u_getPropertyValueName(UCHAR_GENERAL_CATEGORY, category, U_SHORT_PROPERTY_NAME);

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>CharCategory").Device(DEVICE_CPU), CharCategoryOp);

}  // end namespace miss
}  // namespace tensorflow
