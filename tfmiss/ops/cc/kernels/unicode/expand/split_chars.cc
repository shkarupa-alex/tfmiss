#include "unicode_expand.h"

class SplitCharsOp : public UnicodeExpandOp
{
public:
  explicit SplitCharsOp(OpKernelConstruction *ctx) : UnicodeExpandOp(ctx) {}

protected:
  uint64 expand_rate() override
  {
    return (uint64)5; // Mean word length
  }

  bool expand_unicode(const UnicodeString &unicode_string, std::vector<UnicodeString> &expanded_strings) override
  {
    for (int32_t pos = 0; pos < unicode_string.length(); pos++)
    {
      UnicodeString character = UnicodeString(unicode_string, pos, 1);
      expanded_strings.push_back(character);
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("SplitChars").Device(DEVICE_CPU), SplitCharsOp);
