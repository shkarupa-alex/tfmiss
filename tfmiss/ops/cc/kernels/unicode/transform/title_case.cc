#include <unicode/brkiter.h>
#include <unicode/locid.h>
#include "unicode_transform.h"

class TitleCaseOp : public UnicodeTransformOp
{
public:
  explicit TitleCaseOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Create title-casing BreakIterator instance
    UErrorCode error = U_ZERO_ERROR;
    _titleIterator = BreakIterator::createTitleInstance(Locale::getRoot(), error);
    OP_REQUIRES(ctx, U_SUCCESS(error),
                errors::InvalidArgument("BreakIterator instantiation failed"));
  }

  ~TitleCaseOp()
  {
    delete _titleIterator;
  }

private:
  const BreakIterator *_titleIterator;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    BreakIterator *titleIterator = _titleIterator->clone();
    unicode_string.toTitle(titleIterator);
    delete titleIterator;

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("TitleCase").Device(DEVICE_CPU), TitleCaseOp);
