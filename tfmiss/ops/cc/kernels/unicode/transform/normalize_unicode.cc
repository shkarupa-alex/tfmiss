#include <unicode/normalizer2.h>
#include "unicode_transform.h"

namespace tensorflow {
namespace miss {

class NormalizeUnicodeOp : public UnicodeTransformOp
{
public:
  explicit NormalizeUnicodeOp(OpKernelConstruction *ctx) : UnicodeTransformOp(ctx)
  {
    // Prepare attr
    string form;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("form", &form));
    std::transform(form.begin(), form.end(), form.begin(), ::toupper);

    // Create Normalizer2 instance
    UErrorCode error = U_ZERO_ERROR;
    if ("NFC" == form)
    {
      _normalizer = Normalizer2::getNFCInstance(error);
    }
    else if ("NFD" == form)
    {
      _normalizer = Normalizer2::getNFDInstance(error);
    }
    else if ("NFKC" == form)
    {
      _normalizer = Normalizer2::getNFKCInstance(error);
    }
    else if ("NFKD" == form)
    {
      _normalizer = Normalizer2::getNFKDInstance(error);
    }
    else
    {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unknown normalization form"));
    }
    OP_REQUIRES(ctx, U_SUCCESS(error),
                errors::Internal("Normalizer2 instantiation failed"));
  }

private:
  // Should not be deleted according to reference
  // http://icu-project.org/apiref/icu4c/classicu_1_1Normalizer2.html
  const Normalizer2 *_normalizer;

protected:
  bool transform_unicode(UnicodeString &unicode_string) override
  {
    UErrorCode error = U_ZERO_ERROR;
    unicode_string = _normalizer->normalize(unicode_string, error);

    return U_SUCCESS(error);
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>NormalizeUnicode").Device(DEVICE_CPU), NormalizeUnicodeOp);

}  // end namespace miss
}  // namespace tensorflow
