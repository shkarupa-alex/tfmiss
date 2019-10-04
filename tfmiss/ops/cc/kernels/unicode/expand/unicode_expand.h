#pragma once

#include <unicode/unistr.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace icu;

namespace tensorflow {
namespace miss {

class UnicodeExpandOp : public OpKernel
{
public:
  explicit UnicodeExpandOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx);

protected:
  virtual uint64 expand_rate();
  virtual bool expand_unicode(const UnicodeString &unicode_string, std::vector<UnicodeString> &expanded_strings) = 0;
};

}  // end namespace miss
}  // namespace tensorflow
