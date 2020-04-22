#pragma once

#include <unicode/unistr.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace icu;

namespace tensorflow {
namespace miss {

class UnicodeTransformOp : public OpKernel
{
public:
  explicit UnicodeTransformOp(OpKernelConstruction *ctx);
  void Compute(OpKernelContext *ctx);

protected:
  std::unordered_set<string> _skip;
  virtual bool transform_raw(string &binary_string);
  virtual bool transform_unicode(UnicodeString &unicode_string) = 0;
};

}  // end namespace miss
}  // namespace tensorflow
