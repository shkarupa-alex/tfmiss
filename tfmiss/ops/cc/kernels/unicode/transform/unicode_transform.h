#pragma once

#include <unicode/unistr.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace icu;
using namespace tensorflow;

class UnicodeTransformOp : public OpKernel
{
public:
  explicit UnicodeTransformOp(OpKernelConstruction *ctx) : OpKernel(ctx){};

  void Compute(OpKernelContext *ctx);

protected:
  virtual bool transform_raw(string &binary_string);
  virtual bool transform_unicode(UnicodeString &unicode_string) = 0;
};
