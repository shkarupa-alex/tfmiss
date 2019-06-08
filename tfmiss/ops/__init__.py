from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from tensorflow.python.framework import load_library, ops
from tensorflow.python.platform import resource_loader

_tfmiss_so = None
_ops_lock = threading.Lock()

# preprocessing
ops.NotDifferentiable('SampleMask')
ops.NotDifferentiable('SkipGram')
ops.NotDifferentiable('ContBow')

# unicode transform
ops.NotDifferentiable('LowerCase')
ops.NotDifferentiable('NormalizeUnicode')
ops.NotDifferentiable('ReplaceRegex')
ops.NotDifferentiable('ReplaceString')
ops.NotDifferentiable('TitleCase')
ops.NotDifferentiable('UpperCase')
ops.NotDifferentiable('WrapWith')
ops.NotDifferentiable('ZeroDigits')

# unicode expand
ops.NotDifferentiable('CharNgrams')
ops.NotDifferentiable('SplitWords')
ops.NotDifferentiable('SplitChars')


def load_so():
    """Load tfmiss ops library and return the loaded module."""

    with _ops_lock:
        global _tfmiss_so
        if not _tfmiss_so:
            _tfmiss_so = load_library.load_op_library(
                resource_loader.get_path_to_datafile('_ops.so'))
            assert _tfmiss_so, 'Could not load _ops.so'

    return _tfmiss_so
