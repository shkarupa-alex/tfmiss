from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import threading

_tfmiss_so = None
_ops_lock = threading.Lock()

# preprocessing
tf.no_gradient('Miss>SampleMask')
tf.no_gradient('Miss>SkipGram')
tf.no_gradient('Miss>ContBow')

# unicode transform
tf.no_gradient('Miss>LowerCase')
tf.no_gradient('Miss>NormalizeUnicode')
tf.no_gradient('Miss>ReplaceRegex')
tf.no_gradient('Miss>ReplaceString')
tf.no_gradient('Miss>TitleCase')
tf.no_gradient('Miss>UpperCase')
tf.no_gradient('Miss>WrapWith')
tf.no_gradient('Miss>ZeroDigits')

# unicode expand
tf.no_gradient('Miss>CharNgrams')
tf.no_gradient('Miss>SplitWords')
tf.no_gradient('Miss>SplitChars')


def get_project_root():
    """Returns project root folder."""
    return


def _get_ops_path(ops_name):
    """Get the path to the specified file in the data dependencies.
    Args:
      ops_name: a string resource path relative to tfmiss/
    Returns:
      The path to the specified data file
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(curr_dir, ops_name)


def load_so():
    """Load tfmiss ops library and return the loaded module."""

    with _ops_lock:
        global _tfmiss_so
        if not _tfmiss_so:
            _tfmiss_so = tf.load_op_library(_get_ops_path('_tfmiss_ops.so'))
            assert _tfmiss_so, 'Could not load _tfmiss_ops.so'

    return _tfmiss_so
