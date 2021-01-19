"""Type aliases"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from typing import Callable, Iterable, Mapping, Optional, Sequence, Text, Tuple, Union

# Objects that can be treated like tensors (in TF2).
TensorLike = Union[np.ndarray, tf.Tensor, tf.Variable]

# Note that we have no way of statically verifying the tensor's shape.
BoolLike = Union[bool, np.bool, TensorLike]
IntegerLike = Union[int, np.integer, TensorLike]
FloatLike = Union[float, np.floating, TensorLike]

ActivationFn = Callable[[TensorLike], TensorLike]
Axis = Union[int, slice, Sequence[int]]