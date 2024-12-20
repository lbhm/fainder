from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FArray = NDArray[np.floating[Any]]
F16Array = NDArray[np.float16]
F32Array = NDArray[np.float32]
F64Array = NDArray[np.float64]
UInt32Array = NDArray[np.uint32]

Histogram = tuple[UInt32Array | F32Array, F64Array]

ConversionIndex = tuple[tuple[FArray, UInt32Array], tuple[FArray, UInt32Array]]
RebinningIndex = tuple[tuple[FArray, UInt32Array]]
PercentileIndex = ConversionIndex | RebinningIndex
PercentileIndexPointer = Sequence[tuple[SharedMemory, SharedMemory]]

PercentileQuery = tuple[float, Literal["le", "lt", "ge", "gt"], float]
