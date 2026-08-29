"""
The two array types this library's public surface speaks in.

Every model here takes a redshift and returns a number per redshift,
and both ends are deliberately loose about shape: you may pass a
Python float, a list, or an array, and what comes back matches.

Spelling that out once is what makes the rest of the annotations
honest. ``E(0.5)`` returns ``np.float64`` and ``E([0.1, 0.2])``
returns an ``ndarray`` -- ``np.asarray(z, dtype=float)`` on a scalar
gives a 0-d array, and numpy's ufuncs turn that back into a scalar on
the way out. Annotating the return as ``np.ndarray`` alone would be
wrong for the commonest call in the library.

>>> from CosmoFit.typing import Array, Redshift
>>>
>>> def E(self, z: Redshift) -> Array:
...     ...
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ArrayLike", "Array", "PathLike", "Redshift"]


#: A redshift, scale factor or wavenumber: anything numpy can make an
#: array of floats from -- a scalar, a sequence, or an array.
Redshift = ArrayLike

#: What comes back: a scalar for a scalar, an array for an array.
Array = np.float64 | NDArray[np.float64]

#: Anywhere a file is written or read: a string, or anything
#: implementing the os.PathLike protocol (`pathlib.Path`, most
#: obviously).
PathLike = str | os.PathLike
