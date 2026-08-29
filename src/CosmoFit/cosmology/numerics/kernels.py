"""
Optionally compiled inner loops.

Everything here has a NumPy implementation that is used when
`numba <https://numba.pydata.org>`_ is not installed, so this
module is an accelerator and never a requirement. ``pip install
"cosmofit[speed]"`` provides it.

Only one kernel so far, and it earns its place for a specific
reason. The linear growth equation is integrated by a fixed-step
RK4, which is *sequential*: each step needs the previous one, so
there is no array operation to hand NumPy. Two ways around that
were measured on the 300-step grid this library uses:

===========================================  ==========
stepping                                       per solve
===========================================  ==========
plain Python loop                              493 us
prefix product of 2x2 step matrices            215 us
**compiled loop (this module)**                **5 us**
===========================================  ==========

The middle one is what runs without numba. It exploits the equation
being linear -- one RK4 step is then a fixed 2x2 matrix, and a
product of matrices is a prefix scan, which pairwise doubling does
in log2(n) rounds of batched work. That is a real 2.3x, and it is
also the only reason the compiled path is optional rather than
load-bearing.

The compiled loop agrees with it to 1e-11, which a test asserts by
running both.
"""

from __future__ import annotations

import importlib.util

import numpy as np


# Whether numba is installed, answered without importing it.
#
# `import numba` costs ~115 ms, which is 17% of what `import CosmoFit`
# used to cost -- paid by every user who has the `speed` extra, on
# every import, for a kernel that only a growth fit ever calls. The
# spec lookup is a filesystem check and costs nothing.
HAVE_NUMBA = importlib.util.find_spec("numba") is not None


def _rk4_growth_python(friction_0, source_0, friction_1, source_1,
                       friction_2, source_2, step, initial):
    """
    Reference implementation, kept as the thing the compiled kernel
    is checked against rather than as a fallback -- when numba is
    absent the caller uses the prefix product instead, which is
    faster than this.
    """

    n = friction_0.shape[0]

    D = np.empty(n + 1)
    P = np.empty(n + 1)

    d = p = initial

    D[0] = d
    P[0] = p

    for i in range(n):

        k1d = p
        k1p = -friction_0[i] * p + source_0[i] * d

        k2d = p + 0.5 * step * k1p
        k2p = -friction_1[i] * k2d + source_1[i] * (d + 0.5 * step * k1d)

        k3d = p + 0.5 * step * k2p
        k3p = -friction_1[i] * k3d + source_1[i] * (d + 0.5 * step * k2d)

        k4d = p + step * k3p
        k4p = -friction_2[i] * k4d + source_2[i] * (d + step * k3d)

        d += step / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        p += step / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

        D[i + 1] = d
        P[i + 1] = p

    return D, P


_compiled = None


def rk4_growth(*args):
    """
    The compiled kernel, imported and JIT-decorated on first call.

    Callers must check :data:`HAVE_NUMBA` first -- this raises
    ``ImportError`` without it, deliberately, rather than silently
    running the uncompiled loop, which is *slower* than the prefix
    product the caller would otherwise have used.
    """

    global _compiled

    if _compiled is None:

        # `cache=True` writes the compiled kernel next to the module,
        # so the ~300 ms of LLVM work happens once per install rather
        # than once per process. numba degrades to recompiling if the
        # directory is not writable.
        from numba import njit

        _compiled = njit(cache=True)(_rk4_growth_python)

    return _compiled(*args)
